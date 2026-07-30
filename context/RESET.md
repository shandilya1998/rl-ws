# RESET.md — Investigation: Flat `Episode_Termination/*` plots under `CoptOnPolicyRunner`

> **Scope.** Why do all `Episode_Termination/<term>` TensorBoard curves collapse to flat
> horizontal lines (`base_contact ≈ 1.0`, `time_out ≈ 0.0`) when training with
> `CoptOnPolicyRunner` (grey line in `termination.png`), while the same metrics evolve
> normally under `OnPolicyRunner` (pink/blue lines)?
>
> This document traces every reset pathway end‑to‑end for both runners, compares how the
> termination statistic is computed in each case, correlates the result to the observed
> plots, pinpoints the root cause, audits the design‑update reset pathway for bugs, and
> proposes fixes.

---

## 0. TL;DR — Root cause

The flat lines are **a correct statistic of a degenerate event distribution**, not a logging
bug and not a stale/overwritten value. Two facts combine:

1. **Timeouts are physically unreachable under COPT.** `_update_morphology()` calls
   `unwrapped_env.reset()` every `ea_update_interval=10` iterations
   (`10 × num_steps_per_env(24) = 240` control steps). That full reset zeroes
   `episode_length_buf` for **all** envs. The `time_out` termination only fires when
   `episode_length_buf >= max_episode_length = 1000`. Because `240 ≪ 1000`, no episode
   ever survives long enough to time out, so `time_out` **never fires** → its column is
   always `0`.

2. **The upstream `_last_episode_dones` buffer saturates.** Isaac Lab logs
   `self._last_episode_dones.float().mean(dim=0)` — a *persistent per‑env* record of
   *which term most recently terminated each env*, averaged over **all** envs. A row only
   flips when a *new* term fires for that env. Under COPT the only term that ever fires is
   `base_contact`, so once every env has fallen at least once (which happens almost
   immediately with an early/un‑trained policy), **every row is frozen at
   `[base_contact=1, time_out=0]`** and can never change → `base_contact ≡ 1.0`,
   `time_out ≡ 0.0`, forever.

The design‑swap `env.reset()` is the originating cause: it is a **silent reset** that
bypasses `TerminationManager.compute()`. It therefore (a) is invisible to the termination
statistics, and (b) starves the `time_out` pathway by continually zeroing the episode
clock. The statistics *are* computed and logged every iteration; they are simply pinned to
`1.0 / 0.0` by the above.

> **"Not updated" vs "overwritten"?** Neither. The per‑step `_term_dones` buffer *is*
> updated (for `base_contact`), the persistent `_last_episode_dones` *is* updated for
> fallen envs, and the scalar *is* written to TensorBoard every iteration. The value is a
> genuine `1.0 / 0.0`. The degeneracy comes from the reset **cadence**, amplified by the
> **monotonic saturation** of the persistent buffer.

---

## 1. The metric and how it is computed

### 1.1 `TerminationManager.compute()` — runs once per `step()`
`IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py:154‑182`

```python
def compute(self) -> torch.Tensor:
    self._truncated_buf[:] = False
    self._terminated_buf[:] = False
    for i, term_cfg in enumerate(self._term_cfgs):
        value = term_cfg.func(self._env, **term_cfg.params)
        if term_cfg.time_out:
            self._truncated_buf |= value      # time_out → truncation
        else:
            self._terminated_buf |= value     # base_contact → termination
        self._term_dones[:, i] = value        # per-step buffer
    # persistent "last episode" buffer — only rows that fired this step are touched
    rows = self._term_dones.any(dim=1).nonzero(as_tuple=True)[0]
    if rows.numel() > 0:
        self._last_episode_dones[rows] = self._term_dones[rows]
    return self._truncated_buf | self._terminated_buf
```

Key property: **`_last_episode_dones` is *only* written here, and only for envs that
actually fired a term this step.** A full `env.reset()` does **not** call `compute()`, so it
never updates this buffer.

### 1.2 `TerminationManager.reset()` — produces the logged extras
`IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py:129‑152`

```python
def reset(self, env_ids=None):
    extras = {}
    last_episode_done_stats = self._last_episode_dones.float().mean(dim=0)   # mean over ALL envs
    for i, key in enumerate(self._term_names):
        extras["Episode_Termination/" + key] = last_episode_done_stats[i].item()
    ...
    return extras
```

- The value is the **fraction of *all* envs whose most recent termination was term `i`**.
- It is **independent of `env_ids`** (it always averages the whole buffer).
- This is upstream Isaac Lab **PR #3745** (the `_last_episode_dones` dual‑buffer design),
  *not* a fork modification — verified: the local file matches the submodule `HEAD`
  (`7cf0f95659`) with no local diff. **The same code serves both runners.**

### 1.3 Termination terms & episode length (this project)
`exts/.../tasks/locomotion/cfg/SF/limx_base_env_cfg.py:1167‑1178`

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)              # truncation
    base_contact = DoneTerm(                                          # termination
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
                "threshold": 1.0},
    )
```

`IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:31‑33`
```python
def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length
```

Episode length math (`limx_base_env_cfg.py`):
`episode_length_s = 20.0`, `decimation = 4`, `sim.dt = 0.005` →
```
max_episode_length = ceil(20.0 / (4 × 0.005)) = ceil(1000) = 1000 control steps
```
(Note: `../ARCHITECTURE.md` says 40 s; the live config is 20 s. Either way the timeout
horizon — 1000 or 2000 steps — is far above the 240‑step COPT reset window.)

### 1.4 The logging pipeline (rsl‑rl)
`rsl_rl/rsl_rl/runners/on_policy_runner.py`

- Per step, the env's `extras["log"]` dict is appended to `ep_infos`
  (`copt_on_policy_runner.py:178‑181`, identical to base `on_policy_runner.py:115‑118`):
  ```python
  if "episode" in extras: ep_infos.append(extras["episode"])
  elif "log" in extras:   ep_infos.append(extras["log"])
  ```
- `log()` (`on_policy_runner.py:176‑205`) averages each key across the whole rollout and
  writes keys containing `/` directly:
  ```python
  for key in locs["ep_infos"][0]:
      infotensor = torch.cat([... ep_info[key] ...])
      value = torch.mean(infotensor)
      if "/" in key:
          self.writer.add_scalar(key, value, locs["it"])   # "Episode_Termination/time_out"
  ```
- `CoptOnPolicyRunner` **inherits this `log()` verbatim** and calls `self.log(locals())`
  (`copt_on_policy_runner.py:266`). So both runners log `Episode_Termination/*` the same way.

### 1.5 Where `extras["log"]` is born
`IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:_reset_idx (349‑394)`

```python
self.extras["log"] = dict()                          # rebuilt ONLY inside _reset_idx
...
info = self.termination_manager.reset(env_ids)
self.extras["log"].update(info)                      # Episode_Termination/* injected here
...
self.episode_length_buf[env_ids] = 0                 # episode clock zeroed for reset envs
```

`_reset_idx` is only reached when `reset_env_ids` is non‑empty
(`manager_based_rl_env.py:216‑221`). On steps with no reset, `extras["log"]` keeps its
**previous** value, so rsl‑rl re‑averages stale (but in‑range `[0,1]`) numbers — see
upstream Issue #2977. Under COPT this only reinforces the flatness; it is not the cause.

---

## 2. Reset pathway traces

There are three reset pathways. Pathways **A** and **B** exist for both runners; pathway
**C** is COPT‑only.

### 2.1 Pathway A — Reset on **base contact** (both runners)

```
robot base touches ground
  └─ env.step()  (manager_based_rl_env.py:153‑241)
       ├─ termination_manager.compute()                         # line 204
       │     ├─ illegal_contact(...) → True for fallen envs
       │     ├─ _terminated_buf |= value                        # base_contact is time_out=False
       │     ├─ _term_dones[:, base_contact] = True
       │     └─ _last_episode_dones[fallen] = [base_contact=1, time_out=0]   # row flip
       ├─ reset_env_ids = reset_buf.nonzero()  → fallen envs    # line 216
       └─ _reset_idx(fallen)                                    # line 221
             ├─ extras["log"] = {}                              # rebuilt
             ├─ termination_manager.reset() →
             │     extras["log"]["Episode_Termination/base_contact"] = mean_over_all_envs
             │     extras["log"]["Episode_Termination/time_out"]     = mean_over_all_envs
             └─ episode_length_buf[fallen] = 0
  └─ runner appends extras["log"] → ep_infos → log() → TensorBoard
```
**Identical for both runners.** This pathway *does* update `_last_episode_dones` and *does*
emit the stat every time a robot falls.

### 2.2 Pathway B — Reset on **time out** (the divergence)

`time_out` fires only when `episode_length_buf >= 1000`.

- **`OnPolicyRunner`:** episodes run uninterrupted. Survivors reach 1000 steps → `time_out`
  fires → `_truncated_buf`, `_term_dones[:, time_out]=True`,
  `_last_episode_dones[survivors] = [base_contact=0, time_out=1]` (**row flips the other
  way**). As the policy improves, more envs reach timeout → `time_out` fraction rises,
  `base_contact` fraction falls. **→ exactly the pink/blue curves.**

- **`CoptOnPolicyRunner`:** `episode_length_buf` is zeroed for **all** envs every 240 steps
  by pathway C (below). It can climb to at most 240 before being wiped — **never** to 1000.
  Therefore `time_out` **never fires**, no row ever flips back to `[0,1]`, and the
  `time_out` column stays `0` while `base_contact` saturates to `1`. **→ exactly the grey
  flat lines.**

> This single difference — episodes can or cannot reach the timeout horizon — fully
> explains the divergence between grey and pink/blue.

### 2.3 Pathway C — Reset on **design update** (COPT only)

`copt_on_policy_runner.py:learn (100‑287)`

```python
# once, before the loop
self._update_morphology()                 # line 117 → unwrapped_env.reset() zeroes ALL clocks

for it in range(...):
    is_morph_update_time = ((it+1-start_iter) % self._ea_update_interval == 0) and ...   # 155
    # ... 24-step rollout ...
    if is_morph_update_time:               # every 10th iteration
        self._update_morphology()          # 244
        obs = self.env.get_observations()  # 246
```

`_update_morphology()` (`383‑421`):
```python
sim.stop()                                              # Fabric off
self._design_generator.update_with_fitness(...) / sample_batch()
self.current_population = generate_population(...)
apply_link_length_params(unwrapped_env, ...)            # edits USD + ONE sim.reset() (physics cook)
unwrapped_env.reset()                                   # line 413 — FULL env reset, ALL envs
apply_actuator_params(unwrapped_env, ...)
self._individual_fitness.zero_(); self._individual_episode_counts.zero_()
```

What the design‑update reset does to termination state:

| Action | Effect on termination stats |
|---|---|
| `unwrapped_env.reset()` → `_reset_idx(all)` | zeroes **all** `episode_length_buf` → restarts the timeout clock from 0 for every env |
| `_reset_idx` → `termination_manager.reset()` | emits `extras["log"]["Episode_Termination/*"]` = current saturated `1.0 / 0.0` |
| **does NOT call `compute()`** | `_last_episode_dones` is **not** updated → the forced reset is a *silent* termination, recorded as no term at all |

**Confirmed by code audit of the COPT utilities** (`co_optimisation/utils/`):
`apply_link_length_params` (`update.py`) only edits USD layers + a single `sim.reset()`
(physics cook, *not* an env reset); `apply_actuator_params` (`respawn.py`) only overwrites
actuator tensors; `analysis.py` is read‑only diagnostics. **None of them touch
`episode_length_buf`, the termination manager, `reset_buf`, or `extras`.** The only
env‑level reset in the in‑place cycle is the runner's own `unwrapped_env.reset()` at line
413. So COPT does **not** zero/clear the termination buffers itself — it starves them by
controlling the reset cadence.

> Side note: `init_at_random_ep_len=True` randomizes `episode_length_buf` at
> `learn()` start (`copt_on_policy_runner.py:107‑110`), but the very next line region calls
> the initial `_update_morphology()` (117) whose `env.reset()` immediately **zeroes** those
> random values. So even the desync trick that would normally let some envs time out early
> is wiped out under COPT.

---

## 3. Comparison & correlation to the plot

| Pathway | `OnPolicyRunner` | `CoptOnPolicyRunner` |
|---|---|---|
| **A. base_contact** | fires often early, less as policy improves | fires constantly (only event that ever happens) |
| **B. time_out** | fires increasingly as policy survives to 1000 steps | **never fires** (clock wiped every 240 steps) |
| **C. design reset** | n/a | every 240 steps; silent (bypasses `compute()`) |
| `_last_episode_dones` rows | flip both ways (`base_contact`↔`time_out`) | flip **one way only** → saturate at `[1,0]` |
| `Episode_Termination/base_contact` | decays `~1.0 → ~0.15` | **flat `≈1.0`** |
| `Episode_Termination/time_out` | rises `~0.0 → ~0.85` | **flat `≈0.0`** |

This is precisely `termination.png`: pink/blue (`OnPolicyRunner`) sweep from
fall‑dominated to timeout‑dominated as the policy learns; grey (`CoptOnPolicyRunner`) is
pinned because the timeout event is unreachable and the persistent buffer is saturated.

---

## 4. Root‑cause analysis — "not updated" vs "overwritten"

**Verdict: neither.** Decomposed:

1. **Are the stats computed/logged each iteration?** Yes. Every time a robot falls
   (pathway A), `_reset_idx` rebuilds `extras["log"]` with `Episode_Termination/*`, the
   runner appends it to `ep_infos`, and `log()` writes it to TensorBoard. The scalar is
   emitted every iteration.

2. **Are they overwritten by a COPT bug?** No. The COPT utilities never write
   `episode_length_buf` or the termination buffers (audited). The values are genuine.

3. **So why flat?** Two compounding upstream/structural causes:
   - **(Primary, originating) cause:** the COPT design‑swap reset (pathway C) fires every
     240 steps, far below the 1000‑step timeout horizon, so the `time_out` pathway (B) is
     **starved** — it can never fire. With only `base_contact` ever firing, the *true*
     per‑episode outcome distribution is degenerate (100% falls, 0% timeouts).
   - **(Amplifying) cause:** Isaac Lab's persistent `_last_episode_dones` buffer is
     **monotonic** in the absence of competing terms — a row set to `base_contact=1` never
     resets until a *different* term fires. Since `time_out` never fires, the buffer freezes
     at `[1,0]` for every env, making the curve a perfectly flat line rather than a noisy
     one near 1.0.

The design‑swap reset is the lever: it is a **silent truncation** that is invisible to the
statistics *and* prevents the only event (timeout) that would otherwise diversify them.

---

## 5. Secondary bugs / issues in the design‑update pathway

These are distinct from the flat‑plot symptom but were uncovered while tracing pathway C
and are worth fixing alongside it.

1. **Missing truncation bootstrap (value‑function correctness bug).**
   The forced `env.reset()` at `copt_on_policy_runner.py:413` truncates all in‑flight
   episodes mid‑trajectory **without** signalling a time‑out/truncation to PPO. The
   transitions at the boundary are treated as ordinary (non‑terminal) and the *next*
   rollout starts from a fresh, unrelated state, so GAE/return computation at that boundary
   is corrupted. Per Pardo et al., *Time Limits in RL* (ICML 2018, arXiv:1712.00378),
   external time‑limit/truncation boundaries must bootstrap `V(s')` (partial‑episode
   bootstrapping); conflating them with terminals biases value targets. The design swap is
   exactly such an external truncation and currently gets no bootstrap.

2. **Design‑swap resets are uncounted.** Every 240 steps all 4096 envs are reset, but none
   of these resets is reflected in any `Episode_Termination/*` term (no `compute()` call).
   The logs therefore hide a large fraction of the actual episode endings.

3. **Reward bookkeeping skip on the morph‑update step.**
   `copt_on_policy_runner.py:194` guards the per‑step bookkeeping with
   `if (step < self.num_steps_per_env - 1) or (not is_morph_update_time):`, and the
   post‑update block (`247‑261`) then folds the *entire* `cur_reward_sum` of all envs into
   `rewbuffer`/fitness as if every env completed an episode. This double‑purposes
   `cur_reward_sum` (a running, not per‑episode, sum) as a fitness signal and can bias both
   the reported `Mean/reward` and the EA fitness. (Not the cause of the termination plot,
   but a correctness concern for the co‑optimisation objective.)

4. **`init_at_random_ep_len` is neutralised** (see §2.3 side note) — the initial
   `_update_morphology()` zeroes the randomized clocks, so all episodes are phase‑locked at
   start of training. Minor, but removes the intended desync.

---

## 6. Recommended solutions

Ranked. The first two address the root cause; the rest improve correctness/observability.

### S1 — Make the `time_out` event reachable (choose one; design decision)
The metric is only meaningful if episodes *can* end in both ways. Resolve the cadence
mismatch `240 (reset window) ≪ 1000 (timeout)`:

- **S1a — Shorten the COPT episode horizon** so `max_episode_length < 240`. e.g. set
  `episode_length_s ≈ 4.0 s` (→ 200 steps) for COPT tasks. Natural timeouts then occur
  *within* each morphology generation, restoring a meaningful `time_out` curve and proper
  per‑design episodic evaluation. Trade‑off: shorter episodes evaluate each design over a
  shorter behaviour window.
- **S1b — Lengthen the morphology evaluation window** so episodes can complete: increase
  `ea_update_interval` (e.g. `≥ ⌈1000/24⌉ = 42`) or otherwise reduce reset frequency.
  Trade‑off: fewer EA generations per wall‑clock, slower design search.

> Recommendation: **S1a** for fast co‑design iteration (keep the EA cadence, shrink the
> episode), unless the locomotion task genuinely needs 20 s episodes — then **S1b**.

### S2 — Treat the design swap as an explicit truncation (correctness + observability)
Before `unwrapped_env.reset()` in `_update_morphology()`:
- Flag all envs as timed‑out/truncated so PPO bootstraps the value at the boundary
  (set `reset_time_outs`/`extras["time_outs"]=True` for all envs, or run a final
  `termination_manager.compute()`/manual truncation pass), fixing issue §5.1.
- Optionally record the swap as its own term, e.g. emit
  `Episode_Termination/design_update` so the plot shows the real ending distribution
  (fixes §5.2). This makes the design‑reset visible instead of silent.

### S3 — (Optional) make the metric robust to the degenerate case
If you want the curve to reflect *recent* terminations rather than a saturating
"last‑ever" buffer, log over only the envs reset in the current step
(`_last_episode_dones[env_ids]`/`_term_dones[env_ids]`) rather than `mean(dim=0)` over all
envs. **Caution:** this changes upstream (PR #3745) behaviour for *all* runs; prefer S1+S2
first. Not required once timeouts are reachable again.

### S4 — Fix the morph‑update reward/fitness bookkeeping (§5.3)
Accumulate per‑individual fitness from *completed‑episode* returns only (use the same
`new_ids` masking as the normal path), instead of folding the whole running
`cur_reward_sum` on the morph step. Keeps EA fitness and `Mean/reward` unbiased.

### S5 — Quick diagnostic to confirm
Log `episode_length_buf.float().mean()`/`.max()` and the raw `time_outs.sum()` per
iteration. Under the current setup you should see episode lengths capped at ~240 and
`time_outs == 0`, directly confirming §2.2/§0.

---

## 7. Appendix — key references

**Code**
- `IsaacLab/.../managers/termination_manager.py:154‑182` — `compute()` (per‑step + `_last_episode_dones`)
- `IsaacLab/.../managers/termination_manager.py:129‑152` — `reset()` (`mean(dim=0)` over all envs)
- `IsaacLab/.../envs/mdp/terminations.py:31‑33` — `time_out = episode_length_buf >= max_episode_length`
- `IsaacLab/.../envs/mdp/terminations.py:154‑162` — `illegal_contact` (base_contact)
- `IsaacLab/.../envs/manager_based_rl_env.py:199‑241` — `step()` (compute → reset_idx)
- `IsaacLab/.../envs/manager_based_rl_env.py:349‑394` — `_reset_idx` (`extras["log"]`, `episode_length_buf=0`)
- `IsaacLab/.../envs/manager_based_env.py:341‑393` — `reset()` (env_ids=None → all envs)
- `cfg/SF/limx_base_env_cfg.py:1167‑1178` — `TerminationsCfg`; `:1324‑1328` — dt/decimation/episode_length
- `rsl_rl/.../on_policy_runner.py:115‑118, 176‑205` — `ep_infos` build + `log()` averaging
- `co_optimisation/.../copt_on_policy_runner.py:100‑287` — `learn()`; `:383‑421` — `_update_morphology()` (`env.reset()` @413)
- `co_optimisation/utils/update.py` (`apply_link_length_params`), `utils/respawn.py` (`apply_actuator_params`) — audited; do **not** touch termination/episode buffers

**External**
- Isaac Lab PR #3745 — `_last_episode_dones` dual‑buffer + `float().mean()` logging (the code in use)
- Isaac Lab PR #3107 — move off raw `count_nonzero` termination logging
- Isaac Lab Issue #2977 — stale `extras["log"]` re‑averaging / termination over‑counting
- Pardo et al., *Time Limits in Reinforcement Learning*, ICML 2018 — arXiv:1712.00378 (truncation bootstrapping)
