# rsl_rl.md — Agent A findings (rsl_rl library internals)

All citations are to the installed library under `/ws/rsl_rl/rsl_rl/` and to the
COPT runner at
`/ws/tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`.
This is the newer TensorDict / `obs_groups` API, not the older
`(num_actor_obs, num_critic_obs)` signatures described in `../CO_OPTIMISATION.md`
section 2. Discrepancies with the existing docs are flagged inline with the tag
DISCREPANCY.

Grounding config (the real COPT run): `SFCoptPPORunnerCfg`
(`exts/.../agents/limx_rsl_rl_ppo_cfg.py:131`) inherits
`SF_TRON1AFlatPPORunnerCfg` (`:92`). Confirmed from a saved run
`/ws/IsaacLab/logs/rsl_rl/sf_copt/2026-06-25_07-39-47/params/agent.yaml`:
`num_steps_per_env=25`, `empirical_normalization=false`,
`actor_obs_normalization`/`critic_obs_normalization` MISSING (default `False`),
`policy.class_name=CoptActorCritic`, `algorithm.class_name=PPO`, `rnd_cfg=null`,
`symmetry_cfg=null`, `normalize_advantage_per_mini_batch=false`,
`schedule=adaptive`, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`,
`clip_param=0.2`, `entropy_coef=0.005`, `value_loss_coef=1.0`,
`use_clipped_value_loss=true`, `learning_rate=1e-3`, `num_learning_epochs=5`,
`num_mini_batches=4`, `max_grad_norm=1.0`. `num_envs=4096` (scene cfg).

---

## 1. OnPolicyRunner (`runners/on_policy_runner.py`)

### 1.1 Constructor (`__init__`, lines 26-60)
Member variables set, in order:

- `self.cfg = train_cfg` (`:27`) — the full training dict.
- `self.alg_cfg = train_cfg["algorithm"]` (`:28`).
- `self.policy_cfg = train_cfg["policy"]` (`:29`).
- `self.device` (`:30`), `self.env` (`:31`).
- `_configure_multi_gpu()` (`:34`) sets `self.gpu_world_size`,
  `self.is_distributed`, `self.gpu_local_rank`, `self.gpu_global_rank`,
  `self.multi_gpu_cfg` (`:351-393`). For COPT, single GPU →
  `is_distributed=False`, ranks 0, `multi_gpu_cfg=None`.
- `self.num_steps_per_env = self.cfg["num_steps_per_env"]` (`:37`) → 25.
- `self.save_interval = self.cfg["save_interval"]` (`:38`).
- `obs = self.env.get_observations()` (`:41`) — a `TensorDict`.
- `default_sets = ["critic"]`, plus `"rnd_state"` only if `rnd_cfg` present
  (`:42-44`). For COPT only `["critic"]`.
- `self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)`
  (`:45`). DISCREPANCY with the real agent.yaml: `obs_groups` is a
  `dataclasses._MISSING_TYPE` there, so resolution falls back on group-name
  matching (see `utils/utils.py:203-262`); the `policy` and `critic` groups are
  inferred from the env observation keys, not from an explicit config.
- `self.alg = self._construct_algorithm(obs)` (`:48`) — builds policy + PPO +
  storage. COPT overrides this (see 1.7).
- `self.disable_logs` (`:52`).
- `self.log_dir = log_dir` (`:55`).
- `self.writer = None` (`:56`).
- `self.tot_timesteps = 0` (`:57`).
- `self.tot_time = 0` (`:58`).
- `self.current_learning_iteration = 0` (`:59`).
- `self.git_status_repos = [rsl_rl.__file__]` (`:60`).

Note `self.logger_type` is NOT set here; it is created lazily in
`_prepare_logging_writer()` (`:436-458`, sets `self.logger_type` at `:440`).

### 1.2 `learn()` (lines 62-174)
- `rewbuffer = deque(maxlen=100)` and `lenbuffer = deque(maxlen=100)` created
  fresh per `learn()` call (`:78-79`). These are LOCAL to `learn()`, so on a
  resumed run they start empty. They are passed into `log()` via `locals()`.
- `cur_reward_sum`, `cur_episode_length`: per-env float tensors of size
  `num_envs` (`:80-81`).
- Rollout loop (`:98-145`): for each of `num_steps_per_env` steps
  (`:102`): `actions=self.alg.act(obs)` (`:104`),
  `obs,rewards,dones,extras=self.env.step(...)` (`:106`),
  `self.alg.process_env_step(obs,rewards,dones,extras)` (`:110`).
- Bookkeeping (`:113-138`): `cur_reward_sum += rewards`,
  `cur_episode_length += 1`. On done, `new_ids=(dones>0).nonzero(...)` (`:129`),
  then `rewbuffer.extend(cur_reward_sum[new_ids][:,0]...)` and
  `lenbuffer.extend(cur_episode_length[new_ids][:,0]...)` (`:130-131`), then
  `cur_reward_sum[new_ids]=0`, `cur_episode_length[new_ids]=0` (`:132-133`).
  ONLY completed (done) episodes feed `rewbuffer`. This is the canonical
  rolling window of the last 100 completed-episode returns.
- `self.alg.compute_returns(obs)` (`:145`) at rollout end (inside
  `torch.inference_mode()`), then `loss_dict = self.alg.update()` (`:148`).
- `self.current_learning_iteration = it` (`:152`).
- `self.log(locals())` (`:156`); save every `it % save_interval == 0` (`:158`).
- Final save after the loop (`:173-174`).

### 1.3 COPT `learn()` override — exact differences
(`copt_on_policy_runner.py:105-320`)

1. Pre-loop (`:120-125`): creates `self.respawn_robots`, makes `designs/` dir,
   and calls `self._update_morphology(0)` BEFORE the first rollout. Base does
   none of this.
2. Per iteration it computes `is_late_start_toggle_time` (`:163`) and
   `is_morph_update_time = ((it - start_iter + 1) % ea_update_interval == 0) and (...)`
   (`:164-166`).
3. Inside the rollout, the done-bookkeeping block is GUARDED by
   `if (step < self.num_steps_per_env - 1) or (not is_morph_update_time):`
   (`:203-205`). On the LAST step of a morph-update iteration the normal
   done-path is SKIPPED; partial returns are flooded in afterwards instead.
   When the block runs it additionally accumulates per-individual fitness
   (`:206-214`: `self._individual_fitness[ind_idx] += cur_reward_sum[env_idx]`,
   `self._individual_episode_counts[ind_idx] += 1`).
4. After `self.alg.update()`, if `is_morph_update_time` (`:256-294`): optional
   late-start toggle (`:257-265`), `self._update_morphology(...)` (`:266-267`),
   `obs = self.env.get_observations()` refresh (`:269`), then the KEY
   variance-relevant block (`:270-284`): for EVERY env it adds
   `cur_reward_sum[env_idx]` to its individual fitness AND
   `rewbuffer.extend(cur_reward_sum.cpu()...)`,
   `lenbuffer.extend(cur_episode_length.cpu()...)` over ALL envs (terminated or
   not), then zeroes them. Finally it writes `link_lengths_{it}.csv` (`:289-294`).

   Consequence (objective 1): every `ea_update_interval=120` iters the
   100-slot `rewbuffer` is overwritten by partial, not-yet-terminated returns
   of all `num_envs=4096` envs (only the last 100 of the 4096 survive the
   deque). `Train/mean_reward` at that iteration is therefore the mean of 100
   TRUNCATED returns, producing a periodic downward/distorted spike. See 5.1.

5. COPT does NOT override `save`, `load`, or `log`. It inherits the base ones
   verbatim. (Important for objective 2 — see 6.)

### 1.4 `log()` (lines 176-289) — exact scalar computations
- `collection_size = num_steps_per_env * num_envs * gpu_world_size` (`:178`).
- `self.tot_timesteps += collection_size` (`:180`);
  `self.tot_time += collection_time + learn_time` (`:181`).
- Episode infos (`:186-205`): for each key in `ep_infos[0]`, concatenates the
  per-step tensors over the whole iteration and takes `torch.mean` (`:198`).
  Keys containing `/` are logged under their own name (`:201`); others under
  `Episode/<key>` (`:204`). These are the `Episode/rew_*`, curriculum, and
  termination scalars in the plots.
- `mean_std = self.alg.policy.action_std.mean()` (`:207`). Logged as
  `Policy/mean_noise_std` (`:216`). For COPT `action_std = exp(log_std)`
  broadcast over the batch (`copt_actor_critic.py:136`).
- Losses (`:211-213`): iterates `loss_dict` and writes `Loss/<key>` for each
  key. PPO returns `value_function`, `surrogate`, `entropy`
  (`ppo.py:407-411`). Also writes `Loss/learning_rate = self.alg.learning_rate`
  (`:213`) — the live PPO attribute, not the optimizer param-group lr.
- Perf scalars `Perf/total_fps`, `Perf/collection time`, `Perf/learning_time`
  (`:219-221`).
- Training scalars, only `if len(rewbuffer) > 0` (`:224`):
  - `Train/mean_reward = statistics.mean(rewbuffer)` (`:231`) — arithmetic mean
    over the (up to) 100-entry deque of the most recent COMPLETED-episode
    returns (rolling window). For COPT, contaminated by the partial-return
    flood (1.3 item 4).
  - `Train/mean_episode_length = statistics.mean(lenbuffer)` (`:232`).
  - `Train/mean_reward/time` and `.../mean_episode_length/time` keyed by
    `self.tot_time` (`:234-237`), skipped for wandb.
- DISCREPANCY (objective 1): there is NO `explained_variance` or value-fit
  diagnostic anywhere in rsl_rl (grep confirms only `value_function` loss). The
  CONTEXT open question assumes explained variance is logged; it is not. The
  only critic diagnostic available is `Loss/value_function` (mean clipped value
  loss, see 2.4).

### 1.5 `save(path, infos)` (lines 291-307)
`saved_dict` written by `torch.save` contains EXACTLY:
- `"model_state_dict": self.alg.policy.state_dict()` (`:294`).
- `"optimizer_state_dict": self.alg.optimizer.state_dict()` (`:295`).
- `"iter": self.current_learning_iteration` (`:296`).
- `"infos": infos` (`:297`) — `None` in every COPT `save()` call
  (`copt_on_policy_runner.py:302,316` pass no infos).
- If `self.alg.rnd` truthy: `"rnd_state_dict"` and
  `"rnd_optimizer_state_dict"` (`:300-302`). NOT present for COPT (rnd None).

`model_state_dict` (= `CoptActorCritic.state_dict()`) contains submodule
params/buffers: `actor.*`, `critic.*`, `estimator.*`, the scalar `std`
(inherited dead parameter, see 4.3) and `log_std`. For COPT the
`actor_obs_normalizer`/`critic_obs_normalizer` are `nn.Identity` (no buffers),
so NO running normalizer stats are saved. For configs that DO enable
normalization (e.g. `SD_BRS1FlatPPORunnerCfg`), the
`EmpiricalNormalization` buffers `_mean`, `_var`, `_std`, `count` ARE inside
`model_state_dict` because they are registered buffers (4.1).

### 1.6 `load(path, load_optimizer=True, map_location=None)` (lines 309-326)
- `loaded_dict = torch.load(path, weights_only=False, map_location=...)`
  (`:310`).
- `resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])`
  (`:312`). `ActorCritic.load_state_dict` always returns `True`
  (`actor_critic.py:186-199`), so `resumed_training` is always `True`.
- If `rnd`: loads `rnd_state_dict` (`:314-315`).
- If `load_optimizer and resumed_training`: restores
  `optimizer_state_dict` (`:317-319`) and rnd optimizer if present.
- `if resumed_training: self.current_learning_iteration = loaded_dict["iter"]`
  (`:324-325`). So `current_learning_iteration` IS restored; `learn()` resumes
  the iteration counter at `start_iter = current_learning_iteration` (`:96`).
- RETURNS `loaded_dict["infos"]` (`:326`) — `None` for COPT.

What `load` restores: policy weights + normalizer buffers (if any) + optimizer
moments + iteration counter. What it does NOT restore: `self.tot_timesteps`,
`self.tot_time` (reset to 0 → logging time axis discontinuity), and crucially
`self.alg.learning_rate` (see 2.5 — adaptive LR continuity is broken). The
observation normalizer IS restored when enabled (via model_state_dict), but for
COPT there is none.

### 1.7 `_construct_algorithm(obs)` (lines 395-434) — body COPT replicates
1. `self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, obs_groups, env)`
   (`:398`, see `rnd.py:184-208`).
2. `self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)` (`:401`).
3. Deprecated `empirical_normalization` shim → copies into
   `policy_cfg["actor_obs_normalization"]`/`["critic_obs_normalization"]`
   (`:404-413`). For COPT `empirical_normalization=False`, so this is a no-op
   and both normalization flags stay at their `False` default.
4. `actor_critic_class = eval(self.policy_cfg.pop("class_name"))` (`:416`),
   then `actor_critic = actor_critic_class(obs, obs_groups, num_actions, **policy_cfg)`
   (`:417-419`).
5. `alg_class = eval(self.alg_cfg.pop("class_name"))` (`:422`), then
   `alg = alg_class(actor_critic, device=device, **alg_cfg, multi_gpu_cfg=...)`
   (`:423`).
6. `alg.init_storage("rl", num_envs, num_steps_per_env, obs, [num_actions])`
   (`:426-432`).

COPT replicates this exactly (`copt_on_policy_runner.py:322-377`) with ONE
change: it forwards `self.encoder_cfg` as an extra positional arg to the policy
constructor (`:352-358`), i.e. `CoptActorCritic(obs, obs_groups, num_actions,
self.encoder_cfg, **policy_cfg)`. This matches the guideline in
`../CO_OPTIMISATION.md` section 3 step 1. Note `pop("class_name")` mutates
`policy_cfg`/`alg_cfg` in place, so the COPT override must be the only place
those pops happen (they are).

---

## 2. PPO (`algorithms/ppo.py`)

### 2.1 Constructor (lines 26-123) — hyperparameters and members
Signature defaults (`:26-49`): `num_learning_epochs=5`, `num_mini_batches=4`,
`clip_param=0.2`, `gamma=0.99`, `lam=0.95`, `value_loss_coef=1.0`,
`entropy_coef=0.01` (COPT cfg sets 0.005), `learning_rate=1e-3`,
`max_grad_norm=1.0`, `use_clipped_value_loss=True`, `schedule="adaptive"`,
`desired_kl=0.01`, `normalize_advantage_per_mini_batch=False`,
`rnd_cfg=None`, `symmetry_cfg=None`, `multi_gpu_cfg=None`.

Members:
- `self.rnd`, `self.rnd_optimizer` (`:62-73`) — None for COPT.
- `self.symmetry` (`:76-97`) — None for COPT.
- `self.policy = policy` (`:100`).
- `self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)`
  (`:104`) — single Adam over ALL policy params (actor, critic, estimator, std,
  log_std).
- `self.storage: RolloutStorage | None = None` (`:107`),
  `self.transition = RolloutStorage.Transition()` (`:108`).
- PPO scalars `:111-123`: `clip_param`, `num_learning_epochs`,
  `num_mini_batches`, `value_loss_coef`, `entropy_coef`, `gamma`, `lam`,
  `max_grad_norm`, `use_clipped_value_loss`, `desired_kl`, `schedule`,
  `self.learning_rate` (`:122`), `self.normalize_advantage_per_mini_batch`
  (`:123`).

### 2.2 `act`, `process_env_step` (lines 143-185)
- `act(obs)` (`:143-154`): fills `self.transition` with `actions` (sampled,
  detached), `values = policy.evaluate(obs)`, `actions_log_prob`,
  `action_mean`, `action_sigma`, and stores `observations=obs`. Returns actions.
- `process_env_step(obs, rewards, dones, extras)` (`:156-185`):
  - `self.policy.update_normalization(obs)` (`:160`) — updates
    EmpiricalNormalization running stats IF enabled. For COPT it is Identity,
    so this is a NO-OP (no running stats accumulate). The COPT
    `CoptActorCritic` does NOT override `update_normalization`, so it inherits
    the base which touches only the (Identity) normalizers.
  - `self.transition.rewards = rewards.clone()`, `self.transition.dones = dones`
    (`:166-167`).
  - Time-out bootstrapping (`:177-180`): `if "time_outs" in extras:
    self.transition.rewards += self.gamma * (values * time_outs)`. This adds
    `gamma*V(s)` to the stored reward on timed-out steps so GAE treats them as
    truncations rather than terminations. See 3.2 for the COPT silent-reset
    implication.
  - `self.storage.add_transitions(self.transition)` (`:183`),
    `self.transition.clear()` (`:184`), `self.policy.reset(dones)` (`:185`,
    a no-op for non-recurrent ActorCritic).

### 2.3 `compute_returns(obs)` (lines 187-192)
- `last_values = self.policy.evaluate(obs).detach()` (`:189`).
- `self.storage.compute_returns(last_values, gamma, lam,
  normalize_advantage=not self.normalize_advantage_per_mini_batch)` (`:190-191`).
  Since `normalize_advantage_per_mini_batch=False` for COPT,
  `normalize_advantage=True` → advantages are normalized ONCE, GLOBALLY, over
  the entire rollout batch in storage (see 3.1). This pools all 4096 envs /
  morphologies into one mean and std.

### 2.4 `update()` (lines 194-417) — losses and adaptive schedule
Loop over `mini_batch_generator(num_mini_batches, num_learning_epochs)`
(`:207`) → 4 * 5 = 20 minibatch updates per iteration. Each minibatch:
- Optional per-minibatch advantage normalization (`:226-228`) — DISABLED for
  COPT.
- Recompute `actions_log_prob_batch`, `value_batch`, `mu/sigma/entropy`
  (`:250-256`).
- Adaptive KL LR schedule (`:259-292`), only when
  `desired_kl is not None and schedule=="adaptive"` (true for COPT):
  - KL approximated analytically between old and new Gaussians (`:261-268`).
  - `if kl_mean > desired_kl*2: lr = max(1e-5, lr/1.5)`;
    `elif kl_mean < desired_kl/2 and kl_mean>0: lr = min(1e-2, lr*1.5)`
    (`:279-282`). Updates `self.learning_rate` and every
    `optimizer.param_groups[*]["lr"]` (`:291-292`). RUNS EVERY MINIBATCH (20x
    per iteration), so the LR can move several 1.5x steps within one iteration.
- Surrogate loss (`:295-300`):
  `ratio = exp(new_logp - old_logp)`,
  `surrogate_loss = max(-A*ratio, -A*clamp(ratio, 1-clip, 1+clip)).mean()`.
- Value loss (`:303-311`): clipped by default
  (`use_clipped_value_loss=True`): `value_clipped = target_values +
  clamp(value - target_values, -clip, +clip)`;
  `value_loss = max((value-returns)^2, (value_clipped-returns)^2).mean()`.
- Total `loss = surrogate_loss + value_loss_coef*value_loss
  - entropy_coef*entropy.mean()` (`:313`).
- `optimizer.zero_grad(); loss.backward()`; grad clip to `max_grad_norm`;
  `optimizer.step()` (`:364-377`).
- Accumulates `mean_value_loss`, `mean_surrogate_loss`, `mean_entropy`
  (`:383-385`).
- After the loop, divides each by `num_updates = num_learning_epochs *
  num_mini_batches = 20` (`:394-397`), then `self.storage.clear()` (`:404`).

`loss_dict` returned (`:407-411`): EXACTLY
`{"value_function": mean_value_loss, "surrogate": mean_surrogate_loss,
"entropy": mean_entropy}` (plus `"rnd"`, `"symmetry"` only if enabled — neither
for COPT). These become `Loss/value_function`, `Loss/surrogate`,
`Loss/entropy`. No explained-variance, no clip-fraction, no approx-KL is logged.

### 2.5 Stateful members needing persistence beyond `optimizer_state_dict`
- `self.learning_rate` (`:122`, mutated at `:280-282`). KEY FINDING: it is a
  plain Python float, NOT in `optimizer.state_dict()` in a way the runner uses.
  `save()` writes only `optimizer_state_dict`; `load()` restores the optimizer
  (whose `param_groups[*]["lr"]` does carry the saved lr), but `self.alg`
  is freshly constructed with `learning_rate=1e-3` from cfg, and on the FIRST
  adaptive update after resume (`:291-292`) the code OVERWRITES every
  param-group lr with `self.learning_rate` (=1e-3). Net effect: the adaptive
  learning rate silently RESETS to 1e-3 on every resume. For identical
  continuation, `self.alg.learning_rate` must be saved and restored explicitly.
- `self.rnd.update_counter` (`rnd.py:101`, incremented at `:119`) — a plain int
  NOT in any state_dict; drives the RND weight schedule. Would reset on resume.
  Moot for COPT (no RND) but relevant for the general save/load extension.
- `self.storage` and `self.transition` are transient (storage is cleared at the
  end of every `update()`, `:404`; save happens after update). Nothing to
  persist.

---

## 3. RolloutStorage (`storage/rollout_storage.py`)

### 3.1 GAE returns/advantages + advantage normalization (lines 130-151)
`compute_returns(last_values, gamma, lam, normalize_advantage=True)`:
- Backward recursion over `num_transitions_per_env` (`:134-144`):
  - `next_values = last_values` if last step else `self.values[step+1]` (`:136`).
  - `next_is_not_terminal = 1.0 - self.dones[step].float()` (`:138`).
  - `delta = rewards[step] + next_is_not_terminal * gamma * next_values
    - values[step]` (`:140`).
  - `advantage = delta + next_is_not_terminal * gamma * lam * advantage`
    (`:142`).
  - `returns[step] = advantage + values[step]` (`:144`).
- `self.advantages = self.returns - self.values` (`:147`).
- If `normalize_advantage`:
  `self.advantages = (advantages - advantages.mean()) / (advantages.std()+1e-8)`
  (`:150-151`). This is a SINGLE global mean/std over the FULL
  `[num_transitions_per_env, num_envs]` tensor — i.e. pooled across all 4096
  envs and hence all morphologies in the population.

### 3.2 Truncation vs termination / time-out bootstrapping
- The ONLY place a bootstrap value is injected for a time-out is
  `PPO.process_env_step` (`ppo.py:177-180`), which requires `"time_outs"` in
  `extras`. Storage itself uses ONLY `self.dones` to gate the GAE recursion
  (`:138`); it does not distinguish termination from truncation. So a step is
  treated as a hard terminal (`next_is_not_terminal=0`) UNLESS its reward was
  pre-augmented with `gamma*V` upstream via the `time_outs` signal.
- COPT silent-reset implication (objective 1/2): `_update_morphology` calls
  `unwrapped_env.reset()` (`copt_on_policy_runner.py:449`) OUTSIDE the rollout
  step loop — it runs AFTER `compute_returns` and `update()` for that iteration
  (`:246-249` then `:256-267`). Therefore the reset boundary falls BETWEEN
  rollout windows, not inside one. Consequences:
  - GAE is NOT corrupted: every transition inside a given 25-step rollout
    belongs to one morphology, and the end-of-rollout bootstrap
    (`last_values`) uses the pre-reset obs. There is no GAE bootstrap spanning
    the reset.
  - The reset does NOT emit a `done` or a `time_outs` flag into the runner's
    bookkeeping for the in-progress episodes; their continuation is discarded.
    The only visible effect is the rewbuffer/lenbuffer flood (1.3 item 4) and
    `env.episode_length_buf` being zeroed silently. So the silent reset is a
    LOGGING/metric artefact, not a GAE/advantage artefact.

### 3.3 Mini-batch generator (lines 162-217)
Flattens `[T, N, ...]` to `[T*N, ...]` (`:170-179`), random permutation of
`num_mini_batches*mini_batch_size` indices (`:167`), yields `num_epochs *
num_mini_batches` minibatches. `mini_batch_size = (num_envs *
num_transitions_per_env) // num_mini_batches` (`:165-166`) = (4096*25)/4 =
25600 samples per minibatch. Minibatches mix all morphologies uniformly at
random (no per-morphology grouping).

---

## 4. Observation normalization (`networks/normalization.py`, used in `modules/actor_critic.py`)

### 4.1 `EmpiricalNormalization` (lines 14-68)
- Registered BUFFERS (`:30-33`): `_mean` (zeros), `_var` (ones), `_std`
  (ones), `count` (long 0). Because they are buffers, they ARE included in
  `module.state_dict()` and therefore in `policy.state_dict()` → persisted by
  `OnPolicyRunner.save()` automatically WHEN normalization is enabled.
- `forward` (`:43-45`): `(x - _mean) / (_std + eps)`, `eps=1e-2`.
- `update` (`:47-63`): Welford-style running update, only when `self.training`
  and `count < until`. Updates `count`, `_mean`, `_var`, `_std`. Computed over
  the WHOLE batch axis (`dim=0`), pooling all envs/morphologies.

### 4.2 Where it lives in the policy (`modules/actor_critic.py`)
- `self.actor_obs_normalizer` = `EmpiricalNormalization(num_actor_obs)` if
  `actor_obs_normalization` else `nn.Identity()` (`:62-66`).
- `self.critic_obs_normalizer` similarly (`:73-77`).
- `update_normalization(obs)` (`:178-184`) only updates when the corresponding
  flag is True.

### 4.3 COPT specifics + DISCREPANCY notes
- For the real COPT run BOTH flags are `False` (agent.yaml), so both
  normalizers are `nn.Identity` → NO running stats exist, nothing to persist,
  and `process_env_step`'s `update_normalization` call is a no-op.
- `CoptActorCritic` (`co_optimisation/modules/copt_actor_critic.py`) calls
  `super().__init__(...)` with `noise_std_type="scalar"` and
  `state_dependent_std=False`, so the PARENT creates `self.std =
  nn.Parameter(...)` (`actor_critic.py:92-93`). `CoptActorCritic` then ALSO
  creates `self.log_std = nn.Parameter(...)` (`copt_actor_critic.py:108`) and
  uses ONLY `log_std` in `_update_distribution` (`:136`). So `self.std` is a
  DEAD parameter that is nonetheless saved in `model_state_dict` and tracked by
  the Adam optimizer. Harmless but worth knowing when diffing checkpoints.
- `CoptActorCritic` overrides `act`, `_update_distribution`,
  `act_inference`, `get_actions_log_prob` but NOT `evaluate`. So the CRITIC
  uses the base `ActorCritic.evaluate` (`actor_critic.py:162-165`) over
  `obs_groups["critic"]` and does NOT receive the estimator latent. Whether the
  critic can see morphology depends entirely on whether design/link-length
  parameters are present in the `critic` obs group (an Agent E question). The
  actor receives `cat(actor_obs, estimator(privilegedObs))`
  (`copt_actor_critic.py:127-137`), so morphology reaches the actor only via
  whatever `privilegedObs` encodes.

### 4.4 RND normalizers (`modules/rnd.py`, not used by COPT)
`RandomNetworkDistillation` holds `state_normalizer` and `reward_normalizer`
(EmpiricalNormalization / EmpiricalDiscountedVariationNormalization) as
submodules (`:89-98`) → inside `rnd.state_dict()`, persisted only if RND is
enabled. `self.update_counter` (`:101`) is a plain int and is NOT in the state
dict (would reset on resume).

---

## 5. Objective-specific analysis

### 5.1 Objective 1 — high training variance
- Partial-episode injection: confirmed at `copt_on_policy_runner.py:270-284`.
  Every 120 iters all `num_envs` partial `cur_reward_sum` values are pushed into
  the 100-slot `rewbuffer`, fully overwriting it with truncated returns, then
  zeroed. `Train/mean_reward` (`on_policy_runner.py:231`) is the simple mean of
  that window, so it shows a periodic artefactual dip/spike independent of true
  policy quality. lenbuffer is similarly flooded with truncated lengths. This
  alone produces a sawtooth in the reward plot with period 120 that does NOT
  shrink as designs converge, matching the reported symptom.
- Advantage normalization is GLOBAL across the whole rollout
  (`rollout_storage.py:150-151`, `ppo.py:191`), pooling all morphologies into
  one mean/std. With heterogeneous link lengths the per-morphology return scales
  differ; pooling does not inflate the PLOTTED reward variance (that statistic
  is pre-normalization), but it does blur the advantage signal across
  morphologies, weakening the gradient (large-return morphologies dominate the
  pooled std). Per-minibatch normalization is OFF, so there is no per-batch
  re-normalization that would interact with morphology composition within a
  minibatch.
- Critic across morphologies: the critic minimizes a single clipped MSE value
  loss (`ppo.py:303-311`) over the pooled batch. If the critic obs do NOT encode
  morphology (see 4.3), one state maps to different returns across link lengths,
  so the value target is multi-valued → irreducibly high `Loss/value_function`,
  biased `delta` (`rollout_storage.py:140`), noisier advantages, and hence
  noisier policy gradients. There is NO explained_variance metric to confirm
  this directly (DISCREPANCY with CONTEXT); `Loss/value_function` is the only
  proxy. Recommend adding an explained-variance scalar if this hypothesis is to
  be tested quantitatively.
- Timing sanity: `ea_update_interval=120`, `num_steps_per_env=25` → 3000
  control steps per morphology window. DISCREPANCY: the train.py comment
  (`scripts/rsl_rl/train.py:201`) says "50 * 24 = 1200" and CONTEXT says
  "120*24=2880"; both use stale numbers. The real product is 120*25 = 3000
  steps, comfortably above the ~1000-step episode cap, so episodes DO complete
  within a window (the flood is the dominant artefact, not a lack of
  completions).

### 5.2 Objective 2 — full enumeration of rsl_rl-side state to persist
For byte-identical continuation, the following rsl_rl-side objects matter
(serialization shown). COPT does NOT currently override `save`/`load`, so today
ONLY items (a)-(d) are persisted; the rest are lost.

Currently persisted by base `save()`:
- (a) Policy weights + buffers — `self.alg.policy.state_dict()` → key
  `model_state_dict`. Includes actor/critic/estimator/std/log_std and any
  EmpiricalNormalization buffers (`_mean,_var,_std,count`).
- (b) Optimizer moments — `self.alg.optimizer.state_dict()` → key
  `optimizer_state_dict` (Adam `exp_avg`, `exp_avg_sq`, `step`, and
  `param_groups[*].lr`).
- (c) Iteration counter — `self.current_learning_iteration` → key `iter`,
  restored at `on_policy_runner.py:325` and reused as `start_iter`.
- (d) `infos` (always `None` for COPT).

NOT persisted today but required for identical continuation (rsl_rl side):
- (e) `self.alg.learning_rate` (float). MUST be saved (e.g.
  `saved_dict["learning_rate"]=self.alg.learning_rate`) and restored
  (`self.alg.learning_rate=loaded_dict["learning_rate"]`) BEFORE the first
  adaptive update, else the adaptive LR resets to the cfg default 1e-3 on
  resume (2.5). This is the single most impactful omission for PPO continuity.
- (f) `self.tot_timesteps`, `self.tot_time` (ints/floats). Needed only for
  continuous logging axes; not training-critical but trivial to persist.
- (g) If RND were used: `self.alg.rnd.update_counter` (int) and the RND
  optimizer/state (the latter two are already handled when `rnd` is truthy).
- (h) NOT needed: `self.alg.storage`, `self.alg.transition` (transient, cleared
  every iteration before save).

Out of rsl_rl scope but flagged because COPT inherits the base save (Agent C
owns the fix): `CoptOnPolicyRunner` does NOT persist `self.generation`,
`self.current_population`, `self._design_generator` (the CMA-ES
`CMAEvolutionStrategy` state — mean, sigma, covariance), `self._individual_fitness`,
`self._individual_episode_counts`, `self._env_to_individual`,
`self._copt_started`, or the late-start toggle state. None of these appear in
the inherited `save()`. The CONTEXT objective 2 statement that "current
save/load persists only model weights" is slightly inaccurate on the rsl_rl
side — it ALSO persists the PPO optimizer and the iteration counter — but is
correct that everything EA/env-side is lost.

Recommended approach: override `CoptOnPolicyRunner.save`/`load` to call
`super().save(...)`/`super().load(...)` then add the extra keys (e/f and all the
COPT/EA fields). Use `torch.save` for tensors and pickle for the CMA-ES object
(pycma `CMAEvolutionStrategy` is picklable; confirm via Agent D).

### 5.3 Objective 3 — CMA-ES
No CMA-ES code lives in rsl_rl. The only rsl_rl-side touchpoint is that
`CoptOnPolicyRunner` neither persists nor seeds the generator's RNG through
`save`/`load`, so a resumed run cannot reproduce the CMA-ES trajectory. Defer
substance to Agent C/D.

---

## 6. Summary of discrepancies with existing docs
1. `../CO_OPTIMISATION.md` section 2 documents the OLD `(num_actor_obs,
   num_critic_obs)` ActorCritic API and `act(obs, critic_obs)` /
   `process_env_step(rewards, dones, infos)` / `compute_returns(last_critic_obs)`
   signatures. The installed library uses TensorDict `obs` + `obs_groups`
   throughout: `ActorCritic(obs, obs_groups, num_actions, ...)`,
   `act(obs)`, `process_env_step(obs, rewards, dones, extras)`,
   `compute_returns(obs)`.
2. No explained-variance metric exists anywhere in rsl_rl (CONTEXT assumes one
   is logged). Only `Loss/value_function`, `Loss/surrogate`, `Loss/entropy`,
   `Loss/learning_rate`, `Policy/mean_noise_std` are written.
3. Base `save()` persists model + optimizer + iter (+ rnd if used), not "only
   model weights".
4. Stale step-count arithmetic in `train.py:201` ("50 * 24 = 1200") and in
   CONTEXT ("120*24=2880"); real value is 120 * 25 = 3000 control steps per
   morphology window (`num_steps_per_env=25`, confirmed in agent.yaml).
5. Adaptive learning rate is not persisted and silently resets to 1e-3 on
   resume — not previously documented.

---

## 7. Additions (2026-07-02, learned-model architecture groundwork)

1. `ActorCritic.__init__` (modules/actor_critic.py:20-104) asserts every group
   listed in `obs_groups["policy"]` and `obs_groups["critic"]` is 2D
   (actor_critic.py:45,49). A 3D history group (flatten_history_dim False) must
   therefore be consumed directly via `obs["<group>"]`, never listed in the
   policy or critic observation sets.
2. `ActorCritic` sizes the critic MLP from the sum of `obs_groups["critic"]`
   group widths (actor_critic.py:47-50) and `get_critic_obs`
   (actor_critic.py:171-173) concatenates exactly those groups. Extending the
   critic input is therefore achieved by listing additional groups in
   `obs_groups["critic"]`, no `get_critic_obs` override or critic rebuild needed.
3. `resolve_obs_groups` (utils/utils.py:203-262) accepts an explicit dict such
   as `{"policy": ["policy"], "critic": ["policy", "criticOnly"]}` and validates
   each named group exists in the env observation TensorDict. When the config
   omits "critic" it falls back to a group literally named "critic", else to the
   policy set. `RslRlOnPolicyRunnerCfg.obs_groups`
   (isaaclab_rl/rsl_rl/rl_cfg.py:159) is the config surface for this dict.
4. `PPO.act` (algorithms/ppo.py:143-154) executes
   `self.transition.actions = self.policy.act(obs).detach()`. A policy whose
   `act` returns a tuple breaks this line, so an algorithm subclass must
   override `act` (and `update`, which also calls `self.policy.act`) to unpack.
5. `RolloutStorage` stores the full observation TensorDict per step, so any
   auxiliary group (for example a prediction target) is available in
   `obs_batch` inside `update` minibatches without storage changes.

## Symmetry augmentation consumer contract (added 2026-07-22, for the SD_BRS1 symmetry plan SYMMETRY_PLAN.md)

How the rsl_rl at /ws/rsl_rl consumes a symmetry data augmentation function, read from source, authoritative over the IsaacLab bundled copy since the user trains with this tree.

1. Config surface. `isaaclab_rl.rsl_rl.RslRlSymmetryCfg` (isaaclab_rl/rsl_rl/symmetry_cfg.py) has four fields, `use_data_augmentation`, `use_mirror_loss`, `data_augmentation_func`, `mirror_loss_coeff`. The stock `RslRlPpoAlgorithmCfg` (isaaclab_rl/rsl_rl/rl_cfg.py:128-129) already declares `symmetry_cfg: RslRlSymmetryCfg | None = None`, so any configclass subclass such as the project `RslRlPpoAlgorithmMlpCfg` (an empty subclass) inherits it, and setting it is a purely additive change with no class edit.
2. Serialization. `agent_cfg.to_dict()` in scripts/rsl_rl/train.py converts the nested cfg through `class_to_dict`, which at isaaclab/utils/dict.py:62-63 turns the callable `data_augmentation_func` into a `module:function` string via `callable_to_string`, exactly the form `string_to_callable` re imports inside the PPO constructor (algorithms/ppo.py:83-84). The function MUST therefore be a top level importable module member, a lambda or a closure cannot survive the round trip.
3. Runtime env injection. `resolve_symmetry_config` (modules/symmetry.py:11-25) sets `alg_cfg["symmetry_cfg"]["_env"] = env`, called from runners/on_policy_runner.py:401 with the wrapped VecEnv. The function receives that `RslRlVecEnvWrapper`, reach the managers through `env.unwrapped.observation_manager` and `env.unwrapped.scene["robot"]`.
4. The callable contract, three call sites in algorithms/ppo.py, always by keyword with names env, obs, actions. Data augmentation branch ppo.py:235-239, both inputs non None, returns augmented pair. Mirror loss obs branch ppo.py:319-323, actions None, returns augmented obs. Mirror loss action branch ppo.py:333-335, obs None, returns augmented actions. Signature, accept env, obs, actions by keyword, tolerate either input None, return a two tuple in (obs, actions) order.
5. Shapes. `obs` is a `tensordict.TensorDict` keyed by observation group, `obs["policy"]` and `obs["critic"]`, each `(batch, group_dim)`. `actions` is `(batch, num_actions)`. The function chooses the augmentation factor implicitly, `num_aug = int(returned_batch / original_batch)` at ppo.py:241, two for a biped left right mirror. Originals MUST be the first original_batch rows, entropy and KL keep `[:original_batch]` at ppo.py:254-256 and the mirror loss reads the tail `[original_batch:]` at ppo.py:340.
6. Caller tiling and the Mittal stability point. The caller tiles old_actions_log_prob, target_values, advantages, returns with `.repeat(num_aug, 1)` at ppo.py:243-246, so the function must NOT tile them but must produce augmented actions itself. Tiling old_actions_log_prob from the originals is the correct handling of the importance ratio that Mittal arXiv 2403.04359 requires, the mirrored row is scored against the original transition's old probability, not a recomputed mirrored probability which could explode.
7. Raw observations. The function operates on RAW un normalized obs, normalization is applied inside the network forward (modules/actor_critic.py:150,156,164), never inside update.
8. Mirror loss. ppo.py:316-346, MSE between the actor mean on the mirrored obs and the analytic mirror of the actor mean on the originals, tail only, scaled by mirror_loss_coeff, added only when use_mirror_loss is true, otherwise computed and logged only. Both flags may be true together, the mirror loss branch checks `if not use_data_augmentation` to avoid re augmenting.
9. Recurrent policies are rejected at ppo.py:92-93, not a blocker for the BRS plain ActorCritic.

## Symmetry, the critic group must be mirrored for value invariance (added 2026-07-22)

An augmentation function must mirror EVERY observation group present in the batch TensorDict, not just `policy`. In `algorithms/ppo.py` update, the value head is `value_batch = self.policy.evaluate(obs_batch)` at line 252, evaluated on the augmented `obs_batch`, and trained toward the tiled `returns_batch` at line 307. If the `critic` group is left tiled (the `obs.repeat(num_aug)` default without overwrite), the mirrored rows feed the ORIGINAL privileged state while carrying the replicated return of that same state, so the value loss on the mirrored half merely duplicates the original half and never teaches the value function the invariance V(gs)=V(s). The anymal reference at `isaaclab_tasks/.../mdp/symmetry/anymal.py` only overwrites `obs_aug["policy"]` and tiles the rest, which is that latent defect for any env with a distinct privileged critic group. SymmLoco does it correctly, a separate `critic_in_field_type` transforms `critic_observations` at `ppo_augment.py:123,179`. So a correct data augmentation func mirrors both `policy` and `critic`, each by its own per term map. World frame critic terms (root/foot velocities and forces, absolute root position) cannot be mirrored inside the augmentation func, since it sees only the stored history with no per transition base yaw to rotate world vectors into the base frame, so those specific terms are left identity as a recorded approximation. Full worked example for SD_BRS1 in `/ws/plans/SYMMETRY_PLAN.md` section 3.5 and `mdp/symmetry/brs.py`.
