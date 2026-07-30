# copt.md — Agent C findings (co_optimisation package)

Scope: `/ws/tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/` plus the
entry point `scripts/rsl_rl/train.py` and the COPT env/agent cfg that decide the
observation wiring. All paths absolute. Build copy under `co_optimisation/build/`
is a stale duplicate and is ignored.

## 0. Package tree (source, not build/)

- `co_optimisation/__init__.py` — re-exports `CoptOnPolicyRunner`.
- `runners/__init__.py` — exports `CoptOnPolicyRunner, Population, DesignGeneratorBase, RandomPopulation, RandomDesignGenerator`. NOTE: does **not** export `CMAESDesignGenerator` or `GrowingDesignDistCMAESDesignGenerator`.
- `runners/copt_on_policy_runner.py` — the runner.
- `runners/usd_generator.py` — generators + populations.
- `modules/copt_actor_critic.py` — `CoptActorCritic`.
- `utils/respawn.py` — `respawn_robots`, `apply_actuator_params`.
- `utils/update.py` — `apply_link_length_params`, `update_articulation_links`.
- `utils/analysis.py` — USD prim-tree logging (`log_prototype_and_instance_info`).
- `tests/cmaes_design_generator_test.py`.

---

## 1. CONFIRMED BUG: `GrowingDesignDistCMAESDesignGenerator` is a latent NameError

- `scripts/rsl_rl/train.py:209` instantiates `GrowingDesignDistCMAESDesignGenerator(...)`.
- The only co_optimisation imports in train.py are `train.py:97` (`CoptOnPolicyRunner`) and `train.py:98-102` (`DEFAULT_PARAM_RANGES, CMAESDesignGenerator, RandomDesignGenerator`). The symbol `GrowingDesignDistCMAESDesignGenerator` is **never imported**.
- It is defined only at `usd_generator.py:815` and is **not** re-exported by `runners/__init__.py:1-7`.
- grep over the whole repo finds exactly two hits: the def (usd_generator.py:815) and the use (train.py:209).
- Conclusion: when `--policy_type COPT` is selected, `main()` raises `NameError: name 'GrowingDesignDistCMAESDesignGenerator' is not defined` at train.py:209, before any runner is built. Either train.py was edited after the import block, or COPT runs are launched from a different/patched entry script. This MUST be fixed (add the import) before any COPT objective can be exercised from this train.py.

---

## 2. CoptOnPolicyRunner (`runners/copt_on_policy_runner.py`)

### 2.1 `__init__` member state (lines 69-103)
- `_ea_update_interval` = copt_cfg["ea_update_interval"] (train.py → 120), default 100. (`:79`)
- `_ea_late_start` = copt_cfg["ea_late_start"] (→ 8000), default -1. (`:80`)
- `_num_individuals` = copt_cfg["num_individuals"] (→ 64), default 16. (`:81`)
- `_randomise_before_late_start` = copt_cfg["randomise_before_late_start"] (→ True), default False. (`:82-84`)
- `_design_generator` = injected generator. (`:85`)
- `generation: int = 0`. (`:87`)
- `current_population: Population | None = None`. (`:88`)
- `_individual_fitness` = zeros(num_individuals) float on device. (`:91-93`)
- `_individual_episode_counts` = zeros(num_individuals) long. (`:94-96`)
- `encoder_cfg = train_cfg["encoder"]`. (`:97`)
- `_copt_started = False`. (`:98`)
- `super().__init__(...)` builds PPO/policy and resolves obs_groups. (`:100`)
- `_env_to_individual = self._assign_individuals_to_envs()` = `[i % num_individuals for i in range(num_envs)]` (round-robin, deterministic, recomputable). (`:103`, `:470-472`)
- `respawn_robots` is set later in `learn()` at `:120` (the callable instance), not in `__init__`.

### 2.2 `learn()` rollout + bookkeeping (lines 105-320)
- `_update_morphology(0)` is called once before the first rollout (`:124-125`) — this is the initial spawn path (NOT `_reload_morphology`; see §2.4).
- `rewbuffer`/`lenbuffer` are `deque(maxlen=100)` (`:131-132`); `cur_reward_sum`,`cur_episode_length` are per-env tensors length `num_envs` (`:133-138`).
- Schedule flags per iteration:
  - `is_late_start_toggle_time = self._ea_late_start <= (it - start_iter + 1)` (`:163`). With ea_late_start=8000 this is True once `(it-start_iter+1) >= 8000`.
  - `is_morph_update_time = ((it-start_iter+1) % _ea_update_interval == 0) and (is_late_start_toggle_time or _randomise_before_late_start)` (`:164-166`). Because `_randomise_before_late_start=True`, morph updates fire at **every** multiple of 120 from the start.
- Per-step bookkeeping condition (`:203-205`):
  ```
  if (step < self.num_steps_per_env - 1) or (not is_morph_update_time):
  ```
  - Non-morph iteration: `not is_morph_update_time` is True → block runs every step (normal PPO bookkeeping, fitness accumulation on real `dones`).
  - Morph iteration: block runs for steps `0..num_steps_per_env-2` but is **skipped on the final step**. On that final step, real terminations are NOT recorded through the done-path and `cur_reward_sum`/`cur_episode_length` are NOT zeroed; they are instead handled by the post-rollout morph branch.

### 2.3 BUG (objective 1): rewbuffer/lenbuffer flooding at every morph update (lines 270-294)
After the rollout + `alg.update()`, inside `if is_morph_update_time:` (`:256`) the morph branch (`:270-278`) does, for **all** envs:
```
for env_idx in range(self.env.num_envs):
    ind_idx = self._env_to_individual[env_idx]
    self._individual_fitness[ind_idx] += cur_reward_sum[env_idx]
    self._individual_episode_counts[ind_idx] += 1
rewbuffer.extend(cur_reward_sum.cpu().numpy().tolist())   # num_envs entries
lenbuffer.extend(cur_episode_length.cpu().numpy().tolist())
cur_reward_sum[:] = 0; cur_episode_length[:] = 0
```
- `num_envs` defaults to 4096 (SFCoptEnvCfg scene, `limx_base_env_cfg.py:1374`). `rewbuffer` is `maxlen=100`, so `.extend(4096 partials)` discards all completed-episode returns and leaves the window holding the **last 100 envs' truncated, mid-episode `cur_reward_sum`** values.
- Quantification: every 120th iteration the logged "Mean/std reward" (computed by `OnPolicyRunner.log()` from `rewbuffer`) is the statistic of 100 partial (non-terminated) returns of arbitrary envs, not of completed episodes. This injects a periodic downward/variance artifact into the very curve the variance investigation tracks. The artifact recurs forever (every 120 iters), so plotted variance does not shrink as designs converge — consistent with the observed symptom.
- Same flooding hits `erewbuffer`/`irewbuffer` if RND is enabled (`:280-284`).

### 2.4 Fitness accumulation math (lines 202-224 and 270-278)
- Normal path (`:206-214`): on real terminations, `_individual_fitness[ind] += cur_reward_sum[env]` and `_individual_episode_counts[ind] += 1`. Per-individual mean is computed later by `_compute_individual_fitness()` (`:459-468`) as `fitness/count` (0.0 if count==0).
- Morph-final-step path (`:270-274`): **every** env (terminated or not) is credited as exactly one "episode" with value `cur_reward_sum[env]` (a partial mid-episode return for most envs). So each generation's fitness is contaminated by `num_envs` truncated returns (64 per individual at 4096/64). Over one generation (120 iters × 25 steps = 3000 control steps, episode ≈ 1000 steps → ~3 episodes/env), roughly 64 of ~256 fitness samples per individual are truncated partials (~25%), biasing fitness downward and adding noise to the CMA-ES cost signal.

### 2.5 CONFIRMED BUG (objective 3): CMA-ES tell/ask off-by-one in `_update_morphology` (lines 416-457)
`learn()` calls `_update_morphology` (the in-place pathway), NOT `_reload_morphology`. The CMA branch (`:426-435`) executes in this order:
```
if not self._copt_started:
    self._design_generator.sample_batch()          # es.ask -> _pending_solutions
    self._copt_started = True
self.current_population = self._design_generator.generate_population(self.generation)  # builds pop from _pending; sets _last_solutions = _pending; _pending = None
fitness = self._compute_individual_fitness()        # fitness of the PREVIOUSLY applied population
self._design_generator.update_with_fitness(fitness) # es.tell(_last_solutions, -fitness); es.ask -> _pending
```
- `generate_population` (usd_generator.py:761-772) overwrites `_last_solutions` with the just-built batch **before** `update_with_fitness` reads it.
- Therefore `es.tell` pairs the **current** batch's solution vectors with the **previous** batch's fitness. Trace:
  - gen67 (first CMA): `tell(ask1, cost(F_random))`
  - gen68: `tell(ask2, cost(F_ask1))`
  - gen69: `tell(ask3, cost(F_ask2))` …
  - i.e. `tell(ask_{i+1}, cost(F_ask_i))` — solutions and their fitness are misaligned by one generation, so CMA-ES receives essentially decorrelated feedback.
- This contradicts the documented order in ../CO_OPTIMISATION.md §4.9.4 (`../CO_OPTIMISATION.md:1525-1530`), which calls `update_with_fitness(fitness)` **before** `generate_population` (the correct pairing `tell(prev_solutions, prev_fitness)`). The implementation swapped the two lines, presumably to avoid the `assert _last_solutions is not None` on the first call (usd_generator.py:749-751). This off-by-one is a strong candidate cause for CMA-ES collapsing to a boundary design with no meaningful optimisation (objective 3).
- Recommended fix: restore doc order — on the first CMA generation `sample_batch()` then `generate_population()` without telling; on every subsequent generation compute fitness and `update_with_fitness()` (tell + ask) **before** `generate_population()` so the fitness is attributed to the solutions that produced it.

### 2.6 `_update_morphology` vs `_reload_morphology`
- `_update_morphology(it)` (`:416-457`) is the IN-PLACE pathway actually used (called at `:125` and `:267`). Steps:
  1. stop sim with Fabric guard (`:421-424`).
  2. CMA branch if `_ea_late_start <= it` (`:426-435`): first-time `sample_batch()`; `generate_population`; `_compute_individual_fitness`; `update_with_fitness`. Else random branch (`:436-439`): just `generate_population`.
  3. `self.generation += 1` (`:440`).
  4. `apply_link_length_params(unwrapped_env, current_population.get_link_length_params())` — authors absolute box extents on each design's USD prototype and runs a single `sim.reset()` internally (`:445-447`, update.py).
  5. `unwrapped_env.reset()` (`:449`).
  6. `apply_actuator_params(...)` (`:451-454`) — for CMA-ES populations this is a no-op because `get_actuator_params()` returns `[{}, ...]` (usd_generator.py:793 returns `{}`).
  7. zero `_individual_fitness`, `_individual_episode_counts` (`:456-457`).
- `_reload_morphology()` (`:383-414`) is **DEAD CODE** — defined but never called anywhere (grep confirms only the def). It is the older delete/respawn pathway: calls `update_with_fitness`/`sample_batch`, `generate_population`, then `self.respawn_robots(env, usd_files)` (the heavy 11-step delete+spawn sequence in respawn.py), `apply_actuator_params`, zero accumulators. Its fitness/sample ordering also differs from `_update_morphology` (it calls `update_with_fitness` before `generate_population`, i.e. the correct order, and guards `sample_batch` on `_ea_late_start < 0`). Keeping both is a maintenance hazard; the live path is the buggy one.

### 2.7 Late-start toggling and generator interaction (lines 256-265)
- Inside the morph branch, the toggle fires once: `if _ea_late_start > 0 and _design_generator.late_start and is_late_start_toggle_time: toggle_late_start()` (`:257-265`). The `and late_start` guard prevents re-toggling (toggle flips a bool; once False it stays False). `toggle_late_start()` is usd_generator.py:739-740.
- `_copt_started` gates the very first `es.ask()` (sample_batch) in `_update_morphology` (`:427-429`), independent of the toggle.
- Random phase: while `late_start=True`, `_generate_individual` samples random scales (no CMA solutions); `_pending_solutions` stays None and `_last_solutions` is repeatedly set to None by `generate_population`.

### 2.8 State NOT persisted by base save/load (objective 2)
`CoptOnPolicyRunner` does **not** override `save`/`load`. Base `OnPolicyRunner.save` (`/ws/rsl_rl/rsl_rl/runners/on_policy_runner.py:291-303`) persists only `model_state_dict` (actor, critic, estimator, obs normalizers), `optimizer_state_dict`, `iter`, `infos`, and RND state if present. Base `load` (`:309-326`) restores those and sets `current_learning_iteration`.
None of the following COPT state is saved or restored:
- `generation` (counter that drives the GrowingDesignDist scale schedule and CMA generation index).
- `current_population` (USD paths, link extents, actuator params).
- `_individual_fitness`, `_individual_episode_counts` (in-progress fitness accumulators).
- `_copt_started` (whether CMA has begun).
- `_env_to_individual` (deterministic, recomputed in __init__ — safe).
- `_design_generator` internal CMA-ES state: `_es` (CMAEvolutionStrategy mean/covariance/sigma/step), `_pending_solutions`, `_last_solutions`, `_terminated`, `late_start`.
- `CMAESDesignGenerator.save_state(path)` (usd_generator.py:809-812) pickles `_es` but is **never called anywhere** (grep: only the def). `es_state_path` constructor arg (usd_generator.py:685, 712-718) is the load hook, but train.py never passes it. So CMA-ES checkpoint save/load exists in code yet is entirely inert.
- ADDITIONAL objective-2 BUG: the late-start/morph schedule is computed relative to `start_iter` (`is_late_start_toggle_time = _ea_late_start <= (it - start_iter + 1)`, `:163`; loop `for it in range(start_iter, tot_iter)`, `:160`). On resume, `start_iter = current_learning_iteration` (loaded), so `(it - start_iter + 1)` restarts at 1. Resuming after the 8000-iter random phase would **repeat the entire random phase** and never reach CMA at the right time. This directly blocks objective 2's goal of skipping the first 8000 iterations.

### 2.9 `_construct_algorithm` (lines 322-377)
Mirrors the base method but builds `CoptActorCritic` via `eval(class_name)` (`:351-358`), passing `obs, obs_groups, num_actions, self.encoder_cfg, **policy_cfg`. Storage initialised with `num_envs`, `num_steps_per_env`, obs, `[num_actions]` (`:370-376`).

---

## 3. usd_generator.py

### 3.1 Class hierarchy
- `Population(ABC)` (`:247-275`): abstract `get_usd_files`, `get_actuator_params`, `get_link_length_params`.
- `RandomPopulation(Population)` (`:536-557`): concrete carrier of the three lists.
- `DesignGeneratorBase(ABC)` (`:283-528`): shared URDF→USD machinery, `_sample_scales` (`:354-355`), `_sample_scales_v2` (`:357-364`), `_compute_link_extents` (`:366-373`), `_build_population` (`:338-350`), URDF authoring helpers.
- `RandomDesignGenerator(DesignGeneratorBase)` (`:565-660`): uniform random scales; `_build_actuator_params` (`:613-627`).
- `CMAESDesignGenerator(DesignGeneratorBase)` (`:668-812`).
- `GrowingDesignDistCMAESDesignGenerator(CMAESDesignGenerator)` (`:815-867`).

### 3.2 `CMAESDesignGenerator.__init__` (lines 677-735)
- Restricts param space to `CMAES_PARAM_RANGES` = only `thigh_length_scale`, `shank_length_scale` (`:699-705`, `:236-239`). `num_params = 2`.
- `cma.CMAOptions` set (`:720-728`): `popsize = num_individuals` (64), `bounds = [zeros(N), ones(N)]` i.e. the unit box `[0,1]^2`, `seed`, `verb_disp=1`, `verb_filenameprefix`, `maxiter = max_cma_iter` (train.py passes `(30000-8000)/120 ≈ 183.3`).
- Initial mean `np.full(N, 0.5)` (`:730`) — box centre, which `_denormalise` maps to scale 1.0 (baseline geometry) for both params.
- `sigma0` (train.py → 0.75) passed as `_sigma0` (`:707`, `:730`). NOTE: sigma0=0.75 in a `[0,1]` box is ~3/4 of the full box width → an enormous initial step; combined with hard box bounds this pushes early samples hard against the [0,1] edges and is a plausible contributor to the boundary collapse (objective 3).
- `_pending_solutions`, `_last_solutions`, `_terminated` initialised (`:733-735`).

### 3.3 ask/tell cycle
- `sample_batch()` (`:742-744`): asserts no pending, `_pending_solutions = _es.ask()`.
- `update_with_fitness(fitness)` (`:746-759`): asserts `_last_solutions` set; `costs = [_sanitise_cost(-float(f)) for f in fitness]` (CMA minimises, so negate fitness); `_es.tell(_last_solutions, costs)`; check `_es.stop()` → sets `_terminated`; then `sample_batch()` (next ask).
- `generate_population(generation)` (`:761-772`): if `_terminated`, reuse `_last_solutions` and rebuild (fine-tune on the selected design); else build pop from `_pending_solutions` via `_generate_individual` denormalise, then `_last_solutions = _pending_solutions; _pending_solutions = None`. (This is the line that, called before `update_with_fitness`, causes the §2.5 off-by-one.)
- `_denormalise(x)` (`:795-801`): `scale = lo + clip(x,0,1)*(hi-lo)` per key, range (0.85,1.15).
- `_sanitise_cost(c, 1e6)` (`:803-807`): non-finite → 1e6.
- `save_state(path)` (`:809-812`): pickles `_es`; never called (see §2.8).

### 3.4 Design-resolution floor: `round(val, 2)`
- `_compute_link_extents` (`:366-373`) rounds every extent to 2 decimals: `{key: round(val,2)}` (`:372`). For thigh base z0=0.25 over scale (0.85,1.15) → z ∈ [0.2125, 0.2875] quantised to 0.01 m → at most ~8 distinct values; shank z0=0.30 → [0.255,0.345] → ~10 values. Any sub-cm CMA spread is erased, so near convergence all individuals print identical extents (matches CSV evidence: final thigh z=0.28, shank z=0.34, the rounded upper bounds 0.2875→0.28, 0.345→0.34).
- `_build_actuator_params` (`:626`) also rounds to 2 decimals, but only affects `RandomDesignGenerator`; CMA-ES emits empty actuator params (`:793`).

### 3.5 GrowingDesignDist scale schedule (lines 842-867)
- During `late_start`, `_generate_individual` computes `s=0.1`, `n=late_start_it`, `g=generation`, `scale = 0.95*(g/(n-s)) + 0.05` (`:849-855`), then `scales = _sample_scales_v2(rng, scale)`.
- `_sample_scales_v2` (`:357-364`) shrinks each param's sampling interval toward its midpoint by factor `scale`: `lo_n=(lo-mean)*scale+mean`, `hi_n=(hi-mean)*scale+mean`, sample uniform in `[lo_n,hi_n]`. For thigh/shank, mean=1.0.
- Effect: at g=0 scale=0.05 → designs ≈ baseline (±2.5% of range), nearly deterministic; at g≈66 (late_start_it=66) scale≈1.0 → full (0.85,1.15) range. The morphology distribution **widens monotonically** across the first 8000 iterations, so plotted reward variance grows during the random phase rather than shrinking (objective 1: the variance is partly designed-in during warm-up).
- The base `CMAESDesignGenerator._generate_individual` (`:776-793`) uses flat `_sample_scales` during late_start (no growth); only the Growing subclass adds the schedule. Since train.py references the Growing subclass (modulo the NameError), the growth schedule is the intended behaviour.

---

## 4. utils/

### 4.1 respawn.py — `respawn_robots` (callable class, lines 24-238)
Holds cached spawn props as instance state: `rigid_props`, `articulation_props`, `activate_contact_sensors` (`:26-33`, set from the old spawn cfg at `:81-89`). `__call__(env, new_usd_paths)` performs the heavy hot-swap:
1. `sim.stop()` with `_disable_app_control_on_stop_handle` guard (`:66-70`).
2. `sim_utils.delete_prim` per env `/Robot` prim, deduped (`:79-97`) — fires PRIM_DELETION so old articulation/sensor callbacks clear.
3. `sim_utils.spawn_multi_usd_file` with `MultiUsdFileCfg(usd_path=new_usd_paths, random_choice=False, replicate_physics=False)` — round-robin heterogeneous spawn (`:106-123`).
4. re-register sensor callbacks (`:133-135`).
5. build new `Articulation` from cfg with `spawn=None` before reset (`:146-151`).
6. `sim.reset()` reactivates Fabric and cooks physics (`:158-163`).
7. `log_prototype_and_instance_info` (`:171-172`).
8. patch `scene._articulations["robot"]` (`:177`).
9. re-bind `ActionTerm._asset` and `_offset` (`:187-194`).
10. re-bind `EventManager` class-term `asset`/`robot` (`:206-217`).
11. re-bind `CommandManager` term `robot`/`asset` (`:225-232`); final `env.reset()` (`:237-238`).
Used ONLY by the dead `_reload_morphology`; not exercised in the live path.

### 4.2 respawn.py — `apply_actuator_params` (lines 241-271)
Round-robin per env (`individual_idx = env_idx % num_individuals`), overwrites `IdentifiedActuator` tensor rows `tensor_attr[env_idx,:] = value` for each override that is a Tensor attribute (`:260-270`). Holds no state. No-op for CMA-ES populations (empty overrides).

### 4.3 update.py — in-place geometry update (lines 52-176)
- `apply_link_length_params(env, list)` (`:129-176`): safety-net `sim.stop()` (`:150-153`); iterates first `num_individuals` envs (each design once), for each scalable link resolves the `visuals`/`collisions` instance prototype and calls `update_articulation_links`; then a single timed `sim.reset()` (`:170-175`).
- `update_articulation_links(prototype_path, link_prim_path, extents, stage)` (`:52-126`): edits the prototype `mesh_0` `xformOp:scale` + `xformOp:translate` (`:108-112`), and idempotently the link prim `physics:mass`/`physics:diagonalInertia` (`:114-121`) and the child joint `physics:localPos0` (`:122-126`), each authored on its own resolved source layer via `_source_spec` (`:44-49`). Density recovered from `mass0/(x0*y0*z0)`; joint offset from `abs(p0[2]) - z0`. All inside one `Sdf.ChangeBlock` (`:106`). Holds no Python state; mutates the USD stage in place.

### 4.4 analysis.py — `log_prototype_and_instance_info` (lines 254-357)
Pure diagnostic: walks `stage.GetPrototypes()`, prints prototype→instance mapping, prototype children types, and `xformOp:scale` box dimensions per prototype, plus inferred env stride. No state, no physics mutation. Called in `respawn_robots` (live path commented out in `learn()` at copt runner `:118-119`).

---

## 5. modules/copt_actor_critic.py + observation wiring (DECISIVE for objective 1)

### 5.1 `CoptActorCritic` structure (lines 42-152)
- Actor input width = Σ over `obs_groups["policy"]` of `obs[group].shape[-1]` + `enc_hidden_dims[-1]` per group (`:61-67`). `enc_hidden_dims = encoder_cfg["hidden_dims"]` default `[128,64,16]` → latent size 16.
- Estimator = `MLP(num_one_step_privileged_obs, enc_size=16, hidden=[128,64], activation)` over `obs["privilegedObs"]` (`:93-105`).
- `_update_distribution` / `act` / `act_inference` (`:127-152`): `actor_input = cat(normalizer(get_actor_obs(obs)), estimator(obs["privilegedObs"]))`. So the actor sees the policy obs groups plus a **learned latent estimate** derived from `privilegedObs` (which contains link lengths).
- Critic is inherited unchanged from `rsl_rl.modules.ActorCritic`: `evaluate(obs) = critic(critic_obs_normalizer(get_critic_obs(obs)))`, where `get_critic_obs` concatenates `obs_groups["critic"]` (`/ws/rsl_rl/rsl_rl/modules/actor_critic.py:162-172`).
- NOTE: `EncoderCfg(num_output_dim=19, hidden_dims=[128,64,16])` (limx_rsl_rl_ppo_cfg.py:120-126), but `CoptActorCritic` derives latent size from `hidden_dims[-1]=16` and ignores `num_output_dim=19`. Minor inconsistency, not load-bearing.

### 5.2 Does the value function SEE the morphology? YES.
- The COPT env uses `CoptObservationsCfg` (limx_base_env_cfg.py:360-571) with groups `policy`, `critic`, `commands`, `privilegedObs` (`:567-570`).
- `PrivligedObsCfg` (`:424-457`) contains `link_lengths` (`mdp.robot_link_lengths` over the 4 scalable links), `robot_mass`, `heights`. This feeds the actor's estimator.
- `CriticCfg` (`:459-559`) explicitly contains `link_lengths` at `:534-552`, plus `robot_mass`, `robot_inertia`, `robot_joint_stiffness`, `robot_joint_damping`, `robot_material_properties`, etc.
- `obs_groups` is `MISSING` in the agent cfg (rl_cfg.py:159; saved agent.yaml shows `_MISSING_TYPE`). `OnPolicyRunner.__init__` resolves it via `resolve_obs_groups(obs, cfg["obs_groups"], default_sets=["critic"])` (on_policy_runner.py:42-45): "policy" group → `obs_groups["policy"]=["policy"]`; "critic" default set → since a `critic` group exists in obs, `obs_groups["critic"]=["critic"]`.
- Therefore the critic receives the `critic` group, which **includes the true (history-stacked) link lengths and mass/inertia**. The value function is NOT blind to morphology.
- Implication for objective 1 hypothesis (b): the critic explicitly observes the link lengths it must value, so "critic cannot see the morphology" is FALSE. Any value-estimation difficulty would have to come from (i) the actor receiving only a detached/learned latent estimate rather than the true morphology (encoder `output_detach=True`, limx_rsl_rl_ppo_cfg.py:121), (ii) `critic_obs_normalization` statistics drifting as designs change, or (iii) the metric/buffer artifacts in §2.3. The strongest, clearly-evidenced variance driver in this package is the §2.3 buffer flooding plus the §2.4 truncated-fitness contamination and the §3.5 growing design distribution, NOT critic blindness.

---

## 6. Iteration → generation → branch timeline

Assumptions: fresh run (`start_iter=0`), `ea_update_interval=120`, `ea_late_start=8000`, `randomise_before_late_start=True`, `late_start=True`, `late_start_it=66`, `num_individuals=64`, `num_steps_per_env=25`. `iter1 = it - start_iter + 1`.

| iter1 (it+1) | generation arg passed | `is_late_start_toggle_time` | `is_morph_update_time` | toggle? | `_update_morphology` branch | generator action |
|---|---|---|---|---|---|---|
| pre-loop call | 0 | n/a | n/a | n/a | else (8000≤0 False) | `_update_morphology(0)`: random scales, scale(g=0)=0.05 ≈ baseline; gen→1 |
| 120 | 1 | False | True | no | else | random, GrowingDist scale(g=1)≈0.064; gen→2 |
| 240 | 2 | False | True | no | else | random, scale(g=2)≈0.079; gen→3 |
| … | … | False | True | no | else | random, widening spread |
| 7920 | 66 | False (8000≤7920 F) | True | no | else | random, scale(g=66)≈1.0 (full range); gen→67 |
| 8040 | 67 | True (8000≤8040) | True | YES (late_start True→False) | CMA (8000≤8040) | `_copt_started`F→ `sample_batch`=ask1; `generate_population` builds pop from ask1, `_last`=ask1; fitness=F_random; `tell(ask1, -F_random)`; `sample_batch`=ask2; gen→68 |
| 8160 | 68 | True | True | no (late_start already F) | CMA | `generate_population` builds pop from ask2, `_last`=ask2; fitness=F_ask1; `tell(ask2, -F_ask1)`; ask3; gen→69 |
| 8280 | 69 | True | True | no | CMA | pop from ask3; fitness=F_ask2; `tell(ask3, -F_ask2)`; ask4; gen→70 |
| … | … | True | True | no | CMA | off-by-one persists: `tell(ask_{i+1}, -F_ask_i)` |
| up to 30000 | ~250 | True | True | no | CMA (or `_terminated` fine-tune) | ~183 CMA generations total `(30000-8000)/120` |

Key reading of the table: random/growing phase spans iters 0–7920 (gens 0–66); CMA-ES begins at iter 8040 (gen 67); the very first CMA `tell` pairs ask1 with random-phase fitness, and every subsequent `tell` pairs a batch with the prior batch's fitness (§2.5).

---

## 7. Bug / dead-code / discrepancy register

1. NameError: `GrowingDesignDistCMAESDesignGenerator` used at train.py:209, never imported (§1). Blocks all COPT runs from this entry point.
2. CMA-ES off-by-one tell/ask: `generate_population` called before `update_with_fitness` in `_update_morphology` (copt:430-435) misattributes fitness to the wrong solution batch (§2.5). Doc §4.9.4 has the correct order. Prime suspect for objective 3 collapse.
3. rewbuffer/lenbuffer flooded with `num_envs` partial returns every 120 iters (copt:275-276); deque maxlen=100 → logged reward stat at morph iters built from ~100 truncated partials (§2.3). Prime suspect for objective 1 periodic variance.
4. Fitness contaminated by truncated partials: morph-final step credits every env as one episode with mid-episode `cur_reward_sum` (copt:270-274) (§2.4). Biases CMA cost downward and adds noise.
5. `round(val,2)` design-resolution floor in `_compute_link_extents` (usd:372) erases sub-cm spread → identical designs near convergence (§3.4).
6. sigma0=0.75 in `[0,1]^2` box with hard bounds → oversized initial step, boundary pile-up (§3.2). Objective 3.
7. Save/load persists NO COPT state: generation, accumulators, `_copt_started`, current_population, and CMA-ES `_es`/pending/last/terminated/late_start. `CoptOnPolicyRunner` does not override save/load (§2.8). Objective 2.
8. `CMAESDesignGenerator.save_state` (usd:809) and `es_state_path` load hook exist but are never called/passed → CMA-ES checkpointing is inert (§2.8). Objective 2.
9. Late-start/morph schedule is relative to `start_iter`, not absolute iteration (copt:160,163) → resume repeats the entire random phase (§2.8). Objective 2.
10. Dead code: `_reload_morphology` (copt:383-414) never called; `respawn_robots` only used by it; both diverge from the live `_update_morphology` (different, and correct, fitness/sample ordering).
11. Minor: `CoptActorCritic` ignores `EncoderCfg.num_output_dim=19`, uses `hidden_dims[-1]=16` (§5.1). train.py comment "50 * 24 = 1200" is stale (actual num_steps_per_env=25 → 120×25=3000 control steps/gen; episode ≈1000 steps) (train.py:201-202).
12. Doc drift: ../CO_OPTIMISATION.md §4.9.4 states `learn()` initial spawn uses `_reload_morphology` and the in-loop block was `_reload_morphology`→`_update_morphology`; the actual code uses `_update_morphology(0)` for the initial spawn (copt:125) and the doc's `_update_morphology` body (always `update_with_fitness` then `generate_population`) does not match the live body (CMA gating + reversed order).

---

## 8. Code state update (2026-07-02, supersedes stale entries above)

Re-verification of the live sources shows several defects registered above have
since been fixed. The following facts supersede sections 1, 2.3, 2.5, 2.8 and 7.

1. `train.py` now imports `GrowingDesignDistCMAESDesignGenerator` (train.py:99-104),
   so the NameError of section 1 is resolved.
2. `train.py` COPT branch (train.py:196-237) now uses `ea_late_start = 12000`,
   `sigma0 = 0.25`, explicit `param_ranges = {thigh_length_scale: (0.75, 1.25),
   shank_length_scale: (0.75, 1.25)}`, `ea_update_interval = 120`,
   `_num_individuals = 64`. `SFCoptPPORunnerCfg.max_iterations` is now 45000
   (limx_rsl_rl_ppo_cfg.py:131-133).
3. `CoptOnPolicyRunner` now overrides `save` and `load`
   (copt_on_policy_runner.py:142-217). `save` augments the base checkpoint with
   `learning_rate`, `tot_timesteps`, `tot_time`, a `copt` dict (generation,
   copt_started, fitness accumulators, current population,
   design generator state) and `env_state` via
   `co_optimisation.utils.env_state.capture_env_state`. `load` restores all of it
   and stashes population plus env state for `_apply_restored_state`
   (copt_on_policy_runner.py:538-574), which `learn` consumes on resume.
4. The morph schedule is now absolute, `is_late_start_toggle_time =
   self._ea_late_start <= (it + 1)` (copt_on_policy_runner.py:288), so a resumed
   run no longer repeats the random phase. Section 2.8's schedule bug is fixed.
5. The rewbuffer flood is fixed. At a morph update the runner zeroes
   `cur_reward_sum` and `cur_episode_length` without extending the deques,
   discarding partial returns (copt_on_policy_runner.py:395-404).
6. The tell/ask off-by-one is fixed. `_update_morphology`
   (copt_on_policy_runner.py:576-619) now computes fitness and calls
   `update_with_fitness` BEFORE `generate_population`.
7. `CoptPPO` (algorithms/copt_ppo.py) now subclasses `rsl_rl.algorithms.PPO`
   with (a) `compute_returns_design_wise` implementing per-individual advantage
   normalisation (defined but `learn()` still calls plain `compute_returns`) and
   (b) `update()` appending `explained_variance` to the loss dict.
8. `CoptOnPolicyRunner.__init__` reads `self.encoder_cfg = train_cfg["encoder"]`
   (copt_on_policy_runner.py:125) and `_construct_algorithm` passes it as the
   fourth positional argument to the policy class, then constructs the algorithm
   as `alg_class(actor_critic, self._num_individuals, env_to_individual mapping,
   device=..., **alg_cfg, multi_gpu_cfg=...)` (copt_on_policy_runner.py:470-489).
   Any new policy or algorithm class name must be importable in this module for
   the `eval(class_name)` resolution to succeed.

---

## 9. Learned-model implementation record (2026-07-03)

Section 3 of `../plans/COPT_LEARNED_MODEL.md` was implemented in full against the live
sources described in section 8 above, not against the possibly stale line
numbers the document quotes. Every step compiled cleanly under
`python3 -m py_compile`, and every new class name, import, and task ID was
cross-referenced across files with grep to confirm single, consistent
definitions. The pipeline has not been exercised inside Isaac Sim.

New files created, `co_optimisation/co_optimisation/modules/copt_estimator.py`
(`CoptEstimator`, `get_activation`, ported verbatim from
`himloco/himloco/modules/him_estimator.py:149-168`).

Files edited, `utils/wrappers/rsl_rl/rl_mlp_cfg.py` (`DecoderCfg`, mirrors
`EncoderCfg`), `agents/limx_rsl_rl_ppo_cfg.py` (`SFCoptLearnedModelPPORunnerCfg`),
`cfg/SF/limx_base_env_cfg.py` (`CoptLearnedModelObservationsCfg`,
`SFCoptLearnedModelEnvCfg`), `robots/limx_solefoot_env_cfg.py` (six scenario
classes, `SFCoptLearnedModelBaseEnvCfg[_PLAY]`,
`SFCoptLearnedModelBlindFlatEnvCfg[_PLAY]`,
`SFCoptLearnedModelBlindRoughEnvCfg[_PLAY]`), `robots/__init__.py` (four
`gym.register` calls for `Isaac-Limx-SF-Copt-Learned-{Flat,Rough}[-Play]-v0`),
`co_optimisation/modules/copt_actor_critic.py` (`CoptLearnedModelActorCritic`),
`co_optimisation/modules/__init__.py`,
`co_optimisation/algorithms/copt_ppo.py` (`CoptLearnedModelPPO`),
`co_optimisation/algorithms/__init__.py`,
`co_optimisation/runners/copt_on_policy_runner.py` (import extension,
`self.decoder_cfg`, decoder-conditional branch in `_construct_algorithm`),
`scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`, `scripts/rsl_rl/cli_args.py`,
`/ws/djinn`, `/ws/ARCHITECTURE.md`, `/ws/CO_OPTIMISATION.md`.

Two ambiguities in `../plans/COPT_LEARNED_MODEL.md` were resolved during
implementation, both grounded in cross-checks against the rest of the
document rather than guesswork.

1. Section 3.3 instructs copying `CoptObservationsCfg.PolicyCfg` verbatim into
   the new `CoptLearnedModelObservationsCfg.PolicyCfg` but also states no
   history is used and the policy input is a single step, whereas the
   original `PolicyCfg` carries a ten-step flattened history
   (`history_length=10`, `flatten_history_dim=True`,
   `limx_base_env_cfg.py:416-421`). The observation terms were copied
   verbatim as instructed, but `__post_init__` was written with
   `history_length` left unset (IsaacLab default `0`, confirmed at
   `manager_term_cfg.py:184`), giving a single-step 36-wide policy group. This
   matches the formulation in section 2.1 (`S_a` is a single-step actor
   state) and makes `obs["policy"].shape[-1]` in the section 3.7
   `CoptLearnedModelActorCritic.__init__` code numerically identical to
   `obs["obsHistory"].shape[2]`, since both groups share the same 36-wide term
   set, so the encoder input dimension is correct either way.
2. Section 3.4 instructs appending the rough scenario classes "after the
   definition of `class SFCoptLearnedModelBlindRoughEnvCfg_PLAY` (after line
   432)", a self-referential instruction since that class does not exist yet
   at that point in the plan. This was read as a typo for
   `SFCoptBlindRoughEnvCfg_PLAY`, the existing COPT rough play class that
   occupies that line range, and the new classes were inserted immediately
   after it, mirroring the placement pattern already used for the flat
   scenario classes after `SFCoptBlindFlatEnvCfg_PLAY`.

../README.md updates from section 3.12 were skipped. Neither `/ws/README.md` nor
`/ws/tron1-rl-isaaclab-cozum/README.md` documents `djinn start train copt`
usage today (grep confirms only `base` mode and a generic `train.py`
invocation are shown), so there was no existing sibling block to extend
without inventing unrequested new documentation sections.

## 10. The second learned-model variant and its unreachable branch (2026-07-30)

The package carries a second learned-model variant beyond anything recorded in `../plans/COPT_LEARNED_MODEL.md` or in section 9 above. It comprises two classes, `CoptLearnedModelV2ActorCritic` at `modules/copt_actor_critic.py:266`, which subclasses `CoptLearnedModelActorCritic` and overrides `_get_estimator_input` alone, and `CoptLearnedModelV2PPO` at `algorithms/copt_ppo.py:594`, which subclasses `CoptLearnedModelPPO` and overrides `update`.

Two substantive differences separate the variants, and both point the same way. The first lies in the estimator input, `CoptLearnedModelActorCritic._get_estimator_input` concatenates `obs["morphologyObs"]` with the flattened normalised `obs["obsHistory"]`, whereas the V2 override returns the flattened normalised history alone, discarding the privileged morphology channel. The second lies in the decoder regression target, `CoptLearnedModelPPO.update` regresses the predicted model information against `obs_batch["predictedPrivilegedObs"]` at `copt_ppo.py:500`, whereas `CoptLearnedModelV2PPO.update` regresses it against `obs_batch["predictedMorphologyObs"]` at `copt_ppo.py:719`, so the V2 decoder is asked to reconstruct the morphology rather than the dynamic state.

Taken together the two changes convert the estimator from a privileged encoder into an inference module, the V2 latent must infer the morphology from proprioceptive history rather than read it from a privileged channel, and is supervised directly on that inference. This is the hybrid internal model premise applied to the co-optimisation setting, and it is therefore the variant that could in principle be deployed on hardware without privileged access. No document states this rationale, so it is recorded here as inference from the code rather than as the author's stated intent.

The variant cannot presently be selected. The guard at `scripts/rsl_rl/train.py:196` reads `if args_cli.policy_type in ("COPT", "COPT-LEARNED")`, and the branch that would assign the V2 class names sits at `train.py:225` inside that guarded block, so the condition `elif args_cli.policy_type == "COPT-LEARNED-2"` can never be true where it is evaluated. The mode is nonetheless exposed, `djinn` maps `start train copt-learned-2` to the task `Isaac-Limx-SF-Copt-Learned-Rough-v0` with `--policy-type COPT-LEARNED-2`, which bypasses the co-optimisation runner construction entirely and falls through to the plain `runner_cls` path, so the invocation will not produce a co-optimisation run and will most likely fail when the configured policy class is not found. The remedy is to add the third policy type to the tuple in the guard, which restores the branch without altering the behaviour of either existing mode.

Note also for the record that `play.py` handles the policy types HIMPPO, COPT, and COPT-LEARNED, and carries no branch for COPT-LEARNED-2, so evaluation of a checkpoint from that variant would need one.
