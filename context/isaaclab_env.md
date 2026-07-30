# isaaclab_env.md — IsaacLab Env & Manager State Inventory (Agent B)

> Scope. Enumerate, with file:line precision, every configuration and runtime state held by
> the IsaacLab `ManagerBasedRLEnv` / `ManagerBasedEnv` and their managers, so a save/load
> mechanism can persist and restore it for identical experiment continuation. Also document
> the env reset semantics relevant to the COPT training-variance investigation.
>
> Installed source root: `/ws/IsaacLab/source/isaaclab/isaaclab/`.
> COPT task config: `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`.
> COPT runner: `/ws/tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`.

---

## 0. TL;DR for save/load and variance

- The single most important persistent driver of curriculum progress is
  `ManagerBasedRLEnv.common_step_counter` (`manager_based_rl_env.py:74`, incremented only at
  `:202`). It is **never reset** by `reset()` / `_reset_idx()`. Every project curriculum term
  gates on `env.common_step_counter` (e.g. `curriculums.py:36,64,120,172,243,295,332,381,431,475`).
- The actual "current difficulty" is **not stored in the CurriculumManager**. It lives in the
  objects the term functions mutate, none of which are in any checkpoint:
  - `TerrainImporter.terrain_levels`, `.terrain_types`, `.env_origins`, `.max_terrain_level`
    (`terrain_importer.py:347,348,353,345`), advanced by `update_env_origins`
    (`terrain_importer.py:314-329`).
  - The live command ranges `command_manager.get_term("base_velocity").cfg.ranges.{lin_vel_x,
    lin_vel_y,ang_vel_z}` (mutated at `curriculums.py:449,353,402`).
  - The push event range `event_manager.get_term_cfg("push_robot").params["velocity_range"]`
    (mutated at `curriculums.py:186,191,205,210`).
  - The tracking-reward std `reward_manager.get_term_cfg(...).params["std"]`
    (mutated at `curriculums.py:305`).
- `unwrapped_env.reset()` (called every 120 iters by COPT, `copt_on_policy_runner.py:449`)
  zeroes `episode_length_buf` for ALL envs (`manager_based_rl_env.py:394`) and **calls
  `curriculum_manager.compute(env_ids=all)`** (`manager_based_rl_env.py:356`), which forces a
  synchronized terrain-level promotion/demotion and evaluates every curriculum gate for all
  envs at once. This is the env-side interaction with training variance (see §6).

---

## 1. `ManagerBasedEnv` (`envs/manager_based_env.py`)

### 1.1 Runtime state members (set in `__init__`, lines 77-198)
| Member | Line | Nature |
|---|---|---|
| `self._is_closed` | 92 | bool flag |
| `self.cfg` | 90 | config object (deep state incl. mutated curriculum targets) |
| `self.cfg.seed` | 95-96 | seed, set through `self.seed(...)` |
| `self.sim` | 104 / 110 | `SimulationContext` — PhysX/USD, NOT picklable |
| `self._sim_step_counter = 0` | 133 | total physics-substep counter, increment at `step()` `:478` (base) / `manager_based_rl_env.py:183`. **NOT reset by `reset()`**. Used to derive `global_env_step_count` for reset-mode events (`:567`, `manager_based_rl_env.py:361`). |
| `self.extras = {}` | 136 | log/metrics dict; `extras["log"]` rebuilt every `_reset_idx` (`:573`) |
| `self.scene` | 142 | `InteractiveScene` — USD/PhysX, NOT picklable |
| `self.viewport_camera_controller` | 151/153 | UI handle |
| `self.event_manager` | 158 | created before `sim.reset()` |
| `self._window` | 188/191 | UI |
| `self.obs_buf = {}` | 194 | last observation dict; recomputed by `observation_manager.compute()` |

### 1.2 `reset()` (lines 341-393)
- `env_ids=None` → resets **all** envs (`:361-362`). COPT calls it with no args → all envs.
- Optional `seed` re-seeds globally (`:368-369`).
- Calls `self._reset_idx(env_ids)` (`:372`), then `scene.write_data_to_sim()`, `sim.forward()`,
  recomputes `obs_buf` via `observation_manager.compute(update_history=True)` (`:386`).
- **Does NOT touch** `common_step_counter`, `_sim_step_counter`, or any RNG. It only resets the
  per-env episodic state described in `_reset_idx`.

### 1.3 base `_reset_idx()` (lines 556-585)
`scene.reset(env_ids)`; reset-mode events (`:566-568`); rebuild `extras["log"]={}` (`:573`);
call `observation/action/event/recorder` manager `.reset(env_ids)`. (RL override adds more, §2.3.)

### 1.4 `seed()` (lines 504-522) and RNG location
- Static method → `configure_seed(seed)` (`utils/seed.py:14-45`).
- Seeds **global** generators only: `random.seed` (`seed.py:27`), `np.random.seed` (`:28`),
  `torch.manual_seed` (`:29`), `torch.cuda.manual_seed[_all]` (`:31-32`), `wp.rand_init` (`:33`),
  `os.environ["PYTHONHASHSEED"]` (`:30`), and replicator global seed (`manager_based_env.py:518`).
- **There is no per-env RNG generator object.** All stochasticity (command resampling, event
  randomization, noise models) draws from these global RNGs. For an identical resume the global
  RNG states must be captured/restored: `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`,
  `np.random.get_state()`, `random.getstate()` (warp has no public get-state).

---

## 2. `ManagerBasedRLEnv` (`envs/manager_based_rl_env.py`)

### 2.1 Runtime state members
| Member | Line | Nature / reset behaviour |
|---|---|---|
| `self.common_step_counter = 0` | 74 | incremented at `:202`; **NEVER reset** by reset/`_reset_idx`. Drives all curriculum gates. MUST be persisted. |
| `self.episode_length_buf` | 77 | `torch.zeros(num_envs, long)`; `+=1` at `:201`; zeroed per-env at `_reset_idx` `:394`. |
| `self.reset_buf` | 204 | termination union, recomputed each step |
| `self.reset_terminated` | 205 | from `termination_manager.terminated` |
| `self.reset_time_outs` | 206 | from `termination_manager.time_outs` |
| `self.reward_buf` | 208 | from `reward_manager.compute` |
| `self.command_manager` | 113 | see §3 |
| `self.termination_manager` | 121 | see §5 |
| `self.reward_manager` | 124 | see §4 |
| `self.curriculum_manager` | 127 | see §3 / §7 |
| `self.single_observation_space` / `observation_space` / action spaces | 324-347 | derived, re-derivable |
| `render_fps` metadata | 87 | constant |

`max_episode_length` (property, `:101-103`) = `ceil(episode_length_s / step_dt)`. For SF:
`episode_length_s=20.0` (`limx_base_env_cfg.py:1326`), `decimation=4` (`:1325`), `sim.dt=0.005`
(`:1329`) → `step_dt=0.02` → **`max_episode_length = 1000`**. `num_envs = 4096`
(`limx_base_env_cfg.py:1312`).

### 2.2 `step()` (lines 153-241)
Order: process action → decimation physics loop (increments `_sim_step_counter` `:183`) →
`episode_length_buf += 1` (`:201`) → `common_step_counter += 1` (`:202`) →
`termination_manager.compute()` (`:204`) → `reward_manager.compute()` (`:208`) →
`_reset_idx(reset_env_ids)` for terminated envs (`:221`) → `command_manager.compute()` (`:232`)
→ interval events (`:234-235`) → `observation_manager.compute(update_history=True)` (`:238`).

### 2.3 RL `_reset_idx()` (lines 349-395) — what reset DOES and does NOT do
DOES (in order):
1. `curriculum_manager.compute(env_ids=env_ids)` (`:356`) — **advances curriculum** for those
   envs (terrain level up/down, evaluates command-velocity / push-force / std gates). This is
   the only place curriculum advances.
2. `scene.reset(env_ids)` (`:358`) — scene buffers.
3. reset-mode events (`:360-362`) — e.g. `reset_robot_base`, `reset_robot_joints`
   (`limx_base_env_cfg.py:972,988`); push_robot is interval, not reset.
4. rebuild `extras["log"]={}` (`:367`); call `observation/action/reward/curriculum/command/event/
   termination/recorder` manager `.reset(env_ids)` (`:368-391`).
5. `episode_length_buf[env_ids] = 0` (`:394`).

DOES NOT: reset `common_step_counter`, `_sim_step_counter`, RNG, the terrain levels back to
initial, or the mutated command/push/std config. `curriculum_manager.reset()` (`:378`) is
logging-only here (no project curriculum term is a class, so `_class_term_cfgs` is empty;
see §3).

---

## 3. CurriculumManager + CommandManager state

### 3.1 CurriculumManager (`managers/curriculum_manager.py`)
| Member | Line | Nature |
|---|---|---|
| `_term_names` | 49 | static |
| `_term_cfgs` | 50 | config (params are mutated in place by some terms, e.g. event/reward term refs) |
| `_class_term_cfgs` | 51 | **empty for this project** (all SF curriculum funcs are plain functions, not `ManagerTermBase`) |
| `_curriculum_state` | 57 | dict term→last-returned-state; written by `compute` (`:140`), read for logging by `reset` (`:105-118`). Logging only (e.g. mean terrain level); re-derivable. |

`compute(env_ids)` (`:125-140`): for each term calls `term_cfg.func(env, env_ids, **params)` and
stores the return in `_curriculum_state`. The real progress lives in the targets the funcs mutate
(§7). `reset()` (`:93-123`): logs `_curriculum_state` then calls `func.reset` only for class terms
(none here).

### 3.2 CommandManager (`managers/command_manager.py`)
- `_terms` dict (`:251`), `_commands` dict (`:256`), `cfg.debug_vis` (`:258`).
- `reset(env_ids)` (`:335-359`) calls each term's `reset`. `compute(dt)` (`:361-373`) calls each
  term's `compute`.

### 3.3 `CommandTerm` base state (`command_manager.py`)
| Member | Line | Reset behaviour |
|---|---|---|
| `self.metrics` (dict) | 52 | accumulated in `_update_metrics`; mean-logged and zeroed per-env at `reset` (`:139-143`) |
| `self.time_left` | 54 | resample timer; sampled in `_resample` (`:184`); decremented in `compute` (`:161`) |
| `self.command_counter` | 56 | zeroed per-env at reset (`:146`); `+=1` per resample (`:188`) |

### 3.4 `UniformVelocityCommand` (`envs/mdp/commands/velocity_command.py`, term name `base_velocity`)
| Member | Line | Nature |
|---|---|---|
| `self.vel_command_b` | 82 | current command (num_envs,3); resampled `:130-134` |
| `self.heading_target` | 83 | per-env heading goal |
| `self.is_heading_env` | 84 | mask |
| `self.is_standing_env` | 85 | standing mask (zeroes command in `_update_command` `:162-163`) |
| `self.metrics["error_vel_xy"]`, `["error_vel_yaw"]` | 87-88 | logged metrics |
| `self.cfg.ranges.lin_vel_x / lin_vel_y / ang_vel_z` | (cfg) | **MUTATED by the velocity curriculum** (`curriculums.py:449,353,402`). The live, curriculum-advanced sampling ranges. NOT in any checkpoint. Read via `command_manager.get_term("base_velocity").cfg.ranges`. |

(`NormalVelocityCommand` adds `is_zero_vel_{x,y,yaw}_env` at `:243-245`; not used by SF base.)

---

## 4. RewardManager (`managers/reward_manager.py`)
| Member | Line | Reset behaviour |
|---|---|---|
| `_term_names` / `_term_cfgs` / `_class_term_cfgs` | 52-54 | config; `term_cfg.params["std"]` of `rew_lin_vel_xy` / `rew_ang_vel_z` is **mutated** by `reduce_tracking_rewards_std` (`curriculums.py:305`) |
| `_episode_sums` (dict term→(num_envs,)) | 59-61 | per-term per-env accumulator; `+=` each step (`:154`); mean-logged and zeroed per-env at `reset` (`:119-122`). **Read by curriculum gates**: `env.reward_manager._episode_sums[term_name][env_ids]` (`curriculums.py:299,337,386,436`). |
| `_reward_buf` | 63 | recomputed each step (`:142`) |
| `_step_reward` (num_envs, n_terms) | 66 | per-step per-term value (logging/visualizer) |

---

## 5. TerminationManager (`managers/termination_manager.py`)
| Member | Line | Reset behaviour |
|---|---|---|
| `_term_names` / `_term_cfgs` / `_class_term_cfgs` | 58-60 | config |
| `_term_name_to_term_idx` | 64 | static map |
| `_term_dones` (num_envs, n_terms, bool) | 66 | per-step buffer, overwritten each `compute` (`:175`); read by curriculum via `get_term` (`curriculums.py:176,195,247,264`) |
| `_last_episode_dones` (num_envs, n_terms, bool) | 68 | **PERSISTENT, monotonic-saturating**; only updated in `compute` for envs that fired a term (`:178-180`). A full `env.reset()` does NOT update it (bypasses `compute`). This is the RESET.md flat-`Episode_Termination/*` driver. Must be persisted for identical logging continuation. |
| `_truncated_buf` | 70 | recomputed each step |
| `_terminated_buf` | 71 | recomputed each step |

`reset()` (`:129-152`) logs `_last_episode_dones.float().mean(dim=0)` over ALL envs (`:144`),
independent of `env_ids`.

---

## 6. ObservationManager, ActionManager, EventManager state

### 6.1 ObservationManager (`managers/observation_manager.py`)
| Member | Line | Nature |
|---|---|---|
| `_obs_buffer` | 115 | last computed obs cache; re-derivable via `compute()` |
| `_group_obs_term_history_buffer[group][term]` (`CircularBuffer`) | 475 (created 638-641) | **history buffers** (true runtime state); appended in `compute_group` (`:412`); reset per-env at `reset` (`:307-309`). For SF the policy/critic groups use history; these buffers hold the last `history_length` observations and affect the very next observation after a resume. |
| `_group_obs_class_instances` (modifiers / noise models) | 478 | class instances; may carry internal state (e.g. `NoiseModel`); `reset` at `:311-312` |
| group dims / names / cfgs | 468-473 | static, re-derivable |

There is **no observation normalization state in this manager**. Empirical/running normalization
lives in the rsl_rl policy (Agent A), not IsaacLab.

### 6.2 ActionManager (`managers/action_manager.py`)
| Member | Line | Reset behaviour |
|---|---|---|
| `_action` (num_envs, action_dim) | 215 | set each `process_action` (`:386`); zeroed per-env at reset (`:365`) |
| `_prev_action` | 216 | **previous action used as an observation term**; copied at `:385`; zeroed per-env at reset (`:364`) |
| `_terms` / `_term_names` | 430,429 | config |

### 6.3 EventManager (`managers/event_manager.py`)
| Member | Line | Nature |
|---|---|---|
| `_mode_term_names` / `_mode_term_cfgs` / `_mode_class_term_cfgs` | 72-74 | config; `push_robot` term's `params["velocity_range"]` is **MUTATED by `modify_push_force`** (`curriculums.py:186-211`) |
| `_interval_term_time_left[index]` | 331 (sampled 405/409) | per-env interval timer for `push_robot` (interval mode); decremented in `apply` (`:212`); resampled per-env at `reset` (`:148`) |
| `_reset_term_last_triggered_step_id[index]` | 333 (init 420-421) | per-env last-trigger step for reset-mode terms with `min_step_count_between_reset>0` |
| `_reset_term_last_triggered_once[index]` | 334 (init 423-424) | per-env "fired at least once" flags |

#### Startup randomization (sampled ONCE, never re-sampled at reset)
Applied at `manager_based_rl_env.py:134-135` (`event_manager.apply(mode="startup")`). SF startup
terms (`limx_base_env_cfg.py`): `add_base_mass` (`:855` `randomize_rigid_body_mass`),
`robot_physics_material` (`:874`, friction), `robot_joint_stiffness_and_damping_*`
(`:885,897,910,923` `randomize_actuator_gains`), `joint_offsets` (`:936`
`randomize_joint_default_pos`), `joint_friction` (`:946` `randomize_joint_friction_model`),
`scale_all_joint_armature` (`:1023` `randomize_joint_parameters`). These write randomized values
into the **articulation / PhysX tensors and `robot.data.default_*`** (and depend on the global RNG
sequence at startup). They are NOT re-applied by `env.reset()`. To keep the morphology identical
on resume, either reproduce the exact startup RNG sequence or capture the resulting per-env
physics tensors (mass, friction, stiffness, damping, armature, default joint pos) at save time.
Note: the COPT runner overwrites actuator tensors after each reset via `apply_actuator_params`
(`copt_on_policy_runner.py:452-454`), so live actuator params are design-driven, not from these
startup terms, after the first morphology update.

---

## 7. Project curriculum MDP functions — in-memory state read/written
File: `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/curriculums.py`.
SF configured terms: `limx_base_env_cfg.py:1184-1262`.

| Term (SF) | curriculums.py | Reads | Writes (the persistent progress) |
|---|---|---|---|
| `terrain_levels` = `terrain_levels_vel_delayed` | 454-493 | `asset.data.root_pos_w`, `env.scene.env_origins`, command, `common_step_counter` (gate `>2000*24`, `:475`) | `terrain.update_env_origins` → `TerrainImporter.terrain_levels`, `.env_origins` (`terrain_importer.py:320,329`) |
| `modify_push_force` | 143-213 | `termination_manager.get_term("base_contact"/"time_out")`, `common_step_counter` (gate `%interval`, start `6000*24`) | `event_manager` `push_robot` `params["velocity_range"]["x"/"y"]` (`:186,191,205,210`) |
| `modify_command_velocity_lin_x` = `modify_command_velocity_x` | 407-451 | `reward_manager._episode_sums["rew_lin_vel_xy"]`, term weight, `common_step_counter` | `command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x` (`:449`) |
| `modify_command_velocity_lin_y` = `modify_command_velocity_y` | 309-355 | same | `...cfg.ranges.lin_vel_y` (`:353`) |
| `modify_command_velocity_ang_z` = `modify_command_velocity_angular` | 358-404 | `reward_manager._episode_sums["rew_ang_vel_z"]` | `...cfg.ranges.ang_vel_z` (`:402`) |
| `modify_linear/angular_tracking_reward_std` = `reduce_tracking_rewards_std` | 284-306 | `reward_manager._episode_sums[term]`, `max_episode_length`, weight | `reward_manager.get_term_cfg(term).params["std"]` (`:305`) |

These mutate **config-derived in-memory tensors/scalars that exist in no checkpoint**. The
`CurriculumManager._curriculum_state` only caches the scalar return value for logging; it is not
the source of truth.

`TerrainImporter` state (`terrains/terrain_importer.py`): `terrain_levels` (`:347`),
`terrain_types` (`:348`, constant), `env_origins` (`:353`), `terrain_origins` (`:303`, constant),
`max_terrain_level` (`:345`, constant). `terrain_levels` and `env_origins` are the evolving
curriculum state.

---

## 8. Categorized state table (a) serializable / (b) re-derivable / (c) USD-PhysX bound

### (a) Cheaply serializable tensors / scalars — MUST be saved for identical resume
| State | Location |
|---|---|
| `common_step_counter` (int) | `manager_based_rl_env.py:74` |
| `_sim_step_counter` (int) | `manager_based_env.py:133` |
| `episode_length_buf` (num_envs, long) | `manager_based_rl_env.py:77` |
| `TerrainImporter.terrain_levels`, `.env_origins`, `.max_terrain_level` | `terrain_importer.py:347,353,345` |
| Command `vel_command_b`, `heading_target`, `is_heading_env`, `is_standing_env`, `time_left`, `command_counter`, `metrics` | `velocity_command.py:82-88`, `command_manager.py:54,56` |
| Mutated command ranges `cfg.ranges.{lin_vel_x,lin_vel_y,ang_vel_z}` | via `command_manager.get_term("base_velocity").cfg.ranges` |
| `TerminationManager._last_episode_dones`, `_term_dones` | `termination_manager.py:68,66` |
| `RewardManager._episode_sums` (per-term per-env) | `reward_manager.py:59` |
| Mutated reward std `term_cfg.params["std"]` | `reward_manager.get_term_cfg(...)` |
| Mutated push range `push_robot params["velocity_range"]` | `event_manager.get_term_cfg("push_robot")` |
| `EventManager._interval_term_time_left`, `_reset_term_last_triggered_step_id/_once` | `event_manager.py:331,333,334` |
| `ActionManager._action`, `_prev_action` | `action_manager.py:215,216` |
| `ObservationManager` history `CircularBuffer` contents per term | `observation_manager.py:475` |
| Global RNG states (torch, torch.cuda, numpy, python random) | global (see §1.4) |

### (b) Require re-derivation (recomputed, do NOT need explicit persistence)
| State | Why |
|---|---|
| `reward_buf`, `reset_buf`, `reset_terminated`, `reset_time_outs`, `_truncated_buf`, `_terminated_buf`, `_step_reward`, `_reward_buf` | recomputed every `step()` |
| `obs_buf`, `ObservationManager._obs_buffer` | recomputed by `observation_manager.compute()` |
| `CurriculumManager._curriculum_state` | logging cache; reproduced on next `compute()` |
| gym observation/action spaces | derived at load |
| `TerrainImporter.terrain_types`, `.terrain_origins` | deterministic from cfg + seed (constant after init) |

### (c) Tied to USD / PhysX — cannot be pickled (extract/re-apply, not serialize directly)
| State | Location |
|---|---|
| `SimulationContext` (`self.sim`) | `manager_based_env.py:104` |
| `InteractiveScene` (`self.scene`), articulations, `root_physx_view`, USD stage | `manager_based_env.py:142` |
| Physical robot state (root pose, joint pos/vel) — extract via `scene.get_state()` / restore via `reset_to` | `manager_based_env.py:395-450` |
| Startup-randomized physics tensors (mass, friction, stiffness, damping, armature, default joint pos) | written by startup events (§6.3) into PhysX/`robot.data`; capture tensors or reproduce RNG |
| Timeline / debug-vis callback handles (`_resolve_terms_handle`, `_debug_vis_handle`), UI window, noise-model class instances | `manager_base.py:167`, `command_manager.py:59`, `action_manager.py:60`, `observation_manager.py:478` |
| Bound class-term `func` instances (event class terms) | `manager_base.py:418` |

---

## 9. Variance objective — exact reset semantics

Confirmed for the COPT full reset at `copt_on_policy_runner.py:449` (`unwrapped_env.reset()`,
no args → `env_ids=None` → all 4096 envs):

1. `episode_length_buf` is zeroed for ALL envs (`manager_based_rl_env.py:394`). With
   `ea_update_interval=120` → `120*24 = 2880` control steps between resets, which **exceeds**
   `max_episode_length = 1000`, so episodes CAN complete and `time_out` CAN fire between resets
   (unlike the interval=10/240-step case analysed in RESET.md). The reset still truncates whatever
   episodes are mid-flight at the 120-iter boundary, phase-locking all envs to step 0.

2. The full reset DOES perturb curriculum/command state mid-training, because `_reset_idx` calls
   `curriculum_manager.compute(env_ids=all)` (`manager_based_rl_env.py:356`) on every reset:
   - `terrain_levels_vel_delayed` (active after `common_step_counter > 2000*24`) evaluates
     distance walked from `env_origins` for ALL envs and calls `update_env_origins`, so terrain
     levels are promoted/demoted for the entire population **synchronously** every 120 iters.
     Because the reset truncates episodes early, "distance walked" is whatever was covered since
     the last reset rather than over a full episode, biasing the move_up/move_down decision
     (`curriculums.py:483-491`) and injecting a step change in terrain difficulty (hence the
     observation distribution and reward scale) every 120 iters across all envs at once.
   - `modify_push_force`, `modify_command_velocity_*`, `reduce_tracking_rewards_std` all gate on
     `common_step_counter % interval == 0` (`curriculums.py:174,335,384,434,298`); calling
     `compute` at the extra full-reset can double-trigger a gate whenever the counter lands on a
     multiple of the interval, nudging the live command ranges / push range / std off the schedule
     a single-morphology run would follow.

3. `common_step_counter` is NOT reset (`:74,202`), so the curriculum schedule clock keeps
   advancing across resets — which is correct, but it means the synchronized terrain/command
   perturbations ride on top of a monotonically advancing schedule, producing a sawtooth in
   curriculum-derived quantities at the 120-iter cadence. This is an env-side contributor to the
   plotted variance distinct from (and additive to) the rsl_rl rolling-buffer artefact noted in
   RESET.md.

Net for variance: the every-120-iter full reset both (a) truncates episodes and zeroes
`episode_length_buf` for all envs, and (b) forces a synchronized curriculum step (terrain
up/down + gate evaluation) for all envs, neither of which a baseline single-morphology run does.
The terrain-level curriculum in particular is mutated in `TerrainImporter` memory and is absent
from every checkpoint, so it neither persists across a resume nor stabilises over training.

## Rigid body physics tensor layouts and the reflection of morphology (added 2026-07-22)

Verified from the IsaacLab checkout at `/ws/IsaacLab` and NVIDIA docs, for mirroring privileged observations. Sources, `isaaclab/envs/mdp/events.py` (mass 340-397, inertia 387-397, material 205-283, com 400-438), `isaaclab/assets/articulation/articulation_data.py` (150-165 mass and inertia docstrings, 712-728 com), IsaacLab discussion 506, PhysX PxMaterial docs, MIT OCW 16.07 Lecture L26.

- Body and DOF ordering. Isaac Sim orders articulation links and DOFs BREADTH FIRST from the root via the PhysX stage parser (IsaacLab discussion 506), NOT by URDF declaration order, and the interleaving of left and right siblings within a depth level is whatever the parser emits, so the definitive order must be read at runtime from `env.scene["robot"].body_names` and `.joint_names`. For SD_BRS1 the observed order is LEFT before RIGHT at each level.
- Masses, `root_physx_view.get_masses()` shape (num_envs, num_bodies), body order equals body_names, scalar per body. Surfaced as `asset.data.default_mass`.
- Inertias, `get_inertias()` shape (num_envs, num_bodies, 9), the 9 a flattened symmetric 3x3 documented column major [Ixx, Iyx, Izx, Ixy, Iyy, Izy, Ixz, Iyz, Izz], numerically identical to row major by symmetry. Surfaced as `asset.data.default_inertia`. Under a sagittal reflection R=diag(1,-1,1) the tensor goes to R I R^T, which flips the sign of the products coupling the lateral y axis, flattened indices {1,3,5,7} (Iyx, Ixy, Izy, Iyz), and leaves the diagonal moments and the Ixz pair unchanged, sign vector [+,-,+,-,+,-,+,-,+]. A reflection negates every product of inertia containing the reflected axis, MIT OCW 16.07 L26.
- Material properties, `get_material_properties()` shape (num_envs, num_shapes, 3), the 3 columns [static_friction, dynamic_friction, restitution]. num_shapes is `root_physx_view.max_shapes`, a body may own more than one collision shape (mesh convex decomposition), and shapes are laid out body by body following body order, per body slice [sum(counts[:b]), sum(counts[:b+1])). Per body shape counts are computed in `randomize_rigid_body_material.__init__` by iterating `view.link_paths[0]`. All three are scalars invariant under reflection, so a left right mirror swaps a body's shape block with its partner's, no sign change. For SD_BRS1 the identified mesh gives 19 shapes total, not 13.
- Frames and vector classes. Scalars (mass, friction, restitution) invariant. Center of mass position and linear velocity are polar vectors, flip lateral y. Angular velocity and quaternion vector part are pseudovectors, flip x and z. IsaacLab `body_lin_vel_w`, `net_forces_w`, `root_pos_w`, `root_vel_w` are WORLD frame, so a sagittal reflection about the base plane needs the base yaw to apply and cannot be done by a fixed axis sign flip once the base has any yaw.
