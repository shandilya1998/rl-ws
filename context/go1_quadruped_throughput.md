# Go1 rough versus the quadruped rough task, training throughput

Last revised 2026-09-16. Current.

## Scope

`Isaac-Velocity-Rough-Unitree-Go1-v0`, the rough terrain task IsaacLab ships against the Unitree Go1, trains faster than `Isaac-Quadruped-Blind-Rough-v0`, this repository's quadruped task, a gap the user first observed from two `rsl_rl` progress blocks, Go1 at 31170 steps per second at iteration 47 of 1500, the quadruped at 15805 steps per second at iteration 13322 of 30000. This document establishes the two tasks' full configuration and then isolates the gap by direct, controlled measurement rather than by inference, using two purpose built tasks that hold population, terrain, and in one case the asset itself fixed while the remaining factor varies. Three configurations are compared throughout, `go1-default-debug`, IsaacLab's own Go1 environment and MDP stack with its population and terrain matched to the quadruped's, `go1-debug`, the quadruped's own environment and MDP stack with Go1's asset substituted in, and `quadruped-debug`, the quadruped's task unchanged. All three now share `num_envs=7000` and the same terrain generator, so every ratio reported below is a direct measurement, no population or terrain normalisation is required anywhere in this revision.

## Methodology, `djinn` and the three tasks

Training is launched through `/ws/djinn`'s `start train` dispatch, `djinn:93-203`, which sets a task id and an `rsl_rl` `--policy-type` from its third argument and invokes `scripts/rsl_rl/train.py` inside the per GPU container. Three of its clauses drive this comparison, `djinn:148-169`, `quadruped-debug` selects `Isaac-Quadruped-Blind-Rough-v0` with `policy_type="quadruped-debug"`, `go1-debug` selects `Isaac-Quadruped-Go1Asset-Blind-Rough-v0` with `policy_type="go1-debug"`, and `go1-default-debug` selects `Isaac-Quadruped-Go1Native-Matched-Rough-v0` with `policy_type="go1-default-debug"`. All three policy types route to the same `DebugOnPolicyRunner`, `train.py:197-204`, a subclass of the vendored `OnPolicyRunner` that splits `collection_time` into `act`, `env.step` and `process_env_step`, and `learn_time` into `compute_returns` and `update`, without altering the task's own policy or algorithm class. `djinn`'s training invocation additionally passes `--num_envs 7000` unconditionally to every task it launches, `djinn:201`, which is redundant for all three tasks compared here since each already fixes `num_envs=7000` in its own `__post_init__`, `base_env_cfg.py:1415` for the quadruped stack that `go1-debug` and `quadruped-debug` both build on, and `quadruped_pointfoot_env_cfg.py:434` for `Go1NativeMatchedEnvCfg`, the class `go1-default-debug` resolves to.

`Isaac-Quadruped-Blind-Rough-v0`, `robots/__init__.py:544-561`, is the quadruped's own task, `entry_point="isaaclab.envs:ManagerBasedRLEnv"`, `env_cfg_entry_point=QuadrupedPFBlindRoughEnvCfg`, `rsl_rl_cfg_entry_point=quadruped_runner_cfg`, an instance of `PFQuadrupedPPORunnerCfg`, `robots/__init__.py:44`. `Isaac-Quadruped-Go1Asset-Blind-Rough-v0`, `robots/__init__.py:579-587`, reuses that same environment and MDP stack and the same runner config, `QuadrupedPFGo1AssetBlindRoughEnvCfg`, `quadruped_pointfoot_env_cfg.py:392-397`, substituting `UNITREE_GO1_CFG` for `QUADRUPED_IDENTIFIED_CFG` as the robot and remapping every body and joint name the reward, event, termination and observation terms hardcode for the quadruped's naming onto Go1's, `base_Link` to `trunk`, `foot_.*_Link` to `.*_foot`, `abad_.._Joint`, `hip_.._Joint`, `knee_.._Joint` to `.*_hip_joint`, `.*_thigh_joint`, `.*_calf_joint`, the correspondence stated in `quadruped_identified_cfg.py:11-19,33-36,50-54`. `Isaac-Quadruped-Go1Native-Matched-Rough-v0`, `robots/__init__.py:590-604`, is `Go1NativeMatchedEnvCfg`, `quadruped_pointfoot_env_cfg.py:406-435`, a subclass of IsaacLab's own `UnitreeGo1RoughEnvCfg` with only `num_envs` and the terrain generator overridden, to `7000` and to `QUADRUPED_ROUGH_TERRAINS_CFG` respectively, `quadruped_pointfoot_env_cfg.py:434-435`, its `rsl_rl_cfg_entry_point` is `go1_native_matched_runner_cfg`, an unmodified instance of Go1's own `UnitreeGo1RoughPPORunnerCfg`, `robots/__init__.py:53-57`, so this task trains with byte identical PPO hyperparameters to `Isaac-Velocity-Rough-Unitree-Go1-v0` and differs from it only in population and terrain.

A fourth configuration, `Isaac-Velocity-Rough-Unitree-Go1-v0` at its native `num_envs=4096` and its own smaller terrain, was the original point of comparison and motivated this investigation, its throughput figures from that first exchange are quoted above and reconciled against the matched measurement later in this document, but it is superseded here by `go1-default-debug` for every subsequent table, since population and terrain both confounded it against the quadruped.

`randomize_joint_friction_model` in this repository's own `environments/tasks/locomotion/mdp/events.py` raised `TypeError: 'slice' object is not iterable` the first time `Isaac-Quadruped-Go1Asset-Blind-Rough-v0` was launched, at the line building `actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]`. `ActuatorBase.joint_indices`, `/ws/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base.py:170,225-227,255-262`, is `slice(None)` rather than an explicit index list whenever a single actuator group covers every joint of the articulation, an optimisation IsaacLab's own code defends against at the cited lines but which this repository's `events.py` did not. Go1's `UNITREE_GO1_CFG` carries exactly one actuator group, `"base_legs"`, spanning all twelve joints, `unitree.py:32-43,133-135`, while every other robot in this task registry splits its joints across multiple partial groups, checked directly against each one's identified actuator config, so the branch was unreachable for every existing caller and the crash was specific to Go1's asset. The fix, applied directly since no existing caller ever reached the buggy branch, materialises a `slice(None)` actuator index list the same way `ActuatorBase.__repr__` already does before the membership test, `events.py:315-336`.

## Component comparison

Every quantity below is a single value, one per row, so the three columns can be read straight across. The citation for each row, one column per configuration, is in the table that follows this one, kept separate so this table stays a comparison of values rather than of file paths.

| Quantity | go1-default-debug | go1-debug | quadruped-debug |
|---|---|---|---|
| Entry point class | `ManagerBasedRLEnv` | `ManagerBasedRLEnv` | `ManagerBasedRLEnv` |
| num_envs | 7000 | 7000 | 7000 |
| Steps per env per iteration | 24 | 25 | 25 |
| Steps per iteration | 168000 | 175000 | 175000 |
| sim dt | 0.005 s | 0.005 s | 0.005 s |
| Decimation | 4 | 4 | 4 |
| Control rate | 50 Hz | 50 Hz | 50 Hz |
| PhysX solver position iterations | 4 | 4 | 8 |
| PhysX solver velocity iterations | 0 | 0 | 8 |
| Self collision | disabled | disabled | enabled |
| gpu_max_rigid_patch_count | 327680 | 524288 | 524288 |
| Asset spawn source | pre-baked Nucleus USD | pre-baked Nucleus USD | URDF, converted to USD at spawn |
| Actuator model class | `ActuatorNetMLP` | `ActuatorNetMLP` | `IdentifiedActuator` (`DCMotor` plus a closed form `tanh` friction term) |
| Contact sensor history length | 3 | 4 | 4 |
| Observation groups | 1 | 3 | 3 |
| Critic observation source | shares the actor's `policy` group | dedicated `critic` group, 19 terms | dedicated `critic` group, 19 terms |
| Policy group history length | 0, single frame | 10 frames, flattened | 10 frames, flattened |
| Critic group term count | 8 (= policy group) | 19, including its own height scan raycast | 19, including its own height scan raycast |
| Height scanner grid | 0.1 m, 1.6 by 1.0 m | 0.1 m, 1.6 by 1.0 m | 0.1 m, 1.6 by 1.0 m |
| Reward terms | 10 | 20 | 20 |
| Curriculum terms | 1 | 7 | 7 |
| Startup randomisation events | 3 | 9 | 9 |
| Terrain tile size | 15 by 15 m | 15 by 15 m | 15 by 15 m |
| Terrain rows by cols | 6 by 64 | 6 by 64 | 6 by 64 |
| Terrain tile count | 384 | 384 | 384 |
| Terrain total area | 86400 m^2 | 86400 m^2 | 86400 m^2 |
| Actor hidden dims | [512, 256, 128] | [512, 256, 128] | [512, 256, 128] |
| Critic hidden dims | [512, 256, 128] | [512, 256, 128] | [512, 256, 128] |
| PPO learning epochs | 5 | 5 | 5 |
| PPO mini batches | 4 | 4 | 4 |

## Component comparison, citations

| Quantity | go1-default-debug | go1-debug | quadruped-debug |
|---|---|---|---|
| Entry point class | `robots/__init__.py:592` | `robots/__init__.py:581` | `robots/__init__.py:546` |
| num_envs | `quadruped_pointfoot_env_cfg.py:434` | `base_env_cfg.py:1415` | `base_env_cfg.py:1415` |
| Steps per env per iteration | `go1/agents/rsl_rl_ppo_cfg.py:13` | `quadruped_rsl_rl_ppo_cfg.py:18` | `quadruped_rsl_rl_ppo_cfg.py:18` |
| Steps per iteration | derived | derived | derived |
| sim dt | `velocity_env_cfg.py:311` | `base_env_cfg.py:1433` | `base_env_cfg.py:1433` |
| Decimation | `velocity_env_cfg.py:308` | `base_env_cfg.py:1428` | `base_env_cfg.py:1428` |
| Control rate | derived | derived | derived |
| PhysX solver position iterations | `unitree.py:118` | `unitree.py:118` | `quadruped_identified_cfg.py:78-82` |
| PhysX solver velocity iterations | `unitree.py:118` | `unitree.py:118` | `quadruped_identified_cfg.py:78-82` |
| Self collision | `unitree.py:118` | `unitree.py:118` | `quadruped_identified_cfg.py:79` |
| gpu_max_rigid_patch_count | `velocity_env_cfg.py:314` | `base_env_cfg.py:1431` | `base_env_cfg.py:1431` |
| Asset spawn source | `unitree.py:105-106` | `unitree.py:105-106` | `quadruped_identified_cfg.py:104-114` |
| Actuator model class | `unitree.py:32-43` | `unitree.py:32-43` | `actuator_pd.py:18-40` |
| Contact sensor history length | `velocity_env_cfg.py:74` | `base_env_cfg.py:87` | `base_env_cfg.py:87` |
| Observation groups | `velocity_env_cfg.py:119-146` | `base_env_cfg.py:143-357` | `base_env_cfg.py:143-357` |
| Critic observation source | `/ws/rsl_rl/rsl_rl/utils/utils.py:203-233` | `/ws/rsl_rl/rsl_rl/utils/utils.py:203-233` | `/ws/rsl_rl/rsl_rl/utils/utils.py:203-233` |
| Policy group history length | `velocity_env_cfg.py:141-143` | `base_env_cfg.py:199-204` | `base_env_cfg.py:199-204` |
| Critic group term count | `velocity_env_cfg.py:119-146` | `base_env_cfg.py:265-346` | `base_env_cfg.py:265-346` |
| Height scanner grid | `velocity_env_cfg.py:66-73` | `base_env_cfg.py:75-82` | `base_env_cfg.py:75-82` |
| Reward terms | `velocity_env_cfg.py:230-263` | `base_env_cfg.py:1066-1259` | `base_env_cfg.py:1066-1259` |
| Curriculum terms | `velocity_env_cfg.py:281` | `base_env_cfg.py:1277-...` | `base_env_cfg.py:1277-...` |
| Startup randomisation events | `velocity_env_cfg.py:150-227` | `base_env_cfg.py:925-1064` | `base_env_cfg.py:925-1064` |
| Terrain tile size, rows, cols, count, area | `quadruped_pointfoot_env_cfg.py:435`, `terrains_cfg.py:17-20` | `quadruped_pointfoot_env_cfg.py:397`, `terrains_cfg.py:17-20` | `quadruped_pointfoot_env_cfg.py:213`, `terrains_cfg.py:17-20` |
| Actor and critic hidden dims | `go1/agents/rsl_rl_ppo_cfg.py:21-22` | `quadruped_rsl_rl_ppo_cfg.py:25-26` | `quadruped_rsl_rl_ppo_cfg.py:25-26` |
| PPO learning epochs, mini batches | `go1/agents/rsl_rl_ppo_cfg.py:30-31` | `quadruped_rsl_rl_ppo_cfg.py:46-47` | `quadruped_rsl_rl_ppo_cfg.py:46-47` |

`go1/agents/rsl_rl_ppo_cfg.py` above is `/ws/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/agents/rsl_rl_ppo_cfg.py`, `unitree.py` is `isaaclab_assets/robots/unitree.py` in the same tree, `velocity_env_cfg.py` is `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` in the same tree, and every other bare filename resolves under `tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/` or `tron1-rl-isaaclab-cozum/environments/environments/assets/config/`, following the convention this directory's other documents use.

## Measured throughput

All three configurations were run under `DebugOnPolicyRunner` and their per-iteration debug lines recorded. Every figure is a mean over each run's own steady state iterations, excluding iteration 0, whose first call CUDA graph and JIT warmup inflates `act`, `env.step` and their max columns by one to two orders of magnitude in every run, iteration 0's `env.step` max reaches 3939 ms for `quadruped-debug`, 2954 ms for the original 4096 env Go1 measurement, and 3826 ms for `go1-default-debug` at its now matched population, against steady state maxima in the hundreds throughout.

### Raw steady state averages

| Quantity | go1-default-debug (iters 1-11, n=11) | go1-debug (iters 10-15, n=6, n=7 for the per call row) | quadruped-debug (iters 2-10, n=9) |
|---|---|---|---|
| num_envs | 7000 | 7000 | 7000 |
| steps per env per iteration | 24 | 25 | 25 |
| steps per iteration | 168000 | 175000 | 175000 |
| Computation, steps/s | 24482 | 18490 | 12922 |
| collection_time | 6.681 s | 8.915 s | 12.983 s |
| learn_time | 0.195 s | 0.555 s | 0.565 s |
| act, mean per call | 1.869 ms | 1.773 ms | 1.861 ms |
| env.step, mean per call | 274.22 ms | 352.75 ms | 515.14 ms |
| process_env_step, mean per call | 0.554 ms | 0.597 ms | 0.612 ms |
| compute_returns | 4.213 ms | 4.431 ms | 5.248 ms |
| update | 190.27 ms | 544.35 ms | 559.68 ms |
| env.step, per env | 39.17 µs | 50.39 µs | 73.59 µs |
| Computation, steps/s per env | 3.497 | 2.641 | 1.846 |
| Wall clock per environment step (1 / steps/s) | 40.85 µs | 54.08 µs | 77.39 µs |

The last three rows normalise the population out explicitly, dividing `env.step` and the reciprocal of `steps/s` by `num_envs`, which was already identical across all three runs at 7000, so these rows are a rescaling for readability rather than a correction for an imbalance, unlike every population normalised ratio this document computed before the two matched tasks existed.

### Derived ratios

Because `num_envs` is now 7000 in every run, every ratio below is direct except where `go1-default-debug`'s 24 steps per env against the other two's 25 is called out explicitly, that factor is 1.042 and is applied only to `collection_time`, `learn_time` and `update`, whose cost scales with the batch a single call processes, `env.step`, `act`, `process_env_step` and `compute_returns` are per call figures unaffected by how many calls make up an iteration.

| Comparison | Isolates | collection_time | env.step | learn_time | update |
|---|---|---|---|---|---|
| go1-debug / go1-default-debug | environment and MDP stack alone, asset, population and terrain held fixed | 1.335 (1.281 batch normalised) | 1.286 | 2.855 (2.741 batch normalised) | 2.861 (2.747 batch normalised) |
| quadruped-debug / go1-debug | asset alone, environment, MDP stack, population and terrain held fixed | 1.456 | 1.460 | 1.017 | 1.028 |
| quadruped-debug / go1-default-debug | both, the full gap | 1.943 (1.866 batch normalised) | 1.879 | 2.905 (2.789 batch normalised) | 2.942 (2.824 batch normalised) |

The three rows compose multiplicatively along the `env.step` column exactly, 1.286 times 1.460 equals 1.878, matching 1.879 to within rounding, confirming the two ablations partition the total gap cleanly into an environment and MDP stack factor and an asset factor with no unaccounted residual. The `steps/s` figures give the same story unnormalised, `go1-default-debug` trains 1.324 times faster than `go1-debug`, `go1-debug` 1.431 times faster than `quadruped-debug`, and `go1-default-debug` 1.895 times faster than `quadruped-debug` end to end, 1.324 times 1.431 equalling 1.895 exactly.

The original, population and terrain unmatched comparison opening this document read 31170 against 15805 steps per second, a 1.972 times gap, against the fully matched 1.895 times measured here, the two agree to within eight percent, the difference attributable to sample size, a single iteration each in the original snippets against nine to eleven steady state iterations averaged per run here, and to the small residual population and terrain differences the original comparison carried and this one no longer does.

### What the measurement isolates

`go1-debug` against `go1-default-debug` isolates the environment and MDP stack with the asset, the population and the terrain now all held fixed, the cleanest comparison this document has produced. Its `env.step` ratio of 1.286 is the whole cost of the quadruped's three observation groups, its ten frame history on two of them, and its larger, nineteen term, raycast carrying critic, evaluated once per control step regardless of which asset is spawned. Its `update` ratio, 2.747 once the small batch size difference is divided out, is markedly larger than its `collection_time` counterpart, confirming that the same actor critic asymmetry costs far more during the five PPO epochs than during rollout, since the critic's larger input is processed in both its forward and its backward pass there, and since a naive `env.step` argument would only see the forward evaluation `env.step` itself contains.

`quadruped-debug` against `go1-debug` isolates the asset with the environment, MDP stack, population and terrain now all held fixed, unchanged from the previous revision of this document since both tasks already shared population and terrain there. Its `env.step` ratio of 1.460 is now measured rather than partially inferred to be the heavier PhysX solver settings, 8 and 8 against 4 and 0, and the enabled self collision. Its `update` ratio, 1.028, confirms again that the asset contributes almost nothing to the learning phase.

Composing the two, the full `quadruped-debug` against `go1-default-debug` gap, 1.879 times on `env.step`, splits into 1.286 from the environment and MDP stack and 1.460 from the asset, so of the two, the environment and MDP stack is now the larger contributor to the collection phase gap, a reversal from the previous revision's estimate, which attributed roughly seven eighths of the collection phase gap to the asset. That estimate rested on comparing `go1-debug` against Go1's native task at its original, unmatched population of 4096, dividing by an assumed linear population scaling to normalise it, `go1-debug`'s own `env.step` at 7000 envs was 352.75 ms there as it is here, but the reference it was measured against, at 4096 envs, scaled to 331.49 ms rather than measured at 274.22 ms, an overestimate of 21 percent that inflated the asset's apparent share and understated the stack's. The matched measurement in this revision supersedes that estimate.

## Root causes, ranked by strength of evidence

1. The actor critic asymmetry `ObservationsCfg` builds, `base_env_cfg.py:265-346,354-356`, now measured directly, with the asset, the population and the terrain all held fixed, as a 1.286 times `env.step` cost and a 2.747 times `update` cost, the larger of the two measured effects on the collection phase and by a wide margin the larger effect overall once both phases are weighted by their share of wall clock time.
2. PhysX solver iterations, 8 and 8 against 4 and 0, and enabled self collision, `unitree.py:118`, `quadruped_identified_cfg.py:78-82`, measured directly, with the environment, MDP stack, population and terrain all held fixed, as a 1.460 times `env.step` cost and a negligible 1.028 times `update` cost.
3. Population and terrain no longer contribute to any ratio in this document, both are held at `num_envs=7000` and the same terrain generator across all three configurations compared, the earlier population and terrain normalisation factors this document depended on are retired.
4. Reward, curriculum and event term counts, 20 against 10, 7 against 1, 9 against 3, each individually a small vectorised operation, additive rather than a material contributor on their own, unaffected by this revision's remeasurement.
5. The actuator model, ruled out. Go1's `ActuatorNetMLP` evaluates a small neural network per joint per physics substep, `unitree.py:32-43`, while the quadruped's `IdentifiedActuator` is a `DCMotor` subclass adding one closed form `tanh` friction term, `actuator_pd.py:18-40`, cheaper per call if anything, so this does not explain any part of the gap in Go1's favour.
