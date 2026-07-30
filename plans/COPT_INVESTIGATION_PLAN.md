# Design and Policy Co-optimisation, Investigation and Implementation Plan

> Status, verified against the live sources on 2026-07-30. Implemented but for one item. The complete checkpoint of section 2 exists, as `save` and `load` overrides on `CoptOnPolicyRunner` backed by `co_optimisation/utils/env_state.py`. The CMA-ES initialisation of section 3 has been applied, the initial step size is now 0.25, the parameter ranges are now bounded to 0.75 and 1.25, and the generator import is corrected. All three variance remedies of section 4 have landed, truncated returns are discarded at the morphology update, the delayed terrain curriculum skips its update during a morphology reset, and per individual advantage normalisation is performed in `CoptPPO.compute_returns_design_wise`. The evolutionary schedule has since been retuned beyond this plan, to 256 individuals at an update interval of 240 iterations with the random phase extended to 12000 iterations. The one outstanding item is the removal of the two decimal rounding of link extents, which remains in `usd_generator.py`. This document was formerly named `TODO.md`. See [README.md](README.md) for the full register.

This document records the investigation, analysis, and implementation planning for three deficiencies observed in the design and policy co-optimisation pipeline built on IsaacLab and RSL-RL. The pipeline trains a Proximal Policy Optimisation locomotion policy for the SoleFoot TRON1A biped while an outer Covariance Matrix Adaptation Evolution Strategy optimises the thigh and shank link lengths. The three deficiencies under examination are the persistently high variance of the training curves, the redundant repetition of the first eight thousand iterations on every run, and the biased and low variety sampling produced by the design optimiser. Every claim in this document is grounded in the codebase investigation captured in `../context/knowledge_base.md` and the five section files `../context/rsl_rl.md`, `../context/isaaclab_env.md`, `../context/copt.md`, `../context/cmaes.md`, and `../context/task_plots.md`, with source citations given as `file:line`.

## Table of Contents

1. Introduction
   1. Software Architecture
   2. Hyperparameter List
   3. Improvement Scope
2. Loading and Saving Algorithm
   1. Current Implementation
   2. Relevant Hyperparameters and State
   3. Implementation Plan
3. CMAES Initialisation
   1. Current Implementation
   2. Improvement Scope
   3. Implementation Plan
4. Training Variance Analysis
   1. Problem Description
   2. Current Implementation
   3. Training Implication
   4. Improvement Scope
   5. Implementation Plan
5. References

---

## 1. Introduction

The co-optimisation framework couples a fast inner reinforcement learning loop with a slow outer evolutionary loop. The inner loop is the standard RSL-RL Proximal Policy Optimisation training procedure, and the outer loop replaces the robot morphology used by the parallel environments at a fixed cadence. The coupling is realised by a single custom runner that subclasses the RSL-RL runner and injects the evolutionary update between policy updates. The system reaches a converged design and a seemingly fine tuned policy, yet it exhibits three behaviours that conflict with the intended design, and each behaviour is examined in this document. This introduction first describes the current software architecture together with its call stack, then enumerates the governing hyperparameters and runtime state, and finally frames the three objectives with the evidence gathered during the investigation.

### 1.1 Software Architecture

Training is launched through `tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py`, invoked by the workspace command `djinn start train copt`, which selects the task `Isaac-Limx-SF-Copt-Rough-v0` and passes `--policy-type COPT` (`/ws/djinn:131-142`). When the policy type is COPT the entry point constructs the design generator and the co-optimisation runner rather than the default runner (`train.py:194-238`). The task resolves through the registry to the environment configuration `SFCoptBlindRoughEnvCfg`, which inherits `SFCoptBaseEnvCfg` and `SFCoptEnvCfg` (`robots/limx_solefoot_env_cfg.py:413-418`, `cfg/SF/limx_base_env_cfg.py:1370-1398`), and to the agent configuration `SFCoptPPORunnerCfg` (`agents/limx_rsl_rl_ppo_cfg.py:131-133`). The environment is a `ManagerBasedRLEnv` wrapped by `RslRlVecEnvWrapper`, and the policy is swapped to `CoptActorCritic` at `train.py:220`.

The runner is `CoptOnPolicyRunner` (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`), which extends the RSL-RL `OnPolicyRunner`. Its constructor reads the co-optimisation configuration, namely `ea_update_interval` of one hundred and twenty, `ea_late_start` of eight thousand, `num_individuals` of sixty four, and `randomise_before_late_start` of true, and it allocates per individual fitness accumulators (`copt_on_policy_runner.py:69-103`). The runner overrides `_construct_algorithm` to build `CoptActorCritic` with an additional encoder configuration argument (`copt_on_policy_runner.py:322-377`), and it overrides `learn` to interleave the evolutionary update with the policy update (`copt_on_policy_runner.py:105-320`).

The inner loop follows the canonical RSL-RL structure. For each iteration the runner collects a rollout of twenty five steps per environment, calls `alg.compute_returns` to form the Generalised Advantage Estimation targets, and calls `alg.update` to apply the clipped surrogate update over five epochs and four mini batches (`rsl_rl/rsl_rl/algorithms/ppo.py:194-417`). The reward and length statistics are accumulated into two rolling windows of length one hundred, and the logger reports the arithmetic mean of those windows as `Train/mean_reward` and `Train/mean_episode_length` (`on_policy_runner.py:231-232`). The advantage is normalised once, globally, over the entire pooled rollout (`rollout_storage.py:150`).

The outer loop fires every one hundred and twenty iterations. At that point the runner calls `_update_morphology`, which stops the simulation, advances the design generator, authors new link extents into the per environment USD prototypes through `apply_link_length_params`, calls `unwrapped_env.reset()` for all environments, reapplies actuator parameters, and zeroes the fitness accumulators (`copt_on_policy_runner.py:416-457`, `co_optimisation/utils/update.py:129-176`). The design generator is `GrowingDesignDistCMAESDesignGenerator` (`usd_generator.py:815-867`). For the first eight thousand iterations it samples random designs from a distribution whose spread grows from five percent to the full range, and thereafter it samples from a CMA-ES search distribution restricted to the two length scales (`usd_generator.py:842-867`, `699-735`). The CMA-ES state is held in a `cma.CMAEvolutionStrategy` object operating on the unit square `[0,1]^2`, with the two coordinates denormalised to the physical scale range `(0.85, 1.15)` (`usd_generator.py:720-735`, `795-801`).

The policy network is `CoptActorCritic` (`co_optimisation/modules/copt_actor_critic.py`). The actor receives the proprioceptive policy observations concatenated with a sixteen dimensional latent produced by an estimator network from the privileged observations (`copt_actor_critic.py:127-152`). The critic is the inherited RSL-RL critic and operates directly on the critic observation group. That group contains the true per environment link lengths, the body masses, and the inertias, so the value function does observe the morphology it must evaluate (`cfg/SF/limx_base_env_cfg.py:459-559`, confirmed in `../context/task_plots.md` section 3).

### 1.2 Hyperparameter List

The governing settings are gathered below so that the present configuration is unambiguous. Environment settings derive from `SFCoptEnvCfg` and its parents, PPO settings from `SF_TRON1AFlatPPORunnerCfg`, and co-optimisation settings from the COPT branch of `train.py`.

Environment.

- `num_envs`, four thousand and ninety six (`cfg/SF/limx_base_env_cfg.py:1374`).
- `episode_length_s`, twenty seconds (`:1388`).
- `sim.dt`, zero point zero zero five seconds, `decimation` four, giving a fifty hertz control rate (`:1387`, `:1391`).
- `max_episode_length`, one thousand control steps, computed as twenty divided by the product of `dt` and `decimation`.
- terrain, the generator `BERKELEY_MIMIC_TERRAINS_CFG`; robot asset, `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG` with `replicate_physics` false (`robots/limx_solefoot_env_cfg.py:144-146`, `417-418`).
- `seed`, forty two (`:1392`).

Proximal Policy Optimisation.

- `max_iterations`, thirty thousand; `save_interval`, five hundred in the config and one thousand from the command line (`agents/limx_rsl_rl_ppo_cfg.py:95,133`, `/ws/djinn:139-143`).
- `num_steps_per_env`, twenty five, giving a rollout batch of one hundred and two thousand four hundred transitions (`:93`).
- `learning_rate`, ten to the minus three, schedule adaptive with `desired_kl` zero point zero one (`:112-116`).
- `gamma`, zero point nine nine; `lam`, zero point nine five; `clip_param`, zero point two; `entropy_coef`, zero point zero zero five; `value_loss_coef`, one; `use_clipped_value_loss`, true; `max_grad_norm`, one (`:106-117`).
- `num_learning_epochs`, five; `num_mini_batches`, four (`:110-111`).
- `init_noise_std`, one; actor and critic hidden dimensions `[512, 256, 128]`; activation elu (`:99-103`).
- observation normalisation, disabled; random network distillation, none; symmetry, none; advantage normalisation, global (`../context/rsl_rl.md` section 1).
- encoder hidden dimensions `[128, 64, 16]`, output detached, the latent dimension fed to the actor being sixteen (`:120-126`, `copt_actor_critic.py:62-67`).

Co-optimisation and CMA-ES.

- `num_individuals`, sixty four; `ea_update_interval`, one hundred and twenty; `ea_late_start`, eight thousand; `randomise_before_late_start`, true (`train.py:200-227`).
- optimised parameters, `thigh_length_scale` and `shank_length_scale`, each bounded to `(0.85, 1.15)` (`usd_generator.py:236-239`).
- `sigma0`, zero point seven five; initial mean, zero point five on each coordinate; `seed`, forty two; `bounds`, the unit square; `popsize`, sixty four; `maxiter`, about one hundred and eighty three (`usd_generator.py:720-731`).
- `late_start_it`, sixty six generations, equal to eight thousand divided by one hundred and twenty (`train.py:216`).
- morphology window, three thousand control steps, equal to one hundred and twenty multiplied by twenty five, which exceeds the one thousand step episode length so that episodes complete within a window.

### 1.3 Improvement Scope

The three objectives are framed below, each with the evidence that motivates it.

The first objective concerns the high variance of the training curves. The plots in `/ws/context/artefacts/plots` are a single unsmoothed trace of the run `sf_copt/2026-06-26_05-26-28` taken to step twenty nine thousand nine hundred and ninety nine (`../context/task_plots.md` section 5). Every per component reward panel shows a wide noisy band that stays wide or widens across the entire run and never contracts, and the headline task reward `rew_lin_vel_xy` oscillates between approximately zero and eighteen throughout (`reward7.png`). The value function loss is large and spiky, ending near seventeen and a half (`loss2.png`). The variance does not shrink even though the sampled designs collapse to a single morphology by the end of training, which is the central anomaly. Two hypotheses were posed at the outset, namely a metric buffering and reset artefact caused by the environment reset every one hundred and twenty iterations, and an inability of the critic to estimate value across varying link lengths. The investigation resolves both, and the analysis appears in section four.

The second objective concerns the redundant repetition of the first eight thousand iterations. During those iterations the design generator samples random designs that are independent of any fitness signal, so the segment is identical in expectation across experiments and is wasteful to recompute. The current checkpoint mechanism persists only the policy weights, the optimiser moments, and the iteration counter (`on_policy_runner.py:291-326`), so the state of the environment, its managers, the curriculum, the PPO learning rate, and the entire CMA-ES search distribution are lost on resume. Furthermore the morphology schedule is computed relative to the resumed start iteration rather than the absolute iteration (`copt_on_policy_runner.py:160-166`), so a resumed run would repeat the random phase rather than continue from it. The remedy is examined in section two.

The third objective concerns the biased and low variety design sampling. The design comma separated value files show that by iteration nineteen thousand seven hundred and ninety nine every one of the sixty four individuals is identical, with thigh extent zero point two eight metres and shank extent zero point three four metres, which are the rounded upper bounds of the allowed range (`../context/cmaes.md` section 7). The optimiser pinned both parameters to their maxima rather than discovering an interior optimum. Three compounding causes were identified, namely an order of operations defect that decorrelates the CMA-ES feedback, an oversized initial step size on a bounded unit box that biases sampling toward the boundary, and a coarse two decimal rounding of the link extents that erases sub centimetre variety and flattens the fitness signal. The analysis and the recommended initialisation appear in section three.

The entry point is correctly wired for the recorded run, which is noted here so that the implementation does not chase a phantom defect. The symbol `GrowingDesignDistCMAESDesignGenerator` is imported at `train.py:101` and is re-exported by the runners package through `runners/__init__.py`, and the generator is instantiated at `train.py:210`, so the committed file reproduces the growing distribution trajectory captured in the design files without any patch. One minor inconsistency is worth recording for the implementation, namely that the late start index is computed as `int(ea_late_start / 120)` with a hardcoded interval at `train.py:217` rather than `int(ea_late_start / ea_update_interval)`, which is harmless only because the two values coincide at one hundred and twenty and which should be made to reference the interval directly when the constructor is retuned in section three.

---

## 2. Loading and Saving Algorithm

This section examines the persistence mechanism, identifies what it fails to capture, and plans an extension that allows an experiment to resume on conditions identical to those of the run it continues. The motivating use cases are two, namely the avoidance of recomputing the eight thousand iteration random phase, and the recovery of a run that aborts because of a transient fault such as a failure to acquire GPU memory.

### 2.1 Current Implementation

The base runner persists a deliberately small dictionary. The method `OnPolicyRunner.save` writes exactly four entries, namely the policy state dictionary, the optimiser state dictionary, the iteration counter, and an auxiliary information field, with two further entries added only when random network distillation is active (`on_policy_runner.py:291-303`).

```python
# rsl_rl/rsl_rl/runners/on_policy_runner.py:291
def save(self, path, infos=None):
    saved_dict = {
        "model_state_dict": self.alg.policy.state_dict(),
        "optimizer_state_dict": self.alg.optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "infos": infos,
    }
    if self.alg.rnd:
        saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
    torch.save(saved_dict, path)
```

The method `OnPolicyRunner.load` restores those entries and assigns the iteration counter back to `current_learning_iteration` (`on_policy_runner.py:309-326`). The policy state dictionary carries any registered buffers, so when empirical observation normalisation is enabled its running statistics ride along inside the model state dictionary. In the co-optimisation configuration normalisation is disabled, so the normalisers are identity modules with no buffers (`../context/rsl_rl.md` section 4).

The co-optimisation runner does not override `save` or `load`, so it inherits this behaviour verbatim (`../context/copt.md` section 2.8). The consequence is that three distinct families of state are silently discarded at a checkpoint. The first family is the residual algorithm state inside Proximal Policy Optimisation, of which the adaptive learning rate is the most consequential. The learning rate is a plain Python float held on the algorithm and mutated on every mini batch by the Kullback Leibler schedule (`ppo.py:122`, `280-292`). It is not restored by the loader, so on resume the freshly constructed algorithm begins at the configured value of ten to the minus three, and the first adaptive update overwrites every optimiser parameter group with that value (`ppo.py:291-292`). The adaptive learning rate therefore resets silently on every resume, which alone makes a continuation diverge from an uninterrupted run. The accumulated logging totals `tot_timesteps` and `tot_time` are likewise reset, which discontinues the time axis of the curves.

The second family is the entire evolutionary and environment state. The runner persists none of the generation counter, the current population, the per individual fitness accumulators, the flag that records whether CMA-ES has begun, or the CMA-ES search distribution itself. The generator already provides a `save_state` method that pickles the `cma.CMAEvolutionStrategy` object (`usd_generator.py:809-812`), and the constructor already accepts an `es_state_path` from which to reload it (`usd_generator.py:712-718`), yet neither hook is ever called or supplied (`../context/copt.md` section 2.8). The checkpointing of the optimiser that drives the entire outer loop is thus present in the code and wholly inert.

The third family is the IsaacLab environment and manager state. IsaacLab provides no runtime state checkpoint. The manager `serialize` method returns only the configuration converted to a dictionary, not the evolving state (`manager_base.py:104-106`, `observation_manager.py:437-458`), and `InteractiveScene.get_state` together with `ManagerBasedEnv.reset_to` captures and restores only the physical articulation state, that is the root pose and the joint positions and velocities (`interactive_scene.py:563`, `manager_based_env.py:395`). The true curriculum progress lives elsewhere, in the targets mutated in place by the curriculum functions, and none of those targets appears in any checkpoint (`../context/isaaclab_env.md` sections 0 and 7).

A further defect compounds the loss of state. The morphology schedule is evaluated relative to the resumed start iteration rather than the absolute iteration. The toggle and update predicates read `(it - start_iter + 1)` (`copt_on_policy_runner.py:163-166`) while the loop begins at the loaded iteration (`copt_on_policy_runner.py:160`), so on resume the counter restarts at one and the eight thousand iteration random phase is repeated in full before CMA-ES is allowed to begin. This directly defeats the stated goal of skipping the random phase.

The justification for the extension follows from these three families. Experimental sanctity requires that a continued run begin from the exact curriculum difficulty, command ranges, push schedule, learning rate, fitness accumulators, and search distribution that prevailed at the moment of the checkpoint. Because the present mechanism restores only weights, the optimiser moments, and the iteration counter, a resumed run silently re-randomises the morphology distribution, resets the adaptive learning rate, restarts the curriculum clock implicitly through the lost counters, and re-enters the random phase. None of these is acceptable for either use case, hence the persistence layer must be extended to capture all state that evolves during training.

### 2.2 Relevant Hyperparameters and State

The state required for an identical continuation is enumerated below in three groups, with the attribute path and the serialisation approach for each item. The grouping mirrors the three families identified above.

Proximal Policy Optimisation and runner state, all cheaply serialisable.

- `self.alg.learning_rate`, the adaptive learning rate float (`ppo.py:122`). Must be saved and restored before the first update.
- `self.tot_timesteps` and `self.tot_time`, the logging totals (`on_policy_runner.py:57-58`).
- `self.current_learning_iteration`, already persisted as `iter` (`on_policy_runner.py:296`).
- the policy and optimiser state dictionaries, already persisted.
- if random network distillation were enabled, `self.alg.rnd.update_counter` (`rnd.py:101`), a plain integer absent from any state dictionary. Not used in the co-optimisation configuration but listed for completeness.

Evolutionary and design generator state.

- `self.generation`, the outer loop counter that drives both the growing distribution schedule and the CMA-ES generation index (`copt_on_policy_runner.py:87`).
- `self.current_population`, the active link extents, USD paths, and actuator parameters (`copt_on_policy_runner.py:88`).
- `self._individual_fitness` and `self._individual_episode_counts`, the in progress fitness accumulators (`copt_on_policy_runner.py:91-96`).
- `self._copt_started`, the flag that gates the first CMA-ES sample (`copt_on_policy_runner.py:98`).
- the generator search state, namely the `cma.CMAEvolutionStrategy` object `_es`, the `_pending_solutions`, the `_last_solutions`, the `_terminated` flag, and the `late_start` flag (`usd_generator.py:712-735`). The strategy object is picklable and is captured by the existing `save_state` (`usd_generator.py:809`).
- `self._env_to_individual` is deterministic and is recomputed in the constructor, so it need not be saved (`copt_on_policy_runner.py:470-472`).

IsaacLab environment and manager state. The first list is cheaply serialisable as tensors or scalars, and the second is bound to USD or PhysX and must be extracted and reapplied rather than pickled.

- `ManagerBasedRLEnv.common_step_counter`, the never reset master curriculum clock (`manager_based_rl_env.py:74`).
- `ManagerBasedEnv._sim_step_counter`, the physics substep counter (`manager_based_env.py:133`).
- `episode_length_buf`, the per environment episode clock (`manager_based_rl_env.py:77`).
- `TerrainImporter.terrain_levels` and `env_origins`, the terrain curriculum progress (`terrain_importer.py:347,353`).
- the live command ranges `command_manager.get_term("base_velocity").cfg.ranges`, mutated by the velocity curriculum (`curriculums.py:353,402,449`).
- the push velocity range in the `push_robot` event term parameters, mutated by the push curriculum (`curriculums.py:186-211`).
- the tracking reward standard deviations in the reward term parameters, mutated by the standard deviation curriculum (`curriculums.py:305`).
- `TerminationManager._last_episode_dones`, the persistent saturating termination record (`termination_manager.py:68`).
- `RewardManager._episode_sums`, the per term per environment accumulators read by the curriculum gates (`reward_manager.py:59`).
- the command term buffers, namely `vel_command_b`, `heading_target`, `is_heading_env`, `is_standing_env`, `time_left`, `command_counter`, and `metrics` (`velocity_command.py:82-88`, `command_manager.py:54-56`).
- the event timers `_interval_term_time_left`, `_reset_term_last_triggered_step_id`, and `_reset_term_last_triggered_once` (`event_manager.py:331-334`).
- the action buffers `_action` and `_prev_action`, the latter used as an observation (`action_manager.py:215-216`).
- the observation history circular buffers per group and term (`observation_manager.py:475`).
- the global random number generator states for torch, torch cuda, numpy, and the python random module, since IsaacLab seeds only global generators and holds no per environment generator (`../context/isaaclab_env.md` section 1.4).

USD or PhysX bound state, extracted and reapplied rather than pickled.

- the physical robot state, captured by `scene.get_state` and restored by `env.reset_to` (`interactive_scene.py:563`, `manager_based_env.py:395`).
- the startup randomised physics tensors, namely the body masses, friction coefficients, joint stiffness and damping, joint default positions, and joint armature, which are sampled once at startup and never reapplied at reset (`../context/isaaclab_env.md` section 6.3). These can be reproduced by replaying the startup random number sequence or captured directly from the articulation data buffers. For the co-optimisation runner the actuator tensors are overwritten by `apply_actuator_params` after every reset, so only the mass, friction, and default position offsets need to be captured for an identical morphology.

### 2.3 Implementation Plan

The plan introduces two self contained persistence methods, one for the design generator and one for the environment, and then integrates them with extended runner methods. The design keeps each layer responsible for its own state, which mirrors the guidance in `../CO_OPTIMISATION.md` section 3 and the precedent set by the HIM runner that persists its estimator optimiser.

Step one extends the design generator. The CMA-ES generator already pickles its strategy object, and the method is widened to a full state dictionary that also records the pending and last solutions, the termination flag, the late start flag, and the generation bookkeeping. The growing distribution subclass inherits these methods unchanged.

```python
# usd_generator.py, CMAESDesignGenerator
def get_state(self) -> dict:
    return {
        "es": self._es.pickle_dumps(),
        "pending_solutions": self._pending_solutions,
        "last_solutions": self._last_solutions,
        "terminated": self._terminated,
        "late_start": self.late_start,
    }

def load_state(self, state: dict) -> None:
    self._es = cma.CMAEvolutionStrategy.pickle_loads(state["es"])
    assert self._es.N == self.num_params, "CMA-ES dimension mismatch on resume"
    self._pending_solutions = state["pending_solutions"]
    self._last_solutions = state["last_solutions"]
    self._terminated = state["terminated"]
    self.late_start = state["late_start"]
```

Step two adds an environment state capture and restore pair. Because IsaacLab exposes no runtime checkpoint, the helpers read and write the enumerated attributes directly. They live in a new module `co_optimisation/utils/env_state.py` so that the runner remains uncluttered. The capture returns a dictionary of detached CPU tensors and Python scalars, and the restore writes them back in place.

```python
# co_optimisation/utils/env_state.py
import torch

def capture_env_state(env) -> dict:
    u = env.unwrapped
    cmd = u.command_manager.get_term("base_velocity")
    push = u.event_manager.get_term_cfg("push_robot")
    terrain = u.scene.terrain
    state = {
        "common_step_counter": int(u.common_step_counter),
        "sim_step_counter": int(u._sim_step_counter),
        "episode_length_buf": u.episode_length_buf.detach().cpu().clone(),
        "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
        "env_origins": terrain.env_origins.detach().cpu().clone(),
        "command_ranges": _ranges_to_dict(cmd.cfg.ranges),
        "push_velocity_range": dict(push.params["velocity_range"]),
        "reward_std": _capture_reward_std(u.reward_manager),
        "last_episode_dones": u.termination_manager._last_episode_dones.detach().cpu().clone(),
        "episode_sums": {k: v.detach().cpu().clone()
                          for k, v in u.reward_manager._episode_sums.items()},
        "command_buffers": _capture_command_buffers(cmd),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": __import__("numpy").random.get_state(),
            "python": __import__("random").getstate(),
        },
        "scene_state": u.scene.get_state(is_relative=True),
    }
    return state

def restore_env_state(env, state: dict) -> None:
    u = env.unwrapped
    u.common_step_counter = state["common_step_counter"]
    u._sim_step_counter = state["sim_step_counter"]
    u.episode_length_buf.copy_(state["episode_length_buf"].to(u.device))
    terrain = u.scene.terrain
    terrain.terrain_levels.copy_(state["terrain_levels"].to(u.device))
    terrain.env_origins.copy_(state["env_origins"].to(u.device))
    _restore_ranges(u.command_manager.get_term("base_velocity").cfg.ranges,
                    state["command_ranges"])
    u.event_manager.get_term_cfg("push_robot").params["velocity_range"].update(
        state["push_velocity_range"])
    _restore_reward_std(u.reward_manager, state["reward_std"])
    u.termination_manager._last_episode_dones.copy_(
        state["last_episode_dones"].to(u.device))
    for k, v in state["episode_sums"].items():
        u.reward_manager._episode_sums[k].copy_(v.to(u.device))
    _restore_command_buffers(u.command_manager.get_term("base_velocity"),
                             state["command_buffers"])
    _restore_rng(state["rng"])
    u.reset_to(state["scene_state"], env_ids=None, is_relative=True)
```

The small helpers `_ranges_to_dict`, `_restore_ranges`, `_capture_reward_std`, `_restore_reward_std`, `_capture_command_buffers`, `_restore_command_buffers`, and `_restore_rng` iterate over the named fields and copy the tensors. They are mechanical and are omitted here for brevity. The restore must run after the articulations exist and after the first `sim.reset`, and the physical state restore is delegated to `env.reset_to`, which itself recomputes the observations (`manager_based_env.py:447-450`).

Step three integrates these into the runner by overriding `save` and `load` on `CoptOnPolicyRunner`. The overrides call the base method first, then augment the saved dictionary, which preserves full compatibility with the inherited loader for the policy and optimiser.

```python
# copt_on_policy_runner.py, CoptOnPolicyRunner
def save(self, path, infos=None):
    super().save(path, infos)
    saved = torch.load(path, weights_only=False)
    saved["learning_rate"] = self.alg.learning_rate
    saved["tot_timesteps"] = self.tot_timesteps
    saved["tot_time"] = self.tot_time
    saved["copt"] = {
        "generation": self.generation,
        "copt_started": self._copt_started,
        "individual_fitness": self._individual_fitness.detach().cpu().clone(),
        "individual_episode_counts": self._individual_episode_counts.detach().cpu().clone(),
        "current_population": _population_to_dict(self.current_population),
        "design_generator": self._design_generator.get_state(),
    }
    saved["env_state"] = capture_env_state(self.env)
    torch.save(saved, path)

def load(self, path, load_optimizer=True, map_location=None):
    infos = super().load(path, load_optimizer, map_location)
    loaded = torch.load(path, weights_only=False, map_location=map_location)
    self.alg.learning_rate = loaded["learning_rate"]
    for group in self.alg.optimizer.param_groups:
        group["lr"] = self.alg.learning_rate
    self.tot_timesteps = loaded["tot_timesteps"]
    self.tot_time = loaded["tot_time"]
    c = loaded["copt"]
    self.generation = c["generation"]
    self._copt_started = c["copt_started"]
    self._individual_fitness.copy_(c["individual_fitness"].to(self.device))
    self._individual_episode_counts.copy_(c["individual_episode_counts"].to(self.device))
    self._design_generator.load_state(c["design_generator"])
    self._restored_population = _population_from_dict(c["current_population"])
    self._restored_env_state = loaded["env_state"]
    return infos
```

Step four corrects the schedule so that resume continues rather than repeats. The toggle and update predicates must be evaluated against the absolute iteration `it` rather than `(it - start_iter + 1)`, so that a run resumed at iteration eight thousand enters the CMA-ES phase immediately.

```python
# copt_on_policy_runner.py, learn()
is_late_start_toggle_time = self._ea_late_start <= (it + 1)
is_morph_update_time = ((it + 1) % self._ea_update_interval == 0) and (
    is_late_start_toggle_time or self._randomise_before_late_start
)
```

Step five applies the restored environment state and population inside `learn` after the environment exists and after the initial morphology is spawned, guarded by a flag set during `load`. The restored population is applied through the existing `apply_link_length_params` path so that the physical morphology matches the saved generation, after which `restore_env_state` writes back the curriculum, command, reward, and counter state and the global random number generator. With these five steps a resume reconstructs the policy, the optimiser, the adaptive learning rate, the search distribution, the fitness accumulators, the curriculum progress, and the random number generator, so that continuation is faithful to an uninterrupted run and the random phase is never repeated.

---

## 3. CMAES Initialisation

This section examines why the design optimiser collapses to a single boundary morphology, explains the role of each initialisation choice, and recommends an initialisation that restores exploration without sacrificing convergence. The analysis rests on the installed pycma source at version four point four point four and on Hansen's tutorial, both cited in section five, and on the statistical read of the design files reported in `../context/cmaes.md` section 7.

### 3.1 Current Implementation

The optimiser is a single `cma.CMAEvolutionStrategy` constructed over the unit square. The search dimension is two because CMA-ES drives only the thigh and shank length scales, the initial mean is the box centre `np.full(2, 0.5)`, the initial step size is `sigma0` of zero point seven five, the population size is sixty four, and the bounds are the unit square (`usd_generator.py:699-735`). Each sampled coordinate is denormalised by the affine map from `[0,1]` to the physical scale range `(0.85, 1.15)` (`usd_generator.py:795-801`), and the resulting link extents are rounded to two decimals before they are written to the URDF (`usd_generator.py:366-373`).

Covariance Matrix Adaptation Evolution Strategy maintains a multivariate Gaussian search distribution and adapts it from the ranks of the evaluated samples. Three quantities evolve, namely the mean which tracks the best region, the global step size which controls the overall scale of exploration, and the covariance matrix which captures the shape and orientation of the promising region. The step size is updated by cumulative step size adaptation, which compares the length of an evolution path against the length expected under random selection, and grows the step size when successive steps are correlated and shrinks it when they are not. The covariance is updated by a rank one term from the evolution path and a rank mu term from the current selection. The interpretation of the recorded results follows directly from these mechanics, and each initialisation choice shapes them.

The initial step size is the single most consequential choice, and the value of zero point seven five is far too large for a unit box. The governing rule, stated in the pycma documentation and in Hansen's tutorial, is that the variables should be scaled so that the optimum is expected to lie within approximately the mean plus or minus three times the step size, which places the appropriate step size near one quarter of the search range. On the unit square with the mean at zero point five, a step size of zero point seven five implies an expected optimum bracket of `[-1.75, 2.25]`, which overflows the box by more than two hundred percent on each side and places the bulk of the prior mass outside the feasible region. The appropriate value for a unit box is near zero point two five.

A subtle but important pycma behaviour partly masks this error while preserving its harm. When bounds are set and the maximum step size is left unspecified, pycma caps each coordinate standard deviation at the bound range divided by three, which here is one third (`evolution_strategy.py:1156-1161`, `_stds_into_limits` at `:1506`, applied at construction and after every update). The effective initial standard deviation is therefore the minimum of zero point seven five and zero point three three, namely zero point three three, which is still the widest distribution the box admits. Under a Gaussian centred at zero point five with standard deviation one third, a point only one and a half standard deviations away already reaches a bound, so roughly sixteen percent of the mass on each coordinate falls beyond each bound before repair. The initial distribution is thus nearly uniform across the box with heavy accumulation at the bounds, which is the opposite of the focused local search that the step size is meant to provide.

The boundary handling then converts this near uniform, boundary heavy prior into a bias toward the upper bound. The default boundary handler in this version of pycma is the saturating `BoundTransform`, not the penalty handler that the existing `CMAES_DESIGN_GENERATOR.md` claims (`options_parameters.py:69`, `boundary_handler.py:454`). Genotype samples that fall above the upper bound are all mapped to phenotypes pinned at that bound. Because the inner loop fitness rewards longer thigh and shank segments, the rank based mean update walks the genotype mean steadily upward, eventually beyond the upper bound, where it remains because the transform keeps the phenotype valid. Many distinct genotypes then collapse onto the single boundary phenotype, which destroys variety and pins both parameters at their maxima. The recorded trajectory confirms this exactly. The table below reproduces the per generation mean and standard deviation of the link extents over the sixty four individuals (`../context/cmaes.md` section 7).

| iteration | phase | mean thigh | std thigh | mean shank | std shank |
|---|---|---|---|---|---|
| 119 | random, scale about 0.05 | 0.25000 | 0.00000 | 0.30000 | 0.00000 |
| 959 | random growing | 0.25109 | 0.00437 | 0.30078 | 0.00539 |
| 7799 | end of random | 0.24844 | 0.02101 | 0.29375 | 0.02820 |
| 10439 | CMA-ES early | 0.28047 | 0.00211 | 0.34000 | 0.00000 |
| 15119 | CMA-ES | 0.28203 | 0.00402 | 0.34000 | 0.00000 |
| 19799 | CMA-ES | 0.28000 | 0.00000 | 0.34000 | 0.00000 |
| 29999 | CMA-ES final | 0.28000 | 0.00000 | 0.34000 | 0.00000 |

Three readings emerge from the table. The spread grows through the random phase as the growing distribution widens, then collapses almost immediately once CMA-ES takes over, reaching exactly zero by iteration nineteen thousand seven hundred and ninety nine. The mean drifts to the upper bound, the shank extent of zero point three four being exactly the rounded value of zero point three zero multiplied by one point one five, and the thigh extent of zero point two eight sitting one rounding bin below the rounded upper bound. The population becomes a single repeated design, which is the zero variety endpoint the user observed.

The two decimal rounding accelerates and finalises the collapse. The rounding in `_compute_link_extents` snaps every extent to a one centimetre grid, which on a baseline of zero point two five to zero point three zero metres is about four percent of the link length, and that bin is larger than the entire late stage CMA-ES spread. Any residual genotype variety is therefore erased before it reaches the simulator, so all sixty four individuals receive identical morphologies and return identical fitness. A flat fitness removes the selection gradient entirely and trips the pycma termination criteria `tolfun` and `tolflatfitness` (`evolution_strategy.py:4140-4269`), after which the optimiser freezes and the generator returns the incumbent design unchanged (`usd_generator.py:761-765`).

An order of operations defect decorrelates the feedback the optimiser receives and is a primary cause of the failure independent of the initialisation. In the live update pathway the runner calls `generate_population` before `update_with_fitness` (`copt_on_policy_runner.py:430-435`). The method `generate_population` overwrites the record of the last solutions with the freshly generated batch before `update_with_fitness` reads it (`usd_generator.py:761-772`), so the call to `es.tell` pairs the current batch's genotypes with the previous batch's fitness. The pairing is misaligned by exactly one generation for the whole run, so CMA-ES ranks a set of designs by the fitness of a different set of designs and receives essentially decorrelated feedback. The documented order in `../CO_OPTIMISATION.md` section 4.9.4 is the reverse and correct, namely tell then ask then generate, and the implementation swapped the two lines, presumably to avoid the assertion that guards the first call. This defect alone can drive a covariance adapting optimiser to a degenerate corner regardless of the step size.

A final scheduling consideration weakens the early fitness signal. CMA-ES initialises fresh at the box centre only after eight thousand iterations during which the policy was trained on a wide and growing random design distribution. The policy is therefore adapted to a broad distribution of morphologies rather than to the mean design, so the first CMA-ES evaluations are biased and noisy, which is the morphological innovation protection problem described by Cheney and colleagues. The late start toggle flips to true designs with no warm transfer of the distribution that the optimiser should start from.

### 3.2 Improvement Scope

The diagnosis identifies one defect and three initialisation faults, and the remedies are ordered by impact. The defect is the tell and ask misalignment, which must be corrected first because no amount of tuning compensates for ranking designs by the wrong fitness. The correction restores the documented order so that the fitness of a generation is attributed to the genotypes that produced it, which is the precondition for CMA-ES to function at all.

The first initialisation fault is the oversized step size. Setting `sigma0` near zero point two five aligns the prior with the box, so that the mean plus or minus two standard deviations spans the unit interval without overflowing it, and the search begins as a focused exploration of the interior rather than a near uniform sampling pinned to the bounds. This is the standard quarter of range recommendation, and zero point three is the practical upper limit because the maximum step size cap would clip anything larger.

The second initialisation fault is the coarse quantisation. Removing the two decimal rounding, or replacing it with rounding to four decimals, preserves sub centimetre variety so that the fitness retains a selection gradient and the optimiser does not see flat fitness. This change matters as much as the step size, because even a perfectly initialised optimiser collapses if its proposals are quantised below its working resolution. However, the increase in precision of the link length would drastically increase the search space leading to further difficulties in PPO convergence as seen in previous experiments. Thus, instead of increasing the rounding precision, the range of values is increased.

The third initialisation fault is the narrow range. The interval `(0.85, 1.15)` is only fifteen percent on each side, and against a centimetre grid it admits only a handful of distinct values per parameter, which is too coarse a lattice for a covariance adapting optimiser to navigate. Widening the range to approximately `(0.75, 1.25)` makes an interior optimum more likely and yields a finer effective lattice after any residual rounding. The widened range also reduces the chance that the true optimum lies outside the box, which is the condition under which a boundary collapse is in fact the correct answer rather than a failure.

Two further measures address robustness rather than the immediate collapse. The fitness estimate per individual is noisy because each design is evaluated over a limited number of episodes within one morphology window, so wrapping the update in a noise handler or lengthening the morphology window lowers the variance of the rank signal that CMA-ES consumes. For multimodal gait regimes a restart strategy such as the increasing population restart available through the pycma functional interface recovers a premature collapse by relaunching with a larger population, whereas the bare strategy object used here cannot restart and freezes permanently on the first collapse. The initial mean should remain at the box centre, since the centre is the correct unbiased prior, although a warm start that injects the centroid of the best random phase designs is a reasonable refinement once the preceding faults are fixed.

### 3.3 Implementation Plan

The plan corrects the feedback order, retunes the initialisation, and relaxes the quantisation, with optional robustness measures.

The first change corrects the tell and ask order in the runner so that the fitness is attributed to the solutions that produced it. On the first CMA-ES generation the runner samples and generates without telling, and on every later generation it tells before it generates. Please note that this change has been applied to the code already. Please proceed to the next change directly after verification

```python
# copt_on_policy_runner.py, _update_morphology(), CMA branch
if self._ea_late_start <= it:
    if not self._copt_started:
        self._design_generator.sample_batch()          # ask the first batch
        self.current_population = self._design_generator.generate_population(self.generation)
        self._copt_started = True
    else:
        fitness = self._compute_individual_fitness()    # fitness of the active batch
        self._design_generator.update_with_fitness(fitness)  # tell then ask next
        self.current_population = self._design_generator.generate_population(self.generation)
else:
    self.current_population = self._design_generator.generate_population(self.generation)
```

This ordering matches the documented contract in `../CO_OPTIMISATION.md` section 4.9.4 and ensures that `es.tell` receives the genotypes of the batch that was active during the just completed morphology window together with that batch's fitness.

The second change retunes the constructor of the design generator in the entry point. The step size is reduced to zero point two five and the bounded range is widened, and the boundary handler is set explicitly so that the choice is visible in the code rather than relying on the default.

```python
# train.py, COPT branch
design_generator = GrowingDesignDistCMAESDesignGenerator(
    base_urdf_path=_base_urdf,
    num_individuals=_num_individuals,
    param_ranges=param_ranges,
    sigma0=0.25,                       # was 0.75, aligned to the unit box
    seed=42,
    late_start=True,
    late_start_it=int(ea_late_start / ea_update_interval),
    max_cma_iter=(agent_cfg.max_iterations - ea_late_start) / ea_update_interval,
)
```

The widened range is applied where the CMA-ES ranges are defined, so that both the random phase and the CMA-ES phase explore the same broader interval.

```python
# usd_generator.py
CMAES_PARAM_RANGES = {
    "thigh_length_scale": (0.75, 1.25),   # was (0.85, 1.15)
    "shank_length_scale": (0.75, 1.25),
}
```

The explicit boundary handler is set in the options block of the CMA-ES constructor so that the saturating transform is documented rather than implicit.

```python
# usd_generator.py, CMAESDesignGenerator.__init__ opts.set({...})
"BoundaryHandler": "BoundTransform",
```

The optional robustness measures are listed for completeness. A per coordinate initial step can be supplied through the `CMA_stds` option if the thigh and shank sensitivities differ, the morphology window can be lengthened by raising `ea_update_interval` so that each design accrues more episodes before it is ranked, and a noise handler can be wrapped around the tell call to average noisy fitness. For severe multimodality the outer loop can be routed through the pycma functional interface with restarts, although for a two dimensional search with a population of sixty four the binding constraints are the feedback order, the step size, the quantisation, and the range, not the population size. Together the first three changes restore a correctly initialised, correctly informed optimiser whose proposals are no longer erased by rounding, which is expected to yield a varied population that converges to an interior optimum rather than a quantised corner.

---

## 4. Training Variance Analysis

This section examines the persistently high variance of the training curves. It first describes the observations and their significance, then traces the software architecture to identify candidate root causes and tests the reset hypothesis, then develops the reinforcement learning theory that governs value estimation across a family of morphologies, then frames a high level solution, and finally specifies the implementation.

### 4.1 Problem Description

The evidence is the set of unsmoothed training plots in `/ws/context/artefacts/plots`, all of which belong to the single run `sf_copt/2026-06-26_05-26-28` carried to step twenty nine thousand nine hundred and ninety nine, with no second curve overlaid (`../context/task_plots.md` section 5). Because there is one trace per panel, the visible band is the high frequency oscillation of that single curve from one iteration to the next, not a confidence interval. The plots fall into four families, and each is interpreted below.

The component reward panels `reward1.png` through `reward7.png` show the per term episodic rewards logged by the reward manager. The most important is the headline task reward `rew_lin_vel_xy`, which rises early and then oscillates in a wide band between approximately zero and eighteen for the entire run without contracting (`reward7.png`). The penalty terms behave similarly, with bands that remain wide or widen slightly toward the end. The significance is that the per episode return is highly variable throughout, and crucially the variability does not diminish as training proceeds.

The loss panels `loss1.png` and `loss2.png` show the optimisation diagnostics. The entropy declines smoothly from about six point two to about four point seven, and the policy noise standard deviation in `noise.png` declines gently from about zero point five five to about zero point four six, so the policy is becoming modestly more deterministic and is not collapsing. The surrogate loss is small and noisy about zero. The value function loss in `loss2.png`, by contrast, is large and very spiky, sitting mostly between two and ten with frequent excursions to the axis cap and a final value near seventeen and a half. The adaptive learning rate in `loss1.png` thrashes between near zero and about two times ten to the minus four on nearly every iteration, which is the schedule reacting to a noisy Kullback Leibler estimate.

The curriculum panels `curriculum1.png` through `curriculum3.png` show staircases that are still moving at the end of training. The terrain level rises in a noisy band that has not saturated at iteration thirty thousand, the command velocity ranges for the linear and angular axes are still expanding, the push force oscillates adaptively, and the tracking reward standard deviations are still sharpening. The termination panel `termination.png` shows the fall fraction settle into a noisy band between about twelve and twenty two percent while the survival fraction holds near eighty eight percent, with neither band tightening.

The central anomaly is now precise. The per episode reward variance does not shrink even though the sampled designs collapse to a single identical morphology by iteration nineteen thousand eight hundred, as established in section three. A single morphology run would be expected to settle into a tighter band once the policy matures. Two hypotheses were posed, the first that the variance is an artefact of the metric buffering and the environment reset that fires every one hundred and twenty iterations, and the second that Proximal Policy Optimisation cannot converge because the critic cannot estimate value across varying link lengths. The following subsections resolve both.

### 4.2 Current Implementation

The variance has both an instrumentation cause and a genuine environment cause, and the two must be separated because they call for different remedies.

The instrumentation cause concerns the headline reward statistic. The logger reports `Train/mean_reward` as the arithmetic mean of a rolling window of length one hundred that is meant to hold the most recent completed episode returns (`on_policy_runner.py:231`). In the co-optimisation runner the morphology update branch extends that window, for every one of the four thousand and ninety six environments, with the current partial and not yet terminated reward sum, then zeroes the accumulators (`copt_on_policy_runner.py:270-284`). Because the window holds only one hundred entries while four thousand and ninety six are appended, the window is overwritten wholesale with truncated returns drawn from one hundred arbitrary environments every one hundred and twenty iterations. The headline statistic at those iterations is therefore the mean of one hundred mid episode partial returns rather than of completed episodes, which injects a periodic distortion that recurs forever and does not decay as designs converge. This is a pure logging artefact, and it confirms the first hypothesis for the headline curve.

The genuine cause concerns the per component episodic rewards in `reward1.png` through `reward7.png`, which are computed from completed episode sums and are not corrupted by the buffer flood. Their persistent width has three contributors. The first is the morphology heterogeneity that prevails until iteration nineteen thousand eight hundred, during which sixty four distinct designs are evaluated simultaneously and return genuinely different rewards. The second, and the one that explains why the variance persists after the designs collapse, is the curriculum non-stationarity. Every curriculum term begins before iteration eight thousand and continues to advance to the end of training, so the terrain difficulty, the command ranges, the push magnitude, and the reward shaping are a moving target throughout (`../context/task_plots.md` section 4). A single morphology in an ever hardening environment does not produce a tightening reward band, which accounts for the observation that the variance never shrinks. The third contributor is the synchronized reset.

The reset hypothesis is confirmed as a contributor and characterised precisely. The morphology update calls `unwrapped_env.reset()` with no arguments, which resets all four thousand and ninety six environments (`copt_on_policy_runner.py:449`). That reset zeroes the episode clock for every environment and, critically, invokes `curriculum_manager.compute` for all environments at once (`manager_based_rl_env.py:356`). The terrain curriculum then promotes or demotes the terrain level for the whole population synchronously, and because the reset truncates whatever episodes were in flight, the distance walked that drives the promotion decision is measured over a truncated interval rather than a full episode (`curriculums.py:454-493`, `../context/isaaclab_env.md` section 9). The command, push, and reward standard deviation gates, which fire when the never reset master step counter crosses a multiple of their interval, can also be double evaluated at the extra reset. The result is a synchronized step change in terrain difficulty and shaping for all environments every one hundred and twenty iterations, riding on top of a monotonically advancing schedule. A single morphology baseline run never performs this synchronized reset, so this mechanism is specific to the co-optimisation runner. The morphology update window of three thousand control steps exceeds the one thousand step episode length, so unlike the earlier configuration analysed in `../context/RESET.md` episodes do complete and time outs do fire between resets, yet the truncation at the boundary still phase locks all environments to step zero.

The conclusion of this subsection is that action is required on three fronts. The buffer flood must be removed because it corrupts the headline reward statistic, the synchronized reset must be decoupled from the curriculum advance because it injects a periodic perturbation that a baseline run never sees, and the curriculum induced variance must be acknowledged as partly intrinsic to a hardening environment and addressed by stabilising the design distribution earlier so that morphology ceases to be an additional source of heterogeneity. The next subsection establishes the theoretical basis for the residual concern, namely the difficulty of value estimation across a morphology family.

### 4.3 Training Implication

The co-optimisation problem turns a single Markov Decision Process into a family of processes indexed by the design parameters. Let $\theta_d$ denote the design, here the thigh and shank length scales, and let the family be $\mathcal{M}(\theta_d) = (\mathcal{S}, \mathcal{A}, P_{\theta_d}, r_{\theta_d}, \gamma)$, where the transition kernel $P_{\theta_d}$ and the reward $r_{\theta_d}$ both depend on the morphology through the dynamics and the geometry dependent reward terms such as the base height penalty and the feet distance penalty. The policy is shared across the family, so the relevant value function is the design conditioned function

$$ V^{\pi}(s, \theta_d) = \mathbb{E}_{\pi, P_{\theta_d}}\!\left[\sum_{t\ge 0} \gamma^t\, r_{\theta_d}(s_t, a_t) \,\middle|\, s_0 = s\right]. $$

A correct critic must approximate $V^{\pi}(s, \theta_d)$ as a function of both the state and the design. The investigation establishes that the design is observable to the critic, since the critic observation group contains the per environment link lengths, the body masses, and the inertias (`cfg/SF/limx_base_env_cfg.py:459-559`), and the design is also available to the actor through the estimator network that encodes the privileged observations, which in the co-optimisation policy is trained jointly with the actor because the latent is concatenated without a stop gradient (`copt_actor_critic.py:127-137`). The second hypothesis, that the critic cannot see the morphology, is therefore false as a statement about observability. The difficulty, if any, is one of fitting a moving target, and three mechanisms make the target move.

The first mechanism is the non-stationarity of the design distribution. Proximal Policy Optimisation is an on policy method that assumes the data distribution is approximately stationary within a trust region defined by the clipping parameter and the Kullback Leibler target. The morphology update replaces the design distribution every one hundred and twenty iterations, during the random phase by widening it and during the CMA-ES phase by shifting it, so the joint distribution of states and designs over which the critic is fitted changes discretely at each update. The value target $V^{\pi}(s, \theta_d)$ is consequently re-anchored at each update, which inflates the value function loss immediately after an update and keeps the temporal difference residual

$$ \delta_t = r_t + \gamma\, V(s_{t+1}, \theta_d) - V(s_t, \theta_d) $$

larger and noisier than it would be for a fixed morphology. Because the Generalised Advantage Estimation advantage is the discounted sum of these residuals (`rollout_storage.py:130-144`), the noise propagates into the advantage and hence into the policy gradient.

The second mechanism is the global normalisation of the advantage. The implementation normalises the advantage once over the entire pooled rollout of all four thousand and ninety six environments (`rollout_storage.py:150`, `ppo.py:190-191`). When the population contains morphologies whose return scales differ, the pooled mean and standard deviation are a mixture statistic, so the standardised advantage of a design whose returns sit far from the mixture mean is mis-scaled. Designs with large returns dominate the pooled standard deviation and compress the advantages of the rest, which blurs the credit assignment across morphologies and weakens the gradient for the designs that most need improvement. This effect is strongest exactly when the design distribution is widest, which is the random phase, and it diminishes as the population collapses, which is why the optimisation can still reach a competent policy on the final single morphology.

The third mechanism is the curriculum non-stationarity, which alters both $P_{\theta_d}$ through the terrain and $r_{\theta_d}$ through the shaping schedule. Even for a fixed morphology the value target moves because the environment hardens, and the synchronized reset analysed in the previous subsection adds a periodic shock to the state distribution. The truncation at the reset boundary also matters for the value targets, because the time out bootstrap that converts a truncation into a continuation is applied only when the environment reports a time out signal (`ppo.py:177-180`), whereas the silent morphology reset emits no such signal. Within a rollout this is harmless, since the reset falls between rollout windows rather than inside one, so the Generalised Advantage Estimation of any single window is not corrupted (`../context/rsl_rl.md` section 3.2). The harm is confined to the discarded in flight episodes and to the logging, which returns the analysis to the instrumentation cause of the previous subsection.

The synthesis is that the high plotted variance is not evidence of a critic that is blind to morphology, since the critic observes the design, but of three compounding non-stationarities layered over a logging artefact. The design distribution moves, the advantage normalisation pools heterogeneous morphologies, and the curriculum hardens continuously while a synchronized reset perturbs the state distribution every one hundred and twenty iterations. The large and spiky value function loss in `loss2.png` is the direct signature of a value function chasing this moving target, and the absence of an explained variance metric in the library means this signature cannot at present be read quantitatively, which motivates adding one.

### 4.4 Improvement Scope

The solution attacks the four contributors in order of evidential strength. The first measure removes the logging artefact by ceasing to inject partial returns into the reward window at a morphology update, so that the headline reward statistic reflects completed episodes only and the periodic distortion disappears. The second measure decouples the morphology swap from the curriculum advance, so that the synchronized terrain promotion and gate evaluation no longer fire on the truncated episodes of a forced reset, which removes the periodic perturbation that a baseline run never experiences. The third measure improves the conditioning of the value learning, by adding an explained variance diagnostic so the critic fit can be read directly, and by normalising the advantage in a way that respects the heterogeneity of the population rather than pooling all morphologies into one statistic. The fourth measure reduces the morphology induced component of the variance at its source by adopting the section three corrections to the design optimiser, which stabilise the design distribution earlier and prevent the wide and growing random phase distribution from inflating the early variance. The residual variance that remains after these measures is the intrinsic consequence of training a single policy in a continuously hardening curriculum, which is expected and is best presented with light smoothing rather than removed.

### 4.5 Implementation Plan

The plan specifies four changes, each grounded in the mechanisms above.

The first change removes the partial return flood. In the morphology update branch of `learn`, the lines that extend the reward and length windows with the partial accumulators are deleted, and the accumulators are simply zeroed, so that the windows continue to hold only genuinely completed episode returns from the normal done path. The fitness accumulation for the truncated final step is likewise restricted to environments that actually terminated, which also removes the downward bias in the CMA-ES cost signal.

```python
# copt_on_policy_runner.py, learn(), inside the is_morph_update_time block
# ensure rewards are not logged for incomplete episodes.
# Currently when is_morph_update_time is True, partial episode rewards are logged
# In the reward logging and accumulation block starting at L325 in `copt_on_policy_runner.py`
# change as follows:
new_ids = (dones > 0).nonzero(as_tuple=False)
if (step < self.num_steps_per_env - 1):
    # ---- COPT: accumulate per-individual fitness --------
    completed_env_ids = new_ids[:, 0]
    for env_idx in completed_env_ids.tolist():
        ind_idx = self._env_to_individual[env_idx]
        self._individual_fitness[ind_idx] += cur_reward_sum[
            env_idx
        ]
        self._individual_episode_counts[ind_idx] += 1
    # ---- end COPT injection ----------------------------

    rewbuffer.extend(
        cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
    )
    lenbuffer.extend(
        cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
    )
    cur_reward_sum[new_ids] = 0
    cur_episode_length[new_ids] = 0

    if self.alg.rnd:
        erewbuffer.extend(
            cur_ereward_sum[new_ids][:, 0]
            .cpu()
            .numpy()
            .tolist()
        )
        irewbuffer.extend(
            cur_ireward_sum[new_ids][:, 0]
            .cpu()
            .numpy()
            .tolist()
        )
        cur_ereward_sum[new_ids] = 0
        cur_ireward_sum[new_ids] = 0

# At the morphology update block at Line 380, the following update is to be made:
# ---- COPT: EA generation update ---------------------------------
if is_morph_update_time:
    if (
        self._ea_late_start > 0
        and self._design_generator.late_start
        and is_late_start_toggle_time
    ):
        print(
            f"toggling late start at absolute iteration {it + 1} with configured threshold at {self._ea_late_start} iterations"
        )
        self._design_generator.toggle_late_start()
    with torch.inference_mode():
        self._update_morphology(it + 1)
    # Refresh observations after in-place update
    obs = self.env.get_observations().to(self.device)
    if self.log_dir is not None:
        for env_idx in range(self.env.num_envs):
            ind_idx = self._env_to_individual[env_idx]
            self._individual_fitness[ind_idx] += cur_reward_sum[env_idx]
            self._individual_episode_counts[ind_idx] += 1
        cur_reward_sum[:] = 0
        cur_episode_length[:] = 0
```

The second change decouples the morphology swap from the curriculum advance. The forced full reset is replaced by a morphology aware reset that re-spawns the designs and re-anchors the episodes without invoking the curriculum promotion on truncated episodes. The minimal form introduces a guard that suppresses the terrain promotion within the reset triggered by the morphology update, so that the terrain curriculum advances only on natural episode terminations.

```python
# manager_based_rl_env.py path is external, so the guard is set from the runner
# around the morphology reset. A boolean on the env, read by terrain_levels_vel_delayed,
# skips the move_up / move_down update when a morphology reset is in progress.
unwrapped_env._suppress_terrain_curriculum = True
unwrapped_env.reset()
unwrapped_env._suppress_terrain_curriculum = False
```

```python
# tasks/locomotion/mdp/curriculums.py, terrain_levels_vel_delayed
if getattr(env, "_suppress_terrain_curriculum", False):
    return torch.mean(env.scene.terrain.terrain_levels.float())
```

To implement the next two changes a file containing the derived PPO algorithm class, `class CoptPPO` needs to be created at `algorithms/copt_ppo.py` in the package `co_optimisation`. This class will inherit from `class PPO` defined in `algorithms/ppo.py` in the rsl_rl installation in the workspace. Refer to the parent class and implement the following two changes for improving policy training. The use of a new class will have to be integrated within `runners/copt_on_policy_runner.py` in the package `co_optimisation` and in `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py`. `CoptOnPolicyRunner._construct_algorithm` will be updated to expect `class CoptPPO` and the training and evaluation scripts will set the appropriate algorithm class name for object creation in `runners/copt_on_policy_runner.py` in the package `co_optimisation`

The third change improves the value learning diagnostics and conditioning. An explained variance scalar is added to the update so the critic fit can be read directly, computed from the batched returns and values already available in the update loop.

```python
# copt_ppo.py, update(), after the mini-batch loop, before returning loss_dict
returns = self.storage.returns.flatten(0, 1)
values = self.storage.values.flatten(0, 1)
var_returns = returns.var()
explained_var = 1.0 - (returns - values).var() / (var_returns + 1e-8)
loss_dict["explained_variance"] = explained_var.item()
```

The conditioning of the advantage is improved by normalising within each morphology rather than across the pooled population. Because environments map to individuals by the round robin assignment already held on the runner, the advantage can be standardised per individual before the update, which respects the heterogeneous return scales of the population.

```python
# a per-individual advantage normalisation applied in compute_returns,
# using the env-to-individual map supplied by the runner.
adv = self.returns - self.values                      # [T, N]
for ind in range(num_individuals):
    cols = env_to_individual_mask[ind]                # boolean over N
    block = adv[:, cols]
    adv[:, cols] = (block - block.mean()) / (block.std() + 1e-8)
self.advantages = adv
```

The fourth change is the adoption of the section three corrections to the design optimiser, namely the correction of the tell and ask order, the reduction of the initial step size and the widening of the range. These stabilise the design distribution earlier and prevent the wide random phase distribution from inflating the early variance, which removes the morphology contribution to the variance at its source. Taken together the four changes are expected to eliminate the periodic logging distortion, remove the synchronized curriculum shock, expose and improve the value fit, and shrink the morphology induced variance, leaving only the intrinsic curriculum driven variability that is properly presented with light smoothing.

---

## 5. References

The sources below are numbered for reference throughout the document. Codebase citations appear inline as `file:line` and resolve against the workspace at `/ws`.

1. Hansen, N. (2016). The CMA Evolution Strategy, A Tutorial. arXiv:1604.00772. Source of the quarter of range rule for the initial step size, the statement that the optimum is expected within the mean plus or minus three step sizes, the cumulative step size adaptation update, and the default hyperparameter table.
2. Hansen, N., Akimoto, Y., Baudis, P. (2024). CMA-ES and pycma on GitHub, package `cma` version four point four point four. Behaviour confirmed in the installed source, specifically `options_parameters.py` for the default boundary handler `BoundTransform` and the maximum step size cap `maxstd_boundrange` of one third, `evolution_strategy.py` lines 1129 to 1161, 1506, and 4140 to 4269 for the boundary handler instantiation, the step size cap, and the stop criteria, and `boundary_handler.py` line 454 for the saturating transform.
3. pycma API documentation for `CMAEvolutionStrategy`. Source of the statement that the variables should be scaled so that a single standard deviation is useful and the optimum lies within about the mean plus or minus three step sizes.
4. Cheney, N., Bongard, J., SunSpiral, V., Lipson, H. (2018). Scalable co-optimization of morphology and control in embodied machines. Journal of the Royal Society Interface, 15:20170937. Source of the morphological innovation protection argument relevant to the cold restart of the optimiser after the random phase.
5. Hansen, N., Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2), 159 to 195. Source of the covariance matrix adaptation evolution strategy.
6. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. Source of the clipped surrogate objective and the on policy trust region assumption.
7. Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438. Source of the advantage estimator used by the algorithm.
8. Luck, K. S., Amor, H. B., Calandra, R. (2020). Data efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning. Conference on Robot Learning. Source for the use of an evolution strategy as the outer loop of a deep reinforcement learning co-design, as cited in `CMAES_DESIGN_GENERATOR.md`.
9. Workspace documentation, `../ARCHITECTURE.md`, `../CO_OPTIMISATION.md`, `CMAES_DESIGN_GENERATOR.md`, and `../context/RESET.md`. Sources of the simulation architecture, the environment lifecycle, the design generator theory, and the prior reset pathway investigation.
10. Investigation record for this document, `../context/knowledge_base.md` and the section files `../context/rsl_rl.md`, `../context/isaaclab_env.md`, `../context/copt.md`, `../context/cmaes.md`, and `../context/task_plots.md`, which hold the grounded findings and the `file:line` citations summarised here.
11. Experiment run `IsaacLab/logs/rsl_rl/sf_copt/2026-06-26_05-26-28`, comprising the design files under `designs/`, the training videos under `videos/`, and the saved configurations under `params/`, together with the training plots under `/ws/context/artefacts/plots`. Source of the empirical evidence for all three objectives.
