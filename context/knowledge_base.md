# knowledge_base.md — Shared Investigation Knowledge Base

This file is the shared, accumulating knowledge base for the co-optimisation investigation. Investigation agents read this file at start to learn the objectives and prior findings, write their detailed findings into a dedicated `<area>.md` file in this directory, and return a summary. The orchestrator consolidates those section files into this document between investigation waves. The area files were named `CONTEXT_<area>.md` until 2026-07-30, when the prefix was dropped as redundant within `context/`, and this file was itself renamed from `CONTEXT.md`.

## Investigation Objectives

1. High training variance. The PPO training plots (`/ws/context/artefacts/plots`) show much higher
   variance than a single-morphology RL run, and the variance does not shrink
   over time even though CMA-ES designs converge. Two hypotheses:
   (a) metric buffering / reset artefacts from the env reset every
   `ea_update_interval=120` iterations, and
   (b) PPO/critic inability to estimate value across varying link lengths.
   Both must be investigated to find the root cause, then a fix planned.

2. Redundant re-training of the first 8000 iterations. The first 8000 iterations
   (random-design phase) are identical across experiments and wasteful to repeat.
   Current save/load persists only model weights, not the state of the
   environment, its managers, the PPO algorithm, or the CMA-ES generator.
   Extend save/load to persist and restore all such state for seamless,
   identical continuation (also enables recovery from GPU OOM crashes).

3. Biased / low-variety CMA-ES sampling. The design CSVs show CMA-ES collapsing
   to a single design with no variety. Investigate why, understand the role of
   the initial mean and `sigma0` in `usd_generator.py`, and set hyperparameters
   for maximum exploration and best optimisation.

## Writing constraints for the final COPT_INVESTIGATION_PLAN.md (carry through all notes)

- No `-`, `;`, or `:` to break sentences. Only commas. Use the three symbols
  only in their legitimate non-sentence-breaking contexts (lists, code, ratios).
- No italics or bold unless explicitly asked.
- Coherent high-level structure, grounded in fact and evidence, formal academic
  tone with a succinct regal flair, sentences of moderate length.

## Key entry points and file map

- Training entry point: `tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py`
- COPT runner: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`
- Design generators: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py`
- COPT policy: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/modules/copt_actor_critic.py`
- COPT utils: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/` (respawn.py, update.py, analysis.py, env_state.py)
- Task env cfg: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/`
- rsl_rl library: `/ws/rsl_rl/`
- IsaacLab library: `/ws/IsaacLab/`
- Existing docs to mine: `../ARCHITECTURE.md`, `../CO_OPTIMISATION.md`, `../plans/CMAES_DESIGN_GENERATOR.md`, `RESET.md`, `../tron1-rl-isaaclab-cozum/context/joint_control_analysis.md`
- Document indices, added 2026-07-30 when the markdown corpus was reorganised: `README.md` in this directory registers every context document with a summary and a currency note, and `../plans/README.md` registers every plan document with its implementation status. Consult those two indices before opening any document.
- Experiment logs: `/ws/IsaacLab/logs/rsl_rl/sf_copt/2026-06-26_05-26-28/` (designs/, videos/)
- Plots: `/ws/context/artefacts/plots/` (reward1-7.png, loss1-2.png, curriculum1-3.png, termination.png, noise.png)

## Established facts (orchestrator, from initial read)

### train.py COPT configuration (lines 194-234)
- `policy_type == "COPT"` uses `GrowingDesignDistCMAESDesignGenerator` (note: this
  symbol is USED at line 209 but the visible import block imports only
  `CMAESDesignGenerator, RandomDesignGenerator` — VERIFY whether import resolves
  or is a latent NameError / edited file).
- `_num_individuals = 64`, `ea_update_interval = 120`, `ea_late_start = 8000`.
- `sigma0 = 0.75`, `seed = 42`, `late_start=True`, `late_start_it = 8000/120 = 66`,
  `max_cma_iter = (max_iterations - 8000)/120`.
- params optimised by CMA-ES: only `thigh_length_scale`, `shank_length_scale`,
  each range `(0.85, 1.15)` from `DEFAULT_PARAM_RANGES`.
- agent_cfg_dict["copt"] adds ea_update_interval, ea_late_start, num_individuals,
  randomise_before_late_start=True.
- save/load: `train.py` calls `runner.load(resume_path)` only if `agent_cfg.resume`.
  Config dumped to params/env.yaml, agent.yaml, env.pkl, agent.pkl.

### Design CSV evidence (objective 3)
Columns are 4 scalable links (hip_R_thigh, hip_L_thigh, knee_R, knee_L) as
`{x,y,z}` dicts. Baseline thigh z=0.25, shank z=0.30. CMA-ES drives only z.
- it=959 (gen ~8, random growing phase, scale ~0.17): thigh z in 0.24-0.26,
  shank z in 0.29-0.31. Very tight spread (distribution scale still small).
- it=7799 (gen ~65, end of random phase, scale ~0.99): thigh z in 0.22-0.29,
  shank z in 0.26-0.34. Full spread.
- it=29999 (final CMA-ES): EVERY individual identical, thigh z=0.28, shank z=0.34.
  These equal the UPPER BOUNDS (0.25*1.15=0.2875->0.28, 0.30*1.15=0.345->0.34).
  CMA-ES pinned both params to their maximum, not an interior optimum.
- Two mechanisms collapse variety: (a) CMA-ES sigma collapsed against the box
  boundary; (b) `_compute_link_extents` applies `round(val, 2)`, quantizing
  designs below 1 cm so any residual spread disappears.

### Growing design distribution (objective 1 and 3)
`GrowingDesignDistCMAESDesignGenerator._generate_individual` during late_start:
`scale = 0.95 * (g / (n - s)) + 0.05`, n=late_start_it=66, s=0.1.
So design spread grows from 5% of full range (g=0) to ~100% (g=66) across the
first 8000 iterations. The morphology distribution itself widens over the random
phase, which would tend to INCREASE plotted variance over the first 8000 iters.

### Reset / buffering mechanism (objective 1) — from RESET.md + runner read
- `_update_morphology(it)` runs every 120 iters: stops sim (Fabric off),
  generates/updates population, `apply_link_length_params`, `unwrapped_env.reset()`,
  `apply_actuator_params`, zeroes fitness accumulators.
- At `is_morph_update_time`, `learn()` extends the rolling `rewbuffer`/`lenbuffer`
  (deque maxlen=100) with the PARTIAL, not-yet-terminated `cur_reward_sum` /
  `cur_episode_length` of ALL envs, then zeroes them. This injects truncated
  returns into the very statistic the reward plots track, every 120 iters.
- Current `ea_update_interval=120` -> 120*24=2880 control steps between updates,
  which exceeds max_episode_length (~1000 steps at 50 Hz for 20 s), so episodes
  CAN complete (unlike the old interval=10 case analysed in RESET.md).
- VERIFY num_envs for the COPT task, max_episode_length, and exactly how
  `OnPolicyRunner.log()` computes the logged mean reward from rewbuffer.

## Open questions for investigation waves

- rsl_rl: exact `OnPolicyRunner.save/load` contents; `PPO.update` losses and how
  mean_value_loss / explained variance / entropy are logged; GAE in RolloutStorage;
  observation normalizer (Normalization) state. (Agent A)
- IsaacLab: full enumeration of stateful members of ManagerBasedRLEnv and each
  manager (Curriculum, Command, Termination, Reward, Observation, Action, Event),
  episode_length_buf, common_step_counter, what `reset()` resets, curriculum
  state that advances (terrain levels, push force, command ranges) and is NOT
  persisted. (Agent B)
- co_optimisation: full trace of CoptOnPolicyRunner.learn + _update_morphology,
  fitness math, the dual pathway (_reload_morphology vs _update_morphology),
  apply_link_length_params/apply_actuator_params internals, what CMA-ES state
  exists and where save_state is (not) called. (Agent C)
- CMA-ES: pycma CMAEvolutionStrategy options, meaning of sigma0 in [0,1]^2 box,
  bounds handling, CMA_stds, integer/quantization handling, stop criteria,
  restart strategies (IPOP/BIPOP), and a fuller statistical read of the design
  CSV trajectory. (Agent D)
- Task/plots: COPT task registration and env cfg (num_envs, episode_length_s),
  curriculum cfg details, observation group contents (does the critic see design
  parameters?), encoder cfg, agent PPO cfg values; and a description of each plot
  in /ws/context/artefacts/plots characterising the variance. (Agent E)

## Consolidated findings (wave 1, cross-validated)

Detailed grounding lives in the five section files rsl_rl.md,
isaaclab_env.md, copt.md, cmaes.md, task_plots.md.
Summary of the decision-relevant, cross-validated facts follows.

### Corrected configuration facts (supersede earlier assumptions and ARCHITECTURE.md)
- Task `Isaac-Limx-SF-Copt-Rough-v0`, env `SFCoptBlindRoughEnvCfg` -> `SFCoptEnvCfg`
  (CoptObservationsCfg), agent `SFCoptPPORunnerCfg` (experiment_name "sf_copt",
  max_iterations 30000), launched by `djinn start train copt` via
  `biped/scripts/rsl_rl/train.py --policy-type COPT`. Resume uses
  `--resume true --load_run <run> --checkpoint_path <ckpt>`.
- num_envs 4096, episode_length_s 20 s, dt 0.005, decimation 4 ->
  max_episode_length 1000 control steps. num_steps_per_env is 25 (not 24).
- Morph window = ea_update_interval(120) x num_steps_per_env(25) = 3000 control
  steps > 1000, so episodes complete and time_out can fire (unlike RESET.md's old
  interval=10 case). The train.py comment "50 * 24 = 1200" is stale on both factors.
- PPO cfg, learning_rate 1e-3 adaptive (desired_kl 0.01), gamma 0.99, lam 0.95,
  clip 0.2, entropy_coef 0.005, value_loss_coef 1.0, use_clipped_value_loss True,
  5 epochs x 4 mini-batches, max_grad_norm 1.0, no observation normalization,
  no RND, no symmetry, global advantage normalization.
- Encoder, hidden_dims [128,64,16], output_detach True. CoptActorCritic derives
  the latent size from hidden_dims[-1]=16 and ignores num_output_dim=19.

### Objective 1 (variance), reframed and grounded
- Hypothesis (b) "critic cannot see the morphology" is FALSE. `link_lengths`,
  `robot_mass`, `robot_inertia` are in the `critic` group, and `link_lengths`,
  `robot_mass` are in `privilegedObs` (which feeds the actor estimator). The
  critic does observe the design. The actor sees only a detached 16-dim latent.
- Dominant, clearly evidenced variance driver is the COPT learn() override at
  copt_on_policy_runner.py:270-284, which every 120 iters extends the maxlen-100
  rewbuffer/lenbuffer with the partial, non-terminated returns of all 4096 envs
  and then zeroes them. Because 4096 >> 100, the reward window is overwritten
  wholesale with truncated returns, so Train/mean_reward
  (= statistics.mean(rewbuffer), on_policy_runner.py:231) shows a periodic
  artefact that never decays as designs converge.
- Env-side contributor, the every-120-iter unwrapped_env.reset() (copt:449) zeroes
  episode_length_buf for all envs AND calls curriculum_manager.compute(all)
  (manager_based_rl_env.py:356), forcing a synchronized terrain promote/demote and
  extra gate evaluations on truncated episodes that a baseline run never sees.
- GAE is NOT corrupted by the silent reset, the reset falls between rollout
  windows, after compute_returns and update; advantage normalization is global
  across all pooled morphologies (rollout_storage.py:150).
- Curriculum non-stationarity (terrain, push, command ranges, tracking std) all
  start before iter 8000 and keep moving to iter 30000, confounding any sole
  attribution to morphology. The growing design distribution widens variance over
  the first 8000 iters by construction.
- No explained_variance metric exists in rsl_rl; only Loss/value_function is
  available as a critic-fit proxy, and it is large and spiky (final 17.5).
- Plots are a single raw trace, every Episode_Reward/* band stays wide or widens
  and never contracts; rew_lin_vel_xy oscillates ~0 to 18 throughout.

### Objective 2 (save/load), complete state enumeration
- rsl_rl side, base save persists model_state_dict, optimizer_state_dict, iter,
  infos (+rnd if used). NOT persisted, self.alg.learning_rate (adaptive LR
  silently resets to cfg 1e-3 on resume), tot_timesteps, tot_time.
- COPT does NOT override save/load, so NONE of generation, current_population,
  _individual_fitness, _individual_episode_counts, _copt_started,
  _env_to_individual (recomputable), or the CMA-ES _es / _pending_solutions /
  _last_solutions / _terminated / late_start is persisted.
  CMAESDesignGenerator.save_state and the es_state_path load hook exist but are
  never called or passed.
- Scheduling bug, is_late_start_toggle_time and is_morph_update_time are computed
  relative to start_iter (copt:160,163), so a resume restarts the 8000-iter random
  phase and never reaches CMA-ES at the right time. This directly blocks the
  goal of skipping the first 8000 iterations.
- IsaacLab side, no built-in runtime-state checkpoint, manager serialize() returns
  only class_to_dict(cfg) (config, not state). scene.get_state()/env.reset_to()
  cover only physical robot state (root pose, joint pos/vel). Must manually
  capture, common_step_counter (never reset, master curriculum clock),
  _sim_step_counter, episode_length_buf, TerrainImporter.terrain_levels/env_origins,
  live command ranges, push velocity_range, tracking reward std,
  TerminationManager._last_episode_dones, RewardManager._episode_sums,
  Command term buffers, EventManager interval/reset timers, ActionManager
  _prev_action, ObservationManager history CircularBuffers, and global RNG state
  (torch, cuda, numpy, python). Startup randomization (mass, friction, gains,
  joint offsets, armature) is sampled once and not re-applied at reset, capture
  the resulting physics tensors or reproduce the startup RNG.

### Objective 3 (CMA-ES), root causes
- CMA-ES off-by-one (copt _update_morphology:430-435), generate_population is
  called BEFORE update_with_fitness, and generate_population overwrites
  _last_solutions, so es.tell pairs the current batch's genotypes with the
  previous batch's fitness, decorrelating feedback by one generation. The
  documented order (../CO_OPTIMISATION.md 4.9.4) is the reverse and correct.
- sigma0 0.75 on the [0,1]^2 box with x0 0.5 implies x0 +/- 3 sigma0 = [-1.75,2.25],
  overflowing the box. pycma silently caps the per-coordinate std at
  maxstd_boundrange = bound_range/3 = 0.333, still near-uniform with heavy mass on
  the bounds. Default boundary handler is BoundTransform (saturating), not
  BoundPenalty as ../plans/CMAES_DESIGN_GENERATOR.md claims.
- round(val, 2) in _compute_link_extents quantizes z to a 1 cm grid (about 4% of
  link length), larger than the late-stage CMA-ES spread, erasing variety,
  flattening fitness and tripping tolfun/tolflatfitness.
- Range (0.85, 1.15) is only +/-15%, too narrow; CMA-ES pinned both params to the
  upper bound. Statistical table, std grows over 0-8000 then collapses to 0 by
  iter 19799, mean drifts to thigh z 0.28 and shank z 0.34 (rounded upper bounds).
- Recommended, sigma0 ~0.25, keep x0 0.5, keep [0,1]^2 with explicit BoundTransform,
  widen ranges to ~(0.75,1.25), remove or reduce round(val,2), fix the off-by-one,
  optionally CMA_stds, NoiseHandler, IPOP/BIPOP via cma.fmin2.

### Cross-cutting blocking bug
- train.py:209 uses GrowingDesignDistCMAESDesignGenerator, which is never imported
  (train.py:98-102 import only CMAESDesignGenerator, RandomDesignGenerator) and not
  re-exported by runners/__init__.py. The recorded growing-distribution trajectory
  proves the class did run, so the executed train.py differed from the visible
  import block. Add the import.

### Investigation status, SATURATED
The call stack has been traced end to end (train.py -> rsl_rl -> IsaacLab managers
-> co_optimisation -> CMA-ES library -> empirical designs and plots). No new root
cause is expected from further fan-out. Proceed to analysis and writing.

## LIPM-guided reward work stream (2026-07-10)

A survey and implementation plan for adding the LIPM-guided stability reward of
Su et al. (arXiv:2509.09106, local /ws/2509.09106v1.pdf) to the COPT pipeline is
recorded in /ws/plans/LIPM_REWARD.md. The literature grounding is literature.md
cluster 6. Key facts established during this work, (a) the plots in
/ws/context/artefacts/plots-latest are a NEWER run than /ws/context/artefacts/plots, sf_copt/2026-07-03_08-16-11,
45000 iterations over 3.377 days, same non-contracting variance signature,
rew_lin_vel_xy final 20.25 oscillating 0-20 throughout, Loss/value_function final
11.27, base_contact final 0.1128, explained_variance panel now exists (0.92-0.99,
jittery), curriculum lin_x saturates at 1.5 around iter 35k, angular tracking std
still declining at 45k, (b) the paper's stable-reward equation (4) omits the
negative exponent its own codomain requires, implementations must negate,
(c) implementation seams verified in live code, mdp/rewards.py (append
ManagerTermBase classes), CoptRewardsCfg(RewardsCfg) subclass bound at
cfg/SF/limx_base_env_cfg.py:1663 in SFCoptEnvCfg, IsaacLab fields
body_com_pos_w/body_com_lin_vel_w (articulation_data.py:940,957),
root_physx_view.get_masses(), math_utils.yaw_quat (math.py:565), link_lengths
body-name pairs hip_R_thigh_Link->knee_R_Link and knee_R_Link->
ankle_R_actuator_Link, nominal thigh z 0.25 and shank z 0.30, base height target
0.75, so z_c(d) = 0.20 + 0.25 s_thigh + 0.30 s_shank.

## Learned-model architecture extension (2026-07-02, implemented 2026-07-03)

A new work stream extends COPT with an encoder-decoder estimator trained
jointly with PPO (policy type COPT-LEARNED). The design document with the full
implementation plan is `../plans/COPT_LEARNED_MODEL.md`. The verified literature survey
backing it is `literature.md`. Code-state corrections superseding
older findings in this knowledge base are recorded in `copt.md`
section 8 and `rsl_rl.md` section 7, notably that the runner save/load,
morphology schedule, rewbuffer flood, and CMA-ES tell/ask ordering defects are
already fixed in the live sources, and that `CoptPPO` now logs
explained_variance.

Section 3 of `../plans/COPT_LEARNED_MODEL.md` was implemented in full on 2026-07-03,
following every step verbatim and cross-checked against the live sources
rather than against the possibly stale line numbers quoted in the document.
The implementation record, including two ambiguities in the document that
were resolved during implementation, is recorded in `copt.md`
section 9. The COPT-LEARNED policy type is registered end to end, task
registry, agent config, environment config, the new `CoptEstimator` module,
`CoptLearnedModelActorCritic`, `CoptLearnedModelPPO`, runner plumbing,
train.py and play.py entry points, and djinn, and has been validated by
syntax-checking every touched file and cross-referencing every new class
name, import, and task ID for consistency. It has not been exercised inside
Isaac Sim, since no GPU training run was launched as part of this task.

## Gait quality work stream, SD_BRS1 (2026-07-17, updated 2026-07-31)

The run sd_brs1_flat/2026-07-15_06-59-45 ends with a crouched, forward pitched, asymmetric shuffle. A full diagnosis and a phased improvement strategy are recorded in /ws/plans/GAIT_STRATEGY.md, the code grounded facts in /ws/context/brs_gait.md, and the supporting survey in /ws/context/literature.md cluster 7. Headline root causes, the foot clearance target 0.12 sits below the 0.124 m standing height of the Link6 frame so the term rewards ground level foot sweeping, the sparse feet_air_time at weight 500 punishes every exploratory short step, the orientation penalty is far below reference ratios and no orientation or realistic height termination exists, the all zero default pose rests on the knee hard limit at the straight leg singularity, hip and knee damping ratios are 0.07 to 0.16 against a 0.7 target (BRS.md damping recommendations were never adopted), and no symmetry mechanism exists although the vendored rsl_rl supports symmetry_cfg.

As of 2026-07-29 the pipeline has progressed through eight recorded runs (see experiment history table in brs_gait.md). The latest run sd_brs1_flat/2026-07-27_07-19-39 (Phase A2-b, 16987 iterations, mean reward 1108) walks with alternating feet and bent knees on the new crouched nominal pose. Two residual defects identified from video and TensorBoard analysis, narrow stance width (min_feet_distance 0.21 m barely above the 0.19 m collision threshold, pen_feet_distance actively firing at 0.566/s) causing the feet to plant too close, and hip roll oscillation (pen_hip_deviation 7.7 degrees, pen_ang_vel_xy func value 0.670) with the torso rocking in roll rather than swaying laterally. Root cause is the narrow stance, which requires only 0.105 m of lateral CoM shift per step, achievable through 5.7 degrees of torso roll alone. Two config-only changes are proposed, raising min_feet_distance to 0.32 m (1.23 times hip width, within the stability band) and raising pen_ang_vel_xy from -2.0 to -5.0 (G1 reference proportion), both in brs_base_env_cfg.py with no function edits and no backwards compatibility impact.

### Gait efficiency sub stream (2026-07-31)

The work stream turned on 2026-07-31 from whether the robot walks to how well it walks, on the evidence of the four runs `2026-07-28_06-37-24`, `2026-07-29_04-38-14`, `2026-07-29_10-47-14` and `2026-07-30_07-59-22`, which differ only in `min_feet_distance`, in the weight of `pen_ang_vel_xy`, and in the presence of an ankle deviation penalty, and therefore constitute an ablation already performed. The analysis is the twentieth pass of `brs_gait.md`, the survey is cluster 13 of `literature.md`, and the remedy is `/ws/plans/GAIT_EFFICIENCY_PLAN.md`. It is the first analysis in this work stream computed from the `play.py` numpy dumps directly rather than inverted from logged reward rates.

The verdict is that the policy tracks its command through a manner of walking that would not transfer to hardware. It descends onto the ground at 1.68 m/s, which is 1.26 times free fall from the swing apex, taking 3.08 body weights on the average touchdown and 13.76 at the worst against the 1.0 to 1.5 of human walking. It stands within 0.02 rad of a mechanical stop for a third of every second, its ankle roll actuators sitting above 98 percent of their effort ceiling for 6 to 7 percent of it, and its knee switching between full extension and full fold with only 15 percent of its time in between. It circulates 850 W of joint power to deliver 89, a mechanical cost of transport of 2.38 against the 0.2 of human walking, though the same figure on net power is 0.25, so the trajectory is not expensive and only the manner of producing it is. And it does not turn, in any of the four runs.

Five causes, all of them properties of the reward specification. The clearance reward's maximiser over a swing of fixed duration is a trapezoid, which earns 46 percent more than a sinusoidal arc of the same apex and is opposed by nothing. The single support terms price the weight transfer interval at zero, so although the gait clock commands 19.6 percent double support, close to the human 24, the policy delivers 2.3, and compliance is worth only 3 percent of the positive budget. The effort and smoothness terms supply 6 percent of income of which five sixths is one term, so leaning on a mechanical stop is cheaper than controlling the joint. The `feet_distance` hinge measures the PLANAR separation, which a long stride satisfies while the feet cross the midline beneath it, so stance width has never been regulated by it. And the yaw authority of a machine whose `HipYaw` joints are declared fixed is spent by a narrow stance, a lateral command range of 0.01 m/s, a slide penalty on the stance pivot that turning requires, and the absence of any foot yaw term.

One result reaches beyond this work stream. The articulation joint and body order is LEFT before RIGHT, proved from the joint limits since index 2 spans a range only `HipPitchL` admits and index 3 one only `HipPitchR` admits, so the `JOINT_NAMES` and `FEET_NAMES` tables in `dashboard-brs.py` and the corresponding comment in `play.py` are transposed, and every per joint plot this project has exported carries swapped side labels.

## Objective 1 revisited, the design swap as a non-stationarity (2026-08-10)

Objective 1 was reopened on this date under a sharper question, whether the periodic reassignment of a morphology to an environment index is itself incompatible with proximal policy optimisation. The analysis is `copt_ppo_nonstationarity.md` and the supporting survey is cluster 16 of `literature.md`. Its results revise this document on three points.

The association is exonerated, and provably so. Generalised advantage estimation recurses over time alone, the mini-batch generator flattens the time and environment axes and shuffles them together, the losses are means over the flattened batch, and the advantage normalisation is taken over the whole tensor, so the PPO gradient is exactly invariant under a permutation of the environment indices. Only the fitness accumulator and the per-individual masks read the index as an identifier, and both belong to the evolutionary bookkeeping. The round-robin allocation is moreover an exactly stratified one, sixteen environments per design held fixed across the generation, and should not be disturbed.

The dominant newly identified mechanism is an episode phase synchronisation. `_update_morphology` resets every environment through a bare `unwrapped_env.reset()` at `copt_on_policy_runner.py:672`, which zeroes the episode counter for the whole index range, and the runner's one desynchronising draw happens only at the top of `learn`. A rollout is twenty five control steps against an episode of one thousand, so after a swap a single batch samples a two and a half percent slice of episode phase across all four thousand and ninety six environments rather than the stationary distribution of it. The resulting intra-batch correlation collapses the effective sample size, one percent correlation alone reducing four thousand and ninety six environments to about ninety eight, and the synchronisation does not decay within the generation because a timeout resets an environment on the same beat it started on, leaving roughly forty six percent of the population still phase-locked when the next swap returns all of it to a common phase.

The claim of the 2026-07-30 audit below, that per-individual advantage normalisation with global normalisation disabled is among the variance remedies that have landed, is withdrawn. `compute_returns_design_wise` exists at `copt_ppo.py:96` and has no caller anywhere in the live tree, the runner calling the inherited `compute_returns` at `copt_on_policy_runner.py:390`, which requests global normalisation. The remedy was written and never connected.

Two further corrections. The hypothesis (b) of objective 1 is now doubly false, since the actor as well as the critic observes the design, the runner's observation mapping joining `morphologyObs` to the policy group at `limx_rsl_rl_ppo_cfg.py:139-142`, which supersedes the finding in `task_plots.md` that the policy sees only a detached sixteen dimensional latent. And the iteration ceiling now stands at forty five thousand rather than thirty thousand.

The document was revised the same day on a reading of the design generator's quantisation, and the revision sharpens rather than softens the conclusion. The two decimal rounding at `usd_generator.py:376` makes the design space a finite lattice, the reachable extents of 0.1875 to 0.3125 metres for the thigh and 0.225 to 0.375 for the shank admitting at most about one hundred and ninety five points on a one centimetre grid, of which roughly eighty are realised. That lattice is smaller than the population of two hundred and fifty six, so every generation contains duplicate designs at a mean multiplicity above three, each distinct design is carried by fifty or more environments, and after the opening generations no design arriving at a swap is novel. The out of distribution premise the first draft leaned on is therefore withdrawn, and the value baseline mechanism is ranked below the phase synchronisation in consequence.

The withdrawal yields the inference that decides the investigation. What a swap changes is the weights over a fixed atom set rather than the support, the displacement of the estimand is bounded by the total variation distance between consecutive populations times the diameter of the per-design gradient set, and that distance provably vanishes as covariance matrix adaptation concentrates. Since the record establishes that the distribution did collapse to a single design while the reward bands stayed wide, the estimand motion cannot be the cause of the persistent variance, and the cause must be a mechanism driven by the cadence rather than by design diversity. The phase synchronisation is exactly such a mechanism and would operate at full strength on a population of two hundred and fifty six identical robots.

One further coupling was found in the reset path and is recorded at section 7.7 of that document. Curriculum terms are computed as the first statement of `_reset_idx` (`manager_based_rl_env.py:356`), ahead of the reward manager reset at `:375`, so a curriculum term reads the episode sums of the environments being reset. The angular command curriculum's interval is `250 * 24`, which is six thousand control steps, and the morphology cadence of two hundred and forty iterations at twenty five steps is also six thousand. The two are equal, so that gate can fire only at swap boundaries, where the bare reset presents it with truncated episode sums for all four thousand and ninety six environments. The linear command and push curricula are affected on every sixth and eighth swap by the same mechanism. The terrain suppression flag does not extend to any of them and on this reading should.

Also recorded, that a population of two hundred and fifty six over a lattice of roughly eighty means covariance matrix adaptation is asked to rank duplicate individuals whose robots are identical, so their measured fitness differs only by sampling noise and that noise enters the recombination as though it carried information. This is a distinct defect from the spread collapse `cmaes.md` diagnoses and it argues for removing the rounding, which the investigation plan already recommended and which remains outstanding.

Nothing was changed in the sources. The document specifies three measurements over logs already written, a swap-aligned average of a reward channel, the same average over the explained variance, and a comparison of the ripple amplitude before and after the design distribution collapses, the last of which discriminates the two candidate mechanisms directly.

## Documentation reorganisation and codebase audit (2026-07-30)

The markdown corpus of the workspace was reorganised on this date. Every context document was moved into `/ws/context` and every plan document into `/ws/plans`, each directory receiving a `README.md` that registers its documents with a summary, a revision date, and a currency or implementation status, so that a reader may determine what to read without opening anything speculatively. The two documents describing the project as a whole, `ARCHITECTURE.md` and `CO_OPTIMISATION.md`, remain at the workspace root. Two plans were renamed for clarity, `TODO.md` became `plans/COPT_INVESTIGATION_PLAN.md` and `TODO_himloco.md` became `plans/HIMLOCO_INTEGRATION_PLAN.md`, and every inbound reference to either was updated. All cross references were rewritten to the new locations and validated, every markdown link target in the corpus now resolves.

The reorganisation was accompanied by an audit of every plan against the live sources, which established the following facts.

The three work items of `../plans/COPT_INVESTIGATION_PLAN.md` are implemented but for one. The complete checkpoint of its section 2 exists, `CoptOnPolicyRunner` overrides `save` and `load` and delegates environment state to `capture_env_state` and `restore_env_state` in `co_optimisation/utils/env_state.py`, and the design generator state is persisted through `save_state` and `load_state`. The CMA-ES initialisation of its section 3 has been applied, `train.py` now imports `GrowingDesignDistCMAESDesignGenerator` correctly, passes `sigma0=0.25`, and widens both parameter ranges to the interval 0.75 to 1.25. All three variance remedies of its section 4 have landed, the runner discards the truncated returns at a morphology update rather than extending the reward window with them, `terrain_levels_vel_delayed` in `mdp/curriculums.py` skips its promote and demote decision while a morphology reset is in progress, and `CoptPPO.compute_returns_design_wise` normalises advantages within each individual's block of environments with global normalisation disabled. The outstanding item is the two decimal rounding of link extents at `usd_generator.py:376`, which the plan recommended removing and which survives, so the design resolution floor of roughly one centimetre remains in force.

The evolutionary schedule has been retuned beyond anything the plan proposed. The live configuration in `train.py` is 256 individuals, an update interval of 240 iterations, and a random design phase of 12000 iterations, against the 64 individuals, 120 iterations, and 8000 iterations that every earlier document records. Any document quoting the older triple describes a superseded configuration, including `task_plots.md` and `cmaes.md`, which are correct as records of the runs they analyse.

A latent defect was found in the training entry point. The guard at `train.py:196` admits only the policy types COPT and COPT-LEARNED, yet the branch at `train.py:225` selects the classes `CoptLearnedModelV2ActorCritic` and `CoptLearnedModelV2PPO` for the policy type COPT-LEARNED-2, so that branch can never execute. Both classes exist, at `copt_actor_critic.py:266` and `copt_ppo.py:594`, and the second variant differs from the first in deriving its estimator input from the flattened observation history alone rather than from the privileged observations. The mode is nevertheless exposed to the user, `djinn start train copt-learned-2` passes `--policy-type COPT-LEARNED-2`, which falls through to the plain runner construction and will not produce a co-optimisation run. The remedy is to widen the tuple in the guard, and the variant wants a design record of its own since no document describes it.

An uncommitted mitigation for the NaN crashes was found in `mdp/observations.py`. Five of the Isaac Lab built in policy observation functions, `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, and `joint_vel_rel`, are shadowed by wrappers that pass the result through `torch.nan_to_num`, converting a NaN arising from a physics blow up into a neutral zero before it can reach the policy network and drive the action distribution standard deviation negative. The shadowing works by import precedence, the module is imported last by the package initialiser so its names displace the wildcard imported originals. This is the first recorded countermeasure against the NaN failure that `ARCHITECTURE.md` notes for the Berkeley environment, and it is a silent global override that any future reader of the observation configuration must know about, since the configuration names the terms without revealing which implementation binds.

Three configuration values differ from what the documents record. The working tree carries `min_feet_distance` of 0.25 in `brs_base_env_cfg.py`, raised from the committed 0.21, which corrects the figure of 0.30 asserted in `brs_gait.md` and falls short of the 0.32 that the same document's nineteenth pass proposes. The companion proposal of that pass has been adopted, `pen_ang_vel_xy` stands at a weight of minus 5. The SD_BRS1 runner configuration has been raised from 15000 to 30000 iterations, uncommitted.

The task registry has grown well beyond what `ARCHITECTURE.md` recorded, from the 22 registrations it listed to 47 in `robots/__init__.py`, the additions being the SD_BRS1 family in flat, simplified flat, rough, and hybrid internal model variants, the hybrid internal model variants of the SoleFoot tasks, and the URDF spawned variants. The architecture document has been updated accordingly.

Finally, the mjlab clone that `mjlab.md` cites at `/ws/mjlab` had been removed from the workspace, so its line citations could not be re-verified at the time of this audit. The clone was restored on 2026-07-30 and every citation has since been checked, the record of that verification being section 15 of `mjlab.md`. The findings stand, the line numbers have drifted, and the restored tree is version 1.5.3 at commit `15ebce88`.
