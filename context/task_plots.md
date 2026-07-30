# task_plots.md — Agent E findings (task/env/agent config + plot characterisation)

All file:line citations are absolute. Repo root for the project is
`/ws/tron1-rl-isaaclab-cozum`.

## 0. The COPT task chain (what the run actually used)

- `djinn start train copt` selects task `Isaac-Limx-SF-Copt-Rough-v0` with
  `policy_type=COPT` — `/ws/djinn:131-133`. Play uses `Isaac-Limx-SF-Copt-Rough-Play-v0`
  (`/ws/djinn:168-170`). The copt command passes NO `--num_envs` override and
  `--save_interval 1000` on the CLI (`/ws/djinn:137-143`).
- Registration of `Isaac-Limx-SF-Copt-Rough-v0`
  (`.../tasks/locomotion/robots/__init__.py:373-379`):
  - `env_cfg_entry_point = limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg`
  - `rsl_rl_cfg_entry_point = limx_sf_copt_runner_cfg = SFCoptPPORunnerCfg()`
    (`__init__.py:8, 30`).
- Env cfg inheritance:
  `SFCoptBlindRoughEnvCfg` (`robots/limx_solefoot_env_cfg.py:413-418`,
  sets terrain_type="generator", BERKELEY_MIMIC_TERRAINS_CFG)
  → `SFCoptBaseEnvCfg` (`:141-166`, binds `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG`,
  `replicate_physics=False`, base_Link mass/contact)
  → `SFCoptEnvCfg` (`cfg/SF/limx_base_env_cfg.py:1370-1398`, uses
  `CoptObservationsCfg`).
- Agent cfg: `SFCoptPPORunnerCfg(SF_TRON1AFlatPPORunnerCfg)`
  (`agents/limx_rsl_rl_ppo_cfg.py:131-133`): only overrides
  `experiment_name="sf_copt"`, `max_iterations=30000`. Everything else is
  inherited from `SF_TRON1AFlatPPORunnerCfg` (`:92-126`).
- `train.py` COPT branch (`scripts/rsl_rl/train.py:194-234`) sets
  `agent_cfg.policy.class_name="CoptActorCritic"`, builds
  `GrowingDesignDistCMAESDesignGenerator`, and instantiates `CoptOnPolicyRunner`.
  It does NOT override `num_steps_per_env` or `encoder`, so the inherited
  values apply. `run_name` is unset → default "" (matches the un-suffixed log
  dir `sf_copt/2026-06-26_05-26-28`). `experiment_name="sf_copt"` matches the
  CSV/plot run and `logs/rsl_rl/sf_copt`.

The plots in `/ws/context/artefacts/plots` all carry the run label `sf_copt/2026-06-26_05-26-28`,
i.e. they are this exact COPT run, taken to step 29,999 (= max_iterations 30000).

## 1. Hyperparameter inventory

### Environment (SFCoptEnvCfg / SFCoptBlindRoughEnvCfg)

| Parameter | Value | Source |
|---|---|---|
| `scene.num_envs` | **4096** (NOT reduced; PLAY variant uses 32) | `cfg/SF/limx_base_env_cfg.py:1374`; PLAY `robots/...:175` |
| `episode_length_s` | **20** s | `cfg/SF/limx_base_env_cfg.py:1388` |
| `sim.dt` | 0.005 s (200 Hz physics) | `:1391` |
| `decimation` | 4 (→ 50 Hz control) | `:1387` |
| `sim.render_interval` | 2*decimation = 8 | `:1389` |
| `seed` | 42 | `:1392` |
| terrain | generator, BERKELEY_MIMIC_TERRAINS_CFG | `robots/...:417-418` |
| robot asset | `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG` (multi-USD) | `robots/...:146` |
| `replicate_physics` | False (required for per-env distinct USD morphologies) | `robots/...:144` |

Derived: **max_episode_length = episode_length_s / (dt*decimation) = 20 / (0.005*4)
= 1000 control steps** (≈ 20 s at 50 Hz). The base `SFEnvCfg` also uses 20 s
(`:1326`), so the architecture doc's "40 s episode/time_out" is wrong.

### Agent / PPO (SFCoptPPORunnerCfg → SF_TRON1AFlatPPORunnerCfg)

| Parameter | Value | Source |
|---|---|---|
| `experiment_name` | "sf_copt" | `agents/limx_rsl_rl_ppo_cfg.py:132` |
| `run_name` | "" (default) | inherited RslRlOnPolicyRunnerCfg |
| `max_iterations` | **30000** | `:133` |
| `save_interval` | 500 (CLI passes `--save_interval 1000`) | `:95`; `/ws/djinn:139-143` |
| `num_steps_per_env` | **25** (NOT 24) | `:93` |
| `empirical_normalization` | False | `:97` |
| `policy.class_name` | set to `"CoptActorCritic"` in train.py | `train.py:220` |
| `init_noise_std` | 1.0 | `:99` |
| `actor_hidden_dims` | [512, 256, 128] | `:101` |
| `critic_hidden_dims` | [512, 256, 128] | `:102` |
| `activation` | elu | `:103` |
| algorithm `class_name` | "PPO" | `:105` |
| `value_loss_coef` | 1.0 | `:106` |
| `use_clipped_value_loss` | True | `:107` |
| `clip_param` (ε) | 0.2 | `:108` |
| `entropy_coef` | **0.005** (NOT 0.01) | `:109` |
| `num_learning_epochs` | 5 | `:110` |
| `num_mini_batches` | 4 | `:111` |
| `learning_rate` | 1.0e-3 | `:112` |
| `schedule` | adaptive | `:113` |
| `gamma` | 0.99 | `:114` |
| `lam` (GAE λ) | 0.95 | `:115` |
| `desired_kl` | 0.01 | `:116` |
| `max_grad_norm` | 1.0 | `:117` |

Effective rollout batch = num_steps_per_env * num_envs = **25 * 4096 = 102,400**
transitions per update (the doc's 24*4096=98,304 is wrong).

### Encoder / estimator

Configured `EncoderCfg` (`agents/...:120-126`): `output_detach=True`,
`num_output_dim=19`, `hidden_dims=[128, 64, 16]`, `activation="elu"`,
`orthogonal_init=False`.

NOTE: `CoptActorCritic` ignores `num_output_dim`. It reads
`enc_hidden_dims = encoder_cfg["hidden_dims"] = [128,64,16]`, uses the LAST
element (16) as the latent size and `[128,64]` as the estimator hidden layers
(`modules/copt_actor_critic.py:62, 97-105`). So the estimator is
`MLP(privilegedObs_dim → 16, hidden=[128,64])` and the actor input is
`policy_obs + 16` (`:67, 131-132`). The effective design-latent dimension fed to
the actor is **16**, not 19.

### COPT / CMA-ES knobs (from train.py:200-227)

`_num_individuals=64`, `ea_update_interval=120`, `ea_late_start=8000`,
params=[`thigh_length_scale`,`shank_length_scale`], `sigma0=0.75`, `seed=42`,
`late_start=True`, `late_start_it=8000/120≈66`,
`max_cma_iter=(30000-8000)/120≈183`, `randomise_before_late_start=True`.
Morph-update period in control steps = `ea_update_interval * num_steps_per_env`
= **120 * 25 = 3000 control steps** (the train.py comment "50*24=1200" is stale
on both factors). 3000 > max_episode_length 1000, so ~3 full episodes fit
between morph updates and episodes can terminate normally between resets.

## 2. num_envs vs rewbuffer interaction (training-variance core)

`num_envs = 4096` is confirmed for the COPT train task (only PLAY drops to 32).
Since 4096 >> 100 (the `rewbuffer`/`lenbuffer` deque maxlen=100, per CONTEXT and
Agent A/C), a single morph-update `extend()` of all-env partial `cur_reward_sum`
keeps only the LAST 100 of the 4096 appended values. The entire 100-wide reward
window is therefore overwritten in one shot by **truncated, not-yet-terminated**
returns from 100 envs every 120 iterations. Logged `Train/mean_reward` and
`mean_episode_length` at those iterations are dominated by partial returns rather
than completed-episode statistics. With 4096 envs the overwrite is total
(not a partial blend), which is exactly the buffering artefact of hypothesis (a).

## 3. Observation groups — morphology visibility (DECISIVE)

COPT uses `CoptObservationsCfg` (`cfg/SF/limx_base_env_cfg.py:360-571`) with FOUR
groups: `policy`, `critic`, `commands`, `privilegedObs` (`:567-570`).

| Term | policy (`PolicyCfg` :364) | privilegedObs (`PrivligedObsCfg` :424) | critic (`CriticCfg` :460) |
|---|---|---|---|
| base_lin_vel / ang_vel / proj_gravity | yes (noisy) | – | yes (no noise) |
| joint_pos / joint_vel | yes (noisy) | – | yes |
| last_action, velocity_commands | yes | – | yes |
| **link_lengths (design param)** | **NO** | **YES (:427-445)** | **YES (:534-552)** |
| **robot_mass** | NO | **YES (:446)** | **YES (:515)** |
| heights (height scan) | NO | YES (:447) | YES (:529) |
| **robot_inertia** | NO | NO | **YES (:516)** |
| robot_joint_torque / acc | NO | NO | yes (:509-510) |
| robot_joint_pos / stiffness / damping | NO | NO | yes (:517-519) |
| robot_pos / vel | NO | NO | yes (:520-521) |
| robot_material_properties | NO | NO | yes (:522) |
| feet_lin_vel / feet_contact_force | NO | NO | yes (:511-528) |
| group flags | corruption=True, history_length=10, flatten=True (:416-421) | corruption=True, single-step (no history) (:453-457) | corruption=False, history_length=10, flatten=True (:554-559) |

`link_lengths` is computed per-env from body-frame world positions
(`mdp/observations.py:229-269`); it is the exact per-morphology quantity CMA-ES
optimises (thigh/shank lengths) and is cached, invalidated on reset.

ANSWER to the decisive question: **the morphology/design is visible to BOTH the
critic (directly, as `link_lengths`, `robot_mass`, `robot_inertia`) and the actor
(indirectly, via the 16-dim `estimator(privilegedObs)` latent).** The
`CoptActorCritic` routing (`modules/copt_actor_critic.py:127-152`): the actor
input is `concat(actor_obs(policy), estimator(privilegedObs))`, and the critic is
the inherited rsl_rl `ActorCritic` critic operating on the `critic` group, which
contains `link_lengths` outright. So the critic CAN in principle value different
morphologies — there is NO structural observability gap. Hypothesis (b) must
therefore be re-framed as a learning/capacity/credit-assignment issue (can PPO
actually fit a value function that conditions on link length, given the moving
curriculum and the buffering artefact) rather than "the critic cannot see the
design." The policy network proper does not see the design directly, only the
detached morphology latent.

## 4. Curriculum (CurriculumCfg, cfg/SF/limx_base_env_cfg.py:1181-1262)

All terms gate on `env.common_step_counter` (control steps); iterations ≈
`common_step_counter / num_steps_per_env(25)`. The cfg author wrote thresholds as
`N*24` assuming 24 steps/iter, so real iteration thresholds are slightly earlier.

| Term | func | starting_step | ≈ start iter (÷25) | params |
|---|---|---|---|---|
| terrain_levels | terrain_levels_vel_delayed | 2000*24=48000 | ≈1920 | advances/retreats terrain tiles on tracking perf |
| modify_push_force | modify_push_force | 6000*24=144000 | ≈5760 | max_velocity (3.0,3.0), interval 400*24, inc 1.1 / dec 0.8, min_vel 0.2 |
| modify_command_velocity_lin_x | modify_command_velocity_x | 6000*24=144000 | ≈5760 | max (-1.5,1.5), interval 300*24, update_rate 0.015, thr 0.6 |
| modify_command_velocity_lin_y | modify_command_velocity_y | 6000*24=144000 | ≈5760 | max (-1,1), interval 300*24, rate 0.015, thr 0.4 |
| modify_command_velocity_ang_z | modify_command_velocity_angular | 6000*24=144000 | ≈5760 | max (-1.35,1.35), interval 250*24, rate 0.05, thr 0.4 |
| modify_linear_tracking_reward_std | reduce_tracking_rewards_std | 900*24=21600 | ≈864 | interval 2000*24, rate 0.95, thr 0.67, min_std 0.09 |
| modify_angular_tracking_reward_std | reduce_tracking_rewards_std | 0 | 0 | interval 2000*24, rate 0.975, thr 0.5, min_std 0.09 |

RELATION TO THE 8000-iter RANDOM PHASE: every curriculum term starts BEFORE
iteration 8000 (latest ≈ iter 5760). So terrain difficulty, push force,
command-velocity range expansion and tracking-reward-std sharpening all advance
CONCURRENTLY with random-design sampling (and with the growing design
distribution). The curriculum is a non-stationary moving target throughout, and
crucially is NOT reset/persisted by the morph update (relevant to objectives 1
and 2). Note ../ARCHITECTURE.md's curriculum description (push from iter 1500 every
200; lin_x expands to (-1.5, 3.0)) does NOT match this CurriculumCfg.

## 5. Per-plot variance characterisation (all single run sf_copt, no baseline overlay)

Every TensorBoard panel shows ONE green trace (raw, unsmoothed) for
`sf_copt/2026-06-26_05-26-28`, x-axis to step 29,999. There is NO COPT-vs-baseline
overlay; the visible "band" is the high-frequency per-iteration oscillation of a
single curve. The plot resolution (≈30k points) is far too dense to resolve
individual 120-iteration-periodic spikes; per-iteration noise dominates and no
clean 120-iter periodic spike train is visually separable.

### Reward component panels (Episode_Reward/*) — reward1..7
All are per-component episodic reward terms, not the aggregate Train/mean_reward.

- reward1: `feet_air_time` (final -0.0148), `feet_slide` (-0.2098),
  `keep_balance` (0.0478). Brief early transient then a WIDE noisy band that
  stays wide to the end; no narrowing.
- reward2: `pen_action_rate` (-0.1178), `pen_action_smoothness` (-0.9756),
  `pen_ang_vel_xy` (-0.1126). Very spiky, constant-to-slightly-widening band.
- reward3: `pen_base_height` (-0.1432), `pen_feet_distance` (-0.0145, panel
  truncated at step 27,240), `pen_feet_regulation` (-0.5376, drifts more negative
  with growing spread late).
- reward4: `pen_flat_orientation` (-0.0928), `pen_joint_accel` (-0.1597),
  `pen_joint_pos_limits` (-0.076). Wide noisy bands throughout.
- reward5: `pen_joint_torque` (-0.3005), `pen_joint_vel_l2` (-0.0028),
  `pen_lin_vel_z` (-1.9641). Noisy, no shrink.
- reward6: `pen_undesired_contacts` (-1.5921), `rew_ang_vel_z` (4.5104, ranges
  0→4.5 every-iteration), `rew_keep_ankle_pitch_zero_in_air` (0.6064).
- reward7: `rew_lin_vel_xy` (17.7028, oscillates ~0→18 across the whole run) and
  `rew_no_fly` (0.853). The primary task reward `rew_lin_vel_xy` rises early then
  stays in a very wide oscillating band that does NOT contract — the headline
  evidence that training variance does not shrink even after CMA-ES converges
  (designs are identical by iter ~29999 per CONTEXT objective 3).

VARIANCE VERDICT (rewards): uniformly spiky/noisy; the band is constant or
slightly WIDENS over the run; it never shrinks. This matches objective 1's
description and is the central empirical fact.

### Loss panels — loss1, loss2
- loss1 `Loss/entropy`: the only smooth trace. Declines ~6.2 → ~4.7 with a
  mid-run bump (∼iter 8-12k). Low variance (it tracks the slowly varying log_std).
- loss1 `Loss/learning_rate`: adaptive LR thrashing between ~0 and ~2e-4 every
  iteration — extremely spiky, reflecting the adaptive-KL schedule reacting to
  noisy KL. Final value shown 0.
- loss1 `Loss/surrogate`: noisy around -0.005 with large symmetric spikes; band
  roughly constant. Final 0.0072.
- loss2 `Loss/value_function`: LARGE and very spiky, mostly 2–10 with frequent
  spikes hitting the 10 axis cap; final 17.5484. The band arguably WIDENS later.
  Consistent with the critic struggling to fit value under the moving curriculum
  and per-morph resets — but note this is a fitting/credit-assignment difficulty,
  not an observability gap (the critic does see link_lengths, §3).

### Policy/mean_noise_std — noise.png
Smooth single curve, declines ~0.55 → 0.46 with a hump near mid-run (peak ~0.54).
Already well below init_noise_std=1.0. Low variance; gentle overall decrease, so
the policy is becoming somewhat more deterministic but never collapses.

### Curriculum panels — curriculum1..3 (Curriculum/*)
Monotone/quasi-monotone staircases, confirming curriculum keeps moving to the
very end:
- modify_angular_tracking_reward_std: 0.4 → 0.2806 (sharpening).
- modify_command_velocity_ang: flat ~0.7 then step up to 1.35 plateau (saturates).
- modify_command_velocity_lin_x: 1.0 → 1.375 (still rising at end).
- modify_command_velocity_lin_y: 0.5 → 0.965 (still rising at end).
- modify_linear_tracking_reward_std: 0.4 → 0.1951 (sharpening).
- modify_push_force: climbs from 0.5, very oscillatory (adaptive up/down) peaking
  ~1.3, ends 0.9048.
- terrain_levels: 0 (flat until start) → 2.1278, a NOISY upward band still rising
  at iter 30000 (terrain difficulty not saturated).
Takeaway: command-velocity range expansion, terrain level, push force and
tracking-std sharpening are all live non-stationarities co-occurring with the
morphology variation; they inject reward variance independent of morphology and
confound attribution to morph resets.

### Termination panels — termination.png (Episode_Termination/*)
- base_contact (fall fraction): high early transient (>0.6) → drops to ~0.12 →
  stays in a noisy 0.12–0.22 band; final 0.1192.
- time_out (survival fraction): rises to ~0.85–0.88 plateau, noisy; final 0.8808.
base_contact + time_out ≈ 1.0 (consistent). ~12% of episodes still end in falls
at convergence, and the fall-rate band stays noisy (does not tighten), again
consistent with persistent variance.

## 6. Discrepancies with ARCHITECTURE.md

1. `num_steps_per_env` is **25** for SF (SF_TRON1AFlatPPORunnerCfg:93), not 24;
   batch is 25*4096=102,400 not 98,304.
2. SF `entropy_coef` is **0.005** (`:109`), not 0.01; the doc's PPO table claims
   0.01 for SF_TRON1AFlatPPORunnerCfg and 0.005 only for Berkeley.
3. Encoder `hidden_dims` is **[128, 64, 16]** (`:122`), not [256,128,64,16]; and
   `CoptActorCritic` uses hidden_dims[-1]=16 as the latent size (ignores
   num_output_dim=19), so the design latent is 16-dim.
4. `episode_length_s`/time_out is **20 s** (`:1388`, base `:1326`), not 40 s.
5. COPT observations: the doc describes only `policy`+`critic`; COPT actually uses
   `CoptObservationsCfg` with `policy`, `critic`, `commands`, `privilegedObs`, and
   policy/critic carry a 10-step flattened history. The doc never mentions
   `link_lengths`/`privilegedObs`. For COPT the proprio encoder/estimator IS used
   (the doc says the encoder is unused for non-HIM).
6. Curriculum: the doc's parameters (push from iter 1500 every 200; lin_x to
   (-1.5, 3.0)) do not match CurriculumCfg (push start ≈iter5760, lin_x max
   (-1.5, 1.5), starts ≈iter5760, terrain start ≈iter1920).
7. djinn `him` task in the doc is `Isaac-Limx-SF-Berkeley-HIM-v0`; the actual
   djinn uses `Isaac-Limx-SF-HIM-Identified-Blind-Rough-v0` (minor, not COPT).
</content>
</invoke>
