# Toward a Natural Gait for the SD_BRS1 Biped

> Status, verified against the live sources on 2026-07-30. Partially implemented. Phase A, the swing phase knee flexion reward of section 5.1, is marked superseded within this document, having failed in run 2026-07-23_11-31-57 because its Gaussian kernel sat so far from the operating point that its gradient was of order ten to the minus six. Phase A2 of section 5.2, which moves the nominal pose to the derived crouch and corrects every term referencing it, is implemented, and run 2026-07-27_07-19-39 walks with alternating feet and bent knees on that pose. Phase B, restoring an upright torso, and Phase C, smoothing the torque and joint trajectories, are outstanding. The two residual defects observed after Phase A2, a narrow stance and a hip roll oscillation, are analysed in the nineteenth pass of `../context/brs_gait.md` rather than here. See [README.md](README.md) for the full register.

## Investigation and Plan of Action for Knee Flexion, Upright Posture, and Actuation Smoothness

This document analyses the evaluation of the first successful SD_BRS1 walking policy, run `2026-07-22_11-36-53`, grounds three observed gait defects in the run's own logged data and configuration, surveys the literature that treats each defect, and sets out a phased plan of action. It has since been extended with the outcome of the first phase of that plan, trained as run `2026-07-23_11-31-57`, whose reward budget and post-mortem occupy sections 2.5 and 2.6 and whose failure has reshaped the remainder of the programme. The document is therefore a running record, each phase adding its measured result and revising what follows.

---

## 1. Introduction

The SD_BRS1 work stream has passed, over successive runs, from an asset defect through a reward defect to a policy that finally walks, a history recorded in full in `/ws/context/brs_gait.md`. The run `2026-07-22_11-36-53` is the first whose evaluation exhibits a coordinated alternating gait with faithful command tracking, and it is therefore the correct baseline from which to pursue naturalness rather than mere locomotion. The present document dissects that run across three evidentiary channels, the rendered evaluation video, the numpy dump of per step kinematic and dynamic quantities written by `scripts/rsl_rl/play.py`, and the eight diagnostic plots under `/ws/context/artefacts/plots_latest_play`, and it correlates the observed behaviour with the reward and actuation configuration recorded in the run's own `params/env.yaml`, which is treated throughout as the authoritative statement of what actually trained.

The dump spans 3001 control steps at a control period of 0.01 s, thirty seconds of playback, over thirty two parallel environments, with the plotted environment being index six. The robot masses to 59.85 kg, a body weight of 587 N, distributed over thirteen links whose individual masses the dump also records. Command tracking is already good, the measured forward velocity follows the commanded step from zero to roughly 0.65 m/s with a root mean square error of 0.08 m/s, the feet alternate cleanly in the contact record, and the swing foot clears the ground to about 0.10 m against a target of 0.08 m. The policy has learned to walk. What it has not learned is to walk well, and three specific departures from a natural gait were raised for investigation.

The first is a stiff, straight legged gait in which the knees barely move. The second is a forward inclination of the torso during walking, which threatens a forward tumble on hardware and departs from the upright carriage of a natural walk. The third is a lack of smoothness in the joint torque and joint position trajectories, which present spikes and near discontinuities that impede a clean transfer from simulation to the physical machine. Each observation is first established as fact from the data and the video in section 2, where it is then traced to the reward and actuation configuration and a set of hypotheses is advanced. Sections 2.5 and 2.6 then carry the analysis forward onto the later run `2026-07-23_11-31-57`, in which the first corrective phase was trained, building a full reward budget of that run and a post-mortem of why its knee flexion reward produced no effect whatever. Section 3 grounds the hypotheses in the published literature. Section 4 draws the analysis and the literature together into a phased scope of improvement with mathematical and experimental justification. Section 5 holds the implementation plans, one per phase. Section 6 concludes and section 7 lists the bibliography.

---

## 2. Behaviour Analysis and Root Cause

This section deepens the high level account of section 1. Each of the three observations is treated in turn, the video is described, the relevant plots are read quantitatively against the numbers extracted from the dump, and the behaviour is then correlated with the concrete reward terms and actuation settings of `params/env.yaml`. Hypotheses are stated plainly and reserved for grounding against literature in section 3.

The ten actuated joints appear throughout in their articulation order, HipRoll, HipPitch, KneePitch, AnkleRoll, AnklePitch, each right then left, so that the indices zero through nine read HipRollR, HipRollL, HipPitchR, HipPitchL, KneePitchR, KneePitchL, AnkleRollR, AnkleRollL, AnklePitchR, AnklePitchL. The action is a joint position target with scale 0.25 applied about the default pose, `use_default_offset` true, with no action clipping, at decimation two over a 0.005 s physics step, giving a 100 Hz control rate.

### 2.1 The locked knee and the straight legged gait

In the video the legs read as rigid columns. Across the sampled frames the thigh and shank remain colinear through both stance and swing, the swing foot is carried forward and lifted by a visible rotation at the hip together with the ankle, and at no point does the knee fold into the flexed carriage of a natural step. The rendered posture is that of a compass, two stiff struts pivoting at the hip.

The joint position plot corroborates this without ambiguity. The two Knee Pitch panels are flat against a scale that runs to 0.4 rad, whereas the Hip Pitch and Ankle Pitch panels sweep through wide arcs. The dump makes the contrast quantitative over the steady walking window. The knee position holds a temporal standard deviation of 0.008 rad, about half of one degree, against 0.21 rad at the hip pitch, so the knee moves some twenty five times less than the joint immediately above it. The nominal range of motion of 0.24 rad reported for the knee is an artefact of rare reset transients, the working posture is a knee held essentially motionless at zero, its extension stop.

The dynamic channels reveal why this is pathological rather than merely economical. The knee carries the largest mean absolute torque of any joint on the robot, 177 Nm, nearly three times the 62 Nm borne by the hip pitch, and it peaks at the 601 Nm actuator ceiling. Yet because it barely moves it performs the least mechanical work of the major joints, a mean absolute joint power of 24 W, a 4.9 percent share of the total, while the hip pitch accounts for 22 percent on each side and drives the swing. The torque against velocity scatter for the knee collapses to a vertical stripe at zero velocity spanning zero to minus four hundred Nm, the defining signature of a joint used as a static load bearing strut rather than as a working articulation. The knee is expensive in torque, idle in motion, and the leg it belongs to never bends.

The correlation to configuration is direct, and it is subtler than a single offending term. The reward stack contains no term that rewards knee flexion, no penalty on deviation from a bent nominal posture, and no imitation of a reference trajectory, so nothing in the objective asks the knee to move. The straight knee pose is instead a deep local optimum assembled from several settings that reinforce one another. First, the kinematics leave almost no room for a height incentive to bend the knee. With a thigh of 0.44 m, a shank of 0.43 m, a hip cluster 0.181 m below the torso frame, and a sole 0.124 m below the ankle frame, the torso stands at roughly 1.175 m on fully straight legs, whereas `pen_base_height` targets only 1.14 m at weight minus fifteen. Standing tall and straight overshoots the target by a mere 0.035 m and so costs about 0.018 per step, a negligible toll, while bending each knee to the roughly 0.41 rad, some twenty three degrees, that would place the base exactly at 1.14 m buys back only that fraction, so the height objective supplies essentially no pull toward flexion. Second, the default joint pose sets every joint to zero, and the joint position action carries `use_default_offset` true, so the target is the default pose plus 0.25 times the action and the zero action fixed point of the closed loop is the straight legged stance in every episode, a bias of about 2.3 action units per knee that the policy would have to discover and hold against its own exploration noise merely to stand bent. Third, a straight knee under body weight is a near singular strut, the load line passing close to the joint axis, so the leg bears weight cheaply, which is exactly why the knee shows the largest torque yet the least motion, the 177 Nm is the active extension effort that resists buckling at a joint regulated hard against its zero stop rather than the cost of any movement. The only settings that push the other way are weak, the knee at zero sits just outside its soft limit band, whose lower edge is about 0.074 rad, so `pen_joint_pos_limits` at weight minus two levies a small standing toll, the violation being 0.0742 rad per knee and 0.1483 rad summed, which at that weight costs 0.297 per second and which a slight flexion would remove, but this is far too small to overcome the effort, stability, and control cost of holding a dynamically loaded bent knee. The hypothesis, then, is that the stiff legged gait is not caused by one term but by a straight knee attractor, the coincidence of a height target within 0.035 m of full extension, a zero action default at the knee's extension stop, and a passive load path, standing unopposed by any reward for flexion, with the foot clearance and air time objectives satisfied cheaply by hip and ankle motion that leaves the knee locked.

### 2.2 The forward lean

The video shows the trunk carried ahead of the hips through the stride, most visibly on the environments whose torsos pitch forward over near vertical stance legs. The forward inclination is modest but persistent and is consistent with a body that propels itself by allowing its mass to fall forward over a straight stance leg, the natural companion of the compass gait diagnosed above.

The dump does not log base orientation directly, so the lean is read from the video supported by two indirect quantities. The base pitch rate carries a small positive mean and a large standard deviation of 0.99 rad/s, and the base roll rate a standard deviation of 1.18 rad/s, so the trunk is not held steady but rocks substantially about both horizontal axes, and the vertical velocity of the base carries a standard deviation of 0.20 m/s, a pronounced bob. The hip pitch position is biased away from zero, a stance consistent with an inclined pelvis. The posture is not a stable upright carriage but an oscillating, forward tipped one.

The single term that opposes the lean is `pen_flat_orientation`, a squared penalty on the two horizontal components of the projected gravity vector at weight minus thirty five. This term is large, yet it is blunt. It penalises the magnitude of the tilt symmetrically and does not distinguish a forward pitch from a backward one, nor does it track any reference upright attitude, so a small steady forward lean that materially aids forward propulsion incurs only a small quadratic cost that the plus fifty velocity tracking reward readily outweighs. There is no term that constrains base pitch specifically. The hypothesis is that the forward lean is an emergent consequence of the straight legged propulsion strategy, tolerated because the only orientation penalty is a weak, sign blind, reference free quadratic, and that it will not be corrected without either a stronger or a differently shaped posture objective, and quite possibly not without first unlocking the knee so that the robot no longer needs to fall forward to move.

### 2.3 The non-smooth torque and joint trajectories

The video renders a jittery machine, the feet snapping down and the trunk twitching, rather than a fluid walker. The plots make the roughness quantitative and locate it. The joint torque traces are spiky throughout, the joint acceleration traces show isolated excursions that dwarf their surrounding signal, and the ankle position traces flatten against their limits in bang bang fashion.

The dump quantifies the spikes. The first difference of the joint torque, the change delivered within a single 0.01 s control step, reaches 506 Nm at the hip pitch and 578 Nm at the knee, with a ninety ninth percentile of roughly 160 Nm per step at the hip pitch, so the actuation commands step changes of a size comparable to the whole working torque within a single tick. The joint accelerations spike to between four thousand and nearly six thousand rad/s squared at the ankles and the hip pitch. The hip and knee torques saturate their actuator ceilings, 500 Nm at the hips and 601 Nm at the knee, and the ankle pitch position saturates its 0.454 rad travel limit for some thirteen to fourteen percent of the walking window, both sources of the flat topped, bang bang character. The consequences reach the ground, the vertical contact force peaks at 7444 N under the right foot, 12.7 times body weight, a hard heel strike impact rather than a controlled loading.

The reward stack does contain smoothness shaping, a first order `pen_action_rate` at weight minus 0.01, a second order `pen_action_smoothness` at minus 0.075, a joint acceleration penalty at minus one times ten to the minus seven, and a joint torque penalty at minus one times ten to the minus five. Every one of these is small beside the tracking and gait rewards of fifty, forty, and twenty, so the marginal cost of a torque spike that improves tracking or contact timing is negligible and the policy pays it freely. There is no penalty on the torque rate itself, only on the action rate and the acceleration, and the action is an unclipped position target that the underlying proportional derivative controller converts into whatever torque the tracking error demands up to the actuator ceiling. The hypothesis is that the roughness has two compounding sources, a smoothness objective whose weights are too small to bind against the strong tracking incentives, and a stiff position controlled actuation with no torque rate regularisation and no action clipping, which permits and even rewards abrupt command changes and hard contacts, and that the locked knee aggravates both by forcing the hip and ankle to absorb, through large rapid torques, the shocks that a compliant flexing knee would otherwise attenuate.

### 2.4 Configuration ground truth

The quantities that the three foregoing analyses rest upon are consolidated here, each verified against the run's own `params/env.yaml` and the source cited, so that the plan of section 4 can reason from confirmed numbers rather than recollection. The run instantiates `SDBRS1BlindFlatEnv2Cfg` and binds the robot to `SD_BRS1_IDENTIFIED_CFG2`, the primitive collision twin of `SD_BRS1_Assembly2.urdf`, kinematically and dynamically identical to it, so every figure below transfers unchanged.

The leg is a two link chain hanging from a hip cluster, and with every joint at its zero default the standing height is a scalar sum of vertical offsets.

```
H_straight = 0.181077 (torso to hip) + 0.44 (thigh) + 0.43 (shank) + 0.124 (ankle frame to sole)
           = 1.175077 m
H(theta)   = 0.745077 + 0.43 * cos(theta)        # knee-only crouch, sole held level
H(theta) = 1.14  ->  theta ~ 0.41 rad (23.3 deg)  # knee bend to reach the base-height target
pen_base_height at straight: 15 * (1.175077 - 1.14)^2 = 0.0184 reward/step   # negligible
```

The knee limit is `[0, 1.483]` rad, so its lower bound of zero is the mechanical extension stop and the zero default sits exactly on it, below the soft limit edge of about 0.074 rad, so `pen_joint_pos_limits` at weight minus two levies a standing toll of 0.0742 rad per knee, 0.1483 rad summed, costing 0.297 per second before the policy does anything at all. The default pose for every joint is zero, and the action applies as `target = default + 0.25 * action` with no clipping, so the zero action fixed point is the straight legged stance. A note of record, the asset file's own docstring describes a bent knee inverse kinematics squat at 1.1 m that was never applied to the numbers beside it, an abandoned design intent, and `joint_deviation_l1` is imported in the robot cfg but wired into no reward term, a dead import, so no deviation from a nominal bent pose is penalised anywhere.

The actuator is an identified proportional derivative model under a DC motor torque speed clip with a Coulomb and viscous friction term, its gains and limits as trained being the following.

| group | stiffness | damping | effort limit (Nm) | velocity limit (rad/s) | damping ratio zeta |
|---|---|---|---|---|---|
| hip roll | 150 | 45 | 500 | 40 | 0.72 |
| hip pitch | 200 | 50 | 500 | 40 | 0.69 |
| knee | 200 | 22 | 600 | 40 | 0.69 |
| ankle roll | 20 | 4 | 131 | 25 | 2.46 (overdamped) |
| ankle pitch | 50 | 4 | 262 | 25 | 1.43 (overdamped) |

The damping ratios are those derived in `/ws/plans/GAIT_STRATEGY.md` section 2.5 from the identified link inertias, and they record that the leg joints were tuned to the target near 0.7 in an earlier phase while the two ankle joints were left overdamped, a residual recommendation to bring ankle pitch damping to about 2.0 and ankle roll to about 1.1 standing unapplied. The observed torque ceilings match these limits exactly, the hips saturate at 500 Nm, the knee at 601 Nm, and the ankle roll at its 131 Nm limit, confirming that the bang bang character is in part hard actuator saturation. The joint stiffness and damping are further randomised once per environment at startup by an independent uniform factor in the range 0.9 to 1.1.

The reward terms that the plan will act upon reduce to the following formulas, with the shared or project origin of each noted because it governs how each may be edited.

```
base_height_rough_l2   (project)  sum over sensor of (root_z - target)^2,  target 1.14, weight -15
flat_orientation_l2    (isaaclab) sum(projected_gravity_b[:2]^2),  symmetric, no pitch reference, weight -35
action_rate_l2         (isaaclab) sum((a_t - a_{t-1})^2),  weight -0.01
ActionSmoothnessPenalty(project)  sum((a_t - 2 a_{t-1} + a_{t-2})^2),  first three steps zeroed, weight -0.075
joint_acc_l2           (isaaclab) sum(joint_acc^2),  weight -1e-7
joint_torques_l2       (isaaclab) sum(applied_torque^2),  weight -1e-5
```

There is no penalty anywhere on the torque rate itself, only on the action rate, the action jerk, and the joint acceleration. The terminations are a torso contact fall detector at 1 N and an absolute root height floor at 0.65 m, neither of which constrains orientation, so no term or termination defends an upright attitude beyond the symmetric `flat_orientation_l2`.

The caller map governs the backwards compatibility discipline of `/ws/CLAUDE.md`, since several of these functions are shared across the SD_BRS1 and the TRON1 SF, PF, and WF tasks and an in place edit would silently alter those other runs.

| function | callers | edit freedom |
|---|---|---|
| foot_clearance_reward_v2 | SD_BRS1 only | free to edit in place, no other caller |
| base_height_rough_l2 | SD_BRS1, TRON1 SF | optional argument, default reproduces old behaviour |
| feet_regulation | SD_BRS1, TRON1 SF, TRON1 PF | optional argument or v2 |
| GaitReward | SD_BRS1, TRON1 PF | optional argument or v2 |
| ActionSmoothnessPenalty, feet_distance | all four tasks | optional argument or v2 |
| no_fly, keep_ankle_pitch_zero_in_air, feet_slide | SD_BRS1, TRON1 SF | optional argument or v2 |
| flat_orientation_l2, action_rate_l2, joint_torques_l2, joint_acc_l2, joint_pos_limits | isaaclab library | do not edit, add a project-local term instead |

New reward terms carry no such constraint, so the natural avenue for a knee flexion incentive, a torque rate penalty, or an upright posture term is a new project local function added to `bipedal_locomotion` mdp and wired only into the SD_BRS1 config, which is also self documenting in that run's dumped `params/env.yaml`.

### 2.5 The reward budget of run 2026-07-23_11-31-57

The three foregoing analyses were drawn from the evaluation of run `2026-07-22_11-36-53`. Phase A of section 5 was then implemented and trained as run `2026-07-23_11-31-57`, twenty one thousand logged iterations over sixty thousand environment steps, and that run is the subject of this subsection and the next. Its TensorBoard record was parsed directly from the event file, there being no TensorBoard package in this container, by a minimal TFRecord and protobuf reader retained in the scratchpad, and every figure below is read from that parse.

The purpose of a reward budget is to convert a list of weights, which are not comparable to one another, into a list of realised contributions, which are. The conversion rests on a single fact about the manager, `RewardManager` computes each term as `func * weight * dt` and the logged `Episode_Reward/x` is the episode sum normalised by the maximum episode length, so the logged quantity is a reward RATE per second and dividing it by the configured weight recovers the mean value of the underlying function. That inversion is applied throughout. Its correctness is confirmed by accounting, the twenty four logged rates sum to plus 36.04 per second, which over the twenty second episode horizon predicts a return of 720.7 against a logged `Train/mean_reward` of 710.8, an agreement of better than one and a half percent, the small residue being episodes that terminate early and so never reach the full normalising horizon.

The budget divides into a positive side totalling 55.80 per second and a negative side totalling minus 19.76 per second. The composition of each is the substance of the matter.

| term | function origin | raw range | weight | observed func | observed rate /s | share of its side | impact on training |
|---|---|---|---|---|---|---|---|
| rew_lin_vel_xy | isaaclab | [0, 1] | 50 | 0.544 | +27.19 | 48.7 % pos | dominates the objective, the policy's first loyalty |
| rew_no_fly | project | [-5, 1] | 15 | 0.778 | +11.68 | 20.9 % pos | pays single support, near saturated, little headroom |
| rew_foot_clearance | project | [0, 2] | 20 | 0.445 | +8.90 | 15.9 % pos | active and unsaturated, the main swing shaper |
| rew_ang_vel_z | isaaclab | [0, 1] | 15 | 0.398 | +5.97 | 10.7 % pos | active, tracking |
| feet_air_time | isaaclab | [0, 0.4] | 12.5 | 0.153 | +1.91 | 3.4 % pos | 38 % of its ceiling, mild cadence pressure |
| rew_keep_ankle_pitch_zero_in_air | project | [0, 1] | 1.0 | 0.100 | +0.10 | 0.2 % pos | negligible |
| keep_balance | project | [0, 1] | 0.05 | 0.838 | +0.04 | 0.1 % pos | negligible survival bonus |
| rew_knee_flexion | project | [0, 2] | 10 | 1.0e-8 | +1.0e-7 | 2e-9 of pos | DEAD, see section 2.6 |
| rew_gait | project | [-2, 0] | 40 | -0.170 | -6.79 | 34.4 % neg | largest penalty, cannot ever be positive |
| pen_base_height | project | [0, 0.1225] termination-limited | -300 | 0.0140 | -4.21 | 21.3 % neg | permanent unpaid toll, target never reached |
| pen_action_smoothness | project | [0, unbounded] | -0.075 | 27.80 | -2.08 | 10.6 % neg | binding, the jerk is real |
| pen_ang_vel_xy | isaaclab | [0, unbounded] | -2.0 | 1.029 | -2.06 | 10.4 % neg | binding, torso rocking |
| pen_joint_torque | isaaclab | [0, 1.89e6] | -1e-5 | 177920 | -1.78 | 9.0 % neg | 9.4 % of saturation, moderate |
| feet_slide | project | [0, ~1.6] realistic | -5.0 | 0.181 | -0.90 | 4.6 % neg | moderate |
| pen_flat_orientation | isaaclab | [0, 1] | -50 | 0.0173 | -0.87 | 4.4 % neg | 7.6 deg RMS tilt, weak in practice |
| pen_joint_pos_limits | isaaclab | [0, 0.580] | -2.0 | 0.190 | -0.38 | 1.9 % neg | small, soft limit violations persist |
| pen_joint_torque_rate | project | [0, 7.57e6] | -1e-5 | 26070 | -0.26 | 1.3 % neg | small |
| pen_action_rate | isaaclab | [0, unbounded] | -0.01 | 13.96 | -0.14 | 0.7 % neg | small |
| pen_undesired_contacts | isaaclab | [0, 11] | -2.5 | 0.056 | -0.14 | 0.7 % neg | small |
| pen_feet_regulation | project | [0, unbounded] | -0.2 | 0.448 | -0.09 | 0.5 % neg | negligible |
| pen_lin_vel_z | isaaclab | [0, unbounded] | -0.5 | 0.048 | -0.02 | 0.1 % neg | negligible |
| pen_joint_accel | isaaclab | [0, unbounded] | -1e-7 | 231000 | -0.02 | 0.1 % neg | negligible |
| pen_feet_distance | project | [0, 0.21] realistic | -100 | 0.00006 | -0.01 | 0.03 % neg | satisfied, inactive |
| pen_joint_vel_l2 | isaaclab | [0, 12100] | -1e-5 | 30.0 | -0.0003 | 0.0 % neg | inert, 0.25 % of saturation |

Four readings of this table bear directly on the diagnosis.

The first concerns `rew_gait`, which the name and the positive weight of forty both misrepresent. Its two configured scales, `tracking_contacts_shaped_force` and `tracking_contacts_shaped_vel`, are each minus one, and the implementation branches on that sign into the complementary form, so the force component evaluates to minus one half the sum over feet of `(1 - desired_contact) * (1 - exp(-F^2 / 25))` and the velocity component to minus one half the sum over feet of `desired_contact * (1 - exp(-v^2 / 0.25))`. Both are non positive by construction, the total is bounded in minus two to zero, and the best attainable value is therefore exactly zero. The term is a pure schedule violation penalty, not a reward, and at minus 6.79 per second it is the single largest negative contribution in the entire stack, a third of all penalty. This does not make it wrong, a penalty for being off schedule is precisely what broke the earlier one legged hold, but it means the objective contains no positive incentive to step, only a punishment for stepping at the wrong time, and any future reasoning about the gait clock must price it as a penalty with a ceiling of zero.

The second concerns `pen_base_height`, whose weight was raised to minus three hundred to force a crouch. Inverting the squared kernel, the mean squared deviation of 0.0140 corresponds to a root mean square height error of 0.1185 m, so the torso sits near 1.119 m against a target of 1.0 m, and against a straight legged standing height of 1.175 m established in section 2.4 it has descended only 0.056 m. The target is not being met and the run pays a flat 4.21 per second for the failure across all twenty one thousand iterations, a permanent unpaid toll rather than a gradient the policy follows. Worse, of the 0.056 m actually surrendered, only a small part comes from the knee, since a knee flexed to the measured 0.22 rad lowers the hip by 0.010 m, the remainder arising from the fore and aft splay of the legs and the forward inclination of the trunk that walking produces anyway. A height penalty this large, confronted with a knee the policy cannot cheaply bend, is therefore satisfied preferentially by leaning and splaying, which is to say it rewards precisely the posture defect of section 2.2 rather than the knee flexion it was raised to obtain.

The third concerns the effort and smoothness penalties, which section 2.3 judged too weak. Two of them now bind meaningfully, `pen_action_smoothness` at minus 2.08 and `pen_ang_vel_xy` at minus 2.06, and `pen_joint_torque` at minus 1.78 runs at 9.4 percent of its saturation bound of 1.89 million squared Newton metres. The remainder are still inert, `pen_joint_vel_l2` at a quarter of one percent of saturation and `pen_joint_accel`, `pen_feet_distance`, and `pen_lin_vel_z` each contributing under one tenth of one percent, so those four are decorative and could be removed or raised by an order of magnitude with no risk.

The fourth, and the reason for the next subsection, concerns `rew_knee_flexion`. Its observed contribution is one part in six hundred million of the positive budget. Had it merely been mis weighted the remedy would be arithmetic, but the table shows the weight was in fact chosen correctly, since at saturation the term would supply 15.2 percent of the positive budget, a serious but not domineering share, comparable to the foot clearance reward it was designed to complement. The failure lies elsewhere, in the shape of the kernel rather than the size of the weight, and it is worth its own treatment.

### 2.6 Post-mortem of the Phase A knee flexion reward

The Phase A term of section 5.1 was trained exactly as specified, with target 1.1 rad, tolerance 0.2 rad, and weight ten, and it produced no measurable change in the gait. The video at fifty thousand steps shows the same straight columnar legs, the same compensatory hip rotation, and the same forward inclination of the trunk that the baseline showed, and several environments are visibly on the point of toppling. The reward channel explains why, and does so with unusual precision.

The logged term is never exactly zero. Across all 21161 logged iterations it ranges from 6.57e-10 to 1.05e-6 per second, against a saturation value of ten. At its best it reached one ten millionth of what it could pay. Because the term is a monotone function of the knee angle throughout this range, its magnitude can be inverted to recover the angle that produced it, which turns the dead reward into an unusually direct instrument for measuring the very quantity it failed to move. Writing the logged rate as `w * f * exp(-(q - t)^2 / s^2)` with weight `w` ten, target `t` 1.1 rad, tolerance `s` 0.2 rad, and an airborne duty `f` of about 0.8 leg seconds per second, the implied swing knee angle is

```
q = t - s * sqrt( -ln( rate / (w * f) ) )
```

which over the run yields a swing knee between 0.136 and 0.304 rad, sitting at roughly 0.22 rad, some 12.6 degrees, for essentially the whole of training. The airborne duty assumed here is not a guess, it is confirmed independently by the `no_fly` channel of the same run. That term evaluates to plus one under single support and minus five under flight, so its mean value of 0.778, together with a flight probability indistinguishable from zero, implies single support for 77.8 percent of the time and hence 0.78 airborne leg seconds per second, against the 0.8 assumed. The inference is nonetheless tail weighted, since the kernel is convex here and the mean is dominated by the largest excursions, so 0.22 rad is nearer the upper envelope than the typical value, and the true central knee angle is lower still. The trajectory of this quantity is flat. At iteration 423 it implies 0.218 rad and at iteration 21160 it implies 0.213 rad, with no trend between. Twenty one thousand iterations of training moved the swing knee not at all.

The cause is that the reward supplied no gradient at the policy's operating point. Policy gradient methods do not follow a reward, they follow its derivative along the directions the policy explores, and the derivative of the Gaussian kernel at the measured operating point of 0.22 rad is

```
dr/dq = w * exp(-(q-t)^2/s^2) * (-2 (q - t) / s^2)
      = 10 * 3.9e-9 * 8.8   =  1.7e-6  per radian
```

against, for comparison, a value of 11.1 per radian for a simple linear reward of the same weight and saturation. The implemented term therefore delivered 1.6e-7 of the gradient of the most naive alternative, which is to say none. The kernel was placed 0.88 rad from the operating point with a tolerance of 0.2 rad, so the operating point lay 4.4 tolerances into the tail, and the exponential of minus 4.4 squared is a number no optimiser can act upon. The term was not weak, it was mathematically invisible.

A second and independent obstacle compounds it, and would have limited the term even had the kernel been correctly shaped. The action is a joint position offset about the default pose, `target = default + 0.4 * action`, and the default knee is zero, its mechanical extension stop. Holding the knee at the reward's target of 1.1 rad therefore demands a sustained mean action of 2.75 units on both knees simultaneously, while the policy's own exploration standard deviation over the run is 0.742, so the target sits 3.7 exploration sigmas from the zero action fixed point. The policy would have had to discover, and then hold against both its exploration noise and the entropy bonus that resists extreme mean actions, a large persistent bias on two joints, guided by a reward whose gradient in that direction was 1.7e-6. The following table gives the sustained action each candidate knee angle costs.

| knee angle (rad) | sustained action required | exploration sigmas | note |
|---|---|---|---|
| 0.22 | 0.55 | 0.74 | the measured operating point |
| 0.41 | 1.02 | 1.38 | reaches the old 1.14 m height target |
| 0.60 | 1.50 | 2.02 | a natural swing flexion |
| 0.94 | 2.35 | 3.17 | reaches the current 1.0 m height target |
| 1.10 | 2.75 | 3.71 | the Phase A reward target |
| 1.25 | 3.12 | 4.21 | 1.0 m with the foot held flat |

Read together with section 2.5 this settles the diagnosis. The knee is not locked because the objective is indifferent to it, since Phase A added a term worth 15.2 percent of the positive budget at saturation. It is locked because two structural facts stand between the policy and that reward, a kernel too narrow and too distant to produce a usable gradient, and a zero action attractor at full extension that prices every degree of sustained flexion in exploration sigmas. Both are properties of the parameterisation rather than of the weights, and no adjustment of weights alone will remove either. The corollary, which section 4 develops, is that the flexion should be carried by the default pose, where it costs no sustained action at all, and that any residual shaping reward must be monotone or broad enough to have a gradient where the policy actually stands.

A final observation from the same run bears on the plan. The `Episode_Termination/low_height` channel rises from 0.054 early in training to 0.400 and stays there, so two episodes in five end with the torso below the 0.65 m floor rather than at the time limit. The policy is not merely walking imperfectly, it is falling regularly, and it does so while paying a permanent 4.21 per second for a base height it never reaches. A crouch that the robot cannot hold, demanded by a penalty it cannot satisfy, is a plausible contributor to that failure rate, and the interaction between the base height target, the achievable pose, and the fall rate is taken up in section 4.

### 2.7 A stale contact index in the airborne and single support gates

Investigation of the Phase A term uncovered a defect independent of its kernel, which does not explain the dead reward but does corrupt several terms that are far from dead, and which must be corrected before any swing gated shaping can be trusted.

The contact sensor stores a rolling history whose time ordering is established unambiguously in the IsaacLab source, `sensors/contact_sensor/contact_sensor.py` lines 377 and 378 roll the buffer forward by one along the history axis and then write the newest sample into index zero.

```python
self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(1, dims=1)
self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]
```

Index zero is therefore the MOST RECENT frame and index minus one the OLDEST. The BRS scene configures `history_length=4` on a sensor that refreshes every 0.005 s physics substep, so the buffer spans 20 ms and index minus one carries contact data 15 ms old, index minus two 10 ms old, between one and one and a half control steps behind the present.

Three functions in the project reward module read the wrong end of that buffer. `no_fly` at line 342 names its variable `latest_contact_forces` and then takes `[:, -1]`, `keep_ankle_pitch_zero_in_air` at line 377 takes `[:, -1]` and `[:, -2]`, and the new `knee_flexion_in_swing` at line 417 inherits the same idiom, having been written deliberately to mirror the established one. Each therefore decides whether a foot is on the ground from the two OLDEST samples in the buffer, never consulting the two freshest. That the convention is a mistake rather than an intention is settled inside the same file, since `no_contact` at line 434 reads `[:, 0, :, 2]` under a comment describing it as the latest contact force, and the two idioms cannot both be right. The functions that reduce over the whole history with a maximum, `feet_slide` at line 81 and the two clearance rewards at lines 134 and 210, are indifferent to ordering and are unaffected, as is `GaitReward`, which reads the un-historied `net_forces_w` directly.

The consequence is a phase lag rather than an inverted test. In `keep_ankle_pitch_zero_in_air` and `knee_flexion_in_swing` the two samples are combined by a logical OR, which is commutative, so mislabelling which is current and which is previous changes nothing by itself, the defect is that both samples are drawn from the stale end. The gate therefore trails the true contact state by roughly 10 to 15 ms, continuing to treat a foot as airborne for that window after touchdown and as grounded for a similar window after lift off. Under the configured 1 Hz clock a swing lasts about 0.4 s, so the misalignment is some three percent of the swing phase, a modest smearing rather than a gross error. For the knee term this merely blurs an already invisible reward. For `no_fly` it matters more, since that term supplies 11.68 per second, a fifth of the entire positive budget, and its single support test is being evaluated on contacts that are a control step and a half out of date, which at the touchdown and lift off transitions is precisely when single support is being decided. The correction is to read index zero and index one.

Both `no_fly` and `keep_ankle_pitch_zero_in_air` are shared with the TRON1 SF task by the caller map of section 2.4, so neither may be edited in place. The correction must arrive as an optional argument whose default reproduces the present stale behaviour exactly, set explicitly only in the SD_BRS1 configuration, which also records the choice in that run's dumped `params/env.yaml`. The defect left standing in the unset default is recorded here so that the TRON1 tasks may opt in deliberately.

### 2.8 The base height target lies outside the reachable pose set

Section 2.5 observed that the base height penalty is never satisfied and that its 4.21 per second is paid permanently. The reason is not reluctance on the part of the policy. The target is unreachable, and the constraint that makes it so is the ankle.

Consider the sagittal squat the term implicitly asks for, the torso held vertical and the sole held flat on the ground. Let the thigh tilt back from vertical by alpha and the shank tilt forward by beta, so the knee angle is their sum. For the robot to balance, the hip must lie above the ankle, which couples the two through `L1 sin(alpha) = L2 sin(beta)` with the thigh `L1` 0.44 m and the shank `L2` 0.43 m. Keeping the sole flat while the shank tilts forward by beta requires the ankle pitch joint to rotate by exactly beta. The torso height is then

```
H(alpha, beta) = 0.305077 + 0.44 cos(alpha) + 0.43 cos(beta)
```

where the constant is the 0.181077 m from torso frame to hip plus the 0.124 m from ankle frame to sole. The ankle pitch limit of 0.454 rad therefore places a hard floor under the reachable height, since beta may not exceed it.

| ankle beta (rad) | hip alpha (rad) | knee (rad) | torso height H (m) | status |
|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 1.1751 | full extension |
| 0.100 | 0.098 | 0.198 | 1.1708 | ok |
| 0.200 | 0.195 | 0.395 | 1.1581 | ok |
| 0.288 | 0.282 | 0.570 | 1.1400 | ok, the OLD target |
| 0.300 | 0.293 | 0.593 | 1.1371 | ok |
| 0.400 | 0.390 | 0.790 | 1.1080 | ok |
| 0.454 | 0.443 | 0.897 | 1.0891 | AT the ankle limit, the floor |
| 0.654 | 0.637 | 1.291 | 1.0000 | the CURRENT target, INFEASIBLE |

The deepest balanced flat footed squat this robot can adopt is 1.089 m. The configured target of 1.0 m would require an ankle rotation of 0.654 rad, forty four percent beyond the mechanical limit, and is therefore not merely difficult but impossible. The penalty at weight minus three hundred is asking for a pose that does not exist in the robot's configuration space.

Three consequences follow, and together they reframe the whole diagnosis.

The first is that the penalty cannot be discharged by bending the knee, so the policy discharges it by every other available means. The only remaining ways to lower the torso frame are to lift the heel and abandon the flat foot, to splay the legs fore and aft, and to pitch the trunk forward, since the torso origin sits 0.181 m above the hip and pitching by phi lowers it by `0.181 (1 - cos phi)` while also lowering the whole upper mass. The measured stance confirms exactly this substitution. The robot sits at 1.119 m, whereas a balanced squat at the measured knee angle of 0.22 rad would place it at 1.170 m, so some 0.05 m of the descent is bought not by knee flexion but by splay and trunk pitch. The forward lean of observation 2 is therefore not merely tolerated by a weak orientation penalty, as section 2.2 supposed, it is actively purchased by the base height penalty, which pays 300 per squared metre for precisely the trunk pitch that `pen_flat_orientation` resists at 50 per unit squared gravity projection. Two of the three defects are in direct conflict in the objective, and the height term is the stronger by a wide margin.

The second is that the change of regime recorded in section 5.1.1 was a change from a feasible target weakly enforced to an infeasible target strongly enforced. The earlier configuration asked for 1.14 m at weight minus fifteen, which the table shows is comfortably reachable at a knee of 0.570 rad and an ankle of 0.288 rad, well inside every limit, but priced it so cheaply that the policy ignored it. The present configuration asks for 1.0 m at minus three hundred, which cannot be reached at all. Neither regime ever placed the robot in a posture where a bent knee was both attainable and worth attaining.

The third is that this interacts with the fall rate. A policy driven hard toward an unreachable crouch will approach the ankle limit, where the ankle has no travel left with which to make the small corrective rotations that keep a biped upright, and the ankle pitch position was already measured saturating for thirteen to fourteen percent of the baseline window in section 2.3. The `low_height` termination rising to two episodes in five is consistent with a machine repeatedly driven to the edge of its own ankle authority. The remedy is not a larger weight but a target inside the reachable set, and section 4 adopts one.

### 2.9 The nominal pose, its sign convention, and a feasible crouch

Section 2.8 established which heights the robot can reach. This subsection establishes which joint angles reach them, because the corrective measure that sections 3 and 4 will adopt is a change to the nominal pose, and a nominal pose expressed with a wrong sign would flex one leg and hyperextend the other.

The sign convention is not symmetric and cannot be guessed. Reading the joint axes from `SD_BRS.urdf`, the asset actually simulated, every joint origin carries `rpy="0 0 0"`, so the leg's rotations compose as signed additions about a common lateral axis, and the signs follow from the axis declarations alone. `HipPitchR` carries axis `(0,-1,0)` while `HipPitchL` carries `(0,1,0)`, so the two are mirrored and must take opposite numeric signs for the same physical motion. `KneePitchR` and `KneePitchL` both carry `(0,1,0)`, and `AnklePitchR` and `AnklePitchL` both carry `(0,-1,0)`, so neither the knee nor the ankle is mirrored and both take the same numeric sign on the two legs. This asymmetry, only the hip mirrored, is confirmed independently by the joint limits, since `KneePitch` is `[0, 1.483]` on both legs, which would forbid flexion outright on one leg were the flexion direction negative there, while `HipPitchR` is `(-0.75, 1.25)` against `HipPitchL` at `(-1.25, 0.75)`, a mirrored pair. For a natural forward crouch the pose is therefore

```
HipPitchR = +beta      HipPitchL = -beta       (mirrored, opposite signs)
KneePitchR = KneePitchL = +k                   (same sign, both positive)
AnklePitchR = AnklePitchL = +(k - beta)        (same sign, both positive)
```

The closed chain that places the ankle directly beneath the hip, which is the condition for a balanced stance, fixes the three angles from a single parameter. With the thigh `l1` 0.44 m and the shank `l2` 0.43 m, and writing `d` for the hip to ankle distance, the height is `H = 0.305077 + d` and

```
k    = arccos( (d^2 - l1^2 - l2^2) / (2 l1 l2) )
beta = atan2( l2 sin k,  l1 + l2 cos k )
ankle = k - beta
```

Evaluating this over the candidate heights gives the reachable design space, with the soft limits at ninety percent of range, so `AnklePitch` soft is 0.4086 and `KneePitch` soft is `[0.0742, 1.4089]`.

| H (m) | knee k (rad) | hip beta (rad) | ankle (rad) | ankle margin to soft limit | verdict |
|---|---|---|---|---|---|
| 1.16 | 0.394 | 0.195 | 0.199 | 0.210 | shallow, ample margin |
| 1.15 | 0.481 | 0.238 | 0.244 | 0.165 | RECOMMENDED |
| 1.13 | 0.646 | 0.318 | 0.327 | 0.081 | deeper, margin thin |
| 1.12 | 0.716 | 0.354 | 0.362 | 0.047 | very thin |
| 1.105 | 0.807 | 0.399 | 0.409 | 0.000 | at the soft limit |
| 1.089 | 0.897 | 0.443 | 0.454 | -0.045 | at the HARD limit, the floor |
| 1.00 | 1.291 | 0.637 | 0.654 | -0.245 | INFEASIBLE |

The ankle binds first and binds decisively. The knee does not reach even its soft edge until about 0.968 m, some twelve centimetres below the height at which the ankle has already struck its hard stop, so on this robot the crouch is limited by ankle travel and never by the knee. The recommended pose is the 1.15 m row, which places the knee at 0.481 rad, comfortably inside its soft band and in the region where a swing may both flex further and extend back toward support, while leaving the ankle 0.165 rad, some forty percent of its soft range, in reserve for the balancing rotations a biped needs. The resulting nominal is

```
HipRollR   =  0.0        HipRollL   =  0.0
HipPitchR  = +0.2379     HipPitchL  = -0.2379
KneePitchR = +0.4814     KneePitchL = +0.4814
AnkleRollR =  0.0        AnkleRollL =  0.0
AnklePitchR= +0.2435     AnklePitchL= +0.2435
```

Three facts about the present configuration emerged alongside this and correct the record of section 2.4.

The first is that a crouched nominal pose has already been written once and has never taken effect. The asset file `assets/config/sd_brs1_identified_cfg.py` sets a non zero `init_state.joint_pos` at lines 118 to 130, with hip, knee, and ankle all at 0.2618 rad, but `robots/brs_solefoot_env_cfg.py` overwrites `init_state.joint_pos` with the all zero `_JOINT_POS_DEFAULTS` at lines 48, 82, and 121, after the asset configuration has been constructed. The all zero pose diagnosed in section 2.1 is therefore the operative one, and the crouch in the asset file is dead code. It is also arithmetically wrong on its own terms, since setting hip, knee, and ankle to a common 0.2618 rad does not satisfy the closed chain, it places the ankle 0.114 m forward of the hip rather than beneath it, the correct hip angle for a knee of 0.2618 being 0.1294 rad. The docstring's claim that both soles land beneath the hip does not hold. This is the same abandoned intent that section 2.4 recorded from the docstring, now located in code.

The second is that the spawn height is 1.2 m, set at line 118 of the same file, not the 1.3 m previously supposed, and that `reset_robot_base` randomises only x, y, and yaw, carrying no z key, so 1.2 m is the exact height at every reset. Against a straight legged standing height of 1.175 m this is a 25 mm drop. Against the recommended 1.15 m crouch it would become a 50 mm drop, so the spawn should move with the pose.

The third concerns the reset events, which were all written relative to the zero default and add their offsets to it. `reset_knee_pitch_joints` adds a uniform sample from `(0.0, 0.475)`, so against a new knee default of 0.4814 it would sample `[0.4814, 0.9564]`, a distribution lying entirely on the flexion side of the nominal rather than straddling it. The companion events `reset_hip_pitch_r_joint` at `(-0.25, 0.42)`, `reset_hip_pitch_l_joint` at `(-0.42, 0.25)`, and `reset_ankle_pitch_joints` at `(-0.113, 0.25)` are in the same position. All four must be recentred with the pose, or the initial state distribution will be biased away from the very nominal the change is meant to establish.

---

## 3. Literature Survey

None of the three defects is novel, and each carries an established treatment in the learned locomotion literature. This section reviews that literature problem by problem, states the pertinent formulations, and ties each back to the hypotheses advanced in section 2, so that section 4 may select among grounded options rather than improvise. The reader should keep in mind throughout that the SD_BRS1 objective already contains many of the standard terms, so the question is rarely whether a mechanism exists and usually whether the existing mechanism is shaped or weighted correctly.

### 3.1 The stiff legged gait and the induction of knee flexion

The first and most important lesson of the literature is that a straight stance knee is not a defect but the energetically rational posture, so the objective is not to abolish knee extension but to confine it to the stance phase and to induce flexion in swing. Passive dynamic walking, from McGeer onward, shows that a straight leg transmits body weight through its structure at near zero actuation cost, and learned policies given only an effort or energy objective reliably rediscover this, converging on extended stance legs because bending a loaded knee is expensive. The emergence studies of Heess and colleagues and the more recent biologically grounded results confirm that natural looking, straight kneed support and even heel to toe rollover arise from energy minimisation alone without explicit prescription. Our configuration, an effort penalty together with a base height target within 0.035 m of full extension, is therefore precisely the recipe the literature predicts will produce a straight stance leg, and the diagnosis of section 2.1 is the expected outcome rather than a surprise.

The pathology in the SD_BRS1 gait is narrower and sits in the swing phase. Human walking flexes the knee through swing to shorten the leg and clear the foot, then re extends it for support at touchdown, a pattern documented in the standard gait analysis literature. The SD_BRS1 policy instead clears its swing foot by flexing the hip and rotating the ankle while the knee stays locked, which is why section 2.1 finds the swing foot reaching a healthy 0.10 m of clearance even though the knee never moves. The corrective signal that is missing is not a general knee flexion reward, which would fight the efficient stance posture, but a swing gated one.

The reference free branch of the literature supplies exactly this. Peng, Bao, and Zhou demonstrate on the Unitree G1 a suite of human inspired reward terms that require no motion capture, among them a straight knee during stance term at a small weight, an anti phase arm and leg swing term, and a foot clearance term, achieving natural standing, walking, and running. The only member of that suite with a fully published formula is the yaw angular momentum term, `R = -(L_z)^2 - 0.4 (L_la,z - L_ra,z)^2` at weight 5.0, but the paper establishes the principle that phase gated joint posture shaping yields natural gaits without reference data, and it names the stance straight knee explicitly as a support efficiency device. The complementary term the SD_BRS1 stack lacks is the swing counterpart, a reward for knee flexion while the foot is airborne, which the existing contact clock already makes trivial to gate.

The imitation branch offers a higher fidelity alternative at the cost of reference data and additional machinery. DeepMimic tracks a reference clip through a weighted pose, velocity, end effector, and centre of mass reward with reference state initialisation and early termination, and it produces fully natural knee flexion, but it requires a labelled reference motion and a phase variable. Adversarial Motion Priors remove the phase and the explicit tracking, replacing them with a discriminator trained to separate policy transitions from an unlabelled motion dataset and a style reward of the form `r_S = max(0, 1 - 0.25 (D(s, s') - 1)^2)`, and the legged robot adaptations of Escontrela and colleagues show that this yields natural, knee flexing gaits with good transfer on real quadrupeds and a lower cost of transport than hand designed style rewards. For the SD_BRS1, whose stack already carries a contact phase clock, the swing gated reward is the lowest infrastructure route and an adversarial style term the heavier but more thoroughly natural one.

### 3.2 Torso orientation and the forward lean

The projected gravity penalty in use, `flat_orientation_l2`, is the field standard upright term, carried by the legged_gym reward set of Rudin and colleagues and by the IsaacLab locomotion tasks, and section 2.4 confirms it is a symmetric quadratic bowl minimised at the flat attitude. Its very symmetry is the limitation, it defends flatness in the abstract but names no particular carriage, so a small steady forward pitch that improves velocity tracking pays for itself and persists, which is the finding of section 2.2.

The weight is not obviously the problem. The reward design study of van Marum and colleagues recommends an orientation to linear tracking ratio near two thirds, and the IsaacLab G1 configuration sets its flat orientation penalty at parity with linear tracking, and scaling either rule to the SD_BRS1 tracking weight of 50 brackets the orientation penalty between roughly minus thirty three and minus fifty, so the configured minus thirty five sits at the conservative end of the recommended band rather than below it. This argues that the residual lean is a matter of shaping rather than magnitude, since a symmetric penalty cannot express a preference for one pitch sign over another.

Two directed instruments address that. The first is a reference posture term that tracks a specific base pitch or gravity direction rather than penalising deviation from flat symmetrically, of the kind the gait conditioned paper adds through its waist pitch, roll, and yaw deviation rewards toward an upright reference. The second is an orientation gated termination, the `bad_orientation` term available in IsaacLab and recorded in the project strategy at a limit near 0.8 rad but never wired in, which removes the tumbling and heavily leaned states from the return rather than merely taxing them. The literature on knee flexion also suggests a coupling worth exploiting, since straight leg propulsion advances the centre of mass by a controlled forward fall over an extended stance leg, so a portion of the forward lean is expected to relax once the knee unlocks and the robot can push off rather than topple forward, which bears on the ordering of the plan in section 4.

### 3.3 Actuation smoothness for sim to real

The transfer literature treats non smooth, high frequency actuation as a principal obstacle to deployment, since it drives oscillation, power draw, and mechanical wear and exploits simulator inaccuracies that the hardware does not share. The most direct and best evidenced instrument is the Conditioning for Action Policy Smoothness regularisation of Mysore and colleagues, which augments the policy objective with two terms, a temporal smoothness loss `L_T = || pi(s_t) - pi(s_{t+1}) ||_2` that penalises change in the action between consecutive states, and a spatial smoothness loss `L_S = || pi(s_t) - pi(s_bar) ||_2` with `s_bar` drawn from a Gaussian neighbourhood `N(s_t, sigma)` that penalises sensitivity of the action to small state perturbations, combined as `J_CAPS = J_RL - lambda_T L_T - lambda_S L_S`. On a real quadrotor this produced a ninety six percent improvement in a smoothness metric and an almost eighty percent reduction in power. The distinguishing virtue over a reward penalty is that the regularisation acts on the state to action map itself, damping high frequency response directly rather than paying for it after the fact, and it lives in the policy update rather than the environment.

The reward side of the literature supplies the terms the SD_BRS1 stack already carries, the first order action rate penalty of the legged_gym set, the torque and acceleration penalties, and the second order action smoothness penalty of the Cassie lineage, and section 2.4 shows every one of these is weighted two to seven orders of magnitude below the tracking and gait rewards, so raising them and adding the one term that is absent, a penalty on the torque rate rather than only the action rate and the joint acceleration, are direct and low risk levers. The actuator fidelity literature is the deeper remedy, since Hwangbo and colleagues show that learning the true torque response of the drive with an actuator network closes much of the transfer gap by refusing the policy the fiction of an instantaneous ideal actuator, and the SD_BRS1 already advances part way toward this through its identified proportional derivative model under a DC motor clip and a friction term. The periodic reward composition of Siekmann and colleagues, which is the basis of the SD_BRS1 gait clock, is itself demonstrated to yield smooth, hardware transferable bipedal gaits, the clock regularising contact timing. Finally the twelve times body weight heel strike measured in section 2.3 is a recognised hazard, and the literature couples foot clearance and touchdown velocity shaping with the smoothness terms to soften contact, a coupling that the SD_BRS1 clearance reward is already positioned to carry.

### 3.4 Synthesis and trade-offs

The evidence favours a staged, mostly reference free programme. The knee is addressed first and most cheaply by a swing gated flexion reward that leaves the efficient stance posture intact, with an adversarial motion prior held in reserve as the higher fidelity option should a reference dataset become available. The lean is addressed by a directed upright term and an orientation termination rather than by inflating the already adequately weighted symmetric penalty, and it is expected to ease once the knee unlocks. The roughness is addressed by the policy space smoothness regularisation of the Conditioning for Action Policy Smoothness method, reinforced by raising the existing penalty weights, adding a torque rate penalty, correcting the overdamped ankle, and clipping the action. Three trade offs recur in the sources and must be respected. Smoothness regularisation set too aggressively suppresses the rapid corrections that balance demands, so it is calibrated against tracking rather than maximised. Imitation rewards buy naturalness with the cost of collecting and curating reference motion. And the three defects are coupled, the straight knee both provokes the forward lean and forces the hips and ankles to absorb through large fast torques the shocks a flexing knee would attenuate, so the knee is the correct first intervention and the others are measured after it.

### 3.5 Why a narrow kernel placed far from the operating point cannot teach

The failure documented in section 2.6 is a known one, and the literature is unambiguous about how kernel widths are to be chosen. The convention that legged_gym established, and that IsaacLab inherits, sets `tracking_sigma` to 0.25 in squared error units, so the exponential `exp(-e^2 / sigma)` carries an effective one sigma tolerance near 0.5 m/s on the velocity error. That figure was not chosen from the command range but from the tracking error the policy actually produces, which is the principle our term violated.

The imitation literature makes the same point quantitatively and with more variety. The animal imitation work of Peng and colleagues weights five exponential kernels whose interior coefficients span more than two orders of magnitude, 0.1 for joint velocities, 5 for joint angles, 20 and 10 for root pose, and 40 for end effector Cartesian error, each chosen because the natural error scale of that quantity differs. A single width reused across quantities, or across a target and an operating point that lie far apart, is precisely what those authors avoid. Our term placed a 0.2 rad tolerance around a target 0.88 rad from the operating point, giving an exponent near thirty, and the resulting reward of order ten to the minus fourteen is the arithmetic consequence.

Two structural alternatives to a peaked kernel appear in the literature and both are relevant. The first is the constrained formulation of Kim and colleagues, which replaces several kernel shaped reward terms with constraint costs that are zero inside a desired range and grow outside it, drawn from the joint limits in the model description, reporting that this collapses ten or more hand tuned reward terms to effectively one coefficient and transfers across morphologies. A cost of that form cannot vanish the way a distant Gaussian does, because outside the desired band it grows rather than decays. The second is potential based shaping in the sense of Ng, Harada, and Russell, whose theorem states that a shaping term of the form `F(s, a, s') = gamma Phi(s') - Phi(s)` leaves the optimal policy unchanged, and which Wiewiora sharpens for episodic tasks by requiring the potential to vanish at terminal states, a caveat that matters here because two episodes in five terminate early. The benchmark of Jeon and colleagues on humanoid locomotion tempers the expectation, finding that potential based shaping brings only marginal gains in convergence speed but is markedly more robust to the scaling of its terms, which is to say it addresses exactly the tuning fragility that defeated our reward, though not by accelerating learning.

Finally, curriculum offers a third route. The general argument of Bengio and colleagues, that a hard objective is more easily reached by optimising a smoothed or easier version first and tightening it, applies directly to a reward target, and the termination curriculum of Babadi and colleagues supplies a concrete numerical instance in a closely related mechanism, scheduling a threshold from 0.75 down to 0.5 across training so the constraint is strict early and relaxes as competence grows. Annealing a knee target from the current operating point toward the desired one would keep the gradient alive throughout, at the cost of a schedule to design.

### 3.6 How working systems actually obtain a bent knee

The most consequential finding of this survey is negative, and it overturns the mechanism section 3.1 proposed. Across every mainstream bipedal and humanoid codebase examined, knee flexion is obtained from the DEFAULT JOINT POSE of the articulation, and not from any reward that pulls the knee toward a flexed target.

| system | default knee | default hip pitch | default ankle pitch | knee reward? |
|---|---|---|---|---|
| IsaacLab G1 | 0.42 | -0.20 | -0.23 | none |
| IsaacLab H1 | 0.79 | -0.28 | -0.52 | none |
| unitree_rl_gym G1 | 0.30 | -0.10 | -0.20 | none |
| IsaacLab Digit | bent | bent | bent | deviation from default, weight -0.2 |
| Walk These Ways, Go1 | calf -1.5 | thigh 0.8 to 1.0 | n/a | none |
| SD_BRS1, as trained | 0.0 | 0.0 | 0.0 | Gaussian at 1.1, weight 10 |

The invariant across every configuration surveyed is that the default is never a straight knee, and the SD_BRS1 is the sole exception. Where a reward touches joint posture at all it is a deviation penalty pulling joints back toward that already flexed default, never a reward pulling them away from a straight one. The IsaacLab G1 configuration is explicit about which joints receive it, `joint_deviation_l1`, computed as the summed absolute deviation from the default pose, is applied to hip yaw and hip roll at weight minus 0.1, to the arms at minus 0.1, and to the torso at minus 0.1, and is deliberately NOT applied to hip pitch or to the knee, the two degrees of freedom that must swing freely to walk. The Digit configuration, a true biped and the closest morphological analogue to the SD_BRS1, does add a knee deviation term but still at the modest weight of minus 0.2, alongside hip yaw at minus 0.2 and hip roll at minus 0.1.

This bears directly on the excessive hip rotation of the observed gait. The established remedy for a limb that escapes into an off axis degree of freedom is a small deviation penalty on the off axis joints specifically, hip roll and hip yaw, at a weight near minus 0.1, while leaving hip pitch and the knee unconstrained. The literature does not support penalising hip pitch amplitude directly, since that joint is the primary swing actuator and constraining it would fight locomotion itself. The balance between hip and knee contribution is a consequence of where the nominal pose sits, not of a term that rations hip travel.

One system takes a different route and is instructive for the contrast. Humanoid-Gym keeps all default joint angles at zero, as ours does, but generates knee and hip targets from a phase conditioned reference, `sin_pos = sin(2 pi phase)` scaled so the knee excursion amplitude is about 0.34 rad, tracked by a reward `exp(-2 ||diff||) - 0.2 clamp(||diff||, 0, 0.5)`. Two properties of that construction are exactly the ones ours lacked. The reference moves with gait phase, so it is never far from the current pose, and the kernel is wide, a 0.3 rad error still returning `exp(-0.6)`, about 0.55, rather than a vanishing number. A reference tracking reward can therefore work from a zero default, but only when the target follows the policy rather than standing 0.88 rad away from it.

### 3.7 Termination as a shaping mechanism, and the height threshold

Section 2.6 recorded that two episodes in five end on the low height condition. The literature offers a caution and a correction here.

The caution is that termination shaping matters far less for plain walking than intuition suggests. The DeepMimic ablation reports normalised policy quality with and without early termination across skills, and while the acrobatic skills degrade sharply without it, backflip 0.791 against 0.730 and sideflip 0.823 against 0.717, the walk gait is unchanged, 0.980 against 0.981. The 40 percent termination rate is therefore better read as a symptom of a policy repeatedly driven to the edge of its ankle authority, as section 2.8 argued, than as the primary cause of the gait defects.

The correction concerns the mechanism itself. Of the IsaacLab humanoid reference configurations examined, none uses a height based termination at all. The base velocity template terminates only on time out and on illegal contact with the base body, and the G1, H1, and Digit configurations override that contact body to the torso while adding no height threshold. Digit adds an orientation termination, `bad_orientation` at a limit angle of 0.7 rad, and the teleoperation work of He and colleagues uses a comparable pair, terminating when the projected gravity on x or y exceeds 0.7 or the base height falls below 0.3 m, a floor far beneath the standing height rather than just beneath it. The concern this raises for the SD_BRS1 is direct. A contact based fall detector fires once the robot has actually collapsed, whereas a height threshold set at 0.65 m against a 1.15 m stance fires whenever the torso dips, which is precisely the region of state space a policy must explore to discover that a flexed knee is viable. Truncating those episodes removes the experience from which knee flexion would be learned, compounding the vanishing gradient of section 2.6 with a data availability problem.

### 3.8 Revised synthesis

The evidence overturns the ordering proposed in section 3.4. A swing gated flexion reward is not how working systems obtain a bent knee, and the attempt to make it so failed for reasons section 2.6 now understands. The corrected reading is that the nominal pose is the primary instrument, that deviation penalties on the off axis hip joints are the established remedy for compensatory hip rotation, that the base height target must lie inside the reachable set established in section 2.8 or it will purchase the forward lean rather than the crouch, and that any residual shaping reward must be broad, monotone, or reference following rather than a narrow kernel placed where the policy does not stand. The ranking by evidence strength and implementation cost places the pose change first, a corrected or removed knee reward second, a curriculum on the target third, and an adversarial or reference based prior last, its expressive power being disproportionate to a defect whose root cause is a single dictionary of joint angles.

---

## 4. Improvement Scope

This section draws the analysis of section 2 and the literature of section 3 into a concrete, phased programme. It states for each defect the mechanism proposed, its mathematical form, the experimental intuition behind it, the code level shape of the change together with the backwards compatibility handling that the caller map of section 2.4 demands, and the measurable gate by which its success will be judged. The subsections are ordered as the interventions should be applied, because the defects are coupled and the sequence matters.

### 4.1 First principles and the ordering of interventions

Three facts fix the order. The straight knee is the deepest of the three pathologies, an attractor assembled from the height target, the zero action default, and the passive load path, and it is also causally upstream, since straight legged propulsion is what tips the torso forward and what forces the hips and ankles to absorb contact shocks through large rapid torques. It follows that the knee is corrected first, that the torso lean is measured and only then corrected, since a portion of it is expected to relax once the knee unlocks, and that the smoothness work comes last, since its target metrics will shift once the leg both flexes to cushion contact and no longer drives the hips and ankles into saturation. A single discipline governs every phase, one change is made per experiment and evaluated against this baseline run through the same `play.py` dump and the same eight diagnostic plots, and every proposal is validated against the newest dumped `params/env.yaml` rather than the working tree, which section 2.4 confirms are in agreement today but which the project history shows may diverge. The symmetry data augmentation already prepared in `/ws/plans/SYMMETRY_PLAN.md`, which targets the residual limp, is orthogonal to the three defects treated here and may proceed in parallel, though for a clean attribution of cause the knee experiment is best run without a simultaneous second change.

### 4.2 Phase A, inducing swing-phase knee flexion

The root cause established in section 2.1 is not that the knee extends but that it never flexes, and the literature of section 3.1 is unanimous that extension is correct at stance and flexion is required in swing, so the corrective term must be gated to the swing phase and must leave the stance posture untouched. The contact clock of the existing `GaitReward` already computes, for each foot, a desired contact state that is near one through stance and near zero through swing, so the swing indicator is available at no cost as its complement.

The proposed term rewards the swing leg's knee for approaching a flexed target, in the exponential kernel idiom the codebase already uses for foot clearance.

```
r_knee_flex = sum over feet i of  ( 1 - desired_contact_i ) * exp( -( q_knee_i - q_flex_target )^2 / sigma_k^2 )
```

Here `q_knee_i` is the knee angle of leg `i`, `q_flex_target` a swing flexion target in the region of 0.5 to 0.8 rad well inside the `[0, 1.483]` range, `sigma_k` a tolerance of order 0.2 rad, and `(1 - desired_contact_i)` the swing indicator from the clock. Because the reward is zero whenever the foot is in stance, the term places no pressure on the efficient extended stance knee, and because it rewards flexion only while airborne it directly supplies the signal section 2.1 found missing. The experimental intuition is that the swing leg, freed of load, can flex its knee at little torque cost to collect this reward while still clearing the foot, and that the clearance and air time rewards will then be met by a shortened, flexing leg rather than by a stiff hip driven swing.

Two supporting adjustments are available. The reset distribution already samples the knee into flexion through `reset_knee_pitch_joints`, so it need not change, but the zero action attractor can be moved by recentring the action offset onto a lightly flexed nominal pose, which shifts the closed loop fixed point off the extension stop so that holding a bent knee no longer requires a standing bias against exploration noise, a measure the project strategy already contemplates. This is a change to the SD_BRS1 initial state and action offset alone and touches no shared function. The primary reward term is a new project local function wired only into the SD_BRS1 `RewardsCfg`, so by the caller map of section 2.4 it carries no backwards compatibility burden whatsoever, and it may be paired with the SD_BRS1 exclusive `foot_clearance_reward_v2` if a combined swing shaping proves cleaner. The gate for this phase is a rise in the knee's swing phase range of motion from its current temporal standard deviation of 0.008 rad to a swing flexion peak above roughly 0.5 rad, with the stance knee still resting near extension, the swing foot clearance held near 0.10 m, the velocity tracking error unchanged, and the knee's mean torque falling from its present 177 Nm as the loaded strut gives way to a working joint.

### 4.3 Phase B, restoring an upright torso

Section 2.2 found the lean tolerated because the only orientation term, `flat_orientation_l2`, is a symmetric, reference free quadratic, and section 3.2 confirmed that its weight of minus thirty five already sits within the recommended band, so the remedy is directed shaping and termination rather than a heavier symmetric penalty. Before either is applied the lean is re measured on the Phase A policy, since section 3.2 predicts that unlocking the knee, by allowing the robot to push off an actively extending leg rather than to fall forward over a passive one, will itself reduce the forward pitch.

Should a residual lean remain, two instruments are proposed. The first is an orientation gated termination, the IsaacLab `bad_orientation` term at a limit angle near 0.8 rad, recorded in the project strategy but never wired in, which removes the heavily pitched and tumbling states from the return entirely rather than taxing them, and which by the weight to rate conversion of section 2.4 must be priced through forfeited future income rather than as a per step penalty. The second, if a standing pitch persists below the termination threshold, is a directed base pitch term that penalises forward inclination specifically rather than tilt in the abstract, for instance a one sided penalty on the forward component of the projected gravity vector, `pen = ( relu( g_x - g_x_ref ) )^2`, which is silent while the torso is upright or slightly back and grows only as it pitches forward past a small reference. Both are config local, the termination as a new `TerminationsCfg` entry and the directed term as a new project local reward wired only into SD_BRS1, so neither disturbs a shared caller. The gate is a reduction in the root mean square torso pitch and in the base pitch rate standard deviation of 0.99 rad/s reported in section 2.2, with no increase in the tracking error and no new failure to stand.

### 4.4 Phase C, smoothing the torque and joint trajectories

Section 2.3 traced the roughness to two compounding sources, smoothness penalties too weak to bind against the tracking rewards and a stiff, unclipped position controller with no penalty on the torque rate, aggravated by a locked knee that denies the leg its natural shock absorber and by an overdamped ankle. Section 3.3 supplies a layered remedy, and because the knee correction of Phase A is expected to soften contact on its own, this phase is measured against the Phase A and B policy rather than against the baseline.

The lightest layer is config local and touches no function. The weights of the existing penalties, the first order `pen_action_rate` at minus 0.01, the second order `pen_action_smoothness` at minus 0.075, `pen_joint_accel` at minus one times ten to the minus seven, and `pen_joint_torque` at minus one times ten to the minus five, are all set in the SD_BRS1 `RewardsCfg` and may be raised there without altering the shared functions they call, since only the weight, not the function, changes and only the SD_BRS1 run is affected. Raising them narrows the gap section 2.3 identified against the tracking rewards. The one penalty the stack lacks is on the torque rate itself, which section 2.3 measured at single step jumps of up to 578 Nm, and this is added as a new project local reward.

```
pen_torque_rate = sum over joints of  ( tau_t - tau_{t-1} )^2
```

The term requires only that the previous step's applied torque be retained, and being new it is wired solely into SD_BRS1 and carries no compatibility burden. Two actuator level adjustments complete the layer, both config local to the SD_BRS1 asset and both requiring a retrain since they change the plant. The action may be clipped, since section 2.4 noted `clip` is null, so bounding the raw action to the order of unit magnitude caps the per step target jump at the scale of 0.25 rad and removes the largest command discontinuities. And the ankle damping, left overdamped at a ratio of 1.43 and 2.46 in section 2.4, may be brought toward the target near 0.7 as the project strategy already recommends, restoring a responsive rather than a sluggish then saturating ankle.

The strongest and most literature backed layer is the policy space regularisation of the Conditioning for Action Policy Smoothness method of section 3.3, which unlike a reward penalty damps the state to action map directly. Its two losses, the temporal `L_T = || pi(s_t) - pi(s_{t+1}) ||_2` and the spatial `L_S = || pi(s_t) - pi(s_bar) ||_2` with `s_bar` drawn from `N(s_t, sigma)`, are added to the PPO surrogate as `J - lambda_T L_T - lambda_S L_S`. The temporal term is computable from the consecutive states already present in each rollout, and the spatial term needs one extra forward pass of the policy on perturbed observations. This is an algorithm level change to the rsl_rl PPO update in `/ws/rsl_rl/rsl_rl/algorithms/ppo.py`, of the same character as the symmetry hook already integrated there, and it is therefore proposed as the second step of this phase, applied only if the reward level layer leaves residual chatter. The gate is a fall in the ninety ninth percentile single step torque change and in the peak joint acceleration reported in section 2.3, a reduction of the twelve times body weight contact impact toward a controlled loading, and preservation of the velocity tracking, with the smoothness regularisation calibrated rather than maximised, since section 3.4 warns that an over damped policy cannot make the fast corrections balance requires.

### 4.5 The phased experimentation plan

The programme therefore runs in three ordered phases, each a single change or a tight group of changes evaluated against the preceding policy, with the discipline of section 4.1 throughout.

| phase | change | kind | backwards compatibility | primary gate |
|---|---|---|---|---|
| A | swing gated knee flexion reward, optional action offset recentre | new project reward, plus SD_BRS1 init state | new term, none needed | swing knee flexion above 0.5 rad, stance knee near extension, tracking and clearance held, knee torque falls |
| B | bad_orientation termination near 0.8 rad, optional directed forward pitch penalty | new termination, new project reward | config local to SD_BRS1 | torso pitch and pitch rate reduced, no new fall, tracking held |
| C | raise existing smoothness weights, add torque rate penalty, clip action, correct ankle damping, then CAPS in the PPO loss | weights config local, new project reward, actuator config, algorithm change | weights and actuator config local, new term none needed, CAPS is an rsl_rl change | single step torque change and peak acceleration reduced, contact impact softened, tracking held |

Each phase produces a new checkpoint, a new `play.py` dump, and a fresh set of the eight plots, so that its effect is read against the same instruments that established the baseline in section 2, and no phase advances until its gate is met, since an unmet gate is itself evidence that the hypothesis of its section requires revision. The detailed code, the exact weights and targets, and the precise wiring of each term are deferred to section 5, to be filled when each phase is implemented.

### 4.6 The revised programme, after the Phase A result

Sections 2.5 through 2.9 and 3.5 through 3.8 supersede the plan of sections 4.2 through 4.5 in its mechanism, though not in its ordering. The knee remains the first target and remains causally upstream of the other two defects, but the instrument changes entirely. Phase A treated the missing swing flexion as a missing reward and added one, and the run proved both that the reward could not be reached and that the literature does not obtain flexion that way in any case. The revised programme treats the nominal pose as the primary instrument and the reward stack as a set of constraints that must be made mutually consistent with it.

Four findings drive the revision, and they are worth stating as a single causal chain. The default pose places the knee on its mechanical extension stop, so every degree of sustained flexion must be bought with a sustained action against exploration noise. The base height penalty, raised to minus three hundred in pursuit of a crouch, asks for a height of 1.0 m that lies outside the reachable set, whose floor is 1.089 m, so it can never be discharged by bending the knee and is instead discharged by trunk pitch and leg splay, which is to say it purchases the forward lean. The knee flexion reward that was meant to break the deadlock was placed 4.4 tolerances into its own tail and delivered no gradient. And the height termination at 0.65 m truncates precisely the low excursions in which a flexed knee would be discovered. Every one of these is a property of the configuration rather than of the learning algorithm, and all four must move together, because fixing any one alone leaves the others opposing it.

The revised Phase A is therefore a single coordinated change to the pose and the terms that reference it, not a single reward. It recentres the nominal joint pose onto the feasible crouch derived in section 2.9, so that the flexion costs no sustained action at all and the zero action fixed point becomes a bent leg. It moves the base height target from the infeasible 1.0 m onto the height that pose actually stands at, and reduces its weight from minus three hundred, since with the crouch carried by the pose the term's job reverts to discouraging collapse and hyperextension rather than forcing a posture. It recentres the four reset events onto the new nominal, so the initial state distribution straddles it. It replaces the dead Gaussian knee reward, either removing it in favour of the pose alone, which is what the G1 and H1 recipes do, or replacing it with a broad monotone term that has gradient where the policy stands. And it adds the small deviation penalty on hip roll and hip yaw that section 3.6 identifies as the established remedy for compensatory hip rotation. Phases B and C retain their targets, the upright torso and the actuation smoothness, but both are re measured after the pose change, since section 2.8 predicts that a feasible height target will itself remove a large part of the forward lean, and section 2.3 predicts that a flexing knee will soften the contact impacts that drive the torque spikes.

One control discipline is worth restating, because the revised Phase A is a group of changes rather than one. They are grouped deliberately, since they are not independent and applying them singly would leave each fighting the others, but the group must be applied as one experiment and evaluated as one, and the reward budget of section 2.5 recomputed afterwards so that the new balance is known rather than assumed.

---

## 5. Implementation Plan

This section is filled one phase at a time, each written in full only when it is to be implemented, so that later phases are specified against the policy the earlier ones actually produced rather than against a forecast. Phase A is specified below. Phases B and C remain reserved, to be written once Phase A has been trained and validated.

### 5.1 Phase A, swing-phase knee flexion, SUPERSEDED

This subsection is retained as the historical record of what was implemented and trained as run `2026-07-23_11-31-57`. It is SUPERSEDED by section 5.2 and should not be implemented as written. Its term was reached by the training run but never by the policy, for the reasons set out in section 2.6, and the mechanism it rests upon is contradicted by the survey of section 3.6. The reader who wants the current plan should go directly to section 5.2.

#### 5.1.1 Baseline note, the working tree has moved since the analysed run

The analysis of section 2 rests on the dumped `params/env.yaml` of run `2026-07-22_11-36-53`, but the working tree has since diverged, and four further runs were trained on 2026-07-23 from the changed configuration. The current `cfg/SF/brs_base_env_cfg.py` RewardsCfg, confirmed against the newest run `2026-07-23_10-32-20`, differs from the analysed baseline in ways that bear directly on the knee. The `pen_base_height` target is now 1.0 m at weight minus three hundred, where the analysed run used 1.14 m at minus fifteen, and by the arithmetic of section 2.4 a 1.0 m target sits 0.175 m below the 1.175 m full extension and demands roughly 0.94 rad of knee flexion, so the height objective now pulls hard toward a crouch rather than supplying almost no pull. A `pen_joint_torque_rate` term using the `JointTorqueRatePenalty` class already present in `mdp/rewards.py` has been wired in at weight minus one times ten to the minus three, which is a Phase C instrument already begun, `pen_flat_orientation` has been raised to minus fifty, a Phase B direction, and `pen_action_rate`, `pen_action_smoothness`, and `pen_joint_accel` have all been raised above their analysed values. The consequence for Phase A is twofold. The swing gated knee flexion reward remains the correct instrument, because a base height crouch bends the knee in stance and in swing alike and does not by itself produce the stance extension with swing flexion pattern that a natural gait requires, but the swing flexion target must be set ABOVE whatever stance knee angle the current crouch produces, which is to be read from the newest run's dump rather than assumed. This divergence should be reconciled before the term is trained, since it changes both the necessity and the target of the reward.

#### 5.1.2 The new reward function

The term is a new project local function added to `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py`. It rewards each leg's knee, while that leg's foot is airborne, for approaching a flexed target, in the exponential kernel idiom of `foot_clearance_reward_v2` and with the two frame contact history gating of `keep_ankle_pitch_zero_in_air`, both of which it mirrors so that no new access pattern is introduced. Because `mdp/__init__.py` re exports the rewards module with a star import, the function is reachable as `mdp.knee_flexion_in_swing` with no export edit.

```python
def knee_flexion_in_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target: float = 0.6,
    std: float = 0.2,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swing leg's knee for approaching a flexed target while the foot is airborne.

    A straight stance knee transmits load efficiently and is left untouched, so this term is
    gated to swing and shapes only the airborne knee, supplying the swing flexion signal that
    no other reward provides. The knee joints resolved by ``asset_cfg`` must appear in the same
    order as the feet bodies resolved by ``sensor_cfg``, left before right, so each knee is
    paired with its own foot.

    Args:
        env: The environment object.
        asset_cfg: Robot asset, ``joint_names`` resolving the two knee pitch joints.
        sensor_cfg: Contact sensor, ``body_names`` resolving the two feet in the same order.
        target: Swing knee flexion target (rad), inside the knee range [0, 1.483].
        std: Width of the Gaussian flexion kernel (rad).
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.

    Returns:
        The reward tensor, summed over the two legs, in the interval [0, 2].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_history[:, -2], dim=-1) > force_threshold
    airborne = ~torch.logical_or(current_contact, last_contact)  # (N, F), True while swinging
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]          # (N, F)
    reward = torch.exp(-torch.square(knee - target) / std**2) * airborne.float()
    return torch.sum(reward, dim=1)
```

The mathematics are the following. For each leg i, let q_knee,i be the knee angle and let c_i be the airborne indicator, one when the foot's contact force stays below the threshold across the last two physics substeps and zero otherwise, the same two frame filter that `keep_ankle_pitch_zero_in_air` uses to reject contact chatter. The per leg reward is c_i times the exponential of minus the square of q_knee,i minus the target over the square of the tolerance, and the term returns the sum over the two legs. The reward is bounded in the interval zero to two, it is exactly zero for a stance leg, and it is maximised when an airborne knee sits at the target, so it places no pressure on the efficient extended stance knee and shapes only the swing.

#### 5.1.3 Wiring into the SD_BRS1 reward configuration

A single `RewTerm` is added to the RewardsCfg in `cfg/SF/brs_base_env_cfg.py`, alongside the existing terms, wired only into the SD_BRS1 configuration, so by the caller map of section 2.4 it disturbs no other task. The knee joints and the feet bodies are given as explicit ordered lists, left before right, so that each knee pairs with its own foot, and the contact sensor is the `contact_forces` sensor the other terms already use.

```python
    rew_knee_flexion = RewTerm(
        func=mdp.knee_flexion_in_swing,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["KneePitchL", "KneePitchR"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "target": 1.1,
            "std": 0.2,
            "force_threshold": 1.0,
        },
    )
```

The target is set to 1.1 rad for the current base height regime, above the crouch the 1.0 m at minus three hundred target forces, so the swing knee flexes further than the stance knee. Were the base height reverted toward the analysed 1.14 m at minus fifteen, the target would drop to the region of 0.6 rad per the table in section 5.1.4.

#### 5.1.4 Choosing the target, the weight, and the tolerance

The tolerance and the weight are stable across the two base height regimes, but the target is not, and it must be set above the prevailing stance knee angle so that swinging flexes the knee further than standing does. The stance knee angle is read from the newest run's `data/*/dump.npy`, the knee columns four and five, over the stance windows, and the target is placed roughly 0.3 to 0.4 rad above it, capped well inside the 1.483 rad limit.

| base height regime | approximate stance knee | swing target | std | weight |
|---|---|---|---|---|
| current, 1.0 m at minus three hundred | crouched near 0.9 rad | 1.1 to 1.2 rad | 0.2 rad | 10, tune 5 to 20 |
| analysed, 1.14 m at minus fifteen | near 0 rad, straight | 0.5 to 0.7 rad | 0.2 rad | 10, tune 5 to 20 |

The weight of ten is chosen because the term saturates at one per airborne leg and at most one leg swings at a time in a walking gait, so its realised rate per second is at most about ten, comparable to the foot clearance reward at twenty and the air time reward at 12.5 and well below the linear velocity tracking at fifty, which lets it shape the swing without dominating the objective. The tolerance of 0.2 rad, about eleven degrees, is wide enough to reward progress toward the target from the current locked pose and narrow enough to distinguish a flexed swing knee from a straight one.

#### 5.1.5 The optional action offset recentre, held as a contingency

The action offset recentre is deliberately NOT part of the first Phase A experiment, since the discipline of section 4.1 admits one change at a time, and the swing reward is the cleaner single instrument to test first. It is held as a contingency for the case where the reward alone cannot lift the knee off its zero action attractor. The recentre moves the knee entries of `_JOINT_POS_DEFAULTS` in `robots/brs_solefoot_env_cfg.py` from zero to a lightly flexed value near the stance target, which by `use_default_offset` shifts the zero action fixed point onto a bent knee, and it requires that the `reset_knee_pitch_joints` offset range be recentred so the reset distribution still straddles the new default rather than sampling only above it. This is a change to the SD_BRS1 asset and reset alone and touches no shared function, but it alters the stance posture as well as the swing, so it is a separate experiment with its own evaluation, not a companion to the reward in the same run.

#### 5.1.6 Validation gates

The Phase A policy is evaluated through the same `play.py` dump and the same eight plots that established the baseline in section 2, and it passes only if every one of the following holds. The swing knee flexes to within the tolerance of its target, read as a swing phase knee angle peak above roughly the target minus the tolerance, against the current temporal standard deviation of 0.008 rad. The stance knee remains near the angle the base height objective sets, so the term has not disturbed the efficient stance posture. The swing foot clearance holds near 0.10 m and the feet continue to alternate cleanly. The velocity tracking error is unchanged within noise. And the knee mean torque falls from its baseline of 177 Nm as the loaded strut becomes a working joint, with the knee torque against velocity scatter opening from its vertical zero velocity stripe into a loop. An unmet gate is evidence that the hypothesis of section 2.1 requires revision rather than that the weight needs a further nudge.

#### 5.1.7 The minimal implementation, in order

The change is two small edits to two files, and no more, which is the cheapest form the plan can take. First, the function of section 5.1.2 is appended to `mdp/rewards.py`, requiring no import change since `Articulation`, `ContactSensor`, `SceneEntityCfg`, `ManagerBasedRLEnv`, and `torch` are already imported there and no `__init__` edit since the module is star exported. Second, the `RewTerm` of section 5.1.3 is added to the RewardsCfg in `cfg/SF/brs_base_env_cfg.py`, with its target set per section 5.1.4 once the current stance knee angle has been read. Training and evaluation are the user's action, since the container carries no Isaac Sim.

### 5.2 Phase A2, the nominal pose and the terms that reference it

This is the revised first phase set out in section 4.6, replacing the failed Phase A. It is a coordinated group of changes to the SD_BRS1 configuration and asset. Every item states its file, its previous value, its new value, and the section that justifies it.

The phase has now been IMPLEMENTED, on 2026-07-27, with one deliberate departure from the specification above. The knee flexion reward is retained rather than removed, reimplemented in its monotone form so that the two configurations may be compared by ablation. The contact index correction was held back for a provenance survey, which returned a verdict of porting error, and was then applied. The status of each item is the following.

| item | subject | status |
|---|---|---|
| 5.2.1 | nominal pose and spawn height, both files | APPLIED |
| 5.2.2 | base height target and weight, every instance | APPLIED |
| 5.2.3 | the four reset events | APPLIED |
| 5.2.4 | knee flexion reward | REIMPLEMENTED as v2, retained for ablation |
| 5.2.5 | hip roll deviation penalty | APPLIED |
| 5.2.6 | stale contact index | APPLIED, after the provenance survey confirmed a porting bug |
| 5.2.7 | height termination | APPLIED at 0.4 m |

#### 5.2.1 The nominal pose, APPLIED

The pose is now carried identically in BOTH places that define it, so the two files no longer disagree. The operative one is `_JOINT_POS_DEFAULTS` in `tasks/locomotion/robots/brs_solefoot_env_cfg.py`, which overwrites the asset at lines 48, 82, and 121, and the asset's own `init_state` in `assets/config/sd_brs1_identified_cfg.py` now carries the same numbers rather than the incorrect crouch recorded in section 2.9.

```python
_JOINT_POS_DEFAULTS = {
    "HipRollR": 0.0,      "HipRollL": 0.0,
    "HipPitchR": 0.2379,  "HipPitchL": -0.2379,   # mirrored axes, opposite signs
    "KneePitchR": 0.4814, "KneePitchL": 0.4814,   # same axis, same sign
    "AnkleRollR": 0.0,    "AnkleRollL": 0.0,
    "AnklePitchR": 0.2435, "AnklePitchL": 0.2435, # same axis, same sign
}
```

The spawn height moved from `(0.0, 0.0, 1.2)` to `(0.0, 0.0, 1.17)`, preserving the 20 mm settling margin the previous value carried over its own 1.175 m straight legged standing height, now measured against the 1.15 m standing height of the crouch. The misleading docstring, which described a squat at 1.1 m that the numbers beneath it did not implement, is replaced by the closed chain derivation and the sign convention, so the file now documents what it does.

#### 5.2.2 The base height target and weight, APPLIED

A sweep of the whole extension was made for every place a base height is specified or consumed, so that none was left referring to the infeasible figure. Four sites carry one, and the two belonging to the SD_BRS1 have moved.

| site | quantity | previous | new |
|---|---|---|---|
| `cfg/SF/brs_base_env_cfg.py` `pen_base_height` | `target_height` | 1.0 | 1.15 |
| `cfg/SF/brs_base_env_cfg.py` `pen_base_height` | `weight` | -300.0 | -30.0 |
| `cfg/SF/brs_base_env_cfg.py` `pen_feet_regulation` | `base_height_target` | 1.0 | 1.15 |
| `cfg/PF/limx_base_env_cfg.py` | `target_height`, `base_height_target` | 0.65 | UNTOUCHED, TRON1 PF |

The weight of minus thirty is chosen so the term retains real authority against collapse and hyperextension without dominating. At a 0.05 m deviation it now pays 0.075 per second and at the 0.118 m deviation of the failed run it would pay 0.42 per second, an order of magnitude below the 4.21 that run paid permanently. The `height_decay_scale` of `pen_feet_regulation` is left at 0.03, since its guiding comment sizes it at about 0.025 of the base height target and 0.03 against 1.15 is 0.026, still within that intent.

Two further sites mention a base height without setting the reward target and were checked and left alone. `mdp/rewards.py` `nominal_foot_position` takes a `base_height_target` argument but is wired into no configuration, and `mdp/terminations.py` `root_height_below_minimum` takes the termination floor treated separately in section 5.2.7.

#### 5.2.3 The reset events, APPLIED

All four reset events in `cfg/SF/brs_base_env_cfg.py` add their offsets to the default, so against a flexed nominal they no longer straddle it. Each is recentred so its mean offset is zero while its width is preserved, keeping the reference state initialisation broad, which section 3.5 and the DeepMimic precedent both favour.

```
reset_knee_pitch_joints    (0.0, 0.475)    ->  (-0.24, 0.24)
reset_hip_pitch_r_joint    (-0.25, 0.42)   ->  (-0.24, 0.24)
reset_hip_pitch_l_joint    (-0.42, 0.25)   ->  (-0.24, 0.24)
reset_ankle_pitch_joints   (-0.113, 0.25)  ->  (-0.15, 0.15)
```

The ankle range is held narrower because section 2.9 shows its margin to the soft limit at the new pose is 0.165 rad, so a wider sample would clamp.

#### 5.2.4 The knee flexion reward, REIMPLEMENTED and RETAINED

The recommendation of section 3.6 was to remove the term, since the G1 and H1 recipes carry no knee reward at all and obtain flexion from the pose alone. That recommendation is overridden by direction, and deliberately so, because removing it would confound two changes. Retaining a working term allows the pose change to be measured with and without a knee incentive, which is the cleaner experiment and settles by ablation what the survey can only settle by analogy.

The term is therefore reimplemented rather than removed. Because the new form is a different quantity from the old, rewarding flexion RELATIVE to the stance nominal rather than proximity to an absolute angle, and because its parameters change meaning, no optional argument can carry it while preserving the old behaviour, so by rule 5 of the workspace code discipline it is a new function. The original is left untouched and unwired, which also keeps the dumped `params/env.yaml` of run `2026-07-23_11-31-57` replayable, and its docstring now records the defect it embodies.

The new function in `mdp/rewards.py` is a monotone ramp whose derivative is the constant `1 / (cap - nominal)` across its whole active range.

```python
def knee_flexion_in_swing_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal: float = 0.4814,
    cap: float = 0.9,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_history[:, -2], dim=-1) > force_threshold
    airborne = ~torch.logical_or(current_contact, last_contact)  # (N, F), True while swinging
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]          # (N, F)
    ramp = torch.clamp((knee - nominal) / (cap - nominal), min=0.0, max=1.0)
    return torch.sum(ramp * airborne.float(), dim=1)
```

and it is wired in place of the v1 term, at the same weight of ten.

```python
    rew_knee_flexion = RewTerm(
        func=mdp.knee_flexion_in_swing_v2,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["KneePitchL", "KneePitchR"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "nominal": 0.4814,   # the stance nominal carried by _JOINT_POS_DEFAULTS
            "cap": 0.9,
            "force_threshold": 1.0,
        },
    )
```

The mathematics are the following. For each leg i, with `c_i` the airborne indicator, the per leg reward is `c_i * clip((q_i - nominal) / (cap - nominal), 0, 1)`, and the term returns the sum over the two legs, bounded in zero to two. Setting `nominal` to the stance pose means the term pays nothing for merely holding the crouch and pays only for flexing beyond it, so it cannot be farmed by standing, and setting `cap` to 0.9 rad saturates it at a natural swing flexion well inside the 1.483 rad limit. The two changes work together and neither would suffice alone. The pose change moves the knee's resting angle from zero to 0.4814, which is exactly the foot of the ramp, so the term is live from the first step of training with a gradient of `10 / (0.9 - 0.4814)`, that is 23.9 per radian, against the 1.7e-6 that v1 offered at the place the policy actually stood. Conversely the ramp is deliberately flat below its nominal, returning zero for any knee angle under 0.4814, so had it been introduced without the pose change it would have been just as silent as the term it replaces, since the old operating point of 0.22 rad lies beneath its foot. A monotone reward is only monotone where it is active, and the pose is what places the policy inside that region.

Retaining the term also defines the ablation. The two runs to compare are the pose change with `rew_knee_flexion` wired, and the pose change with that single term commented out. Since every other change is common to both, the difference isolates the contribution of an explicit swing incentive over and above what the nominal pose supplies, which is precisely the question section 3.6 raises and cannot answer from the literature alone.

#### 5.2.5 The compensatory hip rotation, APPLIED

Section 3.6 identifies the established remedy, a small deviation penalty on the off axis hip joints only, leaving hip pitch and the knee free. IsaacLab's `joint_deviation_l1` is a library function computing the summed absolute deviation from the default pose, so it is wired directly with no new code.

```python
    pen_hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,                      # the G1 weight
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["HipRoll[LR]"])},
    )
```

`HipYaw` is a fixed joint on this robot, confirmed from the URDF where both are declared `type="fixed"`, so the articulation carries ten actuated degrees of freedom and hip roll is the entire off axis set. The weight of minus 0.1 is taken directly from the IsaacLab G1 configuration. This also discharges the dead `joint_deviation_l1` import recorded in section 2.4, which had been present and unwired.

#### 5.2.6 The stale contact index, provenance ESTABLISHED and correction APPLIED

This item was held back so that the provenance of a convention shared by three reward terms could be established before it was altered, a shared idiom repeated in several places being as likely to be inherited from an upstream implementation as to be an independent slip. That survey has now been carried out, across the IsaacLab source, the two ancestor repositories from which this project descends, the wider legged_gym family, and the git history of this repository. Its verdict is that the indexing is a porting error, and the evidence is unusually direct.

The first and most decisive point is that IsaacLab documents the ordering explicitly, and the documentation contradicts the code. The docstring of `net_forces_w_history` in `sensors/contact_sensor/contact_sensor_data.py` lines 84 to 91 states that the shape is `(N, T, B, 3)` and that, in the history dimension, the first index is the most recent and the last index is the oldest. This is not an inference from the roll and write mechanism recorded in section 2.7, it is the sensor's own stated contract, and `[:, -1]` therefore reads the oldest sample by definition rather than by accident.

The second is that no first party IsaacLab code reads a trailing index at all. A sweep of every site in the IsaacLab tree that touches `net_forces_w_history` returns two conventions and only two, either index zero, or a reduction by maximum or mean over the whole history axis, which is order independent by construction and therefore immune to the question. A repository wide search for a trailing single index slice of a force history returns no hits whatever. The framework's own debounce primitives, `compute_first_contact` and `compute_first_air`, do not slice the raw history either, they are built on tracked contact and air time scalars, so the idiomatic way to ask whether contact changed within the last step deliberately avoids raw indexing.

The third traces the intended semantics to their source. The Isaac Gym sibling of this project, LimX Dynamics' own codebase for the same TRON1 hardware, contains `_reward_keep_ankle_pitch_zero_in_air` and builds its contact gate as `contact_filt = contact OR last_contacts`, where `contact` is the instantaneous force test and `last_contacts` is a manually cached copy of the previous step, reassigned every step. That buffer has no history axis at all. The two sample disjunction of the current code is plainly an attempt to reproduce this current OR previous step debounce using a history buffer instead of manual caching, and the debounce intent is sound. What miscarried is only which end of the buffer supplies the two samples, since `[:, 0]` and `[:, 1]` would reproduce the ancestor's semantics exactly while `[:, -1]` and `[:, -2]` reach for the two oldest.

The fourth is the git history of this repository, which shows the defect arriving in two separate steps and neither being deliberate. The genesis commit already carries the inconsistency, `no_fly` and `no_contact` both correct at index zero while `keep_ankle_pitch_zero_in_air` is already reading the trailing pair, and it arrives inside a vendor squash whose entire commit message is a version string, so no rationale survives. `no_fly` was then REGRESSED from correct to incorrect by a later commit whose message concerns a velocity curriculum and two new reward terms and says nothing about contact timing at all, the change to `no_fly` appearing as an unremarked side effect of editing next to the function that already carried the pattern. A third commit subsequently fixed a DIFFERENT bug in `keep_ankle_pitch_zero_in_air`, replacing hardcoded joint indices with resolved ones, and its message records that fix while the timing defect survives untouched in the same diff. The idiom has therefore now survived two dedicated bug fixing passes over the very function that originated it, which is the signature of an unnoticed defect rather than a defended choice.

The direct IsaacLab ancestor of this project settles the last doubt. Its `no_fly` reads index zero, correctly, in its first commit and in every commit thereafter, and `keep_ankle_pitch_zero_in_air` does not exist in it at all, so the function and its defect were authored downstream of that repository with no upstream precedent to copy.

No implementation, no paper, and no comment found in any of these sources deliberately uses a delayed contact signal in a reward. The only plausible motive, chatter rejection, is fully served by a disjunction over the two NEWEST samples, so it cannot account for reaching to the oldest.

One correction to the caller set recorded in section 2.7 follows from the same sweep. `no_fly` and `keep_ankle_pitch_zero_in_air` are called from the TRON1 SF configuration and from the SD_BRS1 configuration, and from nowhere else. The TRON1 PF configuration does NOT call either, contrary to what section 2.7 implied by referring to two TRON1 tasks, and `no_contact` currently has no callers at all and is dead code. The blast radius of a correction is therefore one other task, not two.

On that evidence the correction is APPLIED, by direction, and in the same experiment as the pose change rather than after it. Both shared functions take an optional argument whose default reproduces the present behaviour exactly, so the TRON1 SF caller is untouched, and the correct index is set explicitly only in the SD_BRS1 configuration, where it is also recorded in that run's dumped `params/env.yaml`.

```python
def no_fly(env, sensor_cfg, threshold: float = 1.0, history_index: int = -1):
    ...
    contacts = torch.norm(latest_contact_forces[:, history_index], dim=-1) > threshold
```

`keep_ankle_pitch_zero_in_air` takes the same argument and derives its second sample as the adjacent OLDER slot, which is one lower when counting from the tail and one higher when counting from the head, since index zero is the newest.

```python
    previous_index = history_index - 1 if history_index < 0 else history_index + 1
```

so a default of minus one pairs with minus two and reproduces the stale behaviour bit for bit, while an explicit zero pairs with one and reproduces the ancestor's current or previous debounce exactly. Both are set to zero in `cfg/SF/brs_base_env_cfg.py` and nowhere else. `knee_flexion_in_swing_v2` reads indices zero and one directly and takes no such argument, since it is SD_BRS1 exclusive and has no prior run whose behaviour must be preserved, which supersedes the interim position recorded in section 5.2.4.

The semantics were verified numerically rather than by inspection, by emulating the sensor's roll and write over four steps with distinguishable forces. Index zero returns the newest sample and index minus one the oldest, the default pair returns the two oldest exactly as the previous code did, and the explicit pair returns the two newest. The defect left standing in the unset default is recorded in the docstrings of both functions so the TRON1 SF task may opt in deliberately.

One consequence for the experiment should be noted plainly. `no_fly` supplies 11.68 per second, a fifth of the positive budget, and its single support test now fires on current rather than stale contacts, so its realised value will shift for reasons unrelated to the pose. The A2 runs therefore carry two changes against the previous baseline, the pose group and this correction, and the budget of section 2.5 must be recomputed against the new run rather than differenced term by term against the old one. The A2-a against A2-b comparison is unaffected, since the correction is common to both arms.

#### 5.2.7 The height termination, APPLIED at 0.4 m

The `low_height` termination moves from 0.65 m to 0.4 m, deeper than the 0.45 m originally proposed. The reason is specific to this robot's geometry and supersedes the general argument of section 3.7. That section observed that no IsaacLab humanoid reference configuration uses a height termination at all, preferring contact detection on the torso, and inferred that this one might simply be removed. It cannot be. The `base_contact` termination fires on contact with `Part_Torso`, but the SD_BRS1 is built such that its base may never reach the ground even in a genuine fall, so contact detection alone does not reliably detect one and the height floor is doing necessary work as a fall detector rather than being redundant with it.

The change therefore keeps the mechanism and lowers the threshold until it discriminates a fall from a crouch. At 0.4 m against a 1.15 m stance the floor sits well beneath any posture the robot can adopt on its feet, the deepest balanced crouch being 1.089 m by section 2.8, so it no longer truncates the low excursions in which knee flexion is discovered while still catching a collapse. The figure is comparable to the 0.3 m floor that He and colleagues use in the teleoperation work cited in section 3.7.

#### 5.2.8 The experiment, and the validation gates

The items above are one coordinated change, for the reason given in section 4.6, that they are mutually dependent and each alone would be opposed by the others. Section 5.2.4 then splits that single change into a two run ablation, since the knee reward is retained rather than removed.

| run | pose and terms of 5.2.1 to 5.2.3, 5.2.5, 5.2.7 | `rew_knee_flexion` |
|---|---|---|
| A2-a | applied | wired, v2 monotone, weight 10 |
| A2-b | applied | commented out |

Comparing the two isolates the contribution of an explicit swing incentive over and above what the nominal pose supplies, which is the one question section 3.6 raises that the literature cannot settle, every surveyed humanoid obtaining flexion from the pose alone but none of them being this robot.

The gates, measured from a fresh `play.py` dump and the same eight plots that established the baseline, are the following. The stance knee should rest near the new nominal of 0.48 rad rather than near zero, which tests the pose change directly. The swing knee should exceed the stance knee, which is the property Phase A was meant to obtain and never did, and which is now readable straight from the joint position plot rather than inferred from a reward. `pen_base_height` should fall from its permanent 4.21 per second toward a small residue, confirming the target is now attainable. `pen_joint_pos_limits` should lose the 0.297 per second structural component that the zero default knee contributed. The forward lean and the base pitch rate should fall, per the prediction of section 2.8 that the height penalty was purchasing them. The `low_height` termination fraction should fall well below 0.4, both because the robot is no longer driven toward an unreachable crouch and because the floor has moved to 0.4 m. And the velocity tracking should hold within noise of the present 27.19 per second.

One instrument is worth carrying forward from the failure. Because `rew_knee_flexion` is now monotone in the swing knee angle, its logged value again measures that angle, this time usefully rather than as a post-mortem, since the realised rate divided by the weight and the airborne duty of about 0.78 gives the mean normalised flexion beyond the nominal directly. A value near zero means the pose is holding but the swing is not flexing, a value near one means the swing is reaching the 0.9 rad cap.

Two outcomes would each be decisive in a useful way. If the stance knee settles at the nominal but the swing knee does not exceed it in either run, the pose hypothesis is confirmed for stance and refuted for swing, and the next instrument is a phase conditioned reference of the Humanoid-Gym kind described in section 3.6 rather than a static incentive. If run A2-b matches A2-a, the knee reward is redundant and should be retired in favour of the pose alone, which is what the surveyed recipes would predict.

### 5.3 Phase B and Phase C

Reserved, to be written after Phase A2 is trained and validated, since section 4.6 predicts that both the forward lean and the contact roughness will partly resolve as a consequence of the pose change and must therefore be re measured before being targeted.

---

## 6. Conclusion

The run `2026-07-22_11-36-53` marks a genuine milestone, the first SD_BRS1 policy that walks with a coordinated alternating gait and follows its velocity commands to within eight centimetres per second, and the analysis of section 2 leaves no doubt that its three remaining defects are matters of naturalness rather than of competence. The knee is locked not by accident but by a deep attractor, the coincidence of a base height target within three and a half centimetres of full leg extension, a zero action default resting on the knee's own extension stop, and a passive load path that bears weight cheaply, standing unopposed by any reward for flexion, and it is this locked knee that both tips the torso forward into its controlled fall and forces the hips and ankles to absorb through five hundred Newton metre torque steps and twelve times body weight impacts the shocks a flexing knee would cushion. The forward lean is permitted by an orientation penalty that is adequately weighted but symmetric and reference free, and the roughness by smoothness penalties two to seven orders of magnitude below the tracking rewards atop a stiff, unclipped position controller with no penalty on the torque rate at all.

The literature of section 3 confirms that each defect is well understood and each has an established remedy, that a straight stance knee is correct and only its swing counterpart is missing, that an upright carriage is a matter of directed shaping rather than heavier symmetric penalty, and that policy space smoothness regularisation of the Conditioning for Action Policy Smoothness kind is the strongest single instrument for a clean transfer to hardware. Section 4 assembles these into a phased programme that treats the knee first, because it is causally upstream of the other two, then measures and corrects the lean, then smooths the actuation against a plant that the knee correction will already have gentled.

The first phase of that programme was then implemented and trained, and it failed, which is the subject of the later half of this document and the reason its conclusion is now longer than its original. The failure was instructive rather than merely disappointing. A swing gated Gaussian reward on the knee, correctly weighted at fifteen percent of the positive budget were it ever to saturate, was placed 0.88 rad from where the policy stood with a tolerance of 0.2 rad, and so delivered a gradient of 1.7e-6 per radian, one part in six million of the most naive monotone alternative. Across twenty one thousand iterations it moved the swing knee not at all, and because the reward is monotone in the knee angle its own magnitude could be inverted to prove as much, fixing the knee at 0.22 rad throughout. The lesson generalises beyond this term, a reward is only an instruction if it has a derivative where the policy actually is.

Three deeper findings emerged from the same investigation and between them redirect the programme. The base height target of 1.0 m, raised to weight minus three hundred in pursuit of a crouch, lies outside the robot's reachable set, whose floor is 1.089 m because the ankle pitch limit binds long before the knee does, so the penalty could never be discharged by bending the knee and was instead discharged by trunk pitch and leg splay, which is to say the height term has been buying the very forward lean that a second term was paying to prevent. The gait reward, despite its name and its positive weight of forty, is bounded in minus two to zero and is the largest single penalty in the stack. And three contact gated terms, one of them supplying a fifth of the positive budget, decide whether a foot is grounded from the two oldest samples of the contact buffer rather than the newest.

Against all of this the literature returns an unusually clean verdict. No mainstream humanoid system obtains knee flexion from a reward. The G1, the H1, the Go1, and Digit all carry it in the default joint pose of the articulation, and where a reward touches posture at all it is a small deviation penalty pulling joints back toward that already flexed default, applied deliberately to the off axis hip degrees of freedom and withheld from the hip pitch and the knee that must swing. The SD_BRS1 is the sole surveyed configuration whose default knee sits on its mechanical extension stop. The revised plan of sections 4.6 and 5.2 therefore moves the flexion into the nominal pose, where it costs no sustained action at all, brings the height target and its penalty weight onto the pose the robot can actually hold, recentres the reset distribution and the contact indices, adds the established hip roll deviation penalty, and retires the dead reward rather than reweighting it. These changes are grouped as one experiment because they are mutually dependent, and each is wired only into the SD_BRS1 configuration or carried on an optional argument whose default preserves the shared callers, in keeping with the backwards compatibility discipline of the workspace. What remains is to train it, and to recompute the budget of section 2.5 afterwards so that the new balance is measured rather than assumed.

---

## 7. Bibliography

The following sources ground the survey of section 3 and the plan of section 4. Verified formulations are drawn from the primary papers, and the internal project files cited alongside record the SD_BRS1 specific facts on which section 2 rests.

Knee flexion, natural gait, and imitation

1. McGeer, T. (1990). Passive Dynamic Walking. International Journal of Robotics Research, 9(2), 62 to 82.
2. Heess, N., et al. (2017). Emergence of Locomotion Behaviours in Rich Environments. arXiv:1707.02286.
3. Peng, X. B., Abbeel, P., Levine, S., van de Panne, M. (2018). DeepMimic, Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills. ACM Transactions on Graphics, 37(4). arXiv:1804.02717.
4. Peng, X. B., Ma, Z., Abbeel, P., Levine, S., Kanazawa, A. (2021). AMP, Adversarial Motion Priors for Stylized Physics-Based Character Control. ACM Transactions on Graphics, 40(4). arXiv:2104.02180.
5. Escontrela, A., Peng, X. B., Yu, W., Zhang, T., Iscen, A., Goldberg, K., Abbeel, P. (2022). Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions. IROS 2022. arXiv:2203.15103.
6. Peng, T., Bao, L., Zhou, C. (2025). Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion. arXiv:2505.20619.
7. Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement (4th ed.). Wiley. Stance extension and swing flexion of the knee in human gait.

Torso orientation, posture, and reward design

8. Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL 2021. arXiv:2109.11978. The legged_gym reward set, action rate, orientation, and base height penalties.
9. van Marum, B., et al. (2024). Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking. IROS 2024. arXiv:2404.19173.

Actuation smoothness and sim to real

10. Mysore, S., Mabsout, B., Mancuso, R., Saenko, K. (2021). Regularizing Action Policies for Smooth Control with Reinforcement Learning, the Conditioning for Action Policy Smoothness method. ICRA 2021. arXiv:2012.06644.
11. Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., Hutter, M. (2019). Learning agile and dynamic motor skills for legged robots. Science Robotics, 4(26), eaau5872. The actuator network.
12. Siekmann, J., Godse, Y., Fern, A., Hurst, J. (2021). Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. ICRA 2021. arXiv:2011.01387. The periodic contact clock underlying the GaitReward.

Reward shaping, kernel design, and curriculum, added for sections 3.5 to 3.8

18. Ng, A. Y., Harada, D., Russell, S. (1999). Policy Invariance Under Reward Transformations, Theory and Application to Reward Shaping. ICML 1999, 278 to 287. The potential based shaping theorem.
19. Wiewiora, E. (2003). Potential-Based Shaping and Q-Value Initialization are Equivalent. JAIR, 19, 205 to 208. The episodic caveat, the potential must vanish at terminal states.
20. Jeon, S. H., Heim, S., Khazoom, C., Kim, S. (2023). Benchmarking Potential Based Rewards for Learning Humanoid Locomotion. ICRA 2023. arXiv:2307.10142.
21. Peng, X. B., Coumans, E., Zhang, T., Lee, T. W. E., Tan, J., Levine, S. (2020). Learning Agile Robotic Locomotion Skills by Imitating Animals. RSS 2020. arXiv:2004.00784. The five kernel imitation reward whose interior coefficients span 0.1 to 40.
22. Kim, Y., Oh, H., Lee, J., Choi, J., Ji, G., Jung, M., Youm, D., Hwangbo, J. (2024). Not Only Rewards But Also Constraints, Applications on Legged Robot Locomotion. IEEE T-RO. arXiv:2308.12517. Constraint costs in place of reward kernels.
23. Bengio, Y., Louradour, J., Collobert, R., Weston, J. (2009). Curriculum Learning. ICML 2009.
24. Babadi, A., Naderi, K., Hamalainen, P. (2019). Self-Imitation Learning of Locomotion Movements through Termination Curriculum. MIG 2019. arXiv:1907.11842. The 0.75 to 0.5 scheduled threshold.
25. Gu, X., Wang, Y. J., Chen, J. (2024). Humanoid-Gym, Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer. arXiv:2404.05695. The phase conditioned reference and its wide kernel.
26. Margolis, G. B., Yang, G., Paigwar, K., Chen, T., Agrawal, P. (2022). Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior. CoRL 2022. arXiv:2212.03238.
27. He, T., Luo, Z., Xiao, W., Zhang, C., Kitani, K., Liu, C., Shi, G. (2024). Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation. IROS 2024. arXiv:2403.04436. Orientation and height termination thresholds.

Codebases consulted for default poses and reward weights, sections 3.6 and 3.7

28. isaac-sim/IsaacLab, main branch. The G1, H1, and Digit configurations, `mdp/rewards.py`, `mdp/terminations.py`, `velocity_env_cfg.py`, and `isaaclab_assets/robots/unitree.py`.
29. leggedrobotics/legged_gym. The `tracking_sigma` convention of 0.25 in squared error units.
30. unitreerobotics/unitree_rl_gym. The G1 default joint angles and reward scales.
31. Improbable-AI/walk-these-ways. The Go1 default joint angles.
32. roboterax/humanoid-gym. The reference trajectory and reward implementation.

Internal project records

13. `/ws/context/brs_gait.md`, the SD_BRS1 gait work stream, kinematic facts, joint limits, and reward history.
14. `/ws/plans/GAIT_STRATEGY.md`, the strategy and phase plans, the damping ratio derivation of section 2.5, and the reward anchor lines.
15. `/ws/context/literature.md`, the running literature notes, clusters 6 to 9.
16. `/ws/context/sd_brs_urdf.md`, the URDF link lengths, the sole offset, the joint limits, and the default pose.
17. `/ws/IsaacLab/logs/rsl_rl/sd_brs1_flat/2026-07-22_11-36-53/params/env.yaml`, the authoritative reward, actuator, and action configuration of the analysed run.
