# The Actuator Parameterisation of the K-Scale Labs K-Bot, as Published in the Open Source Reinforcement Learning Stack

> Status, established 2026-09-04 from nineteen K-Scale Labs repositories cloned to `/ws/kscale` on that date. Every parameter quoted below is read from a named file and line in that tree, and every derived quantity is computed from the shipped MuJoCo model by the scripts described in section 14, so a later reader may reproduce the whole of it without re-cloning. Where a figure could not be established from the tree it is recorded as absent rather than estimated.

## 1. Introduction

The question this document answers is narrow and the answer is not, because the K-Scale Labs stack distributes the actuator parameterisation of its humanoid across three layers that were written at different times and do not agree with one another. A reader who opens the robot description finds an armature and a friction loss but no stiffness and no damping. A reader who opens the training script finds no gains either, only a scale factor applied to something the script does not itself contain. The gains live in a third place, a JSON metadata file that travels beside the model and is consumed by the simulator, by the deployment runtime and by the firmware bridge alike, and the consequence of that arrangement is that a change to one consumer silently diverges from the others. Section 11 records one such divergence that is live in the tree today.

The document proceeds from provenance to parameters to analysis. Sections 2 and 3 establish what was cloned and settle a nomenclature hazard that would otherwise corrupt every comparison a reader of this workspace might draw. Sections 4 through 7 report the parameters themselves, the per joint gains, the per actuator class inertias and limits, and the nominal pose from which every training episode begins. Sections 8 through 12 are the analysis proper, deriving the effective inertia, the natural frequency and the damping ratio at each joint, reconstructing the rule by which the published damping was almost certainly chosen, and testing that rule against the load the robot actually carries when it stands. Section 13 consolidates every figure into one table.

## 2. Provenance

Nineteen repositories were cloned, selected as those bearing on reinforcement learning for K-Bot locomotion. The shallow clones and their head commits at the time of reading are recorded below, since several of these repositories are under weekly revision and a figure quoted without a commit is not a figure a later reader can check.

| Repository | Commit | Date | Bearing on the question |
|---|---|---|---|
| `kbot` | 76aba70 | 2025-11-07 | Hardware documentation only, carries no model and no gains |
| `kbot-models` | 08defce | 2025-10-02 | The canonical models and metadata, four variants |
| `kbot-joystick` | 211f9a2 | 2025-10-23 | The current full body locomotion policy and its training script |
| `ksim` | 52265b5 | 2025-10-29 | The training library, actuator classes and randomisers |
| `ksim-gym` | 60f49a7 | 2025-06-05 | The reference walking task, unscaled gains |
| `ksim-gym-saw` | d229e42 | 2025-05-25 | A standing and walking variant of the same |
| `ksim-kbot` | 517eae9 | 2025-04-25 | An earlier K-Bot training repository, superseded API |
| `ksim-legacy` | c1b92fe | 2024-12-12 | The pre-JAX simulation code, Stompy era |
| `kscale-assets` | f51d6ea | 2025-05-01 | Archived, carries the historical K-Bot v1 and v2 gain sets |
| `urdf2mjcf` | 7954f8b | 2025-09-26 | The converter that writes the model, and the reason damping is absent |
| `actuator` | 01c8f98 | 2025-09-14 | The RobStride driver, source of the firmware gain and rate limits |
| `kos-sim` | dd44e1b | 2025-05-06 | The deployment simulator, applies the metadata gains unscaled |
| `kinfer-sim` | 033c8f9 | 2025-10-24 | The policy inference simulator, likewise unscaled |
| `kinfer` | 0a3d8bf | 2025-10-07 | The model export format, carries no gains |
| `sysid` | af29fe7 | 2025-04-05 | Actuator system identification, Feetech only, no RobStride |
| `ktune` | bcf6694 | 2025-05-27 | Real to simulation tuning utility, Feetech oriented |
| `mujoco-scenes` | b221b81 | 2025-10-28 | Scene construction, no actuator content |
| `ksim-zbot` | ed1ec5e | 2025-04-22 | The Z-Bot task, the contrasting identified actuator model |
| `kbot-inference` | 9c7e382 | 2025-01-31 | Historical inference code |

The four repositories that carry the answer are `kbot-models`, `ksim`, `kbot-joystick` and `actuator`. The remainder either consume those parameters, produce them, or establish by their silence what has not been measured.

## 3. A caution on nomenclature

This workspace already contains a robot called KScale, documented at `tron1-rl-isaaclab-cozum/context/KScale.md`, and it is not the robot described here. That robot is a sole footed biped of twelve actuated degrees of freedom, six per leg including an ankle roll, massing 20.281 kg, parameterised in Isaac Lab from a URDF and analysed against a target damping ratio near 0.70. The K-Bot described here is a full humanoid of twenty actuated degrees of freedom, five per leg with a single ankle pitch and five per arm, massing 36.719 kg as measured from the shipped model, parameterised in MuJoCo and analysed by its authors against a target this document reconstructs as near 0.96. The two share a vendor name and nothing else, and any figure carried from one document to the other is wrong by a factor the mass ratio does not predict.

The distinction matters for a second reason. The analytical order used below, effective inertia, then natural frequency and damping ratio, then stance load and sag, is deliberately the order that `KScale.md` and `BRS.md` use, so that the K-Bot may be compared against the robots this workspace already characterises. The conclusions differ sharply, and section 10 explains why.

## 4. Where the parameters live, and the contract between the layers

The MuJoCo model at `kbot-models/kbot/robot.mjcf` declares four default classes, one per actuator model, and each class sets three joint attributes and one actuator attribute. Lines 4 through 19 of that file give the whole of it, an armature, a friction loss, an actuator force range and a matching control range, repeated for the RobStride 00, 02, 03 and 04. The actuators themselves are plain torque sources, twenty `<motor>` elements at lines 193 through 212, each naming its joint and its class. No `<position>` element and no joint stiffness appears anywhere in the file, which establishes at once that the proportional derivative loop is not a MuJoCo actuator but a computation performed outside the physics step.

That computation is `ksim.PositionActuators`, at `ksim/ksim/actuators.py:96`, whose docstring at line 99 names it an MIT Cheetah style controller operating on position, after the impedance control interface that platform popularised [2]. Its control law is a single line at `ksim/ksim/actuators.py:213`, the torque being the product of the proportional gain and the position error plus the product of the derivative gain and the velocity error, clipped at `ksim/ksim/actuators.py:215-217` to a soft torque limit. The velocity target is identically zero, set at `ksim/ksim/actuators.py:208`, so the derivative term is a pure damper on joint velocity rather than a tracker of a commanded rate. This is the action interface that the legged reinforcement learning literature has settled on, in which the policy emits a joint position offset and a fixed gain servo converts it to torque [1].

The gains that law consumes come from neither the model nor the script. They are read at `ksim/ksim/actuators.py:132-134` from a `JointMetadata` record whose schema is at `ksim/ksim/types.py:129-134`, carrying a proportional gain, a derivative gain, an armature, a friction, an actuator type and a soft torque limit. That record is populated from `kbot-models/kbot/metadata.json`, a file shipped beside the model, and the same file is read by the deployment simulator at `kos-sim/kos_sim/simulator.py:124-129` and by the inference simulator at `kinfer-sim/kinfer_sim/actuators.py:118-119`. The metadata file is therefore the single point of truth for the gains and the only place a reader should look for them.

The absence of joint damping from the model has a specific and traceable cause. The converter writes a damping attribute only where the actuator metadata declares one, at `urdf2mjcf/urdf2mjcf/convert.py:248-249`, and the K-Bot metadata declares damping as null for all four actuator classes. The consequence is that `dof_damping` is identically zero across the whole articulation, which was confirmed by counting the occurrences of the attribute in all four shipped variants and finding none. Section 12 records what this does to one of the domain randomisers.

## 5. The joint gain table

The twenty joints and their published parameters are reproduced below verbatim from `kbot-models/kbot/metadata.json`. The four model variants, `kbot`, `kbot-headless`, `kbot-full-collisions` and `kbot-headless-full-collisions`, carry byte identical joint metadata differing only in whether the numbers are quoted as strings, and the copies vendored into `kbot-joystick/robot/kbot/metadata.json` and `ksim/examples/kbot/robot/kbot-headless/metadata.json` agree with them exactly. There is one gain set for this robot and it is the one below.

| Joint | Bus id | NN index | Actuator | kp, Nm/rad | kd, Nm s/rad | Soft torque limit, Nm | Range, degrees |
|---|---|---|---|---|---|---|---|
| dof_left_shoulder_pitch_03 | 11 | 5 | robstride_03 | 100.0 | 8.284 | 42.0 | -80 to 180 |
| dof_left_shoulder_roll_03 | 12 | 6 | robstride_03 | 100.0 | 8.257 | 42.0 | -20 to 95 |
| dof_left_shoulder_yaw_02 | 13 | 7 | robstride_02 | 40.0 | 0.945 | 11.9 | -95 to 95 |
| dof_left_elbow_02 | 14 | 8 | robstride_02 | 40.0 | 1.266 | 11.9 | -142 to 0 |
| dof_left_wrist_00 | 15 | 9 | robstride_00 | 20.0 | 0.295 | 9.8 | -100 to 100 |
| dof_right_shoulder_pitch_03 | 21 | 0 | robstride_03 | 100.0 | 8.284 | 42.0 | -180 to 80 |
| dof_right_shoulder_roll_03 | 22 | 1 | robstride_03 | 100.0 | 8.257 | 42.0 | -95 to 20 |
| dof_right_shoulder_yaw_02 | 23 | 2 | robstride_02 | 40.0 | 0.945 | 11.9 | -95 to 95 |
| dof_right_elbow_02 | 24 | 3 | robstride_02 | 40.0 | 1.266 | 11.9 | 0 to 142 |
| dof_right_wrist_00 | 25 | 4 | robstride_00 | 20.0 | 0.295 | 9.8 | -100 to 100 |
| dof_left_hip_pitch_04 | 31 | 15 | robstride_04 | 150.0 | 24.722 | 84.0 | -60 to 127 |
| dof_left_hip_roll_03 | 32 | 16 | robstride_03 | 200.0 | 26.387 | 42.0 | -12 to 130 |
| dof_left_hip_yaw_03 | 33 | 17 | robstride_03 | 100.0 | 3.419 | 42.0 | -90 to 90 |
| dof_left_knee_04 | 34 | 18 | robstride_04 | 150.0 | 8.654 | 84.0 | 0 to 155 |
| dof_left_ankle_02 | 35 | 19 | robstride_02 | 40.0 | 0.990 | 11.9 | -72 to 13 |
| dof_right_hip_pitch_04 | 41 | 10 | robstride_04 | 150.0 | 24.722 | 84.0 | -127 to 60 |
| dof_right_hip_roll_03 | 42 | 11 | robstride_03 | 200.0 | 26.387 | 42.0 | -130 to 12 |
| dof_right_hip_yaw_03 | 43 | 12 | robstride_03 | 100.0 | 3.419 | 42.0 | -90 to 90 |
| dof_right_knee_04 | 44 | 13 | robstride_04 | 150.0 | 8.654 | 84.0 | -155 to 0 |
| dof_right_ankle_02 | 45 | 14 | robstride_02 | 40.0 | 0.990 | 11.9 | -13 to 72 |

Four properties of this table are worth stating before any analysis touches it. The gains are perfectly left and right symmetric, so no asymmetry has been introduced by tuning. The proportional gains take only five distinct values, 20, 40, 100, 150 and 200, which is the signature of a hand chosen ladder rather than a per joint optimisation. The derivative gains, by contrast, are quoted to three decimal places and take a different value at almost every joint, which is the signature of a computed quantity, and section 9 recovers the computation. And the per joint armature and friction fields are null throughout, the metadata deferring those to the actuator class of section 6.

## 6. The actuator classes

The four RobStride models and their declared properties are below, read from the `actuator_type_to_metadata` block of the same metadata file and cross checked against the default classes at `kbot-models/kbot/robot.mjcf:4-19`, which agree exactly.

| Class | Peak torque, Nm | Soft limit, Nm | Ratio | Armature, kg m^2 | Friction loss, Nm | Declared max velocity, rad/s | Firmware velocity limit, rad/s | Firmware kd ceiling |
|---|---|---|---|---|---|---|---|---|
| robstride_00 | 14.0 | 9.8 | 0.700 | 0.001 | 0.1 | 27.227 | 33.0 | 5.0 |
| robstride_02 | 17.0 | 11.9 | 0.700 | 0.0042 | 0.1 | 37.699 | 44.0 | 5.0 |
| robstride_03 | 60.0 | 42.0 | 0.700 | 0.02 | 0.2 | 18.849 | 20.0 | 100.0 |
| robstride_04 | 120.0 | 84.0 | 0.700 | 0.04 | 0.2 | 17.488 | 15.0 | 100.0 |

The soft torque limit is exactly seven tenths of the peak torque at every class without exception, which identifies it as a derating policy rather than four independent judgements. The firmware limits in the last two columns are read from the driver, at `actuator/actuator/robstride/src/actuators/robstride00.rs:15-26` and the corresponding blocks in the sibling files, and they are the ceilings the on board impedance mode enforces. Two disagreements follow. The declared maximum velocity of the robstride_04 exceeds the firmware ceiling by 2.488 rad/s, so a simulation permitting that rate is permitting a rate the hardware will refuse, and this is the only class where the simulation is the more permissive of the two. And the derivative gain ceiling of 5.0 on the robstride_00 and robstride_02 classes is a real constraint on any future retuning of the ankle, the elbow, the shoulder yaw and the wrist, all four of which currently sit an order of magnitude below it but would not survive the derivative gains the hip joints carry.

The armature deserves separate emphasis because it is not a numerical convenience. MuJoCo adds it to the mass matrix diagonal, so it is part of the inertia the control loop actually sees, and the published result that reflected rotor inertia measurably shifts the best fit simulation gains against hardware is the reason it must be carried through every calculation below [3]. On this robot it dominates at the wrist, where the armature of 0.001 stands against a link contribution of 0.0004, and it is a two thirds correction at the ankle, where 0.0042 stands against 0.0027. At the hips it is negligible, 0.04 against 0.998. A derivation that omits it would misplace the distal joints by a factor of three and the proximal ones not at all, which is the worst possible pattern of error because it is invisible where it is checked.

## 7. The nominal pose and the initial state of a training episode

The pose from which every episode begins is defined in the training script rather than in the model, and it is identical across the two current repositories. It appears as `ZEROS` at `ksim-gym/train.py:23-44` and as `JOINT_BIASES` at `kbot-joystick/train.py:24-45`, in both cases ordered to match the neural network outputs, and the two lists agree joint for joint.

| Joint | Nominal, degrees | Nominal, radians |
|---|---|---|
| dof_left_hip_pitch_04 | 20.0 | 0.34907 |
| dof_left_hip_roll_03 | 0.0 | 0.0 |
| dof_left_hip_yaw_03 | 0.0 | 0.0 |
| dof_left_knee_04 | 50.0 | 0.87266 |
| dof_left_ankle_02 | -30.0 | -0.52360 |
| dof_right_hip_pitch_04 | -20.0 | -0.34907 |
| dof_right_hip_roll_03 | 0.0 | 0.0 |
| dof_right_hip_yaw_03 | 0.0 | 0.0 |
| dof_right_knee_04 | -50.0 | -0.87266 |
| dof_right_ankle_02 | 30.0 | 0.52360 |
| dof_left_shoulder_pitch_03 | 0.0 | 0.0 |
| dof_left_shoulder_roll_03 | 10.0 | 0.17453 |
| dof_left_shoulder_yaw_02 | 0.0 | 0.0 |
| dof_left_elbow_02 | -90.0 | -1.57080 |
| dof_left_wrist_00 | 0.0 | 0.0 |
| dof_right_shoulder_pitch_03 | 0.0 | 0.0 |
| dof_right_shoulder_roll_03 | -10.0 | -0.17453 |
| dof_right_shoulder_yaw_02 | 0.0 | 0.0 |
| dof_right_elbow_02 | 90.0 | 1.57080 |
| dof_right_wrist_00 | 0.0 | 0.0 |

The sign convention is mirrored between the sides, so the two legs adopt the same physical crouch and the two arms the same physical posture despite the opposite numerical signs. The stance is a symmetric crouch of twenty degrees at the hip pitch, fifty at the knee and thirty at the ankle, which closes the leg chain such that the shank leans forward of the thigh and the sole remains level, and the arms hang with the elbows flexed to a right angle and the shoulders rolled outward by ten degrees. It is a conventional humanoid ready posture and it is the same posture that serves three distinct roles in the pipeline, the reset target, the additive bias the policy's action is measured against, and the pose the deviation penalties are computed from.

The base is not reset at all. The reset list at `kbot-joystick/train.py:1146-1153` randomises joint positions, joint velocities, planar base velocity, heading and planar position, and the joint position reset at `ksim/ksim/resets.py:132-143` writes only `qpos[7:]`, leaving the first seven entries of the configuration untouched. The base pose therefore comes from the model's own rest configuration, which is the `<body name="base">` position at `kbot-models/kbot/robot.mjcf:64`, giving a height of 0.80884103 m and an identity orientation.

That number invites a check which the tree does not itself perform. Placing the model at the nominal joint pose and measuring the lowest point of the two sole capsules gives 0.04291 m above the ground plane. An episode therefore begins with the robot suspended 42.9 mm in the air, dropping onto its feet within the first few control steps rather than starting in contact, and the base height at which the soles actually rest is 0.76593 m. This is a deliberate and common arrangement rather than an error, since a small drop removes any dependence on an exactly consistent initial contact state, but it should be known to anyone comparing the first steps of an episode against a standing reference.

The randomisation about this pose is uniform and curriculum scaled. The joint position reset carries a scale of 0.1 rad and the joint velocity reset a scale of 2.0 rad/s at `kbot-joystick/train.py:1148-1149`, both multiplied by the curriculum level at `ksim/ksim/resets.py:134-135` and `ksim/ksim/resets.py:185-186`, so early training begins almost exactly at the nominal pose and the spread opens as the curriculum advances. The reference task in `ksim-gym` is gentler, carrying a joint position scale of 0.1 with no velocity randomisation beyond the default, at `ksim-gym/train.py:397-398`.

## 8. Effective inertia

The inertia a joint's control loop sees is the corresponding diagonal entry of the mass matrix, which for a floating base articulation is the composite inertia of the entire distal chain about that joint's axis plus the armature. It was computed by loading `kbot-models/kbot/robot.mjcf` in MuJoCo 3.12.0, setting the twenty joints to the nominal pose of section 7, and reading the diagonal of the full mass matrix. The model reports a total mass of 36.719 kg and a physics timestep default of 0.002 s.

The distinction between this quantity and the inertia the joint sees in stance is the subject of section 10, and until that section is reached every figure below should be read as the free limb or swing quantity, which is what the mass matrix of a floating base model returns.

| Joint | Armature | Link contribution | Effective inertia at nominal, kg m^2 | Effective inertia at zero pose, kg m^2 |
|---|---|---|---|---|
| dof_left_shoulder_pitch_03 | 0.0200 | 0.1667 | 0.1867 | 0.2388 |
| dof_left_shoulder_roll_03 | 0.0200 | 0.1614 | 0.1814 | 0.2378 |
| dof_left_shoulder_yaw_02 | 0.0042 | 0.0150 | 0.0192 | 0.0063 |
| dof_left_elbow_02 | 0.0042 | 0.0122 | 0.0164 | 0.0164 |
| dof_left_wrist_00 | 0.0010 | 0.0004 | 0.0014 | 0.0014 |
| dof_right_shoulder_pitch_03 | 0.0200 | 0.1615 | 0.1815 | 0.2332 |
| dof_right_shoulder_roll_03 | 0.0200 | 0.1557 | 0.1757 | 0.2322 |
| dof_right_shoulder_yaw_02 | 0.0042 | 0.0148 | 0.0190 | 0.0061 |
| dof_right_elbow_02 | 0.0042 | 0.0122 | 0.0164 | 0.0164 |
| dof_right_wrist_00 | 0.0010 | 0.0004 | 0.0014 | 0.0014 |
| dof_left_hip_pitch_04 | 0.0400 | 0.9979 | 1.0379 | 1.0915 |
| dof_left_hip_roll_03 | 0.0200 | 0.8215 | 0.8415 | 0.9364 |
| dof_left_hip_yaw_03 | 0.0200 | 0.0576 | 0.0776 | 0.0298 |
| dof_left_knee_04 | 0.0400 | 0.0961 | 0.1361 | 0.1400 |
| dof_left_ankle_02 | 0.0042 | 0.0027 | 0.0069 | 0.0069 |
| dof_right_hip_pitch_04 | 0.0400 | 0.9978 | 1.0378 | 1.0914 |
| dof_right_hip_roll_03 | 0.0200 | 0.8215 | 0.8415 | 0.9363 |
| dof_right_hip_yaw_03 | 0.0200 | 0.0576 | 0.0776 | 0.0298 |
| dof_right_knee_04 | 0.0400 | 0.0961 | 0.1361 | 0.1400 |
| dof_right_ankle_02 | 0.0042 | 0.0026 | 0.0068 | 0.0068 |

The two pose columns differ most at the joints whose axis is vertical or nearly so. The hip yaw sees 0.0298 with the leg straight and 0.0776 with it folded into the nominal crouch, a factor of 2.6, because folding the knee swings the shank and foot mass away from the yaw axis. The shoulder yaw moves by a factor of three for the same reason, the elbow flexion of ninety degrees placing the forearm perpendicular to the yaw axis rather than along it. Every other joint moves by less than a quarter. This sensitivity is the key to section 9.

## 9. Natural frequency, damping ratio, and the rule that produced the published gains

Treating each joint as an isolated second order system of inertia I under the control law of section 4 gives a natural frequency of the square root of the proportional gain divided by the inertia, and a damping ratio of the derivative gain divided by twice the square root of the product of the proportional gain and the inertia. The critical derivative gain, the value placing the joint exactly at unit damping ratio, is twice that square root.

Evaluating both at the nominal pose gives the damping ratios in the third column below, and evaluating them at the zero configuration, meaning every joint at its own zero rather than at the nominal crouch, gives the fourth.

| Joint | Natural frequency at nominal, Hz | Damping ratio at nominal | Damping ratio at zero pose |
|---|---|---|---|
| dof_left_shoulder_pitch_03 | 3.68 | 0.958 | 0.848 |
| dof_left_shoulder_roll_03 | 3.74 | 0.969 | 0.847 |
| dof_left_shoulder_yaw_02 | 7.27 | 0.539 | 0.941 |
| dof_left_elbow_02 | 7.85 | 0.781 | 0.781 |
| dof_left_wrist_00 | 18.94 | 0.878 | 0.878 |
| dof_right_shoulder_pitch_03 | 3.74 | 0.972 | 0.858 |
| dof_right_shoulder_roll_03 | 3.80 | 0.985 | 0.857 |
| dof_right_shoulder_yaw_02 | 7.30 | 0.542 | 0.955 |
| dof_right_elbow_02 | 7.85 | 0.781 | 0.781 |
| dof_right_wrist_00 | 18.94 | 0.878 | 0.878 |
| dof_left_hip_pitch_04 | 1.91 | 0.991 | 0.966 |
| dof_left_hip_roll_03 | 2.45 | 1.017 | 0.964 |
| dof_left_hip_yaw_03 | 5.71 | 0.614 | 0.990 |
| dof_left_knee_04 | 5.28 | 0.958 | 0.944 |
| dof_left_ankle_02 | 12.16 | 0.946 | 0.946 |
| dof_right_hip_pitch_04 | 1.91 | 0.991 | 0.966 |
| dof_right_hip_roll_03 | 2.45 | 1.017 | 0.964 |
| dof_right_hip_yaw_03 | 5.71 | 0.614 | 0.990 |
| dof_right_knee_04 | 5.28 | 0.958 | 0.944 |
| dof_right_ankle_02 | 12.16 | 0.946 | 0.946 |

The zero pose column is the one that explains the gain set. Across the five leg joints it runs from 0.944 to 0.990 with a mean of 0.962 and a spread of under five per cent, which is far too tight to be an accident given that the underlying inertias span a factor of a hundred and sixty and the derivative gains a factor of twenty seven. The published derivative gains are therefore the output of the formula for critical damping applied at a damping ratio near 0.96 against the zero configuration mass matrix, and the three decimal places they are quoted to are the residue of that computation rather than a claim to that precision.

Two consequences follow immediately. The first is that the derivation was performed at the zero configuration and not at the nominal pose from which training actually begins, which is why the hip yaw and the shoulder yaw fall to 0.614 and 0.540 once the crouch is adopted, both of them substantially underdamped in the pose the robot spends its whole episode near. Those two joints are precisely the ones whose inertia section 8 showed to be pose sensitive, and they are the only two whose damping ratio moves by more than a tenth between the two poses. The second is that the arm joints do not fit the rule as cleanly, the shoulder pitch and roll sitting at 0.85 and the elbow at 0.78 in the zero configuration, which suggests the arm gains were computed against an earlier version of the model whose arm masses differed. The vendored copy in `kbot-joystick/robot/kbot/robot.mjcf` supports that reading, its forearm inertial block at line 148 declaring a mass of 1.118293 kg against 0.469028 kg in `kbot-models/kbot/robot.mjcf`, a factor of 2.4 in the very links whose gains fit worst.

The whole set sits materially above the 0.7 to 1.0 damping ratio band that the Isaac Sim gain tuning guidance recommends and that this workspace targets from below [4], clustering instead at its upper edge. That is a defensible choice rather than an error, an overdamped joint costing response speed where an underdamped one costs stability, but it is a different choice from the one this workspace has made for its own robots and the difference should not be carried across silently.

For the proportional gains there is no comparable rule to recover, the five value ladder of section 5 being a design decision rather than a computation. It may still be placed against the published range. The Mini Cheetah is trained with a proportional gain of 17 Nm/rad [5] and the A1 with 55 [6], and the K-Bot's leg gains of 150 and 200 sit far above both, which is consistent with its being some four times the mass of either. The survey position that a large proportional gain leads to instabilities in training while a low one leaves the joint behaving as a torque controller [7] frames the choice but does not resolve where in that range a 36 kg humanoid should sit, and the K-Bot gains are best understood as the load driven answer that section 10 derives.

## 10. Stance load, sag, and the inertia the gains were not derived against

The analysis of section 9 answers how the joint behaves when the limb swings freely. It does not answer whether the joint can hold the robot up, and those are two different questions answered by two different inertias, a distinction this workspace established at `tron1-rl-isaaclab-cozum/context/KScale.md` section 3 at the cost of a full training run.

The static load was computed by placing the model at the nominal pose, locating the centre of pressure of each sole as the mean of that foot's two collision capsules projected to the contact plane, and taking the moment of the supported weight about each leg joint axis. The robot weighs 360.21 N, so each foot carries 180.1 N in a balanced double support and the full weight in single support.

| Joint | Double support torque, Nm | Single support torque, Nm | Fraction of soft limit, double support | Sag at published kp, degrees | Minimum kp for a 0.05 rad sag, Nm/rad |
|---|---|---|---|---|---|
| dof_left_hip_pitch_04 | 4.12 | 8.24 | 0.05 | 1.57 | 82.4 |
| dof_left_hip_roll_03 | 0.87 | 1.73 | 0.02 | 0.25 | 17.3 |
| dof_left_hip_yaw_03 | 0.32 | 0.63 | 0.01 | 0.18 | 6.3 |
| dof_left_knee_04 | 16.55 | 33.10 | 0.20 | 6.32 | 331.0 |
| dof_left_ankle_02 | 4.89 | 9.77 | 0.41 | 7.00 | 97.7 |
| dof_right_hip_pitch_04 | 3.86 | 7.72 | 0.05 | 1.47 | 77.2 |
| dof_right_hip_roll_03 | 0.87 | 1.75 | 0.02 | 0.25 | 17.5 |
| dof_right_hip_yaw_03 | 0.32 | 0.64 | 0.01 | 0.18 | 6.4 |
| dof_right_knee_04 | 16.81 | 33.62 | 0.20 | 6.42 | 336.2 |
| dof_right_ankle_02 | 4.62 | 9.25 | 0.39 | 6.62 | 92.5 |

Three joints clear the workspace's customary sag budget of 0.05 rad comfortably and two do not. The hip pitch, hip roll and hip yaw hold their nominal angles to within 1.6 degrees under the standing load, which is well inside the budget. The knee sags 6.4 degrees and the ankle 7.0, both roughly twice the budget, and in single support both double again. This was confirmed dynamically by dropping the model onto a ground plane under the published gains and letting it settle, which produced a knee sag of 8.1 degrees, the discrepancy against the static figure being the additional load the sagging pose itself transfers.

The ankle figure carries a second warning. At 41 per cent of its soft torque limit merely standing still, the ankle has less than a factor of two and a half of headroom before saturation, and a policy commanding a substantial ankle offset while bearing load will clip. This is the same condition the BRS and KScale analyses record and accept for their own robots, on the ground that a trained policy commands small increments around the default pose, and it is accepted here on the same ground, but it does place a firm upper bound on any future increase of the ankle proportional gain.

The damping ratio is affected far more severely than the stiffness. Computing for each leg joint the inertia of everything above it, taken about that joint's axis with the sole pinned, gives the stance inertia against which the joint's damping must actually work.

| Joint | Swing inertia, kg m^2 | Stance inertia, kg m^2 | Ratio | Swing damping ratio | Stance damping ratio |
|---|---|---|---|---|---|
| dof_left_hip_pitch_04 | 1.0379 | 3.1074 | 2.99 | 0.991 | 0.573 |
| dof_left_hip_roll_03 | 0.8415 | 4.2619 | 5.06 | 1.017 | 0.452 |
| dof_left_hip_yaw_03 | 0.0776 | 1.6922 | 21.80 | 0.614 | 0.131 |
| dof_left_knee_04 | 0.1361 | 9.9613 | 73.20 | 0.958 | 0.112 |
| dof_left_ankle_02 | 0.0069 | 20.1664 | 2943.84 | 0.946 | 0.017 |
| dof_right_hip_pitch_04 | 1.0378 | 3.1075 | 2.99 | 0.991 | 0.573 |
| dof_right_hip_roll_03 | 0.8415 | 4.2775 | 5.08 | 1.017 | 0.451 |
| dof_right_hip_yaw_03 | 0.0776 | 1.7076 | 22.00 | 0.614 | 0.131 |
| dof_right_knee_04 | 0.1361 | 9.9613 | 73.21 | 0.958 | 0.112 |
| dof_right_ankle_02 | 0.0068 | 20.1664 | 2944.52 | 0.946 | 0.017 |

The ratio rises monotonically with distality, which is the expected pattern since a distal joint has little limb below it and the whole body above, and it reaches nearly three thousand at the ankle. Because the damping ratio falls as the square root of the inertia, a joint tuned to 0.95 in the air arrives in stance at 0.55 at the hip pitch, 0.11 at the knee and 0.017 at the ankle. On this measure the K-Bot's ankle is essentially undamped whenever it is carrying weight.

This figure must be read with its assumption stated plainly. The stance inertia above treats the sole as rigidly pinned and the entire body as rotating about that single joint, with every other joint locked, which is the convention `KScale.md` adopts and which overstates the constrained inertia because in double support the two legs form a closed chain and the remaining joints share the resistance. The true value lies between the swing and stance columns and nearer the latter. The conclusion is not that the robot cannot stand, since the training results published in `kbot-joystick` demonstrate that it does, but that the published damping was derived against the free limb and that the margin it appears to carry in section 9 is not the margin the joint enjoys in contact. Section 11 offers a reading of the training script that is consistent with the authors having discovered this empirically.

## 11. The scale factors, and the divergence between training and deployment

The current locomotion policy does not use the published gains. At `kbot-joystick/train.py:1097-1103` the actuator is constructed with a proportional scale of 1.4 and a derivative scale of 1.4, carrying the inline comment that a scale of 2.0 works but is unstable in edge cases. The example task vendored inside the library repeats the same two values at `ksim/examples/kbot/train.py:1095-1102`. The reference task in `ksim-gym`, by contrast, constructs the actuator with no scales at all at `ksim-gym/train.py:367-370`, so it trains against the published gains unmodified, and the same is true of `ksim-gym-saw` at `ksim-gym-saw/train.py:554-557`.

Applying the pair of scales uniformly preserves neither the natural frequency nor the damping ratio. Because both gains are multiplied by the same factor, the natural frequency rises by the square root of 1.4, a factor of 1.183, and the damping ratio rises by the same factor, carrying the leg joints from a nominal pose range of 0.95 to 1.02 into a range of 1.13 to 1.20 and the whole articulation into the overdamped regime. The stiffness increase is the part that does real work, reducing the knee sag of section 10 from 6.4 degrees to 4.5 and the ankle from 7.0 to 5.0, which is precisely the remedy the stance analysis prescribes and which supports the reading that the factor was found empirically in response to a robot that stood too low.

The comment about instability at a scale of 2.0 admits a quantitative check. The proportional derivative law is evaluated once per physics step rather than once per control step, the actuator update interval defaulting to a single physics step at `ksim/ksim/engine.py:111-115` and no training script setting it otherwise, so the loop runs at 250 Hz in `kbot-joystick`, whose physics timestep is 0.004 s at `kbot-joystick/train.py:1775`, and at 500 Hz in `ksim-gym`, whose timestep is 0.002 s at `ksim-gym/train.py:650`. The dimensionless number governing explicit integration of a velocity feedback term is the derivative gain multiplied by the timestep and divided by the inertia, which must remain below two.

| Joint | Number at published gains, dt 0.004 | At scale 1.4 | At scale 2.0 |
|---|---|---|---|
| dof_left_wrist_00 | 0.84 | 1.17 | 1.68 |
| dof_left_ankle_02 | 0.58 | 0.81 | 1.16 |
| dof_left_elbow_02 | 0.31 | 0.43 | 0.62 |
| dof_left_knee_04 | 0.25 | 0.36 | 0.51 |
| dof_left_shoulder_yaw_02 | 0.20 | 0.28 | 0.40 |
| dof_left_hip_roll_03 | 0.13 | 0.18 | 0.26 |
| dof_left_hip_pitch_04 | 0.10 | 0.13 | 0.18 |

The wrist is the binding joint, reaching 1.68 of a bound of 2 at the scale the comment describes as unstable, and it is the binding joint because it combines the smallest inertia on the robot with the largest armature to link inertia ratio. The relationship is consistent with the reported instability without proving it, since the true stability boundary of the coupled articulation is not the boundary of the isolated joint, and it is offered here as the most economical explanation available from the tree rather than as a demonstration.

The divergence that matters most is not inside training at all. The deployment simulator applies the metadata gains without any scale, at `kos-sim/kos_sim/simulator.py:242-252`, and the inference simulator does the same at `kinfer-sim/kinfer_sim/actuators.py:118-119` and `kinfer-sim/kinfer_sim/actuators.py:140`. A policy trained in `kbot-joystick` therefore learns against a servo forty per cent stiffer than the servo it will meet on deployment unless something outside the cloned tree writes the scaled gains to the firmware, and nothing in the cloned tree does. The checkpoint conversion at `kbot-joystick/convert.py:67` builds a metadata record for export and does not carry the gains at all. This should be treated as an open question about the deployment path rather than as a confirmed defect, since the repositories that would settle it, the robot's own configuration files, are not among those published.

## 12. Domain randomisation of the actuator parameters

The randomisers applied in the current locomotion task are listed at `kbot-joystick/train.py:1108-1113` and their ranges are read from `ksim/ksim/randomization.py`.

| Randomiser | Range | Definition | Effect on the K-Bot |
|---|---|---|---|
| StaticFrictionRandomizer | 0.5 to 2.0 of nominal | `ksim/ksim/randomization.py:60-71` | Scales the 0.1 and 0.2 Nm joint friction losses |
| ArmatureRandomizer | 0.95 to 1.05 of nominal | `ksim/ksim/randomization.py:111-122` | Scales the four class armatures by five per cent |
| JointDampingRandomizer | 0.5 to 2.5 of nominal | `ksim/ksim/randomization.py:229-241`, range set at `kbot-joystick/train.py:1110` | None, the nominal being zero |
| FloorFrictionRandomizer | 0.5 to 1.5 | `ksim/ksim/randomization.py:79-84` | Ground contact only |
| kp_scale and kd_scale | Fixed at 1.4, magnitude zero | `ksim/ksim/actuators.py:162-166` | No randomisation, a constant multiplier |
| torque_limit_scale | Passed as a low bound of 0.5 | `kbot-joystick/train.py:1102` | Randomises the clipping threshold downward |

Two entries deserve comment. The joint damping randomiser is a complete no operation on this robot, because it multiplies `dof_damping` and section 4 established that `dof_damping` is identically zero, so the widened range of 0.5 to 2.5 that the training script deliberately sets in place of the library default of 0.9 to 1.1 achieves nothing whatever. Whoever widened it intended to randomise the joint viscous damping and in fact randomised nothing, and the only viscous term the K-Bot possesses is the derivative gain of the control law, which no randomiser touches.

The second is that the proportional and derivative scales are not randomised despite being expressed through the randomisation machinery. The actuator accepts either a float or a random variable at `ksim/ksim/actuators.py:162-166`, and a float is promoted to a uniform variable of zero magnitude, so passing 1.4 fixes the value rather than centring a distribution on it. Randomising the servo gains is among the standard measures for closing the transfer gap [8], and this stack does not do it. What it randomises instead is the armature, at a narrow five per cent, and the torque ceiling, at a wide fifty per cent downward.

## 13. What is absent

Three absences in the published tree are as informative as the parameters themselves.

There is no system identification for the RobStride actuators. The `sysid` repository contains only Feetech STS3215 and STS3250 results, which belong to the Z-Bot, and the `sysid` field of every RobStride record in the K-Bot metadata is the empty string. The contrast is instructive because the Feetech path shows what an identified actuator record looks like in this stack, `kscale-assets/actuators/feetech_sts3250.json` carrying a torque constant, a winding resistance, a supply voltage, a maximum duty cycle and an error gain alongside the armature and damping, and `ksim-zbot/ksim_zbot/zbot2/common.py:133-197` implementing a controller that computes a duty cycle, converts it to volts and then to torque through the electrical model. The K-Bot has no such model and is simulated as an ideal torque source behind a clipped proportional derivative law. Closing that gap is what an actuator network addresses [9], and nothing in the published tree attempts it.

There is no joint viscous damping, for the traceable reason given in section 4, and consequently no dissipation at the joints other than the constant Coulomb friction loss and the derivative term of the control law. A joint at rest under load therefore sits inside a friction deadband whose width is the friction loss divided by the proportional gain, which is 0.14 degrees at the ankle and 0.06 degrees at the hip roll, small enough to be neglected but not zero.

There is no backlash. The converter can insert it, adding a parallel joint with a damping of 0.01 and an armature of 0.01 over a specified angular range at `urdf2mjcf/urdf2mjcf/postprocess/add_backlash.py:37-41`, but the K-Bot conversion does not request it, the shipped model containing no such joints.

## 14. Reproduction

Every derived figure above was computed with MuJoCo 3.12.0 in an isolated virtual environment against the unmodified `kbot-models/kbot/robot.mjcf`. The mass matrix diagonals of section 8 and the frequencies and damping ratios of section 9 come from setting the twenty joints to the pose of section 7 and reading the full mass matrix. The stance torques of section 10 come from the moment of the supported weight about each joint axis, with the sole centre of pressure taken as the mean of that foot's two collision capsule centres projected to the contact plane. The stance inertias come from the parallel axis sum of every body not in the joint's own subtree, taken about that joint's axis, plus the armature. The settling test quoted in section 10 integrates the published control law at a timestep of 0.001 s against a ground plane placed at the sole height.

## 15. Consolidated summary

The table below carries every per joint quantity this document establishes, so that a reader needing a single reference need consult no other section. Inertias are in kg m^2, torques in Nm, frequencies in Hz and angles in degrees. The damping ratio columns are at the nominal pose, at the zero configuration and in pinned foot stance respectively, the last being defined only for the leg joints.

| Joint | Actuator | kp | kd | tau_soft | Armature | Nominal angle | I_nominal | I_zero | I_stance | f_n | zeta_nom | zeta_zero | zeta_stance | Stance torque | Sag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dof_left_hip_pitch_04 | RS04 | 150 | 24.722 | 84.0 | 0.0400 | 20.0 | 1.0379 | 1.0915 | 3.1074 | 1.91 | 0.991 | 0.966 | 0.573 | 4.12 | 1.57 |
| dof_left_hip_roll_03 | RS03 | 200 | 26.387 | 42.0 | 0.0200 | 0.0 | 0.8415 | 0.9364 | 4.2619 | 2.45 | 1.017 | 0.964 | 0.452 | 0.87 | 0.25 |
| dof_left_hip_yaw_03 | RS03 | 100 | 3.419 | 42.0 | 0.0200 | 0.0 | 0.0776 | 0.0298 | 1.6922 | 5.71 | 0.614 | 0.990 | 0.131 | 0.32 | 0.18 |
| dof_left_knee_04 | RS04 | 150 | 8.654 | 84.0 | 0.0400 | 50.0 | 0.1361 | 0.1400 | 9.9613 | 5.28 | 0.958 | 0.944 | 0.112 | 16.55 | 6.32 |
| dof_left_ankle_02 | RS02 | 40 | 0.990 | 11.9 | 0.0042 | -30.0 | 0.0069 | 0.0069 | 20.1664 | 12.16 | 0.946 | 0.946 | 0.017 | 4.89 | 7.00 |
| dof_right_hip_pitch_04 | RS04 | 150 | 24.722 | 84.0 | 0.0400 | -20.0 | 1.0378 | 1.0914 | 3.1075 | 1.91 | 0.991 | 0.966 | 0.573 | 3.86 | 1.47 |
| dof_right_hip_roll_03 | RS03 | 200 | 26.387 | 42.0 | 0.0200 | 0.0 | 0.8415 | 0.9363 | 4.2775 | 2.45 | 1.017 | 0.964 | 0.451 | 0.87 | 0.25 |
| dof_right_hip_yaw_03 | RS03 | 100 | 3.419 | 42.0 | 0.0200 | 0.0 | 0.0776 | 0.0298 | 1.7076 | 5.71 | 0.614 | 0.990 | 0.131 | 0.32 | 0.18 |
| dof_right_knee_04 | RS04 | 150 | 8.654 | 84.0 | 0.0400 | -50.0 | 0.1361 | 0.1400 | 9.9613 | 5.28 | 0.958 | 0.944 | 0.112 | 16.81 | 6.42 |
| dof_right_ankle_02 | RS02 | 40 | 0.990 | 11.9 | 0.0042 | 30.0 | 0.0068 | 0.0068 | 20.1664 | 12.16 | 0.946 | 0.946 | 0.017 | 4.62 | 6.62 |
| dof_left_shoulder_pitch_03 | RS03 | 100 | 8.284 | 42.0 | 0.0200 | 0.0 | 0.1867 | 0.2388 | n/a | 3.68 | 0.958 | 0.848 | n/a | 0.87 | 0.50 |
| dof_left_shoulder_roll_03 | RS03 | 100 | 8.257 | 42.0 | 0.0200 | 10.0 | 0.1814 | 0.2378 | n/a | 3.74 | 0.969 | 0.847 | n/a | 1.22 | 0.70 |
| dof_left_shoulder_yaw_02 | RS02 | 40 | 0.945 | 11.9 | 0.0042 | 0.0 | 0.0192 | 0.0063 | n/a | 7.27 | 0.539 | 0.941 | n/a | 0.18 | 0.26 |
| dof_left_elbow_02 | RS02 | 40 | 1.266 | 11.9 | 0.0042 | -90.0 | 0.0164 | 0.0164 | n/a | 7.85 | 0.781 | 0.781 | n/a | 0.97 | 1.38 |
| dof_left_wrist_00 | RS00 | 20 | 0.295 | 9.8 | 0.0010 | 0.0 | 0.0014 | 0.0014 | n/a | 18.94 | 0.878 | 0.878 | n/a | 0.00 | 0.00 |
| dof_right_shoulder_pitch_03 | RS03 | 100 | 8.284 | 42.0 | 0.0200 | 0.0 | 0.1815 | 0.2332 | n/a | 3.74 | 0.972 | 0.858 | n/a | 0.87 | 0.50 |
| dof_right_shoulder_roll_03 | RS03 | 100 | 8.257 | 42.0 | 0.0200 | -10.0 | 0.1757 | 0.2322 | n/a | 3.80 | 0.985 | 0.857 | n/a | 1.15 | 0.66 |
| dof_right_shoulder_yaw_02 | RS02 | 40 | 0.945 | 11.9 | 0.0042 | 0.0 | 0.0190 | 0.0061 | n/a | 7.30 | 0.542 | 0.955 | n/a | 0.18 | 0.26 |
| dof_right_elbow_02 | RS02 | 40 | 1.266 | 11.9 | 0.0042 | 90.0 | 0.0164 | 0.0164 | n/a | 7.85 | 0.781 | 0.781 | n/a | 0.97 | 1.38 |
| dof_right_wrist_00 | RS00 | 20 | 0.295 | 9.8 | 0.0010 | 0.0 | 0.0014 | 0.0014 | n/a | 18.94 | 0.878 | 0.878 | n/a | 0.00 | 0.00 |

For the arm joints the stance torque and sag columns give the load and deflection with the torso held fixed and the limb hanging, since those joints carry no ground reaction. The robot as a whole masses 36.719 kg, weighs 360.21 N, spawns with its base at 0.80884 m and its soles 0.04291 m clear of the ground, rests with its base at 0.76593 m, is controlled at 50 Hz against a physics rate of 250 Hz in the current locomotion task, and is simulated with no joint damping anywhere.

## 16. The historical gain sets

Two earlier gain sets survive in the archived `kscale-assets` repository and are recorded here so that a reader encountering them is not misled into treating them as current. The K-Bot v1 set at `kscale-assets/kbot-v1/metadata.json` carries leg proportional gains of 100, 50, 50, 100 and 20 against the current 150, 200, 100, 150 and 40, and derivative gains quoted to five significant figures which fit the same critical damping construction against a different and lighter model. Its armatures are also smaller by a factor of four to eight, 0.0005 through 0.007 against the current 0.001 through 0.04, and its torque ceilings are looser, the class named motor_03 permitting 100 Nm against the present 60. The K-Bot v2 set at `kscale-assets/kbot-v2/metadata.json` reaches the current proportional ladder exactly but pairs it with a crude derivative ladder of 4.0, 8.0 and 2.0 applied by actuator class rather than computed per joint. The progression from v2 to the present set is therefore the replacement of a class wise derivative guess by the per joint critical damping computation that section 9 recovers, at unchanged proportional gains, and it is the single most consequential change the gain history records.

## 17. Findings

The published gains are a five value proportional ladder chosen by hand, paired with derivative gains computed to place each joint at a damping ratio near 0.96 against the free limb inertia at the zero configuration. The rule fits the five leg joints to within five per cent and the arm joints considerably less well, the misfit at the shoulder and elbow being consistent with a derivation performed against a forearm 2.4 times heavier than the one the current model ships.

The derivation was performed against the wrong configuration and the wrong inertia for the use the gains are put to. Evaluated at the nominal crouch rather than the zero configuration, the hip yaw and shoulder yaw fall to damping ratios of 0.61 and 0.54, and evaluated against the inertia a pinned foot presents rather than a free limb, the knee falls to 0.11 and the ankle to 0.017. The proportional gains reflect the same omission, the knee and ankle sagging 6.4 and 7.0 degrees under a static double support load against a customary budget of 2.9.

The proportional and derivative scale of 1.4 that the current locomotion task applies is the empirical correction to that omission, raising the stiffness where the stance analysis says stiffness is short. It is applied in training and in nothing else, the deployment and inference simulators reading the unscaled metadata, and the tree as published does not show how the two are reconciled.

The joint damping randomiser is inert on this robot, multiplying a quantity that the converter never wrote because the metadata never declared it, and the widened range the training script sets for it has no effect of any kind.

No system identification exists for the RobStride actuators, and the K-Bot is accordingly simulated as an ideal clipped torque source, where the Z-Bot in the same stack is simulated through an electrical model fitted to bench data.

## 18. Bibliography

1. Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Proceedings of Machine Learning Research 164, 91-100. arXiv:2109.11978.
2. Katz, B., Di Carlo, J., Kim, S. (2019). Mini Cheetah, A Platform for Pushing the Limits of Dynamic Quadruped Control. IEEE International Conference on Robotics and Automation (ICRA) 2019, 6295-6301. DOI 10.1109/ICRA.2019.8793865.
3. Guan, N., Yu, S., Zhu, S., Kim, D. (2024). Impedance Matching, Enabling an RL-Based Running Jump in a Quadruped Robot. Ubiquitous Robots 2024. arXiv:2404.15096.
4. NVIDIA. Tutorial 6, Joint Gains Tuning. Isaac Sim OpenUSD Tuning Tutorials, `docs.isaacsim.omniverse.nvidia.com`.
5. Ji, G., Mun, J., Kim, H., Hwangbo, J. (2022). Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. IEEE Robotics and Automation Letters 7(2). arXiv:2202.05481.
6. Kumar, A., Fu, Z., Pathak, D., Malik, J. (2021). RMA, Rapid Motor Adaptation for Legged Robots. Robotics, Science and Systems 2021. arXiv:2107.04034.
7. Spoljaric, Yashuai and Lee (2025). Variable Stiffness for Robust Locomotion through Reinforcement Learning. 16th IFAC Joint Symposia of Mechatronics and Robotics. arXiv:2502.09436. Author list as recorded in `literature.md` cluster 23, not expanded here.
8. Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Hafner, D., Bohez, S., Vanhoucke, V. (2018). Sim-to-Real, Learning Agile Locomotion For Quadruped Robots. Robotics, Science and Systems XIV. arXiv:1804.10332.
9. Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., Hutter, M. (2019). Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26), eaau5872. arXiv:1901.08652, DOI 10.1126/scirobotics.aau5872.
