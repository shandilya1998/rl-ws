# SD_BRS1 Symmetry Data Augmentation, Implementation Plan

> Status, verified against the live sources on 2026-07-30. Implemented. The robot specific mirror exists as `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/symmetry/brs.py` and is wired into the agent configuration at `agents/limx_rsl_rl_ppo_cfg.py`, which imports it and passes an `RslRlSymmetryCfg`. The run that first enabled it, 2026-07-22_11-36-53, produced the first SD_BRS1 policy to walk with a coordinated alternating gait, and its evaluation is analysed in [NATURAL_GAIT_PLAN.md](NATURAL_GAIT_PLAN.md). The requirement that the critic group be mirrored as well as the policy group, raised in section 3.5 of this document, is recorded as established fact in `../context/rsl_rl.md`. See [README.md](README.md) for the full register.

This document is the complete implementation brief for enabling left-right symmetry on the SD_BRS1 biped through the rsl_rl symmetry hook. It carries the full codebase investigation as validation material so that the implementing agent needs no further exploration, followed by the exact code and the precise edits. The immediate goal is observation 1 from the run 2026-07-21_06-03-36 video, the asymmetric limping gait. Observations 2 and 3, the stiff straight legs and the standing shuffle, are recorded at the end for later phases and are out of scope here.

## 1. Problem grounding

The overnight run 2026-07-21_06-03-36 enabled the GaitReward contact clock at weight 40, and the clock succeeded at its one job, the cadence collapse is gone and the robot now alternates its feet rather than holding one leg. The tensorboard for that run, parsed over 21992 iterations, shows strong velocity tracking with rew_lin_vel_xy near 31.6, feet_air_time risen to 2.14, rew_no_fly risen to 12.2, and rew_foot_clearance risen to 10.3, so the policy genuinely steps. The residual gait defects are three, and they read directly off the same logs and video. The gait is asymmetric, a visible limp, corroborated by pen_flat_orientation pinned near minus 1.72 across the whole run, meaning the base leans rather than sitting level. The legs are stiff and near straight, corroborated by pen_joint_torque worsening from minus 0.48 to minus 1.09 and feet_slide worsening from minus 0.52 to minus 0.73 as the policy shuffles on straight legs. The robot shuffles its feet when it should stand, a consequence of the always on clock and the absence of a zero command gate.

This plan addresses the first defect only. The chosen instrument is the symmetry mechanism from "Leveraging Symmetry in RL-based Legged Locomotion Control" and its SymmLoco codebase, ported onto the RslRlSymmetryCfg interface that the user's rsl_rl already consumes.

## 2. Literature framing

The idea rests on the observation that a legged robot is symmetric across its sagittal plane, so its Markov decision process carries a discrete reflection symmetry, the order two group C2 equal to {e, g} with g squared equal to e. When the transition density, the initial state density, and the reward are all invariant under this reflection, the optimal policy is equivariant, g applied to the optimal action equals the optimal action at the reflected state, and the optimal value is invariant, the value at the reflected state equals the value at the original state. Unconstrained reinforcement learning ignores this structure and under explores the mirrored modality of the state space, which is why the learned gait drifts into the asymmetric limp seen in the run video. This framing and its equations are established in the primary paper, Su and colleagues, "Leveraging Symmetry in RL-based Legged Locomotion Control", arXiv 2403.17320, IROS 2024, whose SymmLoco code the user referenced.

The literature offers four practical ways to inject this symmetry, catalogued by the foundational work of Abdolhosseini and colleagues, "On Learning Symmetric Locomotion", Motion in Games 2019. Their names are DUP, LOSS, PHASE, and NET. DUP duplicates every transition through the mirror and adds both the original and the mirror to the update, which is data augmentation. LOSS adds an auxiliary penalty that pulls the policy toward equivariance, the mirror loss. PHASE learns half a gait cycle and replays it mirrored, which suits imitation of a periodic reference. NET builds a hard equivariant network whose weights guarantee exact equivariance. Their reported symmetry indices, lower is better, place NET best on Walker2D at 2.00 against a baseline 3.97 with LOSS at 2.56 and DUP at 3.77, and they record two cautions that bear on our choice, symmetry enforcement gives no reliable speedup and can even slow learning, and a strictly symmetric policy cannot leave a symmetric neutral pose, the neutral state problem, which is why training must start from a noised non neutral posture.

The mirror loss has a precise and reusable form, introduced by Yu, Turk, and Liu, "Learning Symmetric and Low-Energy Locomotion", arXiv 1801.08093, and adopted verbatim by Abdolhosseini as equation two,

```
L_sym(theta) = sum_t || pi_theta(s_t) - M_a( pi_theta( M_s(s_t) ) ) ||^2
```

where M_s mirrors the state and M_a mirrors the action, added to the PPO objective with a scalar weight, Yu and Abdolhosseini use a weight of 4. This is exactly the loss that the IsaacLab RslRlSymmetryCfg computes when use_mirror_loss is set, as section 3.1 records, so the option is available to us without new machinery.

The primary paper compares three of these on quadrupeds, a vanilla PPO baseline, PPOaug which is the DUP data augmentation route, and PPOeqic which is the NET hard equivariant route built with the MorphoSymm and ESCNN libraries, and it deliberately excludes the mirror loss on the ground, citing the IsaacLab note of Mittal and colleagues, arXiv 2403.04359, that the loss is consistently outperformed by augmentation. Its verified findings are that both symmetry methods reach a far lower gait symmetry index than the baseline, the unconstrained policy scores a symmetry index four times higher than PPOaug and eight times higher than PPOeqic, and that the hard equivariant policy is the most sample efficient and the most accurate in simulation. The decisive result for the SD_BRS1 is the paper's own recommendation on when to prefer each. For task space symmetry such as door pushing the hard equivariant policy excels, but for intrinsic motion symmetry, real bipedal walking where actuator and mass asymmetries break the perfectly symmetric assumption, the hard constraint becomes brittle under distribution shift and the softer data augmentation is more robust because it adapts to the robot's real asymmetries. The SD_BRS1 is a real biped with an identified and therefore asymmetric inertial model, so the data augmentation route is the grounded first choice, and the plan adopts it as primary with the mirror loss held in reserve as an additive soft strengthening.

The gait symmetry metric to watch is the Robinson symmetry index, SI equal to 2 times the absolute difference of the left and right measurements divided by their sum, times 100, lower being more symmetric, used by both the primary and the foundational papers. The biped sagittal sign table used throughout section 3 follows directly from the physics of the reflection, a polar vector such as base linear velocity or projected gravity flips only its lateral component, a pseudovector such as angular velocity flips its in plane roll and yaw components and keeps pitch, and the joints swap left for right with a sign flip on the degrees of freedom whose rotation axis leaves the sagittal plane, all consistent with the MorphoSymm framework of Ordonez Apraez and colleagues, arXiv 2302.10433, whose authors overlap the primary paper.

## 3. Codebase investigation, the validation record

Every claim in this section was read from source. File paths are absolute or rooted at the workspace. This is the material an implementer should check the code against before and after the change.

### 3.1 The rsl_rl consumer contract

The consumer is `/ws/rsl_rl/rsl_rl/algorithms/ppo.py`. The symmetry configuration is stored as a plain dict on `self.symmetry`, constructed at `ppo.py:76-97`, and the augmentation callable is `self.symmetry["data_augmentation_func"]`. It is invoked at three sites, always by keyword, with argument names `env`, `obs`, `actions`.

- Data augmentation branch, `ppo.py:235-239`, active when `use_data_augmentation` is true, called as `data_augmentation_func(obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"])`, both inputs non None, must return an augmented `(obs, actions)` pair each with leading batch dimension multiplied by the augmentation factor.
- Mirror loss branch when augmentation was not already done, `ppo.py:319-323`, called as `data_augmentation_func(obs=obs_batch, actions=None, env=...)`, must return augmented obs and may return None for actions.
- Mirror loss action transform, `ppo.py:333-335`, called as `data_augmentation_func(obs=None, actions=action_mean_orig, env=...)`, must return augmented actions and may return None for obs.

The required signature is therefore a function accepting `env`, `obs`, and `actions` by keyword, tolerating either `obs` or `actions` being None, and returning a two tuple in `(obs, actions)` order.

The shape contract. `obs_batch` is a `tensordict.TensorDict`, not a flat tensor, keyed by the observation groups `policy` and `critic`, each value a two dimensional tensor of shape `(batch, group_dim)`. The augmentation factor num_aug is inferred purely from the returned batch, `num_aug = int(obs_batch.batch_size[0] / original_batch_size)` at `ppo.py:241`, so the function alone chooses the factor, two for a biped left-right mirror. The tensors old_actions_log_prob, target_values, advantages, and returns are tiled by the caller with `.repeat(num_aug, 1)` at `ppo.py:243-246`, so the function must not tile them, but the function is responsible for producing the augmented actions of shape `(batch times num_aug, num_actions)`.

This tiling is exactly the stability fix the literature demands. Mittal and colleagues, arXiv 2403.04359, show that when a mirrored sample enters the PPO surrogate the importance ratio must keep the original transition's old action log probability in the denominator, not the mirrored sample's own probability under the old policy, since the latter can be arbitrarily small and would make the ratio and its variance explode. Because rsl_rl tiles old_actions_log_prob from the originals rather than recomputing it on the mirrored action, the mirrored row is scored against the original transition's old probability, which is the correct and stable handling, so our function need do nothing further here beyond placing the originals in the first half.

The load bearing invariant, the untouched originals must occupy the first original_batch_size rows. Entropy and KL keep only `[:original_batch_size]` at `ppo.py:254-256`, RND keeps `[:original_batch_size]` at `ppo.py:354`, and the mirror loss compares the tail `[original_batch_size:]` against the analytic mirror at `ppo.py:340`. The comment at `ppo.py:329-330` states this assumption.

The mirror loss itself, `ppo.py:316-346`, computes the actor mean over the augmented observations, passes the original slice back through the function to obtain the analytic mirror of those means, and takes the plain MSE between the actor mean on the mirrored observations and the analytic mirror, applying `mirror_loss_coeff` only when `use_mirror_loss` is true. When both flags are false the function is still called once per minibatch for logging.

Hard constraints. Recurrent policies are rejected at `ppo.py:92-93`, the BRS policy is the plain ActorCritic so this is not a blocker. The function receives raw un-normalized observations, because normalization is applied inside the network forward at `actor_critic.py:150,156,164`, never inside update, so the mirror operates in raw observation space exactly as the anymal reference does.

Config resolution. The dict keys PPO reads are `use_data_augmentation`, `use_mirror_loss`, `data_augmentation_func`, `mirror_loss_coeff`, plus `_env` injected at runtime by `resolve_symmetry_config` at `/ws/rsl_rl/rsl_rl/modules/symmetry.py:11-25`, called from `on_policy_runner.py:401`. The `data_augmentation_func` may be a string in the `module:attribute` form, resolved by `string_to_callable` at `ppo.py:83-84`, which forbids lambdas, so the function must be a top level importable module member.

### 3.2 The configuration surface and why no class change is needed

The BRS runner is `SD_BRS1FlatPPORunnerCfg` at `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:237`, whose `algorithm` is a `RslRlPpoAlgorithmMlpCfg`. That class at `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/utils/wrappers/rsl_rl/rl_mlp_cfg.py:13-19` is an empty configclass subclass of the stock `RslRlPpoAlgorithmCfg`, which at `/ws/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:128-129` already declares `symmetry_cfg: RslRlSymmetryCfg | None = None`. The field is therefore inherited, and passing `symmetry_cfg=RslRlSymmetryCfg(...)` into the existing algorithm instance is accepted with no change to any class.

The serialization path is safe. In `scripts/rsl_rl/train.py` the config is converted by `agent_cfg.to_dict()`, and `class_to_dict` at `/ws/IsaacLab/source/isaaclab/isaaclab/utils/dict.py:62-63` detects the callable `data_augmentation_func` and converts it to a `module:function` string via `callable_to_string`, exactly the form `string_to_callable` re-imports inside the PPO constructor. This is the second reason the function must be a top level importable function and not a lambda or a closure. The `_env` object is injected after conversion, so there is no serialization concern for the live environment.

The environment the function receives at runtime is the `RslRlVecEnvWrapper` produced at `train.py:187`, whose `.unwrapped` returns the `ManagerBasedRLEnv`. The observation manager is `env.unwrapped.observation_manager`, the scene is `env.unwrapped.scene`, and the articulation is `env.unwrapped.scene["robot"]`, matching the anymal convention.

The additive change touches only the BRS instance, setting an inherited field whose default is None, so no other TRON1 SF, PF, WF, Berkeley, or Copt runner is affected, and the workspace backwards compatibility rule is satisfied without a v2.

### 3.3 The SymmLoco construction and how it maps

SymmLoco enforces symmetry through two modes only, `aug` pure data augmentation that duplicates every transition through the robot symmetry group and trains vanilla PPO on the doubled buffer, and `emlp` a hard equivariant network. It contains no MSE mirror loss. The robot symmetry group for the CyberDog2 quadruped is C2, a single sagittal mirror with two elements, loaded from MorphoSymm's `a1.yaml`. The augmentation stacks the original block first and the mirrored block second, `torch.cat([x] + [transform(x, g) for g in G.elements[1:]])` at `rsl_rl/rsl_rl/algorithms/ppo_augment.py:165-184`, and it transforms observations, critic observations, actions, action mean, and action sigma, while merely tiling the symmetry invariant log_prob, value, reward, and done.

The per term representations are the reusable content. `rep_O3` equals diag(1, -1, 1) and applies to base frame vectors and projected gravity. `rep_O3_pseudo` equals minus diag(1, -1, 1) which is diag(-1, 1, -1) and applies to angular velocity and to the velocity command, a pseudovector. `rep_QJ` is the joint permutation with sign flips and applies to joint position residual, joint velocity, and actions. A two dimensional swap representation applies to per leg scalar pairs such as the clock inputs. SymmLoco mirrors the command, `c2_standdance_env.py` applies `rep_O3_pseudo` to the command after a manual x y reindex, and it mirrors the critic observation with its own field type. The joint map from `a1.yaml` swaps the left and right legs and negates the hip abduction joint of each leg, confirming the structure of a permutation with a small set of sign flips.

The mapping onto the target interface is direct for the augmentation path, SymmLoco `aug` corresponds to `use_data_augmentation=True`, and the per term reps become the per term maps of our `compute_symmetric_states`, with the same original first ordering that rsl_rl assumes. The IsaacLab mirror loss has no SymmLoco counterpart, so if it is enabled its coefficient is our own design and not copied from the paper code. The equivariant `emlp` mode is a network change and is out of scope. Because the group is a fixed C2, the whole construction reduces to a fixed permutation with signs on a fixed layout, so we hand code it and do not import MorphoSymm, escnn, or hydra.

### 3.4 The SD_BRS1 observation and action layout

Source `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/brs_base_env_cfg.py`, ObservationsCfg at lines 152 to 267. The runner sets no obs_groups, so the policy set maps to the `policy` group and the critic set to the `critic` group.

Correction one, the observations are history flattened. Both PolicyCfg and CriticCfg set history_length equal to 10 and flatten_history_dim true, so each term becomes a contiguous block of ten time major frames and the blocks are concatenated in declaration order. The policy observation is therefore 440 wide, not a single 44 wide frame, and the mirror must operate per block by reshaping each block to shape (N, 10, term_width) and applying the per frame map on the last axis.

The policy per frame terms in declaration order, with per frame width and flattened block range in the 440 vector.

| term | func | per frame width | flattened block range |
|------|------|-----------------|-----------------------|
| base_lin_vel | mdp.base_lin_vel | 3 | [0:30] |
| base_ang_vel | mdp.base_ang_vel | 3 | [30:60] |
| proj_gravity | mdp.projected_gravity | 3 | [60:90] |
| joint_pos | mdp.joint_pos_rel | 10 | [90:190] |
| joint_vel | mdp.joint_vel_rel | 10 | [190:290] |
| last_action | mdp.last_action | 10 | [290:390] |
| velocity_commands | mdp.generated_commands | 3 | [390:420] |
| gait_phase | mdp.get_gait_phase | 2 | [420:440] |

The velocity command is the three vector [lin_vel_x, lin_vel_y, ang_vel_z] from vel_command_b, three dimensional even though heading_command is true. The gait_phase term returns concat of sin(2 pi phi) and cos(2 pi phi) for the single shared clock phase of foot index 0, width two, defined at `mdp/observations.py:154-179`, and the second foot is implicitly at phi plus the offset of 0.5.

Correction two, the joint sign flips. The action term is a single JointPositionActionCfg over all ten actuated joints with scale 0.25 and use_default_offset true, and joint_pos_rel, joint_vel_rel, and last_action are all in the same articulation joint order. The resolved order, confirmed against the dumped tensors as recorded in section 3.5, is depth major with the left leg first at each level, since Isaac Sim orders the articulation breadth first and this parser emitted the left sibling before the right.

| index | joint | URDF axis | mirror sign |
|-------|-------|-----------|-------------|
| 0 | HipRollL | (1,0,0) | flip |
| 1 | HipRollR | (1,0,0) | flip |
| 2 | HipPitchL | (0,1,0) | flip |
| 3 | HipPitchR | (0,-1,0) | flip |
| 4 | KneePitchL | (0,1,0) | keep |
| 5 | KneePitchR | (0,1,0) | keep |
| 6 | AnkleRollL | (-1,0,0) | flip |
| 7 | AnkleRollR | (-1,0,0) | flip |
| 8 | AnklePitchL | (0,-1,0) | keep |
| 9 | AnklePitchR | (0,-1,0) | keep |

The left right permutation is the pairwise swap `perm = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]`, and the flipped indices are the HipRoll pair, the HipPitch pair, and the AnkleRoll pair, that is {0, 1, 2, 3, 6, 7}, while KneePitch and AnklePitch keep their sign. HipPitch flips because HipPitchR and HipPitchL carry opposite joint axes, decisively corroborated by three pieces of config evidence, the URDF limits are negated across the pair with HipPitchR at [-0.75, 1.25] and HipPitchL at [-1.25, 0.75], the reset events split HipPitch by side with negated ranges, and KneePitch and AnklePitch use identical ranges on both sides. KneePitch and AnklePitch keep their sign because their left and right axes match. The permutation and the flip set are invariant to whether the parser lists the left or the right sibling first, since every swap is between adjacent partners of the same joint kind, so the change from the earlier right first assumption to the confirmed left first order does not alter a single index.

A word of reconciliation, since this deviates from the textbook rule. The general physics of the reflection says a pitch joint rotates about the lateral axis and lies in the sagittal plane, so its physical degree of freedom is preserved and one expects no sign flip, which would flip only the roll joints and give the set {0, 1, 6, 7}. That rule holds only when the left and right joint frames share a consistent axis convention. This robot does not, its HipPitchR and HipPitchL frames are defined with opposite axes, so the numeric joint value must flip to represent the same physically mirrored configuration, which is precisely the case that MorphoSymm encodes as a per joint entry of its reflection vector rather than as an abstract roll versus pitch rule. The authoritative flip set for this robot is therefore {0, 1, 2, 3, 6, 7}, grounded in the URDF axes and the three config corroborations above, and the name based joint map in the implementation encodes it by listing HipPitch among the flipped kinds.

The implementation builds the joint map from `env.unwrapped.scene["robot"].joint_names` at runtime, pairing each joint with its opposite side partner by name and assigning the flip by joint kind, so it is correct for either sibling order. The left first values above are the confirmed result and serve as the validation assertion.

The per frame mirror sign table for the non joint terms, verified against the anymal left-right case.

| term | per frame order | mirror op |
|------|-----------------|-----------|
| base_lin_vel | [vx, vy, vz] | multiply [1, -1, 1] |
| base_ang_vel | [wx, wy, wz] | multiply [-1, 1, -1] |
| proj_gravity | [gx, gy, gz] | multiply [1, -1, 1] |
| velocity_commands | [lin_x, lin_y, ang_z] | multiply [1, -1, -1] |
| gait_phase | [sin, cos] | multiply [-1, -1] |

The gait_phase sign follows from the fixed offset of 0.5. A left right foot swap exchanges the two antiphase feet, which shifts the reported clock phase by half a cycle, phi to phi plus 0.5, and since sin and cos of an argument shifted by pi both negate, gait_phase maps to multiply by [-1, -1]. This equivalence holds only while the offset is exactly 0.5, the current config fixes it there. If the offset is ever widened, this rule breaks and the phase must be recomputed by shifting phi by the true offset.

The critic group is much larger, 489 wide per frame and 4890 flattened, adding privileged terms such as the height scan of width 187, per joint torque and acceleration, per foot linear velocity and contact force, per body mass and inertia, default joint properties, root pose and velocity, and material properties of width 57. Its inversion is the subject of section 3.5.

### 3.5 Critic Observation inversion

Outcome of the investigation, stated first. The critic group must be mirrored, not left tiled, and this supersedes the earlier decision to tile it. In the update loop the value head reads `obs_batch["critic"]` and is trained toward the tiled `returns_batch`, `value_batch = self.policy.evaluate(obs_batch)` at `ppo.py:252` against `returns_batch` at `ppo.py:307`. When the critic block is left tiled, each mirrored row presents the original privileged state while carrying the replicated return of that same original state, so the value loss on the mirrored half merely repeats the loss on the original half, teaches the value function nothing new, and never induces the sagittal invariance the theory requires, the value at the reflected state equals the value at the original state. To make the predicted value at the mirrored observation match the replicated target, the critic observation must itself be the mirror of the state, which is exactly what SymmLoco does, it defines a separate `critic_in_field_type` and transforms `critic_observations` alongside the actor observations at `ppo_augment.py:123, 179`, and the primary paper states that augmenting the critic biases the learned value function toward an approximately invariant function, which in turn sharpens the advantage estimates that guide the actor. Mirroring the critic is therefore the correct implementation, and the plan now does so for every critic term whose reflection is well defined, leaving identity only the four world frame terms whose reflection the augmentation function cannot recover, for the reason given below.

The load bearing result, the body order, read from the dump. The investigation rests on the tensors dumped by run 2026-07-21_06-03-36, since deleted, a dict of logged arrays read for one env at one timestamp and correlated against the URDF. The dumped `robot_mass` reads [8.2464, 2.726, 2.726, 3.86, 3.8609, 9.6512, 9.645, 5.8054, 5.805, 0.0722, 0.0722, 3.691, 3.691], and matching the slightly asymmetric identified masses, Link2L at 3.86 against Link2R at 3.8609, Link3L at 9.6512 against Link3R at 9.645, Link4L at 5.8054 against Link4R at 5.805, pins the order to left before right at every level. The sign of the inertia products confirms it independently, the dumped body 5 carries a positive ixy matching Link3L while body 6 carries a negative ixy matching Link3R. The thirteen bodies are therefore ordered Part_Torso, then Link1L, Link1R, Link2L, Link2R, Link3L, Link3R, Link4L, Link4R, Link5L, Link5R, Link6L, Link6R, and the body permutation is the pairwise swap [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]. This left first order also settles the joint ordering used throughout section 3.4, though as noted there it leaves every permutation unchanged.

Significance of each observation and what its indices carry. The critic sees the whole policy observation first, then a privileged block that a real robot could not measure but that a simulator knows, so the value function can judge a state more accurately during training. The single timestep index map of the 489 wide critic frame, with the mirror operation applied to each term, is the following. The material properties term is 57 wide here, since the identified mesh assets decompose into 19 collision shapes rather than 13, three columns per shape.

| range | term | width | meaning | mirror operation |
|-------|------|-------|---------|------------------|
| [0:3] | base_lin_vel | 3 | base frame linear velocity | polar vector, multiply [1, -1, 1] |
| [3:6] | base_ang_vel | 3 | base frame angular velocity | pseudovector, multiply [-1, 1, -1] |
| [6:9] | proj_gravity | 3 | gravity direction in base frame | polar vector, multiply [1, -1, 1] |
| [9:19] | joint_pos | 10 | joint position minus default | joint permutation, sign flip {0,1,2,3,6,7} |
| [19:29] | joint_vel | 10 | joint velocity | joint permutation, sign flip {0,1,2,3,6,7} |
| [29:39] | last_action | 10 | previous action | joint permutation, sign flip {0,1,2,3,6,7} |
| [39:42] | velocity_commands | 3 | commanded lin_x, lin_y, ang_z | multiply [1, -1, -1] |
| [42:229] | heights | 187 | height scan clearances, 11 lateral rows by 17 forward columns | reverse the 11 lateral rows |
| [229:239] | robot_joint_torque | 10 | applied actuator torque per joint | joint permutation, sign flip {0,1,2,3,6,7} |
| [239:249] | robot_joint_acc | 10 | joint acceleration | joint permutation, sign flip {0,1,2,3,6,7} |
| [249:255] | feet_lin_vel | 6 | world frame foot linear velocity, Link6 R then L | identity, world frame, see below |
| [255:268] | robot_mass | 13 | per body mass | body permutation, no sign change |
| [268:385] | robot_inertia | 117 | per body inertia, 13 bodies by 9 | body permutation, then within each 9 flip {1,3,5,7} |
| [385:395] | robot_joint_pos | 10 | default joint position | joint permutation, sign flip {0,1,2,3,6,7} |
| [395:405] | robot_joint_stiffness | 10 | per joint actuator stiffness gain | joint permutation, no sign change |
| [405:415] | robot_joint_damping | 10 | per joint actuator damping gain | joint permutation, no sign change |
| [415:418] | robot_pos | 3 | world frame root position | identity, world frame, see below |
| [418:424] | robot_vel | 6 | world frame root linear and angular velocity | identity, world frame, see below |
| [424:481] | robot_material_properties | 57 | per collision shape static friction, dynamic friction, restitution | shape permutation by body, no sign change |
| [481:487] | feet_contact_force | 6 | world frame foot contact force, Link6 R then L | identity, world frame, see below |
| [487:489] | gait_phase | 2 | sin and cos of the shared gait clock | multiply [-1, -1] |

The three morphology terms, resolved from the dump. The masses are scalars per body, so the mirror only swaps left for right by the body permutation, with no sign change, and the dump shows they are constant across the episode as expected of a per env randomized parameter. The inertia term is thirteen flattened three by three tensors, in the column major order [Ixx, Iyx, Izx, Ixy, Iyy, Izy, Ixz, Iyz, Izz] documented at `articulation_data.py:156-165`, though by the symmetry of an inertia tensor the row and column major layouts are numerically identical. A reflection across the sagittal plane sends the tensor to R I R transpose with R equal to diag(1, -1, 1), which flips the sign of exactly the products of inertia that couple the lateral axis, the local elements {1, 3, 5, 7}, and leaves the diagonal moments and the front to back product unchanged, so the per body operation is the body swap followed by the fixed sign vector [1, -1, 1, -1, 1, -1, 1, -1, 1] on each nine block. This was verified numerically on the dump, mirroring the stored Link3R block reproduces the sign structure of the stored Link3L block and the operation is an exact involution. The material properties term is per collision shape rather than per body, three columns being static friction, dynamic friction and restitution in that order, all scalars invariant under reflection, so the mirror swaps the shapes of each body with those of its partner body and changes no sign. The dump shows 57 values, that is 19 shapes, since the identified meshes decompose into more than one convex shape for some links, and the restitution column is all zero since that range is not randomized. Shapes are laid out body by body following the body order, so the shape permutation is built at runtime from the per body shape counts, swapping each body's contiguous shape block with its partner's, which is well defined because partner bodies carry identical meshes and therefore equal shape counts.

The four world frame terms that stay identity, and why. The terms feet_lin_vel, feet_contact_force, robot_pos and robot_vel are expressed in the global simulation frame, confirmed by the dump where a planted foot's contact force is almost purely vertical in world z. A left right mirror of this robot is a reflection across the base sagittal plane, and expressing that reflection on a world frame vector requires the base orientation, in particular the yaw, to rotate the vector into the base frame, reflect, and rotate back. The augmentation function operates on the stored history batch and has no access to the per transition base yaw, the yaw appears in no observation term, so it cannot perform this rotation, and a naive world y flip would be a different and inconsistent operation valid only when the base faces along the world x axis. These four terms are therefore left identity, a recorded approximation, so the mirrored critic sample is exact on the base frame proprioception, the joint space, the morphology, the height scan and the gait clock, and only approximate on the four world frame privileged coordinates. This is honest and it does not corrupt training, the four terms are privileged inputs to a value function that the network learns to tolerate, and the bulk of the critic that carries the sagittal signal is mirrored exactly.

The gait clock term flips, the same as in the actor. The clock is a single shared phase, but a left right foot swap exchanges the two feet whose schedules are half a cycle apart, so the mirrored state must advance the clock by that half cycle, phi to phi plus 0.5, which negates both the sin and the cos channels. This was derived from the GaitReward schedule, the reward pairs foot zero with phase phi and foot one with phase phi plus the offset of 0.5, and only a phase shift of the offset preserves the reward under the foot swap. The value [-1, -1] holds while the offset is fixed at 0.5.

## 4. Implementation plan

The change is three parts, a new module holding the augmentation function, one export line, and the additive symmetry_cfg on the BRS runner. Nothing shared is edited, so backwards compatibility is absolute.

### 4.1 New module, the robot specific mirror

The mirror map is specific to this robot's joint set, body set and observation layout, so it lives in its own robot named module rather than a single shared file, which leaves room for a TRON1 mirror beside it later. Create the directory `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/symmetry/` as a package, with a small `__init__.py` and the implementation in `brs.py`.

The `__init__.py`.

```python
from .brs import compute_symmetric_states  # noqa: F401

__all__ = ["compute_symmetric_states"]
```

The `brs.py`, verbatim. The term layout of each group is read at runtime from the observation manager, so no width is hard coded, and the joint, body and collision shape maps are built from the runtime names, so the module is correct whichever way the parser orders the siblings. The one value the module cannot read from the running sim without a heavier introspection is the number of collision shapes per body, which the material properties mirror needs. It is left as a module constant to fill once from the runtime print in section 4.3, and until it is filled the material term degrades safely to identity.

```python
# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the SD_BRS1 biped.

Robot-specific module. Implements ``compute_symmetric_states`` in the signature the
rsl_rl PPO symmetry hook expects (``/ws/rsl_rl/rsl_rl/algorithms/ppo.py``) and referenced
through ``isaaclab_rl.rsl_rl.RslRlSymmetryCfg``. A biped has one sagittal mirror, so the
augmentation doubles the batch, originals in the first half, mirrored copies in the second.

Both the policy and the critic observation groups are mirrored, so the value function is
biased toward the sagittal invariance the theory requires. The term layout is read at
runtime from the observation manager, and the joint, body, and collision-shape maps are
built from the runtime names, so the module is correct regardless of whether Isaac Sim
resolves the sibling links left-first or right-first.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# history length shared by the policy and critic groups (flatten_history_dim=True)
_HISTORY = 10

# joints whose sign flips under the sagittal mirror, by name substring. HipPitch flips
# despite being a pitch joint because its left and right URDF axes are opposite.
_FLIP_JOINT_KINDS = ("HipRoll", "AnkleRoll", "HipPitch")

# height-scan grid, lateral y rows by forward x columns (GridPatternCfg size (1.6, 1.0),
# resolution 0.1, ordering "xy" -> meshgrid shape (len(y), len(x)) = (11, 17)).
_HEIGHT_ROWS = 11  # lateral y, the axis reversed by the mirror
_HEIGHT_COLS = 17  # forward x

# collision shapes per body in body order, for robot_material_properties. Leave None to
# skip mirroring that term (identity fallback). Fill after the one-time runtime print in
# the plan, section 4.3. The observed total for SD_BRS1 is 19 shapes (width 57 = 19 * 3).
# Partner bodies must carry equal counts.
_NUM_SHAPES_PER_BODY = None  # e.g. [1, 1,1, 1,1, 2,2, 2,2, 1,1, 2,2]

# fixed per-frame sign vectors
_VEC = (1.0, -1.0, 1.0)       # polar vector, flip lateral y
_PVEC = (-1.0, 1.0, -1.0)     # pseudovector, flip x and z
_CMD = (1.0, -1.0, -1.0)      # velocity command, flip lin_y and ang_z
_GAIT = (-1.0, -1.0)          # gait phase, half-cycle shift under the foot swap
_INERTIA9 = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # flip y-coupling products

# per-term mirror kind, keyed by observation term name
_POLICY_KIND = {
    "base_lin_vel": "vec", "base_ang_vel": "pvec", "proj_gravity": "vec",
    "joint_pos": "joint", "joint_vel": "joint", "last_action": "joint",
    "velocity_commands": "cmd", "gait_phase": "gait",
}
_CRITIC_KIND = {
    "base_lin_vel": "vec", "base_ang_vel": "pvec", "proj_gravity": "vec",
    "joint_pos": "joint", "joint_vel": "joint", "last_action": "joint",
    "velocity_commands": "cmd", "heights": "heights",
    "robot_joint_torque": "joint", "robot_joint_acc": "joint",
    "feet_lin_vel": "identity", "robot_mass": "body_scalar",
    "robot_inertia": "body_inertia", "robot_joint_pos": "joint",
    "robot_joint_stiffness": "joint_nosign", "robot_joint_damping": "joint_nosign",
    "robot_pos": "identity", "robot_vel": "identity",
    "robot_material_properties": "material", "feet_contact_force": "identity",
    "gait_phase": "gait",
}

# module-level cache, one entry per (device); env is a singleton
_CACHE: dict = {}


def _build_joint_map(joint_names, device):
    """out[i] = sign[i] * x[perm[i]], partner is the same joint with its side letter swapped."""
    n = len(joint_names)
    perm, sign = list(range(n)), [1.0] * n
    for i, name in enumerate(joint_names):
        partner = name[:-1] + "L" if name.endswith("R") else name[:-1] + "R" if name.endswith("L") else name
        perm[i] = joint_names.index(partner)
        if any(k in name for k in _FLIP_JOINT_KINDS):
            sign[i] = -1.0
    return (torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=torch.float, device=device), perm)


def _build_body_map(body_names):
    perm = list(range(len(body_names)))
    for i, name in enumerate(body_names):
        partner = name[:-1] + "L" if name.endswith("R") else name[:-1] + "R" if name.endswith("L") else name
        perm[i] = body_names.index(partner)
    return perm


def _build_material_perm(num_shapes_per_body, body_perm, device):
    if num_shapes_per_body is None:
        return None
    counts = list(num_shapes_per_body)
    for b, pb in enumerate(body_perm):
        assert counts[b] == counts[pb], "partner bodies must carry equal collision-shape counts"
    starts, acc = [], 0
    for c in counts:
        starts.append(acc)
        acc += c
    perm = []
    for b in range(len(counts)):
        src = body_perm[b]
        perm.extend(range(starts[src], starts[src] + counts[src]))
    return torch.tensor(perm, dtype=torch.long, device=device)


def _layout(env, group):
    """Return (term_names, block_widths) in concatenation order, widths already history-flattened."""
    om = env.unwrapped.observation_manager
    names = list(om.active_terms[group])
    dims = om.group_obs_term_dim[group]
    widths = [int(math.prod(d)) for d in dims]
    return names, widths


def _maps(env, device):
    if device in _CACHE:
        return _CACHE[device]
    robot = env.unwrapped.scene["robot"]
    jperm, jsign, _ = _build_joint_map(list(robot.joint_names), device)
    bperm_list = _build_body_map(list(robot.body_names))
    om = env.unwrapped.observation_manager
    m = {
        "jperm": jperm,
        "jsign": jsign,
        "bperm": torch.tensor(bperm_list, dtype=torch.long, device=device),
        "mat_perm": _build_material_perm(_NUM_SHAPES_PER_BODY, bperm_list, device),
        "vec": torch.tensor(_VEC, device=device),
        "pvec": torch.tensor(_PVEC, device=device),
        "cmd": torch.tensor(_CMD, device=device),
        "gait": torch.tensor(_GAIT, device=device),
        "inertia9": torch.tensor(_INERTIA9, device=device),
        "pol_layout": _layout(env, "policy"),
        "cri_layout": _layout(env, "critic") if "critic" in om.active_terms else None,
    }
    _CACHE[device] = m
    return m


def _apply(kind, blk, n, m):
    """blk is (n, block_width); return the mirrored block, same shape."""
    if kind == "identity":
        return blk
    sw = blk.shape[1] // _HISTORY
    x = blk.view(n, _HISTORY, sw)
    if kind in ("vec", "pvec", "cmd", "gait"):
        return (x * m[kind]).reshape(n, -1)
    if kind == "joint":
        return (x[..., m["jperm"]] * m["jsign"]).reshape(n, -1)
    if kind == "joint_nosign":
        return x[..., m["jperm"]].reshape(n, -1)
    if kind == "body_scalar":
        return x[..., m["bperm"]].reshape(n, -1)
    if kind == "body_inertia":
        xi = x.view(n, _HISTORY, sw // 9, 9)[..., m["bperm"], :] * m["inertia9"]
        return xi.reshape(n, -1)
    if kind == "heights":
        xh = torch.flip(x.view(n, _HISTORY, _HEIGHT_ROWS, _HEIGHT_COLS), dims=[-2])
        return xh.reshape(n, -1)
    if kind == "material":
        if m["mat_perm"] is None:
            return blk  # identity fallback until _NUM_SHAPES_PER_BODY is filled
        xm = x.view(n, _HISTORY, sw // 3, 3)[..., m["mat_perm"], :]
        return xm.reshape(n, -1)
    raise ValueError(f"unknown mirror kind {kind}")


def _mirror_group(flat, names, widths, kind_map, n, m):
    out = torch.empty_like(flat)
    start = 0
    for name, w in zip(names, widths):
        out[:, start:start + w] = _apply(kind_map.get(name, "identity"), flat[:, start:start + w], n, m)
        start += w
    assert start == flat.shape[1], f"width mismatch, mapped {start} of {flat.shape[1]}"
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    device = obs["policy"].device if obs is not None else actions.device
    m = _maps(env, device)

    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)  # tiles every group, originals in [:n]
        pnames, pwidths = m["pol_layout"]
        obs_aug["policy"][n:] = _mirror_group(obs["policy"], pnames, pwidths, _POLICY_KIND, n, m)
        if m["cri_layout"] is not None and "critic" in obs.keys():
            cnames, cwidths = m["cri_layout"]
            obs_aug["critic"][n:] = _mirror_group(obs["critic"], cnames, cwidths, _CRITIC_KIND, n, m)
    else:
        obs_aug = None

    actions_aug = None if actions is None else torch.cat([actions, actions[:, m["jperm"]] * m["jsign"]], dim=0)
    return obs_aug, actions_aug
```

The index math of this module, the joint and body permutations, the inertia nine block sign flip, the height row reversal, the material shape swap, and the original first doubling, were validated in numpy against the dumped tensors, the mirror is an exact involution and the base frame sign flips are correct.

### 4.2 Wiring

Wire the symmetry_cfg on the BRS runner in `/ws/tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`. Add two imports near the top, mirroring the anymal reference which imports its symmetry module in the ppo cfg.

```python
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
from bipedal_locomotion.tasks.locomotion.mdp.symmetry.brs import compute_symmetric_states
```

Then add the `symmetry_cfg` argument to the existing `RslRlPpoAlgorithmMlpCfg(...)` inside `SD_BRS1FlatPPORunnerCfg`, leaving every other field of that instance untouched.

```python
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.0,
        ),
```

This is the primary configuration, pure data augmentation faithful to the SymmLoco `aug` mode, one moving part. The mirror loss is available as an additive strengthening by setting `use_mirror_loss=True` with a small `mirror_loss_coeff`, for example 1.0, which the ppo update adds on top of the already augmented batch. Do not enable the mirror loss in the first run, establish the augmentation only baseline first.

### 4.3 Mandatory runtime confirmation and the collision shape count

Two things must be read from the running sim once and confirmed, and one of them must be written back into the module.

First, the joint and body names. The module builds its maps from `env.unwrapped.scene["robot"].joint_names` and `.body_names`, so correctness does not depend on the sibling order, but the confirmed left first order should still be checked. Print both once at the first training start. The expected joint names are `['HipRollL','HipRollR','HipPitchL','HipPitchR','KneePitchL','KneePitchR','AnkleRollL','AnkleRollR','AnklePitchL','AnklePitchR']`, giving perm `[1,0,3,2,5,4,7,6,9,8]` and flipped indices `{0,1,2,3,6,7}`, and the expected body names are `['Part_Torso','Link1L','Link1R','Link2L','Link2R','Link3L','Link3R','Link4L','Link4R','Link5L','Link5R','Link6L','Link6R']`, giving body perm `[0,2,1,4,3,6,5,8,7,10,9,12,11]`. If the parser lists the right sibling first instead, the name based builders produce the mirrored perms automatically and the result is still correct.

Second, the collision shape count per body, which the material properties mirror needs and which the module leaves as `_NUM_SHAPES_PER_BODY = None`. The most reliable source is IsaacLab itself, since the enabled `robot_physics_material` event runs `randomize_rigid_body_material`, whose `__init__` already computes and stores `self.num_shapes_per_body` in body order by the exact logic that indexes the material buffer (`isaaclab/envs/mdp/events.py:205-224`). Read that stored list once and copy it into the module. The event term instance is reachable through `env.unwrapped.event_manager`, the precise accessor differs by IsaacLab version, so if it is awkward, replicate the same loop from that `__init__` over `robot.root_physx_view.link_paths[0]`, or as a last resort print only the total `robot.root_physx_view.max_shapes` and split it across bodies by inspecting the collision prims. Alongside it print the names.

```python
robot = env.unwrapped.scene["robot"]
print("body_names   :", list(robot.body_names))
print("joint_names  :", list(robot.joint_names))
print("total shapes :", robot.root_physx_view.max_shapes)  # expect 19
# num_shapes_per_body: read from the robot_physics_material event term's stored
# self.num_shapes_per_body (events.py:205-224), body order, then paste into brs.py
```

The observed total for the identified asset is 19 shapes, so the filled list must sum to 19 and be equal across partner bodies. If the printed total is not 19, the mesh decomposition changed and the list must be re read, the module asserts partner equality and will refuse an inconsistent list, and until the list is filled the material term stays identity, which is safe while the rest of the critic is mirrored regardless.

### 4.4 Self test before launching

With torch installed, run a standalone check in the scratchpad, not in the repository, that stubs `env` with an object exposing `unwrapped.scene["robot"].joint_names` and `.body_names` and `unwrapped.observation_manager.active_terms` and `.group_obs_term_dim`, builds a random policy and critic TensorDict at the runtime widths, and calls `compute_symmetric_states`. Assert the following.

- The output batch is exactly twice the input, and the first half equals the input unchanged, for both obs groups and for actions.
- Applying the mirror to the mirrored half returns the original within a tight tolerance, the involution property, which holds because every permutation is a pairwise swap and every sign is plus or minus one.
- The internal width assertions pass, each group's mapped width equals its total.
- Spot checks on a random frame, base_lin_vel flips only its y channel, velocity_commands flips its second and third channels, gait_phase negates both channels, robot_mass swaps partner bodies, and the robot_inertia nine block flips exactly its lateral coupling elements.

The index math of the module was already validated in numpy against the dumped tensors, the involution error is zero and the base frame sign flips are correct, so this self test is a confirmation on the real TensorDict path rather than a first check. Only after it passes and the two prints are confirmed should the overnight run start.

## 5. Validation gates for the training run

The single quantitative target is a reduction in gait asymmetry. Watch pen_flat_orientation, currently pinned near minus 1.72, it should move toward zero as the base stops leaning. Watch the symmetry loss logged by ppo even though it is not optimised, it is computed for logging whenever symmetry_cfg is set, and it should fall as the policy becomes more symmetric. The velocity tracking rew_lin_vel_xy and the gait terms should hold near their current strong values rather than regress, augmentation should not cost tracking. Visually, the left and right stance phases in the play video should match in duration and posture, and the limp should be gone. If augmentation alone does not close the asymmetry, enable the mirror loss with a small coefficient as the second step.

## 6. Notes for later phases, out of scope here

Observation 2, the stiff straight legs. The KneePitch URDF limit is [0, 1.483], so zero is a fully straight leg and the knee flexes only positively, and the init pose sets every joint including KneePitch to zero, a straight legged start. No reward term directly rewards knee flexion, the incentives are indirect through clearance, air time, and the gait clock, so a stiff leg local optimum is available. A future phase should add a direct knee flexion incentive or a reference knee trajectory, and the symmetry established here will make any such incentive apply evenly to both legs.

Observation 3, the standing shuffle. The gait_command frequencies are degenerate at (1.0, 1.0), so the clock never stops and get_gait_phase keeps ticking even under a zero velocity command, and GaitReward has no zero command gate, so the roughly two percent standing environments are still commanded to march in place. A future phase should widen the frequency range to include zero and gate the gait reward on a nonzero command, so that a flat gait_phase input at zero frequency signals standing and the robot holds still. The symmetry augmentation is compatible with this, a zero frequency clock still mirrors to a zero frequency clock.

## 7. Summary of edits

- New package directory, `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/symmetry/`, holding `__init__.py` and the robot specific `brs.py` with `compute_symmetric_states`, which mirrors both the policy and the critic observation groups and the actions.
- Two import lines and one `symmetry_cfg` argument in `limx_rsl_rl_ppo_cfg.py`, inside `SD_BRS1FlatPPORunnerCfg` only, the function referenced as `bipedal_locomotion.tasks.locomotion.mdp.symmetry.brs:compute_symmetric_states`.
- One value written into `brs.py` after the runtime print, `_NUM_SHAPES_PER_BODY`, without which the material properties term degrades safely to identity.
- No edits to any shared mdp function, to rsl_rl, or to any other runner, so all existing callers and historical comparisons are preserved. The symmetry_cfg sets an inherited field whose default is None, so the change is additive and confined to the BRS instance.
