# Quadruped Open Source Comparison

Status, written 2026-09-10 against nine open source quadruped locomotion implementations cloned to `/ws/external/` and read at their HEAD commits on that date. The per repository extraction, with function bodies and file and line citations, is preserved at `/ws/external/COMPARISON_RAW.md` and is the source for every figure below. This document is the distillation, and it exists to answer three questions that the plan `plans/QUADRUPED_DEGENERATE_GAIT.md` puts to the literature and cannot answer from papers alone, namely what the published configurations actually do rather than what they report, which of them can prevent a limb leaving the support pattern, and which differences against this workspace are worth testing.

## 1. What was cloned and what was read

Nine repositories were cloned shallow and their quadruped reward configuration, gait specific terms and PPO hyperparameters extracted, those being legged_gym [1], unitree_rl_gym [2], Extreme Parkour [3], HIMLoco [4], rapid locomotion, Walk These Ways [5], ZiwenZhuang parkour, Argo Robot quadrupeds_locomotion, and legged loco. A tenth, viper_rl, was cloned and deleted, being a video prediction reward framework for DMC and Atari with no quadruped configuration and therefore evidence of nothing. The training code for the energy minimisation result [6] could not be located, only its project website being public, which is worth recording because that result is the one this workspace most wants to reproduce.

## 2. The consolidated table

Weights are as written in each repository, before any per timestep scaling. Two conventions are in play and section 4 explains why the distinction matters.

| Repository | entropy coef | lin vel | ang vel | action rate | torque | feet air time | foot clearance | contact schedule |
|---|---|---|---|---|---|---|---|---|
| legged_gym [1] | 0.01 | 1.0 | 0.5 | -0.01 | -1e-5 | 1.0 at 0.5 s | none | none |
| unitree_rl_gym [2] | 0.01 | 1.0 | 0.5 | -0.01 | -2e-4 | 1.0 at 0.5 s | none | none |
| Extreme Parkour [3] | 0.01 | 1.5 | 0.5 | -0.1 | -1e-5 | absent | none | none |
| HIMLoco [4] | 0.01 | 1.0 | 0.5 | -0.01 | 0.0 | absent | -0.01 at -0.20 | none |
| rapid locomotion | 0.01 | 1.0 | 0.5 | -0.01 | -1e-5 | 1.0 at 0.5 s | none | none |
| Walk These Ways [5] | 0.01 | 1.0 | 0.5 | -0.1 smoothness | -1e-5 | disabled at 0.0 | -30.0 | von Mises clock, force and velocity at 4.0 |
| ZiwenZhuang parkour | 0.01 | absent | 0.05 | absent | absent | absent | none | none |
| quadrupeds_locomotion | 0.01 | 1.0 | 0.2 | -0.005 | absent | none | none | none |
| legged loco | 0.01 | inherited | inherited | -0.02 | -2e-5 power | none | none | none |
| THIS WORKSPACE | 0.001 | 10 | 5 | -0.01 | -2e-4 | 5.0 at 0.15 to 0.45 s | 0.5 | none |

## 3. Only one of nine can charge a foot for staying aloft

This is the finding that governs the plan. Five distinct mechanisms bear on degenerate gaits across the nine repositories, and only one of them can charge a limb that has left the support pattern.

The `feet_air_time` bonus, carried by four repositories at a weight of 1.0 and a threshold of 0.5 seconds, rewards a delayed touchdown and is evaluated at touchdown alone, so a foot that never lands collects neither reward nor penalty. This is the touchdown blindness that section 4.1 of the plan identified in this workspace's own term, now confirmed as a property of the whole family rather than of this implementation, and it is the single most transferable result of the comparison.

The phase clock contact tracking of Walk These Ways [5] is the exception. A commanded frequency, phase, offset and duration are converted through a Gaussian approximation to a von Mises distribution into a per foot desired contact probability, and two terms then charge contact force when a foot should be swinging and foot velocity when it should be in stance, at a weight of 4.0 each in the shipped training configuration. Because the desired contact probability is defined at every instant and does not wait for an event, a foot that should be in stance and is not is charged continuously. This is the only surveyed construction with that property, and it purchases it by committing to a contact schedule.

The foot clearance term of HIMLoco [4] weights its height error by the lateral foot speed, so a foot that is not moving escapes the penalty. This workspace's `foot_clearance_reward_v4` inherits that construction and section 4.2 of the plan measures the consequence, that the gate reads a world frame speed and therefore passes a limb carried along by a translating body at 0.860.

Pose regularisation toward a default joint configuration, used by Extreme Parkour [3] and two others, makes no reference to contact state at all and constrains only how far a joint may sit from neutral. The Raibert heuristic footstep placement of Walk These Ways [5] prices horizontal foot position against a nominal stance rectangle and likewise says nothing about load.

## 4. Four structural differences against this workspace

The first is reward clipping. Seven of the nine clip the summed step reward at zero, and Walk These Ways instead combines the positive and negative parts as `rew_pos * exp(rew_neg / 0.02)`, so in every case a policy cannot be driven to a large negative return by the penalty stack. Isaac Lab's `RewardManager` performs no such clipping and this workspace inherits none, which is visible in the training logs as 212 to 321 iterations of genuinely negative mean reward per run, two to four per cent of all iterations, and as single iteration excursions to -539 and -631. Whether this matters is untested here, and it is recorded as the largest unexamined structural difference rather than as a diagnosis.

The second is the tracking prize. This workspace pays 10 and 5 for linear and angular tracking where every surveyed repository pays 1.0 and 0.5, a factor of ten. That would be a presentational difference if the kernel were correspondingly loose, and it is not. The kernel is exp of the negative squared error over a variance which the curriculum tightened from 0.16 to 0.0151 over run B, tighter than legged_gym's 0.25, and the policy still realised 96.7 per cent of the linear tracking ceiling and 96.3 per cent of the angular. The consequence is that the two tracking terms sum to 289.80 reward units per episode against a realised total of 283.37, so they exceed the whole return and the entire remaining specification, gait terms and penalties together, nets to -6.43. Every question about gait is settled inside that residue. Tightening the kernel further will not change this, having already been tried by the curriculum and answered by the policy simply tracking better.

The third is the entropy coefficient, where this workspace now stands alone at 0.001 against 0.01 everywhere else. That divergence was deliberate, was made on measurement in phase 1, and section 4.2 records that it worked, so the comparison is noted and no change is proposed.

The fourth is the rollout length, `num_steps_per_env` at 25 here against 24 in seven repositories and 100 in HIMLoco [4]. HIMLoco also trains to 200000 iterations against the 1500 to 5000 typical elsewhere, alongside substantially heavier domain randomisation. This workspace sits with the majority and no change is proposed, but the HIMLoco figure is recorded because it is the one surveyed configuration whose horizon resembles this one's.

## 5. Limitations of the current work that the comparison exposes

The configuration carries no term that prices which feet are on the ground, and it is not alone in that, but it is alone in carrying four terms that are individually satisfied by a limb leaving the support pattern while carrying none that objects. Phase 2 of the plan closes the objection and phases 4 and 6 address the rest.

The configuration prices mechanical power at nothing, charging torque and joint velocity separately at -2e-4 and -5e-5 and therefore unable to express their product. legged loco carries `joint_power` at -2e-5 and the energy minimisation result [6] obtains emergent gaits from a power term as essentially its only shaping. `joint_powers_l1` exists in this tree at `tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/mdp/rewards.py:905` and is wired nowhere.

The configuration has no gait command and no contact schedule, which is a deliberate position and is the one Walk These Ways [5] and van Marum and colleagues take opposite sides of. It should be held only as long as a clock free construction is producing coordination, and section 4.2 of the plan records that run B's coordination is partial, the front left and rear right diagonal correlating at 0.412 at zero lag while the other diagonal correlates at -0.410, so the two diagonals are not alternating as a trot requires.

## 6. Configuration changes worth testing, in the order the evidence supports

The first is phase 2 of the plan, applied 2026-09-10, being the only change that addresses the measured failure directly. The second is a mechanical power term at -2e-5, the figure legged loco carries, which is change 4.2 of phase 4 and which must follow phase 2 because an energy term rewards a limb that does no work. The third is the relative velocity correction to the clearance gate, which is phase 6 and which the measurement predicts will recover only 0.73 reward units per episode, so it is a correctness repair and not a remedy. The fourth is reward clipping at zero, matching seven of the nine repositories, which is untested here and is the largest structural divergence remaining. The fifth is a contact schedule in the manner of Walk These Ways [5], which is the only surveyed mechanism known to work and which is held in reserve because adopting it abandons the clock free position the plan has so far defended.

## 7. Bibliography

1. Legged Robotics Group, ETH Zurich. `leggedrobotics/legged_gym`. Software artefact accompanying Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2021), Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning, CoRL 2021, arXiv:2109.11978. Cloned to `/ws/external/legged_gym`.
2. Unitree Robotics. `unitreerobotics/unitree_rl_gym`. Software artefact with no accompanying article identified. Cloned to `/ws/external/unitree_rl_gym`.
3. Cheng, X., Shi, K., Agarwal, A., Pathak, D. (2024). Extreme Parkour with Legged Robots. ICRA 2024. arXiv:2309.14341. Cloned to `/ws/external/extreme-parkour`.
4. Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., Pang, J. (2024). Hybrid Internal Model, Learning Agile Legged Locomotion with Simulated Robot Response. ICLR 2024. arXiv:2312.11460. Cloned to `/ws/external/HIMLoco`.
5. Margolis, G. B., Agrawal, P. (2022). Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior. CoRL 2022. arXiv:2212.03238. Cloned to `/ws/external/walk-these-ways`.
6. Fu, Z., Kumar, A., Malik, J., Pathak, D. (2021). Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots. CoRL 2021. arXiv:2111.01674. No training code located, only the project website.
