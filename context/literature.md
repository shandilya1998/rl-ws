# literature.md — Literature Survey for the Learned-Model Co-Optimisation Architecture

This file records the verified literature survey performed on 2026-07-02 in
support of the learned-model extension of the COPT pipeline (encoder-decoder
privileged estimator trained jointly with PPO inside the CMA-ES morphology
loop). All citations were verified against arXiv, OpenReview, PMLR, Science
Robotics, Nature, AAAI, the Royal Society, and project pages. The main design
document consuming this survey is `../plans/COPT_LEARNED_MODEL.md`.

## Cluster 1: Privileged learning and estimator architectures in legged locomotion

- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. (2020). Learning
  quadrupedal locomotion over challenging terrain. Science Robotics 5(47),
  eabc5986. Teacher policy trained with privileged terrain and contact
  information, distilled into a proprioception-only student deployed zero-shot
  on ANYmal. Canonical two-stage privileged-learning recipe that the proposed
  estimator compresses into a single stage.
- Kumar, A., Fu, Z., Pathak, D., Malik, J. (2021). RMA: Rapid Motor Adaptation
  for Legged Robots. RSS 2021, arXiv:2107.04034. Base policy conditioned on a
  latent encoding of privileged environment parameters (mass, friction,
  terrain), adaptation module regresses the latent from proprioceptive history.
  Direct precedent for conditioning the actor on a low-dimensional latent of
  privileged physical parameters including body-parameter variation.
- Ji, G., Mun, J., Kim, H., Hwangbo, J. (2022). Concurrent Training of a Control
  Policy and a State Estimator for Dynamic and Robust Legged Locomotion. IEEE
  RA-L 7(2), 4630-4637, arXiv:2202.05481. Policy and supervised state estimator
  (base velocity, foot height, contact probability) trained concurrently in one
  loop with the estimator output fed to the policy. Closest single-stage
  precedent for a supervised regression head on privileged dynamic quantities.
- Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., Pang, J. (2024). Hybrid Internal
  Model: Learning Agile Legged Locomotion with Simulated Robot Response. ICLR
  2024, arXiv:2312.11460. HIMLoco learns a hybrid internal embedding (explicit
  velocity plus implicit stability latent) from proprioceptive history,
  optimised jointly with PPO through an auxiliary contrastive objective in a
  single phase. Stated inspiration for the proposal, the MSE decoder is a
  regression-based sibling of HIM's contrastive loss.
- Nahrendra, I. M. A., Yu, B., Myung, H. (2023). DreamWaQ: Learning Robust
  Quadrupedal Locomotion With Implicit Terrain Imagination via Deep
  Reinforcement Learning. ICRA 2023, arXiv:2301.10602. Asymmetric actor-critic
  with a context-aided estimator network, a variational encoder-decoder trained
  jointly with PPO whose latent conditions the actor. The most direct
  architectural precedent for the proposal.
- Miki, T., Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. (2022).
  Learning robust perceptive locomotion for quadrupedal robots in the wild.
  Science Robotics 7(62), eabk2822, arXiv:2201.08117. Attention-based recurrent
  encoder fusing exteroceptive height samples with proprioception under
  privileged learning. Supports feeding a sparse height map into the encoder.

## Cluster 2: Auxiliary and representation losses reducing RL sample complexity

- Jaderberg, M., Mnih, V., Czarnecki, W. M., Schaul, T., Leibo, J. Z., Silver,
  D., Kavukcuoglu, K. (2017). Reinforcement Learning with Unsupervised
  Auxiliary Tasks. ICLR 2017, arXiv:1611.05397. UNREAL, auxiliary control and
  prediction heads sharing the policy trunk, about 10x faster learning on
  Labyrinth. Foundational evidence that auxiliary prediction gradients shape
  shared representations and cut sample complexity.
- Laskin, M., Srinivas, A., Abbeel, P. (2020). CURL: Contrastive Unsupervised
  Representations for Reinforcement Learning. ICML 2020, arXiv:2004.04136.
  Contrastive auxiliary loss on the encoder of a model-free agent, large
  data-efficiency gains, supports single-optimiser joint representation and RL
  objectives.
- Yarats, D., Zhang, A., Kostrikov, I., Amos, B., Pineau, J., Fergus, R. (2021).
  Improving Sample Efficiency in Model-Free Reinforcement Learning from Images.
  AAAI 2021, 35(12), 10674-10681, arXiv:1910.01741. SAC+AE, a regularised
  autoencoder whose shared encoder receives gradients from both reconstruction
  and Q-learning. Most direct evidence that an MSE reconstruction decoder
  combined with the RL loss on one encoder is stable and sample-efficient.
- Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., Bachman, P.
  (2021). Data-Efficient Reinforcement Learning with Self-Predictive
  Representations. ICLR 2021, arXiv:2007.05929. SPR, predicting future latent
  states as an auxiliary objective, 55 percent relative improvement on Atari
  100k. Supports auxiliary losses forcing the latent to carry dynamics
  information.
- Lyle, C., Rowland, M., Ostrovski, G., Dabney, W. (2021). On the Effect of
  Auxiliary Tasks on Representation Dynamics. AISTATS 2021, PMLR 130,
  arXiv:2102.13089. Theoretical analysis of how auxiliary prediction tasks
  shape TD representations. Principled grounding for privileged-dynamics
  targets inducing task-aligned latents.

## Cluster 3: Dynamics and model information for sample efficiency

- Chua, K., Calandra, R., McAllister, R., Levine, S. (2018). Deep Reinforcement
  Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS
  2018, arXiv:1805.12114. PETS, probabilistic ensemble dynamics models matching
  model-free asymptotic performance with orders of magnitude fewer samples.
- Janner, M., Fu, J., Zhang, M., Levine, S. (2019). When to Trust Your Model:
  Model-Based Policy Optimization. NeurIPS 2019, arXiv:1906.08253. MBPO, short
  learned-model rollouts inside model-free optimisation with usefulness
  guarantees. The proposal takes the safer middle path, dynamics prediction as
  a representation-shaping signal only, avoiding compounding rollout error.
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2023/2025). Mastering
  Diverse Domains through World Models (DreamerV3). arXiv:2301.04104, Nature
  640 (2025). Latents shaped by dynamics prediction are sufficient statistics
  for control across more than 150 tasks.

## Cluster 4: Morphology-control co-optimisation

- Sims, K. (1994). Evolving Virtual Creatures. SIGGRAPH 1994, 15-22. Founding
  precedent for outer-loop evolutionary morphology search wrapped around
  controller optimisation.
- Cheney, N., Bongard, J., SunSpiral, V., Lipson, H. (2018). Scalable
  co-optimization of morphology and control in embodied machines. J. R. Soc.
  Interface 15(143), 20170937. Identifies the core pathology of co-design,
  morphology changes invalidate the co-adapted controller, motivating anything
  that accelerates per-candidate adaptation.
- Schaff, C., Yunis, D., Chakrabarti, A., Walter, M. R. (2019). Jointly
  Learning to Construct and Control Agents using Deep Reinforcement Learning.
  ICRA 2019, arXiv:1801.01432. Single design-parameter-conditioned policy over
  a distribution of designs, direct precedent for feeding link lengths to the
  policy.
- Ha, D. (2019). Reinforcement Learning for Improving Agent Design. Artificial
  Life 25(4), 352-365, arXiv:1810.03779. Design parameters optimised jointly
  with the policy by the same machinery.
- Luck, K. S., Ben Amor, H., Calandra, R. (2020). Data-efficient Co-Adaptation
  of Morphology and Behaviour with Deep Reinforcement Learning. CoRL 2019,
  PMLR 100, 854-869, arXiv:1911.06832. Morphology-conditioned Q-function reused
  as a learned surrogate fitness, validating learned design-conditioned models
  in the outer loop.
- Wang, T., Zhou, Y., Fidler, S., Ba, J. (2019). Neural Graph Evolution:
  Towards Efficient Automatic Robot Design. ICLR 2019, arXiv:1906.05370. GNN
  policies transfer skills between related designs, amortising controller
  knowledge across morphologies.
- Gupta, A., Savarese, S., Ganguli, S., Fei-Fei, L. (2021). Embodied
  intelligence via learning and evolution. Nature Communications 12, 5721,
  arXiv:2102.02202. DERL, morphology shapes learnability at scale.
- Gupta, A., Fan, L., Ganguli, S., Fei-Fei, L. (2022). MetaMorph: Learning
  Universal Controllers with Transformers. ICLR 2022, arXiv:2203.11931.
  Morphology as an input modality to a transformer policy across a modular
  design space.
- Huang, W., Mordatch, I., Pathak, D. (2020). One Policy to Control Them All:
  Shared Modular Policies for Agent-Agnostic Control. ICML 2020, PMLR 119,
  4455-4464, arXiv:2007.04976. One shared policy spans a morphology family, a
  latent identifying the current body suffices for specialisation.
- Won, J., Lee, J. (2019). Learning Body Shape Variation in Physics-based
  Characters. ACM TOG 38(6), Article 207, SIGGRAPH Asia 2019. Controllers
  functional across continuous body-shape variation, the same class of design
  variables as the TRON1A thigh and shank scales.

## Cluster 5: Co-design with learned latent representations

- Yuan, Y., Song, Y., Luo, Z., Sun, W., Kitani, K. (2022). Transform2Act:
  Learning a Transform-and-Control Policy for Efficient Agent Design. ICLR 2022
  oral, arXiv:2110.03659. One learned representation serves design
  transformation and control within a single PPO optimisation.
- Schaff, C., Walter, M. R. (2022). N-LIMB: Neural Limb Optimization for
  Efficient Morphological Design. arXiv:2207.11773. Universal design-conditioned
  controller trained jointly with a distribution over morphologies.
- Strgar, L., Kriegman, S. (2025). Accelerated co-design of robots through
  morphological pretraining. arXiv:2502.10862, accepted at ICLR 2026.
  Morphology-agnostic controllers evaluated zero-shot on new candidate designs,
  recent direct evidence that learned morphology-aware controllers reduce the
  per-candidate cost of evolutionary design loops.

## Cluster 6: LIPM-guided rewards and template models (added 2026-07-10 for LIPM_REWARD.md)

- Su, H., Luo, H., Yang, S., Jiang, K., Zhang, W., Chen, H. (2025). LIPM-Guided
  Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots.
  arXiv:2509.09106 (local copy /ws/2509.09106v1.pdf, text extract in scratchpad).
  LimX point-foot TRON1, Isaac Lab, 2048 envs, vision-CTS teacher-student.
  Constraint-plane LIPM generates desired CoM online, p_hat = p_zmp +
  (z_c/g) kp (v_cmd - v) (eq 2). Stable reward r = exp(pe + ze + |w_roll|+|w_pitch|)
  as printed (eq 4), sign must be NEGATED for r in (0,1]. RFM fusion
  r = r_stable + r_stable*r_vel (eq 5); Table II applies the r_stable gate only to
  the magnitude-tracking component, direction component stays unfused, both with
  exponent factor 4. Double critic (stability vs locomotion returns) after
  RobotKeyframing (Zargarbashi 2024, arXiv:2407.11562). Ablations on stairs,
  success 80.3 (full) vs 52.7 (no stable critic) vs 41.4 (no stable reward).
- Kajita et al. (2001) 3D-LIPM IROS, (2002) realtime pattern generator ICRA,
  (2014) Introduction to Humanoid Robotics, Springer. Constraint plane z = kx x +
  ky y + zc keeps dynamics linear with omega^2 = g/zc (intercept, not height),
  hence valid on slopes with slope-parallel plane.
- Pratt et al. 2006 capture point, Englsberger et al. 2011 capture point control,
  Wieber 2006 LMPC, Vukobratovic & Borovac 2004 ZMP survey.
- Gong & Grizzle, arXiv:2008.10763 (JDSMC 2022), ALIP, L about contact point,
  Ldot_y = m g x exact for point contact, robust to impacts. Gibson et al.
  arXiv:2109.14862 (IROS 2022) terrain-adaptive ALIP MPC on Cassie.
- Xiong & Ames, arXiv:2101.09588 (T-RO 2022), H-LIP step-to-step dynamics
  x_{k+1} = A x_k + B u_k, deadbeat stepping, invariant-set error bound.
- Caron et al. arXiv:1801.07022 (T-RO), VHIP capturability with variable height,
  Caron arXiv:1909.07732 (ICRA 2020) VHIP linear-feedback stabiliser.
- Shi et al. arXiv:2504.02255 piecewise-slope LIPM + discrete MPC for uneven
  footholds. Li et al. Sensors 21(4):1082 (2021) LIPM+LPM double support.
- Blickhan 1989 spring-mass model, Full & Koditschek 1999 templates and anchors.
- Green et al. arXiv:2010.11234 (RA-L 2021), SLIP trajectory library guides Cassie
  policy via reward. Castillo et al. IROS 2023 template-model task-space learning.
  Lee, Hong, Kim IROS 2024 LIP footstep planner + model-free RL. Jenelten et al.
  DTC Science Robotics 9(86) 2024 planner-in-the-loop tracking.
- Ng, Harada, Russell ICML 1999 potential-based shaping policy invariance.
  Jeon et al. arXiv:2307.10142 (ICRA 2023) PBRS on humanoid gives only marginal
  convergence gains, physically structured shaping needed.
- MBRL theory, Azar et al. Machine Learning 91 (2013) minimax PAC
  O(SA/((1-gamma)^3 eps^2)) for model-based with generative model, Agarwal et al.
  COLT 2020 certainty equivalence minimax optimal, Strehl et al. ICML 2006 delayed
  Q-learning model-free O(SA/((1-gamma)^8 eps^4)), Moerland et al. FnTML 2023
  survey, Sutton 1991 Dyna, Peng et al. AMP TOG 2021, Rudin CoRL 2021 parallel RL,
  Siekmann RSS 2021 blind stairs.

The document consuming this cluster is /ws/plans/LIPM_REWARD.md (2026-07-10), which
contains the full survey narrative, IPM derivations, the design-parameterised
reward mathematics z_c(d) = 0.20 + 0.25 s_thigh + 0.30 s_shank, and the two-phase
implementation plan (LIPMStableReward + FusedLIPMTrackingReward in mdp/rewards.py,
CoptRewardsCfg bound in SFCoptEnvCfg). Plots grounding the motivation are
/ws/context/artefacts/plots-latest (run sf_copt/2026-07-03_08-16-11, 45k iterations, rew_lin_vel_xy
oscillating 0-20 to the end, value loss final 11.27, fall rate 11.3%, curriculum
still advancing at 45k).

## Synthesis

The locomotion line (Lee, Kumar, Ji, Miki, Nahrendra, Long) shows privileged
simulation quantities are best exploited by encoding them into a latent that
conditions the actor, and the field has moved from two-stage distillation
toward single-stage concurrent estimators trained jointly with PPO, DreamWaQ
and HIMLoco providing the exact encoder-decoder-plus-auxiliary-loss template.
The representation line (Jaderberg, Laskin, Yarats, Schwarzer, Lyle)
establishes empirically and theoretically that auxiliary reconstruction and
prediction losses sharing an encoder with the RL objective in a single
optimiser reliably reduce sample complexity. The model-based line (Chua,
Janner, Hafner) shows dynamics knowledge is the dominant lever on sample
efficiency, imported here safely as a prediction target rather than a rollout
generator. The co-design line (Sims through Strgar and Kriegman) identifies
per-candidate controller adaptation as the bottleneck of morphology
optimisation and shows design-conditioned policies amortise learning across
bodies. The proposed encoder over link lengths, masses, terrain, and history,
with a decoder regressing torques, accelerations, inertia, and contact
quantities, sits at the intersection of all four lines.

## Cluster 7: Gait quality, symmetry, and posture shaping for bipeds (added 2026-07-17 for GAIT_STRATEGY.md)

- Siekmann et al., Sim-to-Real Learning of All Common Bipedal Gaits via Periodic
  Reward Composition, ICRA 2021, arXiv:2011.01387. Probabilistic phase clock
  intervals gate penalties on foot force during swing and foot velocity during
  stance, a (frequency, offset, duration) parameterisation spans standing,
  walking, hopping, running, skipping, transferred to Cassie. The unused
  GaitReward class in this repo's mdp/rewards.py is the Walk These Ways variant
  of exactly this construction.
- van Marum et al., Revisiting Reward Design and Evaluation for Robust Humanoid
  Standing and Walking, arXiv:2404.19173 (Digit). Minimally constraining reward,
  Table I weights, xy velocity 0.15 each, roll-pitch orientation 0.2 (heavier
  than each velocity term), single-foot-contact 0.1 with a 0.2 s grace window,
  base height 0.05, feet airtime sum(t_air - 0.4) at touchdown weight 1.0 (the
  only sparse term, deliberately the largest weight), base acceleration 0.1,
  action difference 0.02, torque 0.02. Velocity-only rewards yield two-footed
  hopping, the single-foot-contact reward is reported as the most reliable,
  tuning-free fix, clocks are rejected because they impede disturbance
  rejection, PPO is extended with a mirror loss, LSTM (64,64), 50 Hz control.
- Mittal et al., Symmetry Considerations for Learning Task Symmetric Robot
  Policies, ICRA 2024, arXiv:2403.04359. Compares symmetry data augmentation
  against mirror loss on PPO, the interface is implemented verbatim in the
  vendored rsl_rl (symmetry_cfg with use_data_augmentation, use_mirror_loss,
  data_augmentation_func) and surfaced as RslRlSymmetryCfg in isaaclab_rl.
- Yu, Turk, Liu, Learning Symmetric and Low-Energy Locomotion, SIGGRAPH 2018.
  Mirror symmetry loss added to the policy objective plus energy terms and
  curriculum produce symmetric gaits without any reference motion.
- Gait-Conditioned RL with Multi-Phase Curriculum for Humanoid Locomotion,
  arXiv:2505.20619. Human-inspired reference-free terms, contact pattern,
  foot clearance, straight knee during stance, anti-phase arm-leg swing via a
  Z-axis angular momentum penalty, three-phase curriculum walk, stand and
  transitions, then run.
- Peng et al., AMP, SIGGRAPH 2021, arXiv:2104.02180, and HumanMimic (Tang et
  al., arXiv:2309.14225, Wasserstein adversarial imitation for humanoids).
  Discriminator style rewards over retargeted mocap replace hand-tuned gait
  shaping when human-likeness is the objective, at the cost of a dataset,
  retargeting, and adversarial training stability work.
- IsaacLab reference configs (local checkout, G1Rewards in
  isaaclab_tasks/.../velocity/config/g1/rough_env_cfg.py). Tracking 1.0 and
  2.0, feet_air_time_positive_biped weight 0.25 threshold 0.4 (dense,
  non-negative, single-stance gated), flat_orientation_l2 -1.0 (half the
  linear tracking weight), joint_deviation_l1 on hip yaw-roll, arms, torso,
  is_terminated penalty -200, commands lin_vel_x (0.0, 1.0). These ratios are
  the field's working defaults for upright biped posture.
- Su et al., arXiv:2509.09106, LIPM-guided stability reward, already surveyed
  in cluster 6 and planned in /ws/plans/LIPM_REWARD.md, complements rather than
  replaces the posture and gait terms above.

## Cluster 8: Cadence, contact scheduling, and the anti-hold structure of gait rewards (added 2026-07-20 for run 2026-07-20_07-52-14)

Cluster 7 was surveyed to answer why a biped will not leave double support. This cluster answers the opposite question, why a biped that has reached single support will not leave it, which is the failure of run 2026-07-20_07-52-14.

- van Marum et al., arXiv:2404.19173, Table I retrieved in full from the HTML version on 2026-07-20 and now recorded exactly, since the earlier note paraphrased it. x and y velocity 0.15 each, yaw orientation 0.1, roll and pitch orientation 0.2, feet contact 0.1, base height 0.05, feet airtime 1.0, feet orientation 0.05, feet position 0.05, arm 0.03, base acceleration 0.1, action difference 0.02, torque 0.02. Two structural facts matter here and neither was recorded before. First, the feet airtime term is SPARSE and touchdown triggered, sum over feet of (t_air - 0.4) times the touchdown indicator, and it carries the LARGEST weight in the table, so their reward pays for step events and pays nothing at all to a robot that holds a foot up without landing it. Second, their feet contact term is 1 if single contact occurred at least once in the preceding 0.2 s, a GRACE WINDOW, so brief double support during weight transfer is not punished. Our no_fly has no such window and therefore prices every instant of double support at zero, which is what drives the transition count down. Their reported velocity only failure mode is two footed hopping. Their three arguments against periodic clocks are the undefined clock input during standing and transitions, the incentive towards low foot velocities in standing mode which impedes disturbance rejection, and the constraint on free foot motion during recovery from a disturbance.
- Humanoid-Gym, Gu et al., arXiv:2404.05695. A periodic stance mask I_p(t) defines the planned contact schedule, two double support and two single support phases per cycle, and the contact pattern reward phi(I_p(t) - I_d(t)) rewards the measured contacts for matching it, at weight 1.0 which is the largest term in their set. The clock pair sin(2 pi t / C_T), cos(2 pi t / C_T) is supplied as a policy input. This is the most direct available answer to the requirement that something in the reward set specify WHEN each foot should be down, and unlike an air time term it cannot be farmed by holding, since holding is off schedule by construction. The paper does not state how C_T is chosen, which must therefore be set from the robot's own pendulum time constant.
- Learning Sim-to-Real Humanoid Locomotion in 15 Minutes, arXiv:2512.01996. Argues that the customary twenty plus term humanoid reward can be reduced below ten terms, and notably uses foot height TRACKING against a reference swing profile rather than a Gaussian on height multiplied by foot speed. That formulation is immune to the parked and waved foot exploit, because a foot that remains high is off the reference just as a foot that never rises is. The paper does not itself discuss degenerate gaits, so this is an inference from its reward form and should be labelled as such.
- Gait-Conditioned RL with Multi-Phase Curriculum, arXiv:2505.20619, already in cluster 7, adds here that a gait conditioned reward routing scheme keeps gait specific contact and push off terms from interfering across walking, running and standing, and that the policy emits gait frequency as an action constrained to [0.7, 1.3] with an action smoothness term on it. Relevant if a commanded cadence is later made adaptive rather than constant.
- Denoising World Model Learning, arXiv:2408.14472. Constructs its reward from the complementary roles of foot force and foot velocity, promoting high foot velocity during swing and appropriate foot force during stance. This is the Siekmann periodic construction restated, and it is the form the unused GaitReward class in this repository already implements.
- Standing caution recorded from this survey. Neither van Marum's sparse touchdown term nor Humanoid-Gym's contact mask rewards the step COUNT directly. van Marum's (t_air - 0.4) actually pays more for one long hold followed by a landing than for several short swings, so it is not by itself an anti hold device, its protection comes from being touchdown gated combined with the grace window on the contact term. Only the clock based family prices cadence explicitly. This distinction was not obvious from the abstracts and is the main reason the full texts were retrieved.

## Cluster 9: Symmetry enforcement in RL legged locomotion (added 2026-07-22 for the SD_BRS1 limp, run 2026-07-21_06-03-36)

This cluster answers why the GaitReward run alternates its feet yet still limps, and grounds the symmetry data augmentation plan in /ws/plans/SYMMETRY_PLAN.md. All equations and numbers below were read from fetched PDFs, the two saved locally are the primary paper and the Abdolhosseini paper.

- Su et al., "Leveraging Symmetry in RL-based Legged Locomotion Control", arXiv 2403.17320, IROS 2024, the paper and SymmLoco code the user referenced. It models the robot's sagittal symmetry as the order two reflection group C2 equal to {e, g} with g squared equal to e, and proves that when the transition density, the initial state density and the reward are all C2 invariant, the optimal policy is equivariant, g applied to the optimal action equals the optimal action at the reflected state, and the optimal value is invariant. It compares THREE methods, a vanilla PPO baseline, PPOaug which is data augmentation, and PPOeqic which is a hard equivariant network built with MorphoSymm and ESCNN, and it DELIBERATELY EXCLUDES the mirror loss, citing the Mittal note that the loss is outperformed by augmentation. Verified results, the unconstrained baseline scores a Robinson symmetry index four times higher than PPOaug and eight times higher than PPOeqic, and PPOeqic is the most sample efficient and accurate in simulation. The decisive recommendation for us, for TASK SPACE symmetry such as door pushing the hard equivariant policy excels and even transfers from one trained side to both, but for INTRINSIC MOTION symmetry, real bipedal walking where actuator and mass asymmetries break the perfectly symmetric assumption, the hard constraint becomes brittle under distribution shift and the softer PPOaug is more robust because it adapts to the robot's real asymmetries. The SD_BRS1 is a real biped with an identified and therefore asymmetric inertial model, so data augmentation is the grounded first choice. A temporal gait phase signal psi is added with g applied to psi equal to psi plus 0.5 modulo 1, a half cycle offset, which also sidesteps the neutral state problem by ensuring the reflected state differs from the original even in a symmetric pose.
- Abdolhosseini et al., "On Learning Symmetric Locomotion", Motion in Games 2019, ACM DOI 10.1145/3359566.3360070, not on arXiv, full ACM PDF read. This is the paper that catalogues the FOUR practical methods, named DUP, LOSS, PHASE, NET. DUP duplicates every transition through the mirror and adds both to the update, the data augmentation route. LOSS adds the auxiliary mirror penalty. PHASE learns half a cycle and replays it mirrored, suited to periodic imitation. NET is the hard equivariant network. Reported symmetry indices lower is better, Walker2D baseline 3.97, DUP 3.77, LOSS 2.56, NET 2.00, Cassie baseline 9.27, PHASE 4.49, NET 5.15, LOSS 15.72, so no single method wins everywhere, LOSS is the most consistent soft method, NET the only exact one, DUP the weakest enforcer but the cheapest to add. Two cautions, symmetry enforcement gives no reliable learning speedup and can even slow it, contradicting the naive 2x expectation, and a strictly symmetric policy cannot leave a symmetric neutral pose, the neutral state problem, so training must start from a noised non neutral posture.
- The mirror loss form, introduced by Yu, Turk, Liu, "Learning Symmetric and Low-Energy Locomotion", arXiv 1801.08093, adopted verbatim by Abdolhosseini as equation two, L_sym(theta) equals the sum over t of the squared norm of pi_theta(s_t) minus M_a applied to pi_theta applied to M_s(s_t), where M_s mirrors the state and M_a mirrors the action, added to the PPO objective with a scalar weight, both papers use 4. This is exactly the loss IsaacLab's RslRlSymmetryCfg computes when use_mirror_loss is set.
- Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies", arXiv 2403.04359, the IsaacLab symmetry note, evaluated on ANYmal. Its central and directly load bearing point for our implementation, when a mirrored sample enters the PPO surrogate the importance ratio must keep the ORIGINAL transition's old action log probability in the denominator, not the mirrored sample's own probability under the old policy, since the latter can be arbitrarily small and would make the ratio and its variance explode. It provides a state visitation ratio correction for the induced distribution shift, and it reports empirically that data augmentation reaches faster convergence and better behaviour than the symmetry loss, which is why the primary paper drops the loss. The rsl_rl at /ws/rsl_rl already satisfies this, it tiles old_actions_log_prob from the originals rather than recomputing it on the mirrored action, so the mirrored row is scored against the original old probability.
- Ordonez Apraez et al., "On discrete symmetries of robotics systems, a group theoretic and data driven analysis", arXiv 2302.10433, RSS 2023 and IJRR 2024, the MorphoSymm library. Establishes the group theoretic framework and the principle that a morphological symmetry propagates to a representation acting on all proprioceptive and exteroceptive measurements, states, actions, dynamics, optimal policy and value. Four of its authors co author the primary paper. SymmLoco builds its reps from this library's a1.yaml, group C2, joint permutation and a per joint reflection vector whose signs are determined by the joint frame orientations rather than by an abstract roll versus pitch rule, which is exactly why the SD_BRS1 HipPitch flips despite being a pitch joint, its left and right frames carry opposite axes.
- The biped sagittal sign table, derived from reflection physics and consistent with the MorphoSymm principle. A polar vector, base linear velocity or projected gravity, flips only its lateral y component, multiply by [1, -1, 1]. A pseudovector, base angular velocity, flips its in plane roll and yaw components and keeps pitch, multiply by [-1, 1, -1]. The velocity command flips lateral and yaw, multiply by [1, -1, -1]. The joints swap left for right with a sign flip on the degrees of freedom whose axis leaves the sagittal plane. The gait phase [sin, cos] negates both channels because a left right swap shifts the shared clock by exactly half a cycle while the command offset is 0.5.
- The gait symmetry metric to log, the Robinson symmetry index, SI equal to 2 times the absolute difference of the left and right measurements divided by their sum, times 100, lower is more symmetric, used by both the primary and foundational papers, originally Robinson, Herzog, Nigg 1987.
- Later phase notes, weakly grounded and marked as such. No fetched paper links the symmetry family directly to knee flexion, the stiff leg fix is generic reward shaping not a symmetry mechanism. For standing on a zero velocity command inside a periodic gait, Siekmann et al., arXiv 2011.01387, treats standing as a first class gait by setting the contact schedule so both feet may remain in stance at zero commanded velocity, the clean formulation for the standing shuffle of observation 3.

## Cluster 10: Natural gait, knee flexion, upright posture, and actuation smoothness (added 2026-07-23 for the SD_BRS1 baseline run 2026-07-22_11-36-53)

This cluster grounds `/ws/plans/NATURAL_GAIT_PLAN.md`, the plan for the three naturalness defects of the first successful SD_BRS1 walker, the locked knee, the forward lean, the non smooth actuation.

Knee and natural gait. The straight STANCE knee is energetically correct, established in passive dynamic walking, McGeer, Passive Dynamic Walking, IJRR 9(2) 1990, and confirmed by emergence under pure effort objectives, Heess et al., Emergence of Locomotion Behaviours in Rich Environments, arXiv 1707.02286. The pathology is the missing SWING flexion, human gait flexes the knee in swing to shorten and clear the leg, Winter, Biomechanics and Motor Control of Human Movement. The reference free fix is phase gated joint posture shaping, Peng, Bao, Zhou, Gait-Conditioned RL with Multi-Phase Curriculum, arXiv 2505.20619, which adds a straight knee during stance term at weight 0.1 on the Unitree G1 with no motion capture, its one fully published biomechanical term being the yaw angular momentum R = minus(L_z)^2 minus 0.4(L_la,z minus L_ra,z)^2 at weight 5.0. The imitation fix is DeepMimic, Peng et al., arXiv 1804.02717, tracking a reference with pose, velocity, end effector, and centre of mass terms and reference state init, or its phase free adversarial form AMP, Peng et al., arXiv 2104.02180, style reward r_S = max(0, 1 minus 0.25(D(s,s') minus 1)^2), shown for legged sim to real by Escontrela et al., arXiv 2203.15103, with a lower cost of transport than hand designed style rewards.

Posture. The projected gravity penalty flat_orientation_l2 is the legged_gym standard, Rudin et al., Learning to Walk in Minutes, arXiv 2109.11978, symmetric and reference free. Weight calibration from van Marum et al., Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking, arXiv 2404.19173, IROS 2024, an orientation to linear tracking ratio near two thirds, and the IsaacLab G1 sets it at parity, which brackets our term at minus thirty three to minus fifty against the 50 tracking weight, so minus thirty five is adequate and the lean is a shaping problem. Note that van Marum AVOIDS clock based rewards, arguing they hinder disturbance recovery, a caveat against leaning further on the GaitReward clock for posture.

## Cluster 11: Reward kernel failure, nominal pose, and termination shaping (added 2026-07-27 after the Phase A knee reward failed in run 2026-07-23_11-31-57)

This cluster answers why a correctly weighted Gaussian knee reward produced literally nothing, and what working systems do instead. It supersedes the cluster 10 assumption that a swing gated flexion reward is the right instrument.

THE HEADLINE, negative and decisive. NO mainstream bipedal or humanoid codebase obtains knee flexion from a reward. Every one carries it in the DEFAULT JOINT POSE of the articulation. IsaacLab `isaaclab_assets/robots/unitree.py` G1 knee 0.42, hip pitch -0.20, ankle pitch -0.23, H1 knee 0.79, hip pitch -0.28, ankle -0.52. unitree_rl_gym G1 knee 0.30, hip pitch -0.10, ankle -0.20. Walk These Ways (Margolis arXiv 2212.03238) Go1 calf -1.5, thigh 0.8 to 1.0. The invariant across every configuration surveyed is that the default is NEVER a straight knee. Where a reward touches joint posture it is `joint_deviation_l1`, the summed absolute deviation FROM that already flexed default, never a reward pulling AWAY from a straight one. IsaacLab G1 applies it to hip yaw and hip roll at -0.1, arms -0.1, torso -0.1, and DELIBERATELY NOT to hip pitch or the knee, the two DOF that must swing. Digit, the closest true biped analogue, does add a knee deviation but only at -0.2, with hip yaw -0.2, hip roll -0.1, `flat_orientation_l2` -2.5 and a `bad_orientation` termination at limit_angle 0.7 rad. The established remedy for compensatory OFF AXIS hip rotation is therefore a small deviation penalty on hip roll and yaw at about -0.1, never a penalty on hip pitch amplitude, which is the primary swing actuator.

KERNEL WIDTH must be sized to the error the policy CURRENTLY produces, not to the task range. legged_gym sets `tracking_sigma` 0.25 in SQUARED error units, an effective tolerance near 0.5 m/s chosen from the achievable tracking error. Peng et al., Learning Agile Robotic Locomotion Skills by Imitating Animals, RSS 2020, arXiv 2004.00784, weights five exponential kernels whose interior coefficients span 0.1 (joint velocity) through 5 (joint angle) and 20 and 10 (root pose) to 40 (end effector Cartesian), each set from that quantity's natural error scale, so a single width is never reused across quantities. Humanoid-Gym arXiv 2404.05695 is the instructive contrast, it keeps ALL defaults at zero but drives the knee from a phase conditioned reference `sin_pos = sin(2 pi phase)` with a knee excursion amplitude of about 0.34 rad, tracked by `exp(-2 ||diff||) - 0.2 clamp(||diff||,0,0.5)`, so the target FOLLOWS the policy and the kernel is wide, a 0.3 rad error still returning 0.55. A reference tracking reward can work from a zero default, but only when the target is never far from the current pose.

STRUCTURAL ALTERNATIVES to a peaked kernel. Kim et al., Not Only Rewards But Also Constraints, IEEE T-RO, arXiv 2308.12517, replaces kernel shaped terms with CONSTRAINT COSTS that are zero inside a desired range read from the URDF limits and grow outside it, collapsing ten or more hand tuned terms to effectively one coefficient. A constraint cost cannot vanish the way a distant Gaussian does. Potential based shaping, Ng, Harada and Russell, ICML 1999, `F(s,a,s') = gamma Phi(s') - Phi(s)` is necessary and sufficient for policy invariance, with Wiewiora, JAIR 19, 205, adding that EPISODIC tasks need `Phi(terminal) = 0` or the return is biased, which matters where episodes terminate early. But Jeon et al., arXiv 2307.10142, benchmarking PBRS on humanoid locomotion, finds only MARGINAL convergence gains and that the real benefit is robustness to term scaling, so PBRS is not a cure for a vanishing gradient. Curriculum on the TARGET is the third route, Bengio et al., ICML 2009, with a concrete scheduled instance in Babadi et al., MIG 2019, arXiv 1907.11842, whose termination threshold anneals 0.75 to 0.5 across training, strict early and relaxed as competence grows.

TERMINATION. DeepMimic's own ablation shows early termination matters enormously for acrobatics and NOT AT ALL for plain walking, backflip 0.791 against 0.730 and sideflip 0.823 against 0.717 but walk 0.980 against 0.981, so a high termination rate on a walking task is better read as a symptom than a cause. More importantly, NO IsaacLab humanoid reference config uses a HEIGHT termination at all, the base velocity template terminates only on time out and illegal contact with the base body and G1, H1 and Digit merely override that contact body to the torso. He et al., H2O, IROS 2024, arXiv 2403.04436, uses projected gravity on x or y above 0.7 plus a base height below 0.3 m, a floor far BENEATH the standing height rather than just below it. A height threshold set just under the stance height truncates precisely the crouched states a policy must explore to discover that knee flexion is viable, which compounds a vanishing reward with a data availability problem.

## Cluster 12: Provenance of the contact history index, and the ancestry of this codebase (added 2026-07-27)

Not a reward design cluster but an implementation archaeology one, recorded because it establishes this project's ANCESTRY, which is useful well beyond the immediate question.

THE LINEAGE. The direct IsaacLab ancestor of `exts/bipedal_locomotion` is `Andy-xiong6/bipedal_locomotion_isaaclab` (github), which carries the identical directory skeleton and the "LimX Point Foot's locomotion task" docstring. The Isaac Gym sibling for the same TRON1 hardware is `limxdynamics/tron1-rl-isaacgym`, formerly `pointfoot-legged-gym`. Both descend from `leggedrobotics/legged_gym`. This repo's own genesis commit `b5217a9` (2025-07-14) is a VENDOR SQUASH from jenkins_ci@limxdynamics.com whose entire message is a version string, so nothing before it has a surviving rationale.

THE VERDICT on the `[:, -1]` contact index, a PORTING BUG on five grounds. IsaacLab DOCUMENTS the ordering, `contact_sensor_data.py:84-91`, first index most recent, last index oldest. No first party IsaacLab code reads a trailing index anywhere, only `[:, 0]` or a max or mean over the whole axis, and the framework's own debounce primitives `compute_first_contact` and `compute_first_air` use tracked contact and air time scalars rather than slicing the history. The Isaac Gym ancestors have NO history axis at all, `self.contact_forces` is instantaneous in legged_gym, unitree_rl_gym and humanoid-gym alike, and the LimX biped debounce is `contact_filt = contact OR last_contacts` with a manually cached previous step, temporally correct by construction, which is exactly the semantics the port was reproducing. The direct IsaacLab ancestor has `no_fly` at `[:, 0]` correct throughout and lacks `keep_ankle_pitch_zero_in_air` entirely. And the local git history shows `no_fly` being REGRESSED from correct to incorrect by commit `a2f1e2d` as an unremarked side effect, with a later commit fixing a different bug in the same neighbourhood without noticing.

GENERAL LESSON worth keeping. When porting a legged_gym derived reward to IsaacLab, the contact representation CHANGES SHAPE, from an instantaneous `(N, B, 3)` tensor plus a hand cached previous step to a rolling `(N, T, B, 3)` history whose newest sample is at index ZERO. The python habit that `[-1]` means most recent is exactly wrong here. Any reward that asks whether a foot is on the ground right now should use `[:, 0]`, and any that wants debouncing should either OR `[:, 0]` with `[:, 1]` or reduce by max over the whole axis, the latter being what all first party IsaacLab code does.

Smoothness for sim to real. The strongest instrument is CAPS, Mysore et al., Regularizing Action Policies for Smooth Control with RL, arXiv 2012.06644, ICRA 2021, two policy space regularisers, temporal L_T = ||pi(s_t) minus pi(s_{t+1})||_2 and spatial L_S = ||pi(s_t) minus pi(s_bar)||_2 with s_bar drawn from N(s, sigma), added as J minus lambda_T L_T minus lambda_S L_S, reporting a 96 percent smoothness improvement and an 80 percent power reduction on a real quadrotor, damping the state to action map directly rather than paying for roughness through a reward penalty. The reward side is the legged_gym action rate and torque penalties already in the stack, plus the one term the stack lacks, a torque rate penalty. Deeper fidelity is the actuator network, Hwangbo et al., Learning agile and dynamic motor skills for legged robots, Science Robotics 4(26) 2019, closing the torque transfer gap. The GaitReward clock is itself from Siekmann et al., arXiv 2011.01387, shown to give smooth hardware transferable bipedal gaits.
