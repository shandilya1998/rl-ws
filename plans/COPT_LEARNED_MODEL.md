# Learned Model Co-Optimisation, Design Document and Implementation Plan

> Status, verified against the live sources on 2026-07-30. Implemented in full. Section 3 was executed on 2026-07-03 following every step, and the implementation record together with the two ambiguities resolved during it is in `../context/copt.md` section 9. The codebase has since grown a second variant beyond this plan, comprising `CoptLearnedModelV2ActorCritic` and `CoptLearnedModelV2PPO`, which derive the estimator input from the observation history alone and are described by no document. That variant is not presently reachable, because the guard at `scripts/rsl_rl/train.py:196` admits only the policy types COPT and COPT-LEARNED, so the branch selecting it can never execute although `djinn start train copt-learned-2` requests it. See [README.md](README.md) for the full register.

This document introduces the learned-model extension of the design and policy co-optimisation (COPT) framework for the Limx TRON1A biped. It presents the motivation drawn from the current experimental results, justifies the proposal through selected published literature, specifies the proposed encoder-decoder estimator architecture, and provides in detail the step-by-step implementation plan. 
<!-- The grounding context lives in `../ARCHITECTURE.md`, `../CO_OPTIMISATION.md`, `../context/knowledge_base.md`, `../context/copt.md`, `../context/rsl_rl.md`, `../context/isaaclab_env.md`, `../context/cmaes.md`, `../context/task_plots.md`, and `../context/literature.md`. -->

## Table of Contents

1. [Introduction](#1-introduction)
   1. [The co-design framework and the case for lower sample complexity](#11-the-co-design-framework-and-the-case-for-lower-sample-complexity)
   2. [Related work](#12-related-work)
2. [Proposed Method](#2-proposed-method)
   1. [Formulation](#21-formulation)
   2. [Network architecture](#22-network-architecture)
   3. [Comparison with HIMLoco and design choices](#23-comparison-with-himloco-and-design-choices)
   4. [Impact of the additional loss](#24-impact-of-the-additional-loss)
   5. [Hyperparameters](#25-hyperparameters)
   6. [Overview, justification, and expected outcome](#26-overview-justification-and-expected-outcome)
3. [Implementation Plan](#3-implementation-plan)
   1. [DecoderCfg](#31-decodercfg-file-extsbipedal_locomotionbipedal_locomotionutilswrappersrsl_rlrl_mlp_cfgpy)
   2. [SFCoptLearnedModelPPORunnerCfg](#32-sfcoptlearnedmodelpporunnercfg-file-extsbipedal_locomotionbipedal_locomotiontaskslocomotionagentslimx_rsl_rl_ppo_cfgpy)
   3. [CoptLearnedModelObservationsCfg and SFCoptLearnedModelEnvCfg](#33-coptlearnedmodelobservationscfg-and-sfcoptlearnedmodelenvcfg-file-extsbipedal_locomotionbipedal_locomotiontaskslocomotioncfgsflimx_base_env_cfgpy)
   4. [Environment scenario classes](#34-environment-scenario-classes-file-extsbipedal_locomotionbipedal_locomotiontaskslocomotionrobotslimx_solefoot_env_cfgpy)
   5. [Task registration](#35-task-registration-file-extsbipedal_locomotionbipedal_locomotiontaskslocomotionrobots__init__py)
   6. [CoptEstimator](#36-coptestimator-new-file-co_optimisationco_optimisationmodulescopt_estimatorpy)
   7. [CoptLearnedModelActorCritic](#37-coptlearnedmodelactorcritic-file-co_optimisationco_optimisationmodulescopt_actor_criticpy)
   8. [CoptLearnedModelPPO](#38-coptlearnedmodelppo-file-co_optimisationco_optimisationalgorithmscopt_ppopy)
   9. [Runner plumbing](#39-runner-plumbing-file-co_optimisationco_optimisationrunnerscopt_on_policy_runnerpy)
   10. [Training entry point](#310-training-entry-point-file-scriptsrsl_rltrainpy)
   11. [Evaluation entry point](#311-evaluation-entry-point-file-scriptsrsl_rlplaypy)
   12. [djinn and documentation](#312-djinn-and-documentation)
4. [Key Classes and Interfaces](#4-key-classes-and-interfaces)
5. [References](#5-references)

---

## 1. Introduction

### 1.1 The co-design framework and the case for lower sample complexity

The COPT pipeline jointly optimises robot morphology and its locomotion policy. An inner loop trains a PPO policy across 4096 parallel Isaac Lab environments, while an outer loop, driven by CMA-ES, replaces the population of 64 designs every 120 policy iterations by rescaling the thigh and shank link lengths (`scripts/rsl_rl/train.py:196-237`, `co_optimisation/runners/copt_on_policy_runner.py:576-619`). The policy that serves this loop, `CoptActorCritic` (`co_optimisation/modules/copt_actor_critic.py:42-152`), conditions its actor on the proprioceptive policy observations concatenated with a latent produced by an estimator MLP with the privileged observation group, which carries the link lengths, the body masses, and the height scan as its input.

The completed reference run, recorded under `logs/rsl_rl/sf_copt/2026-06-26_05-26-28` and plotted in `/ws/context/artefacts/plots`, exposes the cost of this arrangement. The run consumed 30000 iterations at 25 steps per environment across 4096 environments, roughly 3.07 billion environment transitions, and the first 8000 iterations were spent on a random design phase before CMA-ES even began. Despite this expenditure the reward traces never settle. The primary task reward `rew_lin_vel_xy` oscillates between approximately zero and eighteen across the entire run, and every per-component reward band stays wide or widens, never contracting even after the CMA-ES population collapses to a single repeated design (`../context/task_plots.md` section 5, `../context/cmaes.md` section 7). The critic tells the same story from the loss side, `Loss/value_function` remains large and spiky, mostly between two and ten with frequent excursions to the plotting cap and a final value of 17.55, while roughly twelve percent of episodes still terminate in falls at convergence. The curriculum terms, terrain level, push force, command ranges, and tracking reward sharpening, all keep moving to the end of the run and compound the non-stationarity that the critic must absorb.

The following two structural observations explain why the learner struggles to learn as observed:
1. The actor perceives the design it inhabits only through a sixteen dimensional latent whose encoder is trained exclusively by the PPO policy gradient, no supervised signal ever tells that encoder which aspects of the privileged input matter, so the latent must discover morphology, terrain, and dynamics structure from reward alone
2. The critic, although observes the link lengths, masses, and inertias directly in its observation group, must implicitly re-derive the dynamic consequences of each design and correlate the same to the torques, accelerations, and contact forces that a given body produces and are provided as input, for every one of the 64 designs alive at any moment
Both the observations together lead to high sample complexity. A representation that explicitly encodes the morphology and its dynamic signature, and a critic that is handed the dynamic quantities directly, should identify the system faster, stabilise the value target across design swaps, sharpen the per-individual fitness signal that CMA-ES consumes, and thereby reduce the sample complexity of every generation of the co-design loop. That is the purpose of the learned-model architecture proposed here.

### 1.2 Related work

<!-- The proposal draws on four convergent lines of research, surveyed in full with verified citations in `../context/literature.md`. -->

Privileged learning in legged locomotion began with two-stage teacher-student distillation, where a teacher trained on privileged terrain and contact information is distilled into a proprioceptive student (Lee et al. 2020, Science Robotics, Miki et al. 2022, Science Robotics). RMA (Kumar et al. 2021, RSS) showed that a latent encoding of privileged physical parameters, including body properties, can condition a base policy directly, with an adaptation module later regressing that latent from history. The field then moved to single-stage concurrent training. Ji et al. 2022 (RA-L) train a supervised state estimator alongside the policy in one loop, DreamWaQ (Nahrendra et al. 2023, ICRA) attaches a variational encoder-decoder estimator to PPO whose latent conditions the actor, and HIMLoco (Long et al. 2024, ICLR) learns a hybrid internal embedding from proprioceptive history jointly with PPO through an auxiliary contrastive objective. The present proposal instantiates exactly this single-stage template, replacing the contrastive machinery with a regression decoder because ground-truth dynamic targets are available in simulation.

Auxiliary and representation losses are the established mechanism by which such heads reduce sample complexity. UNREAL (Jaderberg et al. 2017, ICLR) demonstrated order-of-magnitude gains from auxiliary prediction heads sharing the policy trunk, CURL (Laskin et al. 2020, ICML) and SPR (Schwarzer et al. 2021, ICLR) confirmed the effect for contrastive and self-predictive objectives, SAC+AE (Yarats et al. 2021, AAAI) showed specifically that a reconstruction decoder whose gradients mix with the RL gradients on one shared encoder is stable and sample-efficient, and Lyle et al. 2021 (AISTATS) supply the theoretical account of how auxiliary prediction shapes representation dynamics. The single-optimiser joint update proposed below rests on this evidence.

Model-based reinforcement learning identifies dynamics knowledge as the dominant lever on sample efficiency, PETS (Chua et al. 2018, NeurIPS), MBPO (Janner et al. 2019, NeurIPS), and DreamerV3 (Hafner et al. 2023, Nature 2025) being the canonical results. The proposal imports this lever in its safest form, dynamics quantities appear only as prediction targets that shape the latent, never as a generative model rolled forward, so no compounding model error can enter the policy optimisation.

Morphology-control co-design, from Sims 1994 through Cheney et al. 2018 (J. R. Soc. Interface), identifies per-candidate controller adaptation as the central bottleneck, every morphology change invalidates the co-adapted controller. Design-parameter-conditioned policies amortise this cost, as shown by Schaff et al. 2019 (ICRA), Ha 2019 (Artificial Life), NGE (Wang et al. 2019, ICLR), SMP (Huang et al. 2020, ICML), MetaMorph (Gupta et al. 2022, ICLR), and N-LIMB (Schaff and Walter 2022). Luck et al. 2020 (CoRL) further validate learned design-conditioned models as surrogates inside the outer design loop, and Strgar and Kriegman 2025 (ICLR 2026) show that morphology-aware pretrained controllers dramatically reduce the per-candidate cost of evolutionary loops such as the CMA-ES loop used here. A latent that explicitly identifies the current body and its dynamics is therefore precisely the object the co-design literature recommends placing between the design distribution and the shared policy.

---

## 2. Proposed Method

### 2.1 Formulation

Let the following symbols denote the observation partition realised by the new environment configuration, with the concrete Isaac Lab terms named for each.
1. S_a, the actor state, base linear and angular velocity, projected gravity, joint positions and velocities, last action, and velocity commands, the existing `policy` group with additive Gaussian noise.
2. H, the history of the last n actor state steps, the same terms as S_a stored as an unflattened rolling buffer of n steps, the new `obsHistory` group.
3. P_1, morphology and terrain privileged information, link lengths, body masses, and the sparse terrain height map, the existing `privilegedObs` group.
4. P_2, the robot dynamic state, joint torques, joint accelerations, body inertia, foot contact forces, and foot linear velocity, the new `predictedPrivilegedObs` group.
5. C_1, constant robot privileged information, joint stiffness, joint damping, and material properties, and C_2, a history of absolute robot state, joint positions, robot position, and robot velocity, together the new `criticOnly` group with C = {C_1, C_2}.

The estimator realises the mapping
```
L        = enc(concat(P_1, flatten(H)))          latent, dimension 16
P_2_pred = dec(L)                                predicted dynamics
S        = {S_a, L}                              actor input
```
where enc is the encoder MLP and dec the decoder MLP of `CoptEstimator`. The latent L is projected to the unit sphere by an L2 normalisation before it is concatenated to the actor input, following the HIMLoco convention, which bounds the magnitude of the latent features feeding the actor. The critic receives the concatenation of S_a, C, P_1, and the ground-truth P_2, an asymmetric actor-critic arrangement in the tradition of Lee et al. 2020 and DreamWaQ, the critic exists only at training time and may consume every privileged quantity the simulator offers.

The training objective augments the PPO objective with a model estimation term. With L_surr the clipped surrogate loss, L_value the clipped value loss, and E the entropy bonus, one gradient step thus minimises
```
L_total = L_surr + c_v * L_value - c_e * E + || P_2 - P_2_pred ||^2
```
where the model estimation term is the mean squared error between the stored ground-truth `predictedPrivilegedObs` batch and the decoder output recomputed on the same minibatch. A single Adam optimiser over all policy parameters performs the update, so the encoder receives gradients from two sources simultaneously, from the surrogate loss through the non-detached latent in the actor input, and from the model estimation loss through the decoder. The decoder receives gradients only from the model estimation loss. This is the decisive difference from both the current `CoptActorCritic`, whose estimator learns from the policy gradient alone, and from HIMLoco, which updates its estimator with a second, separate optimiser.

### 2.2 Network architecture

The complete forward path per control step is as follows.
```
obsHistory [N, n, d_o] --flatten--> [N, n*d_o] --+
                                                 |--concat--> encoder --> z [N,16]
privilegedObs [N, d_p1] -------------------------+                        |
                                            ------------------------------+--> decoder --> P_2_pred [N, d_p2]
                                            |
policy [N, d_pol] --normalizer--> +--concat(z_normalised)--> actor --> mean [N, 8]
                                                                          Normal(mean, exp(log_std)) --> action

policy + criticOnly + privilegedObs + predictedPrivilegedObs --concat--> critic --> value [N, 1]
```

The per-step actor observation width d_o is 36, three each for base linear velocity, base angular velocity, projected gravity, and velocity commands, plus eight each for joint positions, joint velocities, and last action. With a history length n of 10 the flattened history contributes 360 inputs. The `policy` group itself now only has a single step of width 36. P_1 contributes the four link lengths, the per-body masses, and the 187 ray height scan (grid 1.6 m by 1.0 m at 0.1 m resolution). P_2 contributes eight torques, eight accelerations, nine inertia entries per body, six contact force components, and six foot velocity components. All widths are derived at runtime from the observation TensorDict, no dimension is hard-coded.

The encoder uses hidden dimensions [128, 64] with a sixteen dimensional output layer, expressed as `enc_hidden_dims = [128, 64, 16]` whose last entry is the latent width, identical to the current `EncoderCfg`. The decoder mirrors the encoder in reverse, hidden dimensions [64, 128] followed by a linear output layer of width d_p2. Activations are ELU throughout. The actor and critic retain their [512, 256, 128] ELU topology, the actor input width equals the policy group width plus sixteen exactly as the parent `CoptActorCritic` computes it (`copt_actor_critic.py:61-67`), and the critic input width is the sum of the four critic-set group widths, sized automatically by the rsl_rl `ActorCritic` constructor from the `obs_groups` mapping (`rsl_rl/modules/actor_critic.py:47-50`).

### 2.3 Comparison with HIMLoco and design choices

`HIMEstimator` (`himloco/himloco/modules/him_estimator.py:11-129`) served as the structural template, with the following deliberate differences to extend the idea to learning to identify morphology and distinguish dynamics for different morphologies:
1. The latent interpretation: HIM splits its encoder output into an explicit three dimensional velocity estimate and an implicit stability latent, because on hardware the base velocity is unobservable and must be estimated. In the case of the co-optimisation framework, the policy is already privy to privileged observations and the object to be identified is the morphology and its dynamic signature, so a single undifferentiated latent vector suffices and the encoder output layer is exactly `enc_hidden_dims[-1]` wide.
2. The decoder replaces the prototype machinery: HIM trains its implicit latent with a SwAV-style prototype contrastive swap loss between the history encoding and a target encoding of the next observation, because no ground truth exists for the quantity it wants the latent vector to capture. In the co-optimisation framework setting, the simulator provides exact ground truth for the dynamic quantities P_2, so a direct regression decoder is both simpler and better conditioned. Please note that this is the SAC+AE pattern of Yarats et al. 2021 and the estimator pattern of Ji et al. 2022 and DreamWaQ.
3. The optimisation topology: HIM owns a private Adam optimiser inside the estimator and steps it in `HIMEstimator.update`, separately from the PPO optimiser. In case of the co-optimisation framework a single optimiser performs one joint step per minibatch. The justification is threefold. The encoder must receive the PPO gradient as well as the reconstruction gradient so that the latent stays useful for control rather than drifting toward a purely reconstruction loss. Consequently `CoptEstimator.update` does not step anything, it computes and returns the model estimation loss for `CoptLearnedModelPPO.update` to fold into the total loss.

Two further choices deserve justification. The encoder consumes P_1 alongside the history, unlike HIM which encodes history alone, because in co-optimisation framework the morphology varies across environments and generations, and the literature on design-conditioned policies (Schaff et al. 2019, N-LIMB, RMA, MetaMorph) shows that explicit design conditioning is what lets one policy amortise learning across a design space. The critic additionally receives the ground-truth P_2, which the current COPT critic already partially observes through its critic group, the reorganisation makes the partition explicit and hands the critic every quantity the decoder is asked to predict, so the value function never has to infer what the auxiliary head is being taught.

One deployment note completes the picture. The actor at inference requires P_1 and the observation history.  Since P_1 contains simulator-privileged quantities, the learned-model policy is an instrument of the co-optimisation phase rather than a hardware deployment artefact. Once a final design is selected, a deployable controller can be obtained by retraining in the fixed-morphology pipeline, or by distilling the estimator input down to proprioceptive history in the RMA or HIM style. This is consistent with the framework's existing separation between COPT tasks and deployment tasks.

### 2.4 Impact of the additional loss

The model estimation term acts on training through three channels. It shapes the encoder representation toward dynamics-relevant features, regularises the latent against policy-gradient noise (useful because the policy gradient through a sixteen dimensional bottleneck is a weak and high-variance teacher), and it accelerates system identification, the latent space must separate designs whose dynamic signatures differ because the decoder is graded on exactly those signatures.

The risks are scale imbalance and gradient interference. The P_2 targets are heterogeneous and unnormalised, joint torques reach tens of newton metres while inertia entries are small fractions, so the MSE is dominated by the large-scale terms and its absolute magnitude may initially dwarf the surrogate loss. The proposed implementation adds the term with unit weight, while logging it separately as `Loss/model_estimation_loss` so that an imbalance is visible immediately. The natural remedies if loss imbalance is observed is a scalar coefficient on the term or per-term normalisation of the target group, both one-line follow-ups. Interference with the policy objective is bounded by construction, the decoder gradient touches only the encoder and decoder, and the actor and critic see the term only through the improved latent space. The adaptive KL schedule provides a final safety net, if the representation shift ever perturbs the policy distribution too quickly the learning rate contracts automatically.

### 2.5 Hyperparameters

All PPO hyperparameters are inherited unchanged from `SFCoptPPORunnerCfg` and thence `SF_TRON1AFlatPPORunnerCfg` (`agents/limx_rsl_rl_ppo_cfg.py:92-133`). The new architecture adds only the following.

| Parameter | Value | Where set |
|---|---|---|
| Encoder hidden dims | [128, 64, 16], latent 16 | `EncoderCfg.hidden_dims`, inherited |
| Decoder hidden dims | [64, 128] | `DecoderCfg.hidden_dims`, new runner cfg |
| Decoder output width | d_p2, runtime derived | `obs["predictedPrivilegedObs"].shape[-1]` |
| Estimator activation | elu | `EncoderCfg.activation` |
| History length n | 10 steps | `HistoryObsCfg.history_length` |
| History flattening | False, 3D group | `HistoryObsCfg.flatten_history_dim` |
| Model loss weight | 1.0, implicit | `CoptLearnedModelPPO.update` |
| Latent normalisation | L2, p=2, dim=-1 | `CoptEstimator.forward` |
| obs_groups | policy → [policy], critic → [policy, criticOnly, privilegedObs, predictedPrivilegedObs] | `SFCoptLearnedModelPPORunnerCfg.obs_groups` |
| experiment_name | sf_copt_learned | `SFCoptLearnedModelPPORunnerCfg` |
| learning_rate, schedule | 1e-3, adaptive, desired_kl 0.01 | inherited |
| gamma, lam, clip, entropy | 0.99, 0.95, 0.2, 0.005 | inherited |
| epochs, minibatches, steps | 5, 4, 25 | inherited |
| max_grad_norm | 1.0, applied to all params jointly | inherited PPO clip |
| COPT knobs | 64 individuals, interval 120, late start 12000, sigma0 0.25, ranges (0.75, 1.25) | `train.py` COPT block, shared |

The remaining reward, event, curriculum, terrain, and command settings are byte-identical to the COPT environment, the learned-model environment differs from `SFCoptEnvCfg` in its observations member alone.

### 2.6 Overview, justification, and expected outcome

In summary, the proposal converts the existing morphology latent from an incidental by-product of the policy gradient into an explicitly supervised internal model. The encoder is told, through the decoder loss, that its latent must contain whatever is needed to reproduce the torques, accelerations, inertia, and contact behaviour of the current body, which is precisely the information a locomotion policy and its value function need to act and evaluate across a shifting design population. Every element of the construction has published precedent, the single-stage concurrent estimator (Ji et al., DreamWaQ, HIMLoco), the joint single-optimiser auxiliary loss (UNREAL, SAC+AE, CURL), the dynamics-prediction target (SPR, DreamerV3), and the design-conditioned shared policy (Schaff et al., N-LIMB, Strgar and Kriegman).

The expected observable changes against the reference run are a faster early rise and tighter band of `rew_lin_vel_xy`, a materially lower `Loss/value_function` together with a higher `explained_variance`, both now logged by `CoptPPO.update`, a new `Loss/model_estimation_loss` trace that should fall steadily and re-spike briefly at each morphology swap as the population changes, and a cleaner per-individual fitness signal for CMA-ES since a policy that adapts to a design faster produces less noisy returns within each 120 iteration window. The implementation is strictly additive, every new class subclasses an existing one and no registered task, config, or runner behaviour changes for the existing COPT, HIM, or PPO paths. The next section provides the complete plan.

---

## 3. Implementation Plan

The plan is ordered so that each step compiles against the previous ones, configuration first, then modules, then the algorithm, then wiring. File paths are relative to `/ws/tron1-rl-isaaclab-cozum` unless absolute. Line numbers refer to the current state of each file.

### 3.1 DecoderCfg, file `exts/bipedal_locomotion/bipedal_locomotion/utils/wrappers/rsl_rl/rl_mlp_cfg.py`

Append directly below `class EncoderCfg` `class DecoderCfg`, a class that mimics it with only the name changed, in `utils/wrappers/rsl_rl/rl_mlp_cfg.py`.
```python
@configclass
class DecoderCfg:
    output_detach : bool = True
    num_input_dim : int = MISSING
    num_output_dim : int = 3
    hidden_dims : list[int] = [256, 128]
    activation : str = "elu"
    orthogonal_init : bool = False
```
The co-optimisattion on policy runner passes the config to the policy as a plain dict, the policy reads only `hidden_dims` and `activation`, the remaining fields exist for interface symmetry with `EncoderCfg` exactly as requested.

### 3.2 SFCoptLearnedModelPPORunnerCfg, file `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`

Extend the import at lines 10 to 13 to include `class DecoderCfg` export.
```python
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import (
    DecoderCfg,
    EncoderCfg,
    RslRlPpoAlgorithmMlpCfg,
)
```

Append after `class SFCoptPPORunnerCfg` (lines 130 to 133).
```python
# -----------------------------------------------------------------
@configclass
class SFCoptLearnedModelPPORunnerCfg(SFCoptPPORunnerCfg):
    experiment_name: str = "sf_copt_learned"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["policy", "criticOnly", "privilegedObs", "predictedPrivilegedObs"],
    }
    decoder = DecoderCfg(
        output_detach=False,
        num_output_dim=3,
        hidden_dims=[64, 128],
        activation="elu",
        orthogonal_init=False,
    )
```
Inheritance from `SFCoptPPORunnerCfg` preserves every PPO hyperparameter, the encoder, and `max_iterations` 45000, while the class adds the decoder and the observation-set mapping to implement critic observation breakdown into the requested groups. `num_output_dim` is a placeholder, the true decoder output width is derived at runtime from the `predictedPrivilegedObs` group. The `obsHistory`, `commands` groups are listed in neither set, they are consumed directly by the policy and must stay out of the sets because `ActorCritic.__init__` asserts 2D groups (`rsl_rl/modules/actor_critic.py:45,49`) and `obsHistory` is 3D and commands are independent of the policy and not used as such (but still added for posterity).

### 3.3 CoptLearnedModelObservationsCfg and SFCoptLearnedModelEnvCfg, file `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`

Insert a new configuration class after `CoptObservationsCfg` (which spans lines 359 to 571). Two of its inner classes are verbatim copies with renames, the rest are new.

```python
@configclass
class CoptLearnedModelObservationsCfg:
    """Observation specifications for the learned-model co-optimisation MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        # Copy the body of CoptObservationsCfg.PolicyCfg verbatim without
        # copying the __post__init__() method
        # (limx_base_env_cfg.py:364-421), terms base_lin_vel, base_ang_vel,
        # proj_gravity, joint_pos, joint_vel, last_action, velocity_commands,
        # __post_init__ with enable_corruption=True, concatenate_terms=True,
        # no history is used for policy config here as history is available
        # through class HistoryObsCfg . Thus, policy input is only a 
        # single step input
        ...

    @configclass
    class PrivilegedCfg(ObsGroup):
        # Copy the body of CoptObservationsCfg.PrivligedObsCfg verbatim
        # (limx_base_env_cfg.py:424-457), terms link_lengths, robot_mass,
        # heights, __post_init__ with enable_corruption=True,
        # concatenate_terms=True, single step.
        ...

    @configclass
    class PredictedPrivilegedCfg(ObsGroup):
        """P_2, ground-truth dynamic state, the decoder regression target"""

        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*")
            },
        )
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticOnlyCfg(ObsGroup):
        """C_1 constant robot properties and C_2 absolute state history"""

        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        # Copy the observation terms of PolicyCfg above verbatim, the group
        # differs only in its __post_init__.

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    criticOnly: CriticOnlyCfg = CriticOnlyCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    privilegedObs: PrivilegedCfg = PrivilegedCfg()
    predictedPrivilegedObs: PredictedPrivilegedCfg = PredictedPrivilegedCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()
```

Then insert after `class SFCoptEnvCfg` (lines 1369 to 1398) exactly as specified below
```python
@configclass
class SFCoptLearnedModelEnvCfg(SFCoptEnvCfg):
    observations: CoptLearnedModelObservationsCfg = CoptLearnedModelObservationsCfg()
```

All observation functions referenced already exist and are already used by `CoptObservationsCfg.CriticCfg` (limx_base_env_cfg.py:509-528), so no new mdp code is required. `PredictedPrivilegedCfg` disables corruption because it is a regression target, noise on a target only inflates the irreducible loss floor. `HistoryObsCfg` keeps corruption enabled so that the encoder learns from the same noisy signal the actor sees, mirroring the HIM history group (limx_base_env_cfg.py:632-685). The group history length of ten matches the active HIM configuration and keeps the encoder input at 360 plus d_p1, this is the n of the formulation and is tunable in one place. Constant C_1 terms sit in the ten-step `criticOnly` history for configuration simplicity, the redundancy costs a few hundred critic input floats and no correctness.

### 3.4 Environment scenario classes, file `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_env_cfg.py`

Append `class SFCoptLearnedModelBaseEnvCfg` and `class SFCoptLearnedModelBaseEnvCfg_PLAY` after the definition of `class SFCoptBaseEnvCfg_PLAY` after line 190.
```python
@configclass
class SFCoptLearnedModelBaseEnvCfg(SFCoptLearnedModelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.replicate_physics = False

        self.scene.robot = SOLEFOOT_IDENTIFIED_MULTIUSD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFCoptLearnedModelBaseEnvCfg_PLAY(SFCoptLearnedModelEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)
```

Append the following two classes after the definition of `class SFCoptBlindFlatEnvCfg_PLAY` (after line 337). They inherit from `class SFCoptLearnedModelBaseEnvCfg` so that the asset binding, multi-USD spawn, `replicate_physics=False`, termination bodies, viewer, and play-mode reductions are reused byte for byte.
```
@configclass
class SFCoptLearnedModelBlindFlatEnvCfg(SFCoptLearnedModelBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None


@configclass
class SFCoptLearnedModelBlindFlatEnvCfg_PLAY(SFCoptLearnedModelBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
```

Append the rough environment classes after the definition of `class SFCoptLearnedModelBlindRoughEnvCfg_PLAY` (after line 432). These classes define the rough environment to be used with the updated observation configuration.
```
@configclass
class SFCoptLearnedModelBlindRoughEnvCfg(SFCoptLearnedModelBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFCoptLearnedModelBlindRoughEnvCfg_PLAY(SFCoptLearnedModelBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
```
`class SFCoptLearnedModelEnvCfg` from step 3.3 remains the canonical common-layer root for all the added classes.

### 3.5 Task registration, file `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py`

Add `SFCoptLearnedModelPPORunnerCfg` to the agents import (lines 3 to 9), instantiate a runner config beside `limx_sf_copt_runner_cfg` (line 29), and append four registrations after the existing Copt block (lines 361 to 390).
```python
limx_sf_copt_learned_runner_cfg = SFCoptLearnedModelPPORunnerCfg()

.....

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptLearnedModelBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptLearnedModelBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptLearnedModelBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptLearnedModelBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)
```

### 3.6 CoptEstimator, new file `co_optimisation/co_optimisation/modules/copt_estimator.py`

Create the file with the following complete content. `get_activation` is ported verbatim from `himloco/himloco/modules/him_estimator.py:149-168`, which is where the function actually lives.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class CoptEstimator(nn.Module):
    """Encoder-decoder estimator for the learned-model co-optimisation policy.

    The encoder consumes ``temporal_steps`` steps of actor observations
    flattened together with the morphology and terrain privileged
    observations, and emits a latent of width ``enc_hidden_dims[-1]``.  The
    decoder regresses the robot dynamic state (``predictedPrivilegedObs``)
    from that latent.  Unlike ``HIMEstimator`` this module owns no optimiser,
    the model estimation loss returned by :meth:`update` is folded into the
    PPO loss and minimised by the single shared optimiser, so the encoder
    receives gradients from both the PPO objective and the decoder objective.
    """

    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        num_privileged_obs,
        num_predicted_privileged_obs,
        enc_hidden_dims=[128, 64, 16],
        dec_hidden_dims=[64, 128],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "CoptEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(CoptEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.num_predicted_privileged_obs = num_predicted_privileged_obs
        self.num_privileged_obs = num_privileged_obs

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs + self.num_privileged_obs
        print("encoder input dim: ", enc_input_dim)
        print("temporal_steps: ", self.temporal_steps)
        print("num one step observations: ", self.num_one_step_obs)
        print("num privileged observations: ", self.num_privileged_obs)
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1])]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_input_dim = enc_hidden_dims[-1]
        dec_layers = []
        for l in range(len(dec_hidden_dims)):
            dec_layers += [nn.Linear(dec_input_dim, dec_hidden_dims[l]), activation]
            dec_input_dim = dec_hidden_dims[l]
        dec_layers += [nn.Linear(dec_input_dim, self.num_predicted_privileged_obs)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, estimator_input):
        # The outputs are part of the policy computation graph and must NOT be
        # detached, the encoder is trained by both the PPO and decoder losses.
        z = self.encoder(estimator_input)
        pred = self.decoder(z)
        z = F.normalize(z, dim=-1, p=2)
        return z, pred

    def encode(self, estimator_input):
        z = self.encoder(estimator_input.detach())
        z = F.normalize(z, dim=-1, p=2)
        return z

    def get_latent(self, estimator_input):
        return self.encode(estimator_input).detach()

    def update(self, predicted, target):
        """Return the model estimation loss ``|| P_2 - P_2_pred ||^2``.

        No optimiser step happens here, the caller adds the returned tensor to
        the PPO loss so a single optimiser updates encoder, decoder, actor,
        and critic together.
        """
        return F.mse_loss(predicted, target.detach())
```


Export the class, file `co_optimisation/co_optimisation/modules/__init__.py` currently holds a single import line, extend it.

```python
from co_optimisation.modules.copt_actor_critic import CoptActorCritic, CoptLearnedModelActorCritic
from co_optimisation.modules.copt_estimator import CoptEstimator
```

### 3.7 CoptLearnedModelActorCritic, file `co_optimisation/co_optimisation/modules/copt_actor_critic.py`

Append the following class and import to the file:
```python
from co_optimisation.modules.copt_estimator import CoptEstimator

...

class CoptLearnedModelActorCritic(CoptActorCritic):
    """Co-optimisation actor-critic with a learned-model estimator.

    Replaces the parent's plain MLP estimator with a :class:`CoptEstimator`
    whose encoder consumes the privileged morphology observations together
    with the flattened actor observation history, and whose decoder predicts
    the robot dynamic state.  ``act`` additionally returns the prediction so
    that :class:`CoptLearnedModelPPO` can compute the model estimation loss.

    Note, ``actor_obs_normalization`` must remain False for this class, the
    empirical normalizer is sized for the flat actor input and cannot consume
    the 3D observation history.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        encoder_cfg: dict,
        decoder_cfg: dict,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ):
        super(CoptLearnedModelActorCritic, self).__init__(
            obs,
            obs_groups,
            num_actions,
            encoder_cfg,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
            state_dependent_std,
            **kwargs,
        )

        # Estimator, replaces the parent's plain MLP estimator.
        enc_hidden_dims = encoder_cfg.get("hidden_dims", [128, 64, 16])
        dec_hidden_dims = decoder_cfg.get("hidden_dims", [64, 128])
        assert len(obs["obsHistory"].shape) == 3
        self.history_size = obs["obsHistory"].shape[1]
        num_one_step_obs = obs["policy"].shape[-1]
        num_privileged_obs = obs["privilegedObs"].shape[-1]
        num_predicted_privileged_obs = obs["predictedPrivilegedObs"].shape[-1]
        print("observation history shape: ", obs["obsHistory"].shape)
        print("one step observation size: ", num_one_step_obs)
        self.estimator = CoptEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs,
            num_privileged_obs=num_privileged_obs,
            num_predicted_privileged_obs=num_predicted_privileged_obs,
            enc_hidden_dims=enc_hidden_dims,
            dec_hidden_dims=dec_hidden_dims,
            activation=encoder_cfg.get("activation", "elu"),
        )

    def _get_estimator_input(self, obs: TensorDict) -> torch.Tensor:
        history_obs = torch.flatten(obs["obsHistory"], start_dim=1)
        return torch.cat((obs["privilegedObs"], history_obs), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        latent, pred_info = self.estimator(self._get_estimator_input(obs))
        actor_input = torch.cat((actor_obs, latent), dim=-1)
        mean = self.actor(actor_input)
        if torch.any(torch.isnan(self.log_std)):
            print("received nan values in the log std")
        std = torch.exp(self.log_std)
        self.distribution = Normal(mean, std)
        return pred_info

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_info = self._update_distribution(obs)
        return self.distribution.sample(), pred_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        latent = self.estimator.encode(self._get_estimator_input(obs))
        return self.actor(torch.cat((actor_obs, latent), dim=-1))
```

The parent constructor already sizes the actor input as policy width plus `enc_hidden_dims[-1]` (`copt_actor_critic.py:61-67`), builds the actor, the parent estimator, and `log_std`, so the child only replaces `self.estimator`, the parent's MLP estimator is garbage collected before the PPO optimiser is created in `_construct_algorithm`, hence no stale parameters enter the optimiser. The critic and `evaluate` are inherited untouched, the enlarged critic input is handled entirely by the `obs_groups` mapping of step 3.2 through the rsl_rl base class. `get_actions_log_prob` is inherited from the parent. Corrections 2, 3, 5, 6, and 8 from the correction register are applied in this code.

### 3.8 CoptLearnedModelPPO, file `co_optimisation/co_optimisation/algorithms/copt_ppo.py`

Add `import torch.nn as nn` to the imports, then append the class below `CoptPPO`.

```python
class CoptLearnedModelPPO(CoptPPO):
    """CoptPPO whose policy returns a dynamics prediction from ``act``.

    Overrides ``act`` to unpack the ``(actions, prediction)`` tuple during
    rollouts and ``update`` to add the model estimation loss to the PPO loss,
    so a single optimiser step trains actor, critic, encoder, and decoder
    together.
    """

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values, discarding the dynamics prediction
        actions, _ = self.policy.act(obs)
        self.transition.actions = actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def update(self) -> dict[str, float]:
        # Explained variance diagnostic, mirrors CoptPPO.update
        with torch.no_grad():
            flat_returns = self.storage.returns.flatten(0, 1)
            flat_values = self.storage.values.flatten(0, 1)
            var_y = flat_returns.var()
            explained_var = (
                1.0 - (flat_returns - flat_values).var() / (var_y + 1e-8)
            ).item()

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_model_estimation_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob, entropy, and the dynamics prediction
            _, predicted_model_info = self.policy.act(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0]
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptive KL learning rate schedule, identical to the base PPO
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Model estimation loss, || P_2 - P_2_pred ||^2
            model_estimation_loss = self.policy.estimator.update(
                predicted_model_info, obs_batch["predictedPrivilegedObs"]
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + model_estimation_loss
            )

            # Symmetry loss, identical to the base PPO
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"]
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss, identical to the base PPO
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_model_estimation_loss += model_estimation_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_model_estimation_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "model_estimation_loss": mean_model_estimation_loss,
            "explained_variance": explained_var,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
```

The `update` body is the base `rsl_rl` PPO update (`rsl_rl/algorithms/ppo.py:194-417`) with exactly three insertions, the tuple unpacking of `policy.act`, the model estimation loss added to the total, and its accumulation into the loss dictionary, plus the explained variance diagnostic carried over from `CoptPPO.update` (`copt_ppo.py:111-127`), which this override supersedes. The `act` override handles the tuple unpacking of `policy.act`. Note the runner logs every `loss_dict` key under `Loss/<key>` (`rsl_rl/runners/on_policy_runner.py:211-213`), so `Loss/model_estimation_loss` appears in TensorBoard with no logging change.

Extend `co_optimisation/co_optimisation/algorithms/__init__.py` to export `CoptLearnedModelPPO` beside `CoptPPO`.

### 3.9 Runner plumbing, file `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`

Three small edits.

First, extend the imports (lines 30 to 31).

```python
from co_optimisation.algorithms import CoptLearnedModelPPO, CoptPPO
from co_optimisation.modules import CoptActorCritic, CoptLearnedModelActorCritic
```

Second, in `__init__` directly after `self.encoder_cfg = train_cfg["encoder"]` (line 125), add

```python
self.decoder_cfg: dict | None = train_cfg.get("decoder", None)
```

Third, in `_construct_algorithm`, replace the policy construction (lines 470 to 478) with a branch on the decoder config.

```python
        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        if self.decoder_cfg is not None:
            actor_critic: CoptActorCritic = actor_critic_class(
                obs,
                self.cfg["obs_groups"],
                self.env.num_actions,
                self.encoder_cfg,
                self.decoder_cfg,
                **self.policy_cfg,
            ).to(self.device)
        else:
            actor_critic: CoptActorCritic = actor_critic_class(
                obs,
                self.cfg["obs_groups"],
                self.env.num_actions,
                self.encoder_cfg,
                **self.policy_cfg,
            ).to(self.device)
```

The `eval(class_name)` resolution requires both new classes to be importable in this module, hence the import edit. The decoder config is absent from every existing runner configuration, so the `None` branch reproduces the current behaviour bit for bit and the existing COPT path is untouched. No other runner change is required, `save`, `load`, `learn`, and the morphology machinery operate on `self.alg.policy` opaquely, the estimator parameters travel inside `model_state_dict` and the single optimiser inside `optimizer_state_dict` automatically. This reuses the on policy runner class with minimal changes.

### 3.10 Training entry point, file `scripts/rsl_rl/train.py`

Modify the COPT branch (lines 196 to 237) to accept the new policy type. The design generator setup is shared verbatim, only the class names differ.

```python
    if args_cli.policy_type in ("COPT", "COPT-LEARNED"):

        _base_urdf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/isaaclab/biped/exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/base_robot.urdf",
        )
        _num_individuals = 64
        # ea_update_interval * num_steps_per_env (120 * 25 = 3000) should be more
        # than episode_duration / (dt * decimation) (20 / (0.005 * 4) = 1000)
        ea_update_interval = 120
        ea_late_start = 12000
        param_ranges = {
            "thigh_length_scale": (0.75, 1.25),
            "shank_length_scale": (0.75, 1.25),
        }
        design_generator = GrowingDesignDistCMAESDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
            param_ranges=param_ranges,
            sigma0=0.25,
            seed=42,
            late_start=True,
            late_start_it=int(ea_late_start / ea_update_interval),
            max_cma_iter=(agent_cfg.max_iterations - ea_late_start)
            / ea_update_interval,
        )
        if args_cli.policy_type == "COPT-LEARNED":
            agent_cfg.policy.class_name = "CoptLearnedModelActorCritic"
            agent_cfg.algorithm.class_name = "CoptLearnedModelPPO"
        else:
            agent_cfg.policy.class_name = "CoptActorCritic"
            agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
            "ea_update_interval": ea_update_interval,
            "ea_late_start": ea_late_start,
            "num_individuals": _num_individuals,
            "randomise_before_late_start": True,
        }
        runner = CoptOnPolicyRunner(
            env,
            design_generator,
            agent_cfg_dict,
            log_dir=log_dir,
            device=agent_cfg.device,
        )
    else:
        runner = runner_cls(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
```

The decoder and obs_groups fields flow into the runner automatically through `agent_cfg.to_dict()`, because `SFCoptLearnedModelPPORunnerCfg` declares them, no train.py handling is needed. The stale comment above `ea_update_interval` is corrected in passing, the true product is 120 times 25. Also update the help text of the `--policy-type` argument in `scripts/rsl_rl/cli_args.py:61-64` as follows:
```
Type of the policy to use. Can be one of PPO, HIMPPO, COPT, or COPT-LEARNED
```

### 3.11 Evaluation entry point, file `scripts/rsl_rl/play.py`

Insert an `elif` after the COPT block (lines 298 to 331) mirroring it with the learned-model class names, the play-time generator is the plain `CMAESDesignGenerator` exactly as in the COPT block.

```python
    elif args_cli.policy_type == "COPT-LEARNED":

        _base_urdf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/isaaclab/biped/exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/base_robot.urdf",
        )
        _num_individuals = 256
        param_ranges = {}
        params = ["thigh_length_scale", "shank_length_scale"]
        for param in params:
            param_ranges[param] = DEFAULT_PARAM_RANGES[param]
        design_generator = CMAESDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
            param_ranges=param_ranges,
            sigma0=0.1,
            seed=42,
            late_start=False,
        )
        agent_cfg.policy.class_name = "CoptLearnedModelActorCritic"
        agent_cfg.algorithm.class_name = "CoptLearnedModelPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
            "ea_update_interval": 50,
            "ea_late_start": -1,
            "num_individuals": _num_individuals,
        }
        ppo_runner = CoptOnPolicyRunner(
            env,
            design_generator,
            agent_cfg_dict,
            log_dir=log_dir,
            device=agent_cfg.device,
        )
```
No change is needed in the inference loop, `get_inference_policy` resolves to `act_inference`, which returns the action means alone, so the generic `actions = policy(obs)` branch already handles the new policy type.

### 3.12 djinn and documentation

In `/ws/djinn`, extend the `start train` mode ladder after the `copt` branch (lines 131 to 133) as follows:
```bash
                elif [[ "$3" == "copt-learned" ]]; then
                    task="Isaac-Limx-SF-Copt-Learned-Rough-v0"
                    policy_type="COPT-LEARNED"
```

Extend the `start play` mode ladder after its `copt` branch (lines 168 to 170).
```bash
                elif [[ "$3" == "copt-learned" ]]; then
                    task="Isaac-Limx-SF-Copt-Learned-Rough-Play-v0"
                    policy_type="COPT-LEARNED"
```

The install line already covers `co_optimisation` and `himloco` so no change is required there.

Documentation updates, apply the same regal comma-only style as the host documents.
- `/ws/ARCHITECTURE.md`, add `copt-learned` to the `djinn start train` mode table and mode list in section 7, add the four new task IDs to the registries in sections 4.4 and 8, and add a row for the COPT-LEARNED policy type to the policy table in section 4.5 noting `CoptLearnedModelActorCritic` with observation groups policy, criticOnly, commands, privilegedObs, predictedPrivilegedObs, obsHistory.
- `/ws/README.md` and `/ws/tron1-rl-isaaclab-cozum/README.md`, wherever the `djinn start train copt` usage is documented, add the sibling `djinn start train copt-learned [gpu_id]` and `djinn start play copt-learned <checkpoint> <seed> [gpu_id]` commands.
- `/ws/CO_OPTIMISATION.md`, add a short subsection under section 5 pointing to `COPT_LEARNED_MODEL.md` for the learned-model variant.

---

## 4. Key Classes and Interfaces

Existing interfaces consumed by the plan.

- `ActorCritic`, `rsl_rl/rsl_rl/modules/actor_critic.py:17-199`. Constructor `(obs, obs_groups, num_actions, actor_obs_normalization, critic_obs_normalization, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, noise_std_type, state_dependent_std, **kwargs)`. Sizes actor and critic from `obs_groups` (asserts 2D groups at lines 45 and 49), members `actor`, `critic`, `actor_obs_normalizer`, `critic_obs_normalizer`, `std` or `log_std`, methods `act`, `act_inference`, `evaluate`, `get_actor_obs`, `get_critic_obs` (lines 167 to 173), `get_actions_log_prob`, `update_normalization`.
- `PPO`, `rsl_rl/rsl_rl/algorithms/ppo.py`. Members `policy`, `optimizer` (single Adam over all policy parameters), `storage`, `transition`, schedule scalars. Methods `act` (lines 143 to 154, calls `policy.act(obs).detach()`), `process_env_step`, `compute_returns`, `update` (lines 194 to 417, the template for section 3.8).
- `RolloutStorage.mini_batch_generator`, `rsl_rl/rsl_rl/storage/rollout_storage.py:162-217`, yields the ten-tuple consumed in `update`, `obs_batch` is the flattened observation TensorDict containing every environment group including `predictedPrivilegedObs` and the 3D `obsHistory`.
- `resolve_obs_groups`, `rsl_rl/rsl_rl/utils/utils.py:203-262`, validates the explicit `obs_groups` dict against the environment observation keys, called from `OnPolicyRunner.__init__`.
- `CoptActorCritic`, `co_optimisation/modules/copt_actor_critic.py:42-152`. Constructor adds `encoder_cfg` after `num_actions`, computes the actor width as policy width plus `enc_hidden_dims[-1]` (lines 61 to 67), builds `self.estimator` as `MLP(privileged_width, 16, [128, 64])` (lines 93 to 105) and `self.log_std` (line 108). Methods `_update_distribution` (127), `act` (139), `get_actions_log_prob` (143), `act_inference` (146).
- `CoptPPO`, `co_optimisation/algorithms/copt_ppo.py:25-127`. Constructor `(policy, num_individuals, env_to_individual, ...PPO kwargs...)`, methods `compute_returns_design_wise` (94) and `update` (111) which appends `explained_variance`.
- `CoptOnPolicyRunner`, `co_optimisation/runners/copt_on_policy_runner.py:65-634`. Reads `train_cfg["encoder"]` at line 125, `_construct_algorithm` (442 to 499) resolves policy and algorithm classes by `eval` and passes `(obs, obs_groups, num_actions, encoder_cfg, **policy_cfg)` then `(actor_critic, num_individuals, env_to_individual, device=..., **alg_cfg, multi_gpu_cfg=...)`. `save` (142) and `load` (176) persist the extended COPT state, `learn` (219) interleaves the CMA-ES morphology update (`_update_morphology`, 576) with PPO.
- `HIMEstimator`, `himloco/himloco/modules/him_estimator.py:11-129`, the structural template, `get_activation` at lines 149 to 168 is ported.
- `EncoderCfg`, `exts/.../utils/wrappers/rsl_rl/rl_mlp_cfg.py:22-29`, fields `output_detach`, `num_input_dim`, `num_output_dim`, `hidden_dims`, `activation`, `orthogonal_init`.
- `SF_TRON1AFlatPPORunnerCfg` and `SFCoptPPORunnerCfg`, `exts/.../agents/limx_rsl_rl_ppo_cfg.py:92-133`, the PPO hyperparameter source, `SFCoptPPORunnerCfg` overrides `experiment_name` and `max_iterations` 45000.
- `CoptObservationsCfg`, `exts/.../cfg/SF/limx_base_env_cfg.py:359-571`, groups `policy` (364), `PrivligedObsCfg` (424), `CriticCfg` (460), `CommandsObsCfg` (562). `SFCoptEnvCfg` at lines 1369 to 1398.
- `SFCoptBaseEnvCfg` and scenario classes, `exts/.../robots/limx_solefoot_env_cfg.py:140-190, 324-337, 412-432`, the inheritance sources for step 3.4.
- Task registry, `exts/.../robots/__init__.py:361-390` for the Copt block.
- Entry points, `scripts/rsl_rl/train.py:196-241`, `scripts/rsl_rl/play.py:298-336`, `scripts/rsl_rl/cli_args.py:61-64`, `/ws/djinn:131-133` and `:168-170`.
- `RslRlOnPolicyRunnerCfg.obs_groups`, `IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:159`.

New interfaces introduced by the plan.

- `DecoderCfg`, `exts/.../utils/wrappers/rsl_rl/rl_mlp_cfg.py`, fields identical to `EncoderCfg`, consumed as a dict for `hidden_dims` and `activation`.
- `SFCoptLearnedModelPPORunnerCfg(SFCoptPPORunnerCfg)`, `exts/.../agents/limx_rsl_rl_ppo_cfg.py`, adds `experiment_name` "sf_copt_learned", the `obs_groups` mapping, and the `decoder` field.
- `CoptLearnedModelObservationsCfg`, `exts/.../cfg/SF/limx_base_env_cfg.py`, members `policy`, `criticOnly`, `commands`, `privilegedObs`, `predictedPrivilegedObs`, `obsHistory`, inner classes `PolicyCfg`, `CriticOnlyCfg`, `CommandsObsCfg`, `PrivilegedCfg`, `PredictedPrivilegedCfg`, `HistoryObsCfg` as listed in step 3.3.
- `SFCoptLearnedModelEnvCfg(SFCoptEnvCfg)` and the six scenario classes of step 3.4.
- `CoptEstimator(nn.Module)`, `co_optimisation/modules/copt_estimator.py`, constructor `(temporal_steps, num_one_step_obs, num_privileged_obs, num_predicted_privileged_obs, enc_hidden_dims, dec_hidden_dims, activation, learning_rate, max_grad_norm, num_prototype, temperature, **kwargs)`, members `encoder`, `decoder`, `num_latent`, methods `forward` returning `(z_normalised, prediction)`, `encode`, `get_latent`, `update(predicted, target)` returning the MSE tensor.
- `CoptLearnedModelActorCritic(CoptActorCritic)`, `co_optimisation/modules/copt_actor_critic.py`, constructor adds `decoder_cfg` after `encoder_cfg`, members `estimator` (`CoptEstimator`), `history_size`, methods `_get_estimator_input`, `_update_distribution` returning the prediction, `act` returning `(actions, prediction)`, `act_inference` returning actions.
- `CoptLearnedModelPPO(CoptPPO)`, `co_optimisation/algorithms/copt_ppo.py`, overrides `act` and `update`, loss dictionary keys `value_function`, `surrogate`, `entropy`, `model_estimation_loss`, `explained_variance`.
- Task IDs `Isaac-Limx-SF-Copt-Learned-Flat-v0`, `Isaac-Limx-SF-Copt-Learned-Flat-Play-v0`, `Isaac-Limx-SF-Copt-Learned-Rough-v0`, `Isaac-Limx-SF-Copt-Learned-Rough-Play-v0`, policy type `COPT-LEARNED`, djinn modes `copt-learned` for train and play.

---

## 5. References

Code, configuration, and context artefacts.

- `/ws/ARCHITECTURE.md`, framework architecture and task registry.
- `/ws/CO_OPTIMISATION.md`, co-optimisation design and RSL-RL guide.
- `/ws/context/knowledge_base.md`, `/ws/context/copt.md`, `/ws/context/rsl_rl.md`, `/ws/context/isaaclab_env.md`, `/ws/context/cmaes.md`, `/ws/context/task_plots.md`, investigation knowledge base, including the 2026-07-02 code-state corrections.
- `/ws/context/literature.md`, the verified literature survey backing section 1.2 and the design choices of section 2.
- `/ws/context/artefacts/plots`, TensorBoard exports of run `sf_copt/2026-06-26_05-26-28`.
- Source files cited throughout sections 3 and 4.

Literature, full verified citations with relevance notes in `../context/literature.md`.

- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. (2020). Learning quadrupedal locomotion over challenging terrain. Science Robotics 5(47), eabc5986.
- Kumar, A., Fu, Z., Pathak, D., Malik, J. (2021). RMA, Rapid Motor Adaptation for Legged Robots. RSS 2021, arXiv:2107.04034.
- Ji, G., Mun, J., Kim, H., Hwangbo, J. (2022). Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. IEEE RA-L 7(2), arXiv:2202.05481.
- Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., Pang, J. (2024). Hybrid Internal Model, Learning Agile Legged Locomotion with Simulated Robot Response. ICLR 2024, arXiv:2312.11460.
- Nahrendra, I. M. A., Yu, B., Myung, H. (2023). DreamWaQ, Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning. ICRA 2023, arXiv:2301.10602.
- Miki, T., Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics 7(62), arXiv:2201.08117.
- Jaderberg, M., et al. (2017). Reinforcement Learning with Unsupervised Auxiliary Tasks. ICLR 2017, arXiv:1611.05397.
- Laskin, M., Srinivas, A., Abbeel, P. (2020). CURL, Contrastive Unsupervised Representations for Reinforcement Learning. ICML 2020, arXiv:2004.04136.
- Yarats, D., et al. (2021). Improving Sample Efficiency in Model-Free Reinforcement Learning from Images. AAAI 2021, arXiv:1910.01741.
- Schwarzer, M., et al. (2021). Data-Efficient Reinforcement Learning with Self-Predictive Representations. ICLR 2021, arXiv:2007.05929.
- Lyle, C., Rowland, M., Ostrovski, G., Dabney, W. (2021). On the Effect of Auxiliary Tasks on Representation Dynamics. AISTATS 2021, arXiv:2102.13089.
- Chua, K., Calandra, R., McAllister, R., Levine, S. (2018). Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018, arXiv:1805.12114.
- Janner, M., Fu, J., Zhang, M., Levine, S. (2019). When to Trust Your Model, Model-Based Policy Optimization. NeurIPS 2019, arXiv:1906.08253.
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2025). Mastering diverse control tasks through world models. Nature 640, arXiv:2301.04104.
- Sims, K. (1994). Evolving Virtual Creatures. SIGGRAPH 1994.
- Cheney, N., Bongard, J., SunSpiral, V., Lipson, H. (2018). Scalable co-optimization of morphology and control in embodied machines. J. R. Soc. Interface 15(143), 20170937.
- Schaff, C., Yunis, D., Chakrabarti, A., Walter, M. R. (2019). Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning. ICRA 2019, arXiv:1801.01432.
- Ha, D. (2019). Reinforcement Learning for Improving Agent Design. Artificial Life 25(4), arXiv:1810.03779.
- Luck, K. S., Ben Amor, H., Calandra, R. (2020). Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning. CoRL 2019, PMLR 100, arXiv:1911.06832.
- Wang, T., Zhou, Y., Fidler, S., Ba, J. (2019). Neural Graph Evolution, Towards Efficient Automatic Robot Design. ICLR 2019, arXiv:1906.05370.
- Gupta, A., Savarese, S., Ganguli, S., Fei-Fei, L. (2021). Embodied intelligence via learning and evolution. Nature Communications 12, 5721.
- Gupta, A., Fan, L., Ganguli, S., Fei-Fei, L. (2022). MetaMorph, Learning Universal Controllers with Transformers. ICLR 2022, arXiv:2203.11931.
- Huang, W., Mordatch, I., Pathak, D. (2020). One Policy to Control Them All, Shared Modular Policies for Agent-Agnostic Control. ICML 2020, arXiv:2007.04976.
- Won, J., Lee, J. (2019). Learning Body Shape Variation in Physics-based Characters. ACM TOG 38(6), SIGGRAPH Asia 2019.
- Yuan, Y., Song, Y., Luo, Z., Sun, W., Kitani, K. (2022). Transform2Act, Learning a Transform-and-Control Policy for Efficient Agent Design. ICLR 2022, arXiv:2110.03659.
- Schaff, C., Walter, M. R. (2022). N-LIMB, Neural Limb Optimization for Efficient Morphological Design. arXiv:2207.11773.
- Strgar, L., Kriegman, S. (2025). Accelerated co-design of robots through morphological pretraining. arXiv:2502.10862, ICLR 2026.
- Hansen, N. (2016). The CMA Evolution Strategy, A Tutorial. arXiv:1604.00772, the optimisation-theory grounding of the outer loop, see `../context/cmaes.md`.
