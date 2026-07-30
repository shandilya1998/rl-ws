# Design and Controller Co-optimisation: RSL-RL Architecture and Implementation Guide

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. RSL-RL Framework Overview](#2-rsl-rl-framework-overview)
  - [a. Architecture](#a-architecture)
  - [b. Key Classes and Interfaces](#b-key-classes-and-interfaces)
    - [`runners.OnPolicyRunner`](#runnersonpolicyrunner)
    - [`algorithms.PPO`](#algorithmsppo)
    - [`modules.ActorCritic`](#modulesactorcritic)
    - [`storage.RolloutStorage`](#storagerolloutstorage)
  - [c. Sample Usage](#c-sample-usage)
- [3. Implementing Custom RL Algorithm](#3-implementing-custom-rl-algorithm)
  - [a. Guidelines](#a-guidelines)
  - [b. Implementing HIMLoco](#b-implementing-himloco)
- [4. Environment Lifecycle in IsaacLab](#4-environment-lifecycle-in-isaaclab)
  - [4.1 URDF to USD Conversion](#41-urdf-to-usd-conversion)
    - [4.1.1 Link Geometry Types and their USD Mapping](#411-link-geometry-types-and-their-usd-mapping)
    - [4.1.2 Per-Link-Type Conversion Logic](#412-per-link-type-conversion-logic)
    - [4.1.3 The `UrdfConverter` API and its Usage](#413-the-urdfconverter-api-and-its-usage)
  - [4.2 OpenUSD Theory: A Working Overview](#42-openusd-theory-a-working-overview)
    - [4.2.1 Stage, Layer, and Scene Composition](#421-stage-layer-and-scene-composition)
    - [4.2.2 Physics Schemas](#422-physics-schemas)
    - [4.2.3 The Fabric Interface and Runtime Constraints](#423-the-fabric-interface-and-runtime-constraints)
    - [4.2.4 TRON1A USD Sub-layer Architecture](#424-tron1a-usd-sub-layer-architecture)
  - [4.3 Configuration Hierarchy](#43-configuration-hierarchy)
  - [4.4 Environment Lifecycle](#44-environment-lifecycle)
    - [Construction Phase](#construction-phase)
    - [Episode Loop](#episode-loop)
    - [End-of-Life Sequence](#end-of-life-sequence)
  - [4.5 Step Loop and MDP Cycle](#45-step-loop-and-mdp-cycle)
    - [`OnPolicyRunner.learn()` outer loop](#onpolicyrunnerlearn-outer-loop)
    - [`ManagerBasedRLEnv.step()` inner mechanics](#managerbasedrlenvstep-inner-mechanics)
  - [4.6 Physics Parameter Modification](#46-physics-parameter-modification)
    - [4.6.1 Joint Control Parameters Update through PhysX](#461-joint-control-parameters-update-through-physx)
    - [4.6.2 Body Inertial Parameters Update through PhysX](#462-body-inertial-parameters-update-through-physx)
    - [4.6.3 Surface Properties Update through PhysX](#463-surface-properties-update-through-physx)
    - [4.6.4 USD Attribute Modification During Stop](#464-usd-attribute-modification-during-stop)
    - [4.6.5 Primitive-Shape Colliders and Efficient Size Updates](#465-primitive-shape-colliders-and-efficient-size-updates)
    - [4.6.6 What Cannot Be Changed at Runtime](#466-what-cannot-be-changed-at-runtime)
  - [4.7 Robot Morphology Update During Training](#47-robot-morphology-update-during-training)
    - [4.7.1 Configuration Trace: From `train.py` to Physics](#471-configuration-trace-from-trainpy-to-physics)
    - [4.7.2 `MultiUsdFileCfg` at Startup](#472-multiusdfilecfg-at-startup)
    - [4.7.3 Stop → Delete → Spawn → Play](#473-stop-delete-spawn-play)
      - [Asset Reference Map](#asset-reference-map)
      - [Full Respawn Sequence](#full-respawn-sequence)
    - [4.7.4 Procedural In-Memory USD Generation](#474-procedural-in-memory-usd-generation)
    - [4.7.5 Primitive Shape Geometry In-Place Update](#475-primitive-shape-geometry-in-place-update)
    - [4.7.6 Instantiation and Prototypes](#476-instantiation-and-prototypes)
  - [4.8 Recommended Strategy: Hybrid Two-Tier Co-optimisation](#48-recommended-strategy-hybrid-two-tier-co-optimisation)
    - [4.8.1 Architecture Overview](#481-architecture-overview)
    - [4.8.2  Implementation Requirements](#482-implementation-requirements)
    - [4.8.3 Generation of USD Files from EA Parameters](#483-generation-of-usd-files-from-ea-parameters)
  - [4.9 DesignGeneratorBase API](#49-designgeneratorbase-api)
    - [4.9.1 `class DesignGeneratorBase`](#491-class-designgeneratorbase)
    - [4.9.2 `class RandomDesignGenerator`](#492-class-randomdesigngenerator)
    - [4.9.3 `class CMAESDesignGenerator`](#493-class-cmaesdesigngenerator)
    - [4.9.4 `class CoptOnPolicyRunner`](#494-class-coptonpolicyrunner)
      - [The Committed `_update_morphology`](#the-committed-_update_morphology)
    - [4.9.5 Additional Major Changes](#495-additional-major-changes)
    - [4.9.6 Architecture Conclusion](#496-architecture-conclusion)
      - [Class and Population Hierarchy](#class-and-population-hierarchy)
      - [`generate_population()` Call Tree](#generate_population-call-tree)
      - [Dual Morphology-Update Pathway in `CoptOnPolicyRunner`](#dual-morphology-update-pathway-in-coptonpolicyrunner)
      - [`apply_link_length_params` and `update_articulation_links` Flow](#apply_link_length_params-and-update_articulation_links-flow)
      - [Key Invariants and Design Rationale](#key-invariants-and-design-rationale)
  - [4.10 Key Classes and Interfaces](#410-key-classes-and-interfaces)
    - [`SimulationContext`](#simulationcontext)
    - [`InteractiveScene`](#interactivescene)
    - [`ManagerBasedEnv`](#managerbasedenv)
    - [`ManagerBasedRLEnv`](#managerbasedrlenv)
    - [`AssetBase`](#assetbase)
    - [`Articulation`](#articulation)
    - [`ArticulationCfg`](#articulationcfg)
    - [`UsdFileCfg`](#usdfilecfg)
    - [`MultiUsdFileCfg`](#multiusdfilecfg)
    - [`UrdfFileCfg`](#urdffilecfg)
    - [`ImplicitActuatorCfg`](#implicitactuatorcfg)
    - [`IdentifiedActuatorCfg`](#identifiedactuatorcfg)
    - [`EventManager`](#eventmanager)
    - [`OnPolicyRunner`](#onpolicyrunner)
    - [`spawn_from_usd()`](#spawn_from_usd)
    - [`spawn_from_urdf()`](#spawn_from_urdf)
    - [`_spawn_from_usd_file()`](#_spawn_from_usd_file)
    - [`create_prim()`](#create_prim)
    - [`delete_prim()`](#delete_prim)
    - [`randomize_actuator_gains`](#randomize_actuator_gains)
    - [`randomize_rigid_body_mass`](#randomize_rigid_body_mass)
    - [`randomize_joint_parameters`](#randomize_joint_parameters)
    - [`SFSceneCfg`](#sfscenecfg)
    - [`SFEnvCfg`](#sfenvcfg)
    - [`SOLEFOOT_CFG`](#solefoot_cfg)
    - [`SOLEFOOT_IDENTIFIED_CFG`](#solefoot_identified_cfg)
    - [`SOLEFOOT_IDENTIFIED_MULTIUSD_CFG`](#solefoot_identified_multiusd_cfg)
- [5. Robot Design and Controller co-optimisation](#5-robot-design-and-controller-co-optimisation)

## 1. Introduction

This document provides a comprehensive overview of the software architecture of the `rsl_rl` reinforcement learning package and serves as a detailed guideline for implementing custom learning algorithms within this framework. Following a detailed dive into `rsl_rl`, the document discusses the simulation pipeline and the implementation of a design generation and update framework within the RL training pipeline.

The primary context for this documentation is the broader goal of design and controller co-optimization using IsaacLab. While IsaacLab provides a robust, GPU-accelerated simulation environment for complex robotic systems, `rsl_rl` acts as the learning engine—a fast, PyTorch-based RL library designed for high-performance training of locomotion policies. Standard algorithms like PPO are often insufficient for advanced tasks requiring memory, state estimation, or morphological adaptations. Therefore, understanding how to extend `rsl_rl` to implement custom training loops, auxiliary losses, and specialized neural network architectures is crucial.

Please note that this document focuses on the implementation of the learning algorithm using RSL-RL and IsaacLab. The specific application to design and controller co-optimization will be detailed in subsequent sections. Refer to [ARCHITECTURE.md](ARCHITECTURE.md) for more information regarding the configuration of the algorithms and environments implemented within this workspace.

## 2. RSL-RL Framework Overview

### a. Architecture

The `rsl_rl` package is designed with a highly modular, object-oriented architecture that cleanly separates the environment interaction loop, the mathematical algorithm logic, and the neural network definitions. This separation of concerns allows researchers to swap components without rewriting the entire training pipeline.

The flow of data during a standard training iteration involves four main components:

1.  Runner (`OnPolicyRunner`): The orchestrator. It manages the training loop, initializing the environment, algorithm, and policy based on configuration. It is responsible for executing rollouts (interacting with the environment) and triggering the algorithm's update phase.
2.  Algorithm (`PPO`): The core learning algorithm. It holds references to the policy and the storage buffer. After the runner completes a rollout, the algorithm computes advantages and returns, and then executes the optimization steps (e.g., computing surrogate, value, and entropy losses and applying gradient descent).
3.  Modules / Networks (`ActorCritic`): The neural network definitions (subclasses of `torch.nn.Module`). These modules take observations (typically as a `TensorDict`), route them through the appropriate sub-networks (Actor and Critic MLPs), and handle the sampling of action distributions.
4.  Storage (`RolloutStorage`): A specialized buffer that accumulates transitions during rollouts. It stores observations, actions, rewards, values, and environment termination signals, providing mini-batches to the algorithm during the update phase.

### b. Key Classes and Interfaces

Below is the API documentation for the core classes, detailing their constructor signatures, key member variables, and essential methods.

#### `runners.OnPolicyRunner`
- Role: Orchestrates the training process, logging, and checkpointing.
- Constructor Args:
  - `env`: The RL environment (wrapped to provide the expected interface).
  - `train_cfg` (dict): Training configuration parameters.
  - `log_dir` (str): Directory for saving logs and checkpoints.
  - `device` (str): Compute device (e.g., 'cuda:0').
- Key Methods:
  - `learn(num_learning_iterations, init_at_random_ep_len=False)`: The main training loop. Alternates between collecting data (rollouts) and updating the policy.
  - `_construct_algorithm()`: Internal method to instantiate the RL algorithm and policy based on `train_cfg`.
  - `save(path, infos=None)`: Saves the model state, including the policy and optimizer.
  - `load(path, load_optimizer=True)`: Loads a saved model state.

#### `algorithms.PPO`
- Role: Implements the Proximal Policy Optimization algorithm.
- Constructor Args:
  - `actor_critic` (nn.Module): The policy and value network module.
  - `device` (str): Compute device.
  - `**kwargs`: Hyperparameters (e.g., `clip_param`, `gamma`, `lam`, `learning_rate`).
- Key Variables:
  - `transition`: A specialized dictionary to hold the current step's data before it is added to storage.
  - `storage`: Instance of `RolloutStorage`.
- Key Methods:
  - `act(obs, critic_obs)`: Queries the policy for actions and values during rollouts.
  - `process_env_step(rewards, dones, infos)`: Adds the current transition to the storage buffer.
  - `compute_returns(last_critic_obs)`: Calculates generalized advantage estimations (GAE) at the end of a rollout.
  - `update()`: Iterates over mini-batches, computes losses (surrogate, value, entropy), and performs backpropagation to update the `actor_critic` network.

#### `modules.ActorCritic`
- Role: Defines the neural network architecture for the policy (actor) and value function (critic).
- Constructor Args:
  - `num_actor_obs` (int): Dimension of the actor's observation space.
  - `num_critic_obs` (int): Dimension of the critic's observation space.
  - `num_actions` (int): Dimension of the action space.
  - `actor_hidden_dims` (list): Architecture of the actor MLP.
  - `critic_hidden_dims` (list): Architecture of the critic MLP.
- Key Methods:
  - `update_distribution(observations)`: Passes observations through the actor network and instantiates a `torch.distributions.Normal` distribution.
  - `act(observations, **kwargs)`: Samples actions from the distribution and returns them.
  - `evaluate(critic_observations, **kwargs)`: Passes observations through the critic network to estimate the value.
  - `act_inference(observations)`: Used during deployment; returns the mean of the action distribution deterministically.

#### `storage.RolloutStorage`
- Role: Manages the collection and mini-batch generation of rollout data.
- Key Methods:
  - `add_transitions(...)`: Inserts a step's data into the buffer.
  - `compute_returns(last_values, gamma, lam)`: Computes GAE.
  - `mini_batch_generator(num_mini_batches, num_epochs)`: Yields randomized mini-batches for the algorithm's update loop.

### c. Sample Usage

A standard training setup using IsaacLab and RSL-RL can be found in `@tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py`. The typical workflow is as follows:

1. Configuration Loading: The script parses arguments to load the environment configuration (e.g., `LimxBaseEnvCfg`) and the RSL-RL agent configuration (e.g., `LimxBothPpoRunnerCfg`) from the task registry.
2. Environment Creation: The IsaacLab simulation is launched, and the environment is created using `gym.make(task_name, cfg=env_cfg)`.
3. Environment Wrapping: The core IsaacLab environment is wrapped using `RslRlVecEnvWrapper`. This crucial step bridges the gap between IsaacLab's output format and the `TensorDict` structure expected by `rsl_rl`.
4. Runner Initialization: An instance of `OnPolicyRunner` (or a custom derivative) is created, passing the wrapped environment and configuration.
5. Execution: `runner.learn()` is invoked to begin the training process. Logging (via TensorBoard) and model checkpoints are automatically handled within the runner's log directory.

## 3. Implementing Custom RL Algorithm

To implement advanced architectures (like state estimators or morphological co-optimization), one must extend the base classes provided by `rsl_rl`.

### a. Guidelines

The process of implementing a new algorithm involves the following structured steps. Let us consider the example of a custom learning algorithm whose package name is `sample_algorithm` All custom code for any RL algorithm should be placed in a self-contained package directory under the workspace (e.g., `tron1-rl-isaaclab-cozum/sample_algorithm/`), mirroring the sub-package layout of the base `rsl_rl` library:

<!-- TODO Need to update this section. This is AI Generated.  -->

```
sample_algorithm/
├── sample_algorithm/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── copt_ppo.py              # Custom PPO subclass
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── copt_actor_critic.py     # Custom ActorCritic subclass
│   │   └── copt_embedding.py        # Standalone nn.Module networks
│   ├── runners/
│   │   ├── __init__.py
│   │   └── copt_on_policy_runner.py # Custom OnPolicyRunner subclass
│   └── storage/
│       ├── __init__.py
│       └── copt_rollout_storage.py  # Custom RolloutStorage subclass
└── setup.py
```

---

1. Runner Implementation (`runners/copt_on_policy_runner.py`):

   - Create `CoptOnPolicyRunner(OnPolicyRunner)` inheriting from `rsl_rl.runners.OnPolicyRunner`
   - Override `_construct_algorithm(self, obs: TensorDict) -> CoptPPO`:
     - Replicate the body of `OnPolicyRunner._construct_algorithm()` (`rsl_rl/rsl_rl/runners/on_policy_runner.py`), which calls `resolve_rnd_config`, `resolve_symmetry_config`, instantiates the policy via `eval(self.policy_cfg.pop("class_name"))`, then instantiates the algorithm via `eval(self.alg_cfg.pop("class_name"))`, and finally calls `alg.init_storage(...)`. Add any additional steps for algorithm construction here. 
     - In the custom override, pass any additional constructor arguments required by the custom policy (e.g., `copt_cfg`) alongside the standard `(obs, obs_groups, num_actions, **policy_cfg)` signature. `HIMOnPolicyRunner._construct_algorithm()` in `himloco/himloco/runners/him_on_policy_runner.py` follows this exact pattern, forwarding `encoder_cfg` as an extra positional argument to `HIMActorCritic`.
   - Override `save(self, path: str, infos: dict | None = None) -> None`: 
     - Extend the base `save()`, which writes `model_state_dict` and `optimizer_state_dict` into a `torch.save` dict. 
     - Add an entry for every auxiliary optimizer or sub-module whose state must survive a checkpoint, e.g. `"copt_optimizer_state_dict": self.alg.copt_optimizer.state_dict()`. `HIMOnPolicyRunner.save()` demonstrates this by persisting `"estimator_optimizer_state_dict"` alongside the main optimizer.
   - Override `load(self, path: str, load_optimizer: bool = True, map_location=None) -> dict`: After restoring `model_state_dict` and `optimizer_state_dict` from the loaded dict, additionally call `load_state_dict` for each auxiliary component keyed in the checkpoint, e.g. `self.alg.copt_optimizer.load_state_dict(loaded_dict["copt_optimizer_state_dict"])`. `HIMOnPolicyRunner.load()` follows this pattern for the estimator optimizer.

2. Learning Implementation (`runners/copt_on_policy_runner.py`):

   - Override `learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None` only when the learning and data collection logic differs from the standard rollout-update loop
   - Inject pre-loop logic (e.g., warm-up epochs for an auxiliary network using a supervised signal) before `for it in range(start_iter, tot_iter):`.
   - Inject intra-rollout logic (e.g., refreshing morphological parameters each episode reset) inside the `with torch.inference_mode():` block after each `self.env.step()` call.
   - Inject post-update logic (e.g., annealing a regularisation coefficient based on a metric in `loss_dict`) immediately after `loss_dict = self.alg.update()`.
   - If none of these extension points are required, do not override `learn()`.

3. Update Logging (`runners/copt_on_policy_runner.py` or `algorithms/copt_ppo.py`):

   - The base `OnPolicyRunner.log()` already iterates over every key in `loss_dict` returned by `update()` and writes each to TensorBoard via `self.writer.add_scalar(f"Loss/{key}", value, locs["it"])`. It is therefore sufficient to ensure every auxiliary loss or metric is included in the `dict` returned by the custom `update()` (e.g., `loss_dict["copt_reg"] = mean_copt_reg_loss`). No override of `log()` is needed for scalar metrics.
   - Override `log(self, locs: dict, width: int = 80, pad: int = 35) -> None` only when non-scalar summaries are needed, such as writing parameter histograms via `self.writer.add_histogram("Copt/params", params_tensor, locs["it"])` or custom console output is required. Always call `super().log(locs, width, pad)` first to preserve the standard metrics.

4. Implement Custom Storage (`storage/copt_rollout_storage.py`):
    - Create `CoptRolloutStorage(RolloutStorage)` inheriting from `rsl_rl.storage.RolloutStorage`
    - Extend the inner `Transition` class: Add the new field (e.g., `self.copt_params: torch.Tensor | None = None`) and clear it in `clear()`. This inner class is not inherited — it must be redefined in full, as shown in `HIMRolloutStorage.Transition`.
    - Override `__init__`: Call `super().__init__(...)` then allocate the additional buffer (e.g., `self.copt_params = torch.zeros(num_transitions_per_env, num_envs, num_copt_params, device=device)`). `HIMRolloutStorage.__init__()` shows this pattern, allocating `self.next_observations` as an additional `TensorDict` buffer.
    - Override `add_transitions(self, transition)`: Call `super().add_transitions(transition)`, which writes all standard fields and increments `self.step`. Then copy the extra field at index `self.step - 1` (the slot just written), e.g. `self.copt_params[self.step - 1].copy_(transition.copt_params)`. `HIMRolloutStorage.add_transitions()` demonstrates this pattern.
    - Override `mini_batch_generator(self, num_mini_batches, num_epochs)`: Flatten the extra buffer alongside the standard ones (`self.copt_params.flatten(0, 1)`) and yield it as an additional element in the mini-batch tuple. The consuming `CoptPPO.update()` loop must unpack it in the same order. `HIMRolloutStorage.mini_batch_generator()` illustrates this by yielding `next_obs_batch` as the second element of the tuple.
    - Implement only when the algorithm needs to buffer additional per-step data beyond the standard fields (`observations`, `actions`, `rewards`, `dones`, `values`, `log_probs`, `mu`, `sigma`)

5. Implement Custom Algorithm (`algorithms/copt_ppo.py`):

   - Create `CoptPPO(PPO)` inheriting from `rsl_rl.algorithms.PPO`.
   - `__init__(self, policy, copt_lr: float = 1e-4, copt_reg_coef: float = 0.01, **kwargs)`: 
     - Call `super().__init__(policy, **kwargs)` to initialise all standard PPO components (the main `self.optimizer` over `policy.parameters()`, `clip_param`, `gamma`, etc.).
     - Instantiate a separate `self.copt_optimizer = optim.Adam(self.policy.copt_embedding.parameters(), lr=copt_lr)` so the auxiliary network's learning rate is controlled independently. Store `self.copt_reg_coef`.
     - Reassign `self.transition = CoptRolloutStorage.Transition()` so the extended fields are available during rollouts. `HIMPPO.__init__()` in `himloco/himloco/algorithms/him_ppo.py` follows this same pattern.
   - Override `init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape)`:
     - Instantiate `CoptRolloutStorage` instead of the base `RolloutStorage`, preserving the same call signature `(training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device)`. `HIMPPO.init_storage()` demonstrates this substitution with `HIMRolloutStorage`.
   - Override `process_env_step(self, obs, rewards, dones, extras)`:
     - Capture any additional per-step data before delegating to `super().process_env_step(obs, rewards, dones, extras)` (which calls `self.storage.add_transitions(self.transition)` and clears the transition).
   - Override `update(self) -> dict[str, float]`:
     - Replicate the mini-batch update loop from `ppo.py`.
    - After computing the standard PPO losses, compute the auxiliary loss (e.g., `copt_reg_loss = self.copt_reg_coef * embed.pow(2).mean()`). Call `self.optimizer.zero_grad()` and `self.copt_optimizer.zero_grad()`, call `.backward()` on the combined loss, clip gradients via `nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)`, then call `.step()` on both optimizers.
    - Include the auxiliary loss scalar in the returned dict: `loss_dict["copt_reg"] = mean_copt_reg`. `HIMPPO.update()` in `himloco/himloco/algorithms/him_ppo.py` illustrates this pattern with `estimation_loss` and `swap_loss`.

6. Implement Policy Module (`modules/copt_actor_critic.py`):

   - Create `CoptActorCritic(ActorCritic)` inheriting from `rsl_rl.modules.ActorCritic`.
   - `__init__(self, obs, obs_groups, num_actions, copt_cfg, **kwargs)`:
     - Call `super().__init__(obs, obs_groups, num_actions, **kwargs)` to build the standard actor/critic MLPs and observation normalisers.
     - Instantiate the `CoptEmbedding` sub-module (see step 6). Then rebuild `self.actor` as an `MLP` (from `rsl_rl.networks`) whose input dimension is `num_actor_obs + embed_dim`, since the parent constructed `self.actor` sized for `num_actor_obs` alone.
   - Override `_update_distribution(self, obs: TensorDict) -> None`:
     - Retrieve the auxiliary input from the `TensorDict` (e.g., `copt_params = obs["copt"]`), pass it through `self.copt_embedding` to obtain `embed`, concatenate with the normalised standard actor observation via `torch.cat([actor_obs, embed], dim=-1)`, then call `self.actor(augmented_obs)` and construct `self.distribution = Normal(mean, std)`
     - The base `act()` calls `_update_distribution()` then `self.distribution.sample()`, so overriding this single method is the minimal and preferred change.
   - Override `act_inference(self, obs: TensorDict) -> torch.Tensor`:
     - Mirror the `_update_distribution` augmentation logic but return the actor output deterministically (mean of distribution), used during deployment.
     - The base implementation in `actor_critic.py` returns `self.actor(obs)` directly without any embedding, so this must be overridden.
   - Override `evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor`:
     - Override only if the critic should also receive the morphological embedding.
     - If the critic relies solely on standard privileged observations, fall back to `return super().evaluate(obs, **kwargs)`.

7. Implement Network Elements (`modules/copt_embedding.py`):

   - Create `CoptEmbedding(nn.Module)`: a standalone `torch.nn.Module` responsible for encoding the raw morphological parameter vector into a fixed-size embedding.
   - `__init__(self, num_copt_params: int, embed_dim: int, hidden_dims: list[int], activation: str)`:
     - Build the encoder using `rsl_rl.networks.MLP` or a custom `nn.Sequential` that maps inputs of shape `[batch, num_copt_params]` to `[batch, embed_dim]`
   - `forward(self, copt_params: torch.Tensor) -> torch.Tensor`:
     - Pass the input through the encoder and return the embedding tensor of shape `[batch, embed_dim]`
   - Instantiate this inside `CoptActorCritic.__init__()` as `self.copt_embedding = CoptEmbedding(...)`. Because it is a named sub-module, its parameters are automatically included in `policy.state_dict()` and `policy.parameters()`, and are therefore covered by the base runner's `save()` / `load()` at no extra cost — unless a separate optimizer is used for it (see step 4).

---


### b. Implementing HIMLoco

The implementation of HIMLoco (History Information Model) at `@tron1-rl-isaaclab-cozum/himloco/himloco/` serves as an excellent case study applying these guidelines:

*   Runner (`HIMOnPolicyRunner`): Inherits from `OnPolicyRunner`. It ensures the `HIMPPO` algorithm and `HIMActorCritic` policy are used. Crucially, it overrides `save()` and `load()` to handle the state of the `HIMEstimator`'s separate optimizer, which is trained alongside the PPO actor-critic.
*   Algorithm (`HIMPPO`): Inherits from `PPO`. 
    *   It uses a custom `HIMRolloutStorage` and overrides `process_env_step()` to store `next_observations`, which are required for temporal self-supervised learning.
    *   The `update()` method is significantly modified. It calls `self.policy.estimator.update()` to calculate two new auxiliary losses: `estimation_loss` (an MSE loss for predicting ground truth velocity) and `swap_loss` (a self-supervised contrastive loss using the Sinkhorn-Knopp algorithm, similar to SwAV).
*   Policy (`HIMActorCritic`): Inherits from `ActorCritic`. It acts as a wrapper. Before passing data to the standard actor MLP, it extracts the `obsHistory` from the environment's `TensorDict` and passes it through the `HIMEstimator`. The resulting estimated velocity and latent history vector are concatenated with the raw observations to form the augmented input for the actor.
*   Network Elements (`HIMEstimator`): A custom `nn.Module` representing the core of the HIMLoco approach. It processes observation history using convolutional or dense layers to generate a latent representation, implementing the specific forward passes needed for both the velocity prediction and the contrastive clustering loss.

---

## 4. Environment Lifecycle in IsaacLab

This section provides the relevant background required for robot design and controller co-optimisation using IsaacLab. The full lifecycle of a simulated environment from configuration through initialisation to runtime is introduced and the procedure for physics parameters and robot geometry modification is discussed. The section concludes with a recommendation for a hybrid strategy for jointly optimising robot morphology and locomotion policy using Evolutionary Algorithms (EA) and PPO.

---

### 4.1 URDF to USD Conversion

The Unified Robot Description Format (URDF) is the de-facto industry-standard, XML-based format for describing a robot's physical configuration and is ubiquitous across the ROS ecosystem and robotics tooling ([URDF overview](https://en.wikipedia.org/wiki/URDF)). It is also the modelling choice for this project. Each robot variant is authored as a URDF (for example [`base_robot.urdf`](tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/base_robot.urdf)) and converted to a USD asset before it is loaded into IsaacLab.

IsaacLab performs this conversion through `UrdfConverter` ([`IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py`](IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py)), a thin wrapper around the Isaac Sim URDF Importer extension (`isaacsim.asset.importer.urdf`). The remainder of this subsection describes how the importer maps URDF link geometry to USD prims, the per-link-type conversion logic, and the converter API as it is used in this codebase.

#### 4.1.1 Link Geometry Types and their USD Mapping

A URDF `<link>` carries up to three geometry-bearing children: `<visual>` (rendered appearance), `<collision>` (the shape PhysX simulates), and `<inertial>` (mass, centre of mass, inertia tensor). For each `<visual>`/`<collision>`, the importer authors a `UsdGeomXform` wrapper whose `xformOp:translate`/`orient`/`scale` carry the URDF `<origin>` and a scale, then attaches a typed geometry prim under it ([`UrdfImporter.cpp:242-288`](IsaacSim/source/extensions/isaacsim.asset.importer.urdf/plugins/isaacsim.asset.importer.urdf/UrdfImporter.cpp)). The geometry tag determines the USD prim type:
| URDF geometry | USD prim | How size is encoded | Importer code |
|---|---|---|---|
| `<box>` | `UsdGeomCube` | unit cube (`size`=1.0) + `xformOp:scale = (x, y, z)` | `UrdfImporter.cpp:200-214`, `getScale():147` |
| `<cylinder>` | `UsdGeomCylinder` | intrinsic `height` + `radius`, axis Z | `UrdfImporter.cpp:216-227` |
| `<sphere>` | `UsdGeomSphere` | intrinsic `radius` | `UrdfImporter.cpp:190-198` |
| `<capsule>` | `UsdGeomCapsule` | intrinsic `height` + `radius`, axis Z | `UrdfImporter.cpp:229-240` |
| `<mesh>` | referenced `UsdGeomMesh` | `xformOp:scale`; mesh attached as a USD reference | `addMeshReference():152-188` |

Primitive shapes (box, cylinder, sphere, capsule) are preserved as analytic USD gprims, not baked into meshes. Only `<mesh>` geometry is imported as triangle data and requires cooking. The box dimensions live in `xformOp:scale` (the cube itself is a unit cube), so a box link's size is changed by editing the scale op, the mechanism exploited in [Section 4.6](#46-physics-parameter-modification). Cylinders, spheres, and capsules instead store their size in intrinsic attributes (`height`/`radius`). In `base_robot.urdf`, all length-scalable structural links (the thigh, shank, and foot links) are `<box>`, while the actuator housings are `<cylinder>`.

#### 4.1.2 Per-Link-Type Conversion Logic

Geometry dispatch. `getScale()` ([`UrdfImporter.cpp:140-151`](IsaacSim/source/extensions/isaacsim.asset.importer.urdf/plugins/isaacsim.asset.importer.urdf/UrdfImporter.cpp)) returns the box dimensions for a `<box>` (applied to the wrapper's scale op) and `(1, 1, 1)` for the other primitives (whose size is intrinsic). The geometry switch authors the corresponding prim via `addBox`/`addCylinder`/`addSphere`/`addCapsule`/`addMeshReference` (`UrdfImporter.cpp:268-285`).

Collision authoring. Every collider receives a bare `UsdPhysics.CollisionAPI` (`UrdfImporter.cpp:571-573`). The mesh-approximation schema — `UsdPhysics.MeshCollisionAPI` with the `convexHull` or `convexDecomposition` token — is applied only when the geometry type is `MESH` (`UrdfImporter.cpp:574-589`); a `<box>`/`<cylinder>` collision therefore becomes a primitive collider with no approximation. Collision prims are tagged `purpose = guide` (`UrdfImporter.cpp:599`), and both the visuals and colliders scopes are referenced and marked `instanceable` (`UrdfImporter.cpp:530-531, 568-569`).

Rigid body, mass and inertia. Each link gets `UsdPhysics.RigidBodyAPI` (`UrdfImporter.cpp:366`) and `UsdPhysics.MassAPI` (`UrdfImporter.cpp:378`). Mass is taken from the URDF `<mass>` when present, otherwise computed from `link_density` if the link has colliders (`UrdfImporter.cpp:379-394`). The inertia matrix is diagonalised and written as `diagonalInertia` + `principalAxes` (`UrdfImporter.cpp:398-410`); the centre of mass is written from the `<inertial>` origin (`UrdfImporter.cpp:413-431`). A link with neither colliders nor inertia is given a small isotropic inertia to avoid an ill-defined body (`UrdfImporter.cpp:603-616`).

Fixed-joint merging. When `merge_fixed_joints` is enabled, `collapseFixedJoints()` (`UrdfImporter.cpp:108-111`; implementation `ImportHelpers.cpp:112-210`) folds links joined by fixed joints into their parent, combining masses and inertias by the parallel-axis theorem. A fixed child carrying non-zero mass is not silently merged — the joint is flagged `dontCollapse` instead (`ImportHelpers.cpp:124-131`).

Articulation root. With `fix_base = True`, the importer synthesises a world-to-root fixed joint and applies `UsdPhysics.ArticulationRootAPI` to it (`UrdfImporter.cpp:1217-1237`); with `fix_base = False` (the floating base used for the biped) the root API is applied to the root link, provided the robot has joints (`UrdfImporter.cpp:1515-1531`).

Joints and drives. URDF revolute/continuous/prismatic/fixed joints map to the corresponding `UsdPhysics` joint types (`UrdfImporter.cpp:990-1127`); angular limits are converted to degrees (×180/π) (`UrdfImporter.cpp:980-981`). `configureDriveAPI()` (`UrdfImporter.cpp:851-890`) authors the `UsdPhysics.DriveAPI` stiffness, damping, and target. Note that `target_type = "none"` yields zero stiffness and damping (`UrdfImporter.cpp:885-889`) — the configuration appropriate for explicit feed-forward actuator models such as `IdentifiedActuator` (see [Section 4.2.2](#422-physics-schemas)).

#### 4.1.3 The `UrdfConverter` API and its Usage

`UrdfConverter` is configured by `UrdfConverterCfg` ([`urdf_converter_cfg.py`](IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py)). The fields most relevant to this project:

| Field | Default | Purpose |
|---|---|---|
| `asset_path` | — | Absolute path to the source URDF |
| `usd_dir` / `usd_file_name` | derived | Output USD location/name (`usd_path` returns their join) |
| `force_usd_conversion` | `False` | Always regenerate, bypassing the cache |
| `make_instanceable` | `True` | Emit an instanceable USD to reduce per-instance memory |
| `fix_base` | (required) | Fixed vs floating base |
| `link_density` | `0.0` | Fallback density for links missing `<inertial>` |
| `merge_fixed_joints` | `True` | Fold fixed-joint links into their parent |
| `collider_type` | `"convex_hull"` | Mesh-collision approximation (`convex_hull`/`convex_decomposition`) — affects mesh colliders only |
| `self_collision` | `False` | Enable intra-articulation self-collision |
| `replace_cylinders_with_capsules` | `False` | Convert cylinder colliders to capsules (PhysX simulates capsules exactly) |
| `joint_drive` | `JointDriveCfg()` | Drive type/target/gains; set to `None` for no drive |

(Full field set: [`urdf_converter_cfg.py:13-132`](IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py).)

The converter inherits lazy, hash-based conversion from `AssetConverterBase` ([`asset_converter_base.py:101`](IsaacLab/source/isaaclab/isaaclab/sim/converters/asset_converter_base.py)): a USD is regenerated only when `force_usd_conversion` is set, the output is missing, or the config/asset hash changed. Importantly, the hash covers the config and the main URDF file only — edits to externally referenced mesh files do not trigger re-conversion (`asset_converter_base.py:164-198`). At construction, `_get_urdf_import_config()` maps each cfg field onto an importer setter (`set_convex_decomp`, `set_merge_fixed_joints`, `set_fix_base`, `set_self_collision`, `set_replace_cylinders_with_capsules`, …) (`urdf_converter.py:117-154`), parses the URDF, and writes the USD; the result is exposed via the `usd_path` property (`asset_converter_base.py:132-135`).

In this codebase the converter is driven by the design generator: `RandomDesignGenerator._convert_urdf_to_usd()` ([`usd_generator.py:489-506`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)) builds a `UrdfConverterCfg` with `collider_type="convex_hull"`, `merge_fixed_joints=False`, `fix_base=False`, `self_collision=False`, `joint_drive=None`, and `force_usd_conversion=True`, then returns `converter.usd_path`. Because the length-scaled links are boxes (Section 4.1.1), `collider_type` is immaterial for them — they import as primitive `UsdGeomCube` colliders regardless. For headless or manual conversion, IsaacLab also ships a CLI tool, `scripts/tools/convert_urdf.py`, exposing the same options (`--fix-base`, `--merge-joints`, `--make-instanceable`, `--joint-target-type`, …).

---

### 4.2 OpenUSD Theory: A Working Overview

IsaacLab uses OpenUSD (Universal Scene Description) as its native scene representation format. OpenUSD is a language agnostic scene and object description format used extensively in the animation and robotics industries. IsaacLab leverages OpenUSD as a universal scene description format before PhysX takes over simulation.

#### 4.2.1 Stage, Layer, and Scene Composition

`UsdStage` is the single authoritative view of the scene. It is a composed, in-memory object assembled from one or more file-backed layers. The stage is not a file itself — it is the result of composing multiple files according to USD's composition rules.

`SdfLayer` is the file-level container. USD supports two formats: `.usda` (human-readable ASCII) and `.usdc` (binary crate format). A single USD asset such as `SF_TRON1A.usd` is typically the root layer of a composition that includes several sub-layers — for example, separate layers for geometry, physics, and sensors. When the stage is opened, all contributing layers are parsed and their values are composed according to the opinion strength rules. An opinion is USD's term for a single layer's assertion about the value of a property or metadata field. Since OpenUSD composes multiple layers together to create the stage, multiple layers may attempt to override the value of the same attribute. Each layer possess an opinion and opinion resolution is performed using a fixed opinion strength hierarchy (from strongest to weakest): local opinions → reference opinions → payload opinions → inherited opinions → built-in fallbacks.

Composition Arcs are the mechanism by which layers and prims reference one another. The most common arc is a reference — a statement that the content of an external USD file should be inserted at a given prim path. This is how `_spawn_from_usd_file()` ([`IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:314`](IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py)) adds a robot to the scene: it calls `create_prim(prim_path, usd_path=usd_path, ...)` ([`IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py:44`](IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py)), which inserts a reference arc to the robot's USD file at the path returned by `GridCloner`.

`SdfPath` is the hierarchical addressing scheme for all scene objects, following a Unix-style path notation (e.g., `/World/envs/env_0/Robot/base_Link`). The `GridCloner` used by `InteractiveScene` ([`IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:125`](IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py)) produces environment-scoped paths of the form `/World/envs/env_{i}/Robot`, giving each parallel environment its own isolated prim subtree.

`UsdPrim` is the fundamental addressable node in the scene graph. Every robot link, joint, and sensor is a prim. Prims carry typed schemas and API schemas that describe their physical behaviour. The `Articulation._initialize_impl()` method ([`IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:1506`](IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py)) searches the prim subtree for a prim bearing `UsdPhysics.ArticulationRootAPI` to anchor the PhysX articulation.

`UsdAttribute` is a typed, namespaced property on a prim. Attributes carry opinions from layers and are composed according to USD's value resolution rules. Physics attributes such as `drive:angular:physics:stiffness` etc. are USD attributes that PhysX reads at `sim.reset()` time. Refer to [Section 4.2.2](#422-physics-schemas) for more information regarding physics simulation.

Within an IsaacLab session, the current stage is accessed via `omni.usd.get_context().get_stage()` or through the `use_stage(self.sim.get_initial_stage())` context manager used during scene construction (line 141 of `manager_based_env.py`). The `AssetBase.__init__()` method ([`IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py:74`](IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py)) also holds a direct reference as `self.stage = get_current_stage()`. The reference to the stage variable in IsaacLab may be used to access specific prims, update prim or stage properties, perform runtime composition etc before the simulation is handled over to PhysX.

#### 4.2.2 Physics Schemas

USD physics is defined by a set of API schemas applied to prims. In object-oriented terms, a typed schema defines what a prim is (e.g. `Mesh`, `Xform`, `Capsule` — analogous to a class). An API schema defines capabilities a prim has, independently of its type — analogous to an interface or mixin. API schemas can, thus, be applied to any prim without changing its type. The following schemas govern articulation behaviour in PhysX:

`UsdPhysics.ArticulationRootAPI`: Applied to the root prim of a robot (e.g., `base_Link`). Marks the entire prim subtree as a reduced-coordinate articulation, enabling PhysX to use the Featherstone algorithm for efficient dynamics. Detected by `Articulation._initialize_impl()` at line 1525 of `articulation.py`. Configured in IsaacLab via `ArticulationRootPropertiesCfg` ([`IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py`](IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py)), which is passed as `articulation_props` in `UsdFileCfg` (e.g., `solefoot_cfg.py:23`).

`UsdPhysics.RigidBodyAPI`: Applied to each link, enabling mass and inertia simulation. Every robot link that contributes to dynamics must have this schema.

`UsdPhysics.MassAPI`: Provides explicit mass (`physics:mass`), centre of mass (`physics:centerOfMass`), and inertia tensor (`physics:diagonalInertia`, `physics:principalAxes`) overrides on a rigid body prim. If absent, PhysX computes these from geometry density. At runtime, mass and inertia are updated via `root_physx_view.set_masses()` and `root_physx_view.set_inertias()` (used in `randomize_rigid_body_mass` at [`IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:286`](IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py)).

`UsdPhysics.CollisionAPI`: Applied to geometry prims to mark them as collision shapes. Collision geometry is cooked (pre-processed for narrow-phase collision) by PhysX at load time (during `sim.reset()`). This cooking is irrevocable for the lifetime of the simulation — collision geometry, and by extension link lengths and joint attachment points, cannot be modified at runtime. This is the fundamental reason why geometric morphology changes require a new USD load.

`UsdPhysics.DriveAPI`: Applied to joint prims, defining the PD drive for each degree of freedom. The force law is:

```
F = Kp × (q* − q) − Kd × (q̇* − q̇)
```

where `Kp` is `drive:angular:physics:stiffness`, `Kd` is `drive:angular:physics:damping`, `q*` is `drive:angular:physics:targetPosition`, and `q̇*` is `drive:angular:physics:targetVelocity`.

IsaacLab manifestations:
- For `ImplicitActuatorCfg` ([`isaaclab/actuators/actuator_cfg.py`](IsaacLab/source/isaaclab/isaaclab/actuators/actuator_cfg.py)): the `stiffness` and `damping` fields are written directly as `DriveAPI` attributes on the joint prim at spawn time. PhysX then applies the force law natively.
- For `IdentifiedActuatorCfg` ([`tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/actuators/actuator_cfg.py:15`](tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/actuators/actuator_cfg.py)): the USD `DriveAPI` stiffness and damping are set to zero; torque is computed in the actuator model by `IdentifiedActuator.compute()` and applied as a feed-forward force command. The `saturation_effort`, `friction_static`, `activation_vel`, and `friction_dynamic` fields model physical actuator non-linearities absent from the standard DriveAPI.

#### 4.2.3 The Fabric Interface and Runtime Constraints

The Fabric interface is a USD/PhysX integration layer that activates when `sim.reset()` is called ([`IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:173`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py)). Fabric takes ownership of all live physics state. It maintains a high-performance, GPU-resident data store that mirrors the PhysX simulation state. Rather than reading from the USD stage on every step, the simulation reads from and writes to the Fabric store directly via the PhysX Tensor API. This is what makes GPU-accelerated parallel simulation possible.

After `sim.reset()` fires, if you attempt to modify a USD prim attribute at runtime — for example, `prim.GetAttribute("drive:angular:physics:stiffness").Set(25.0)` — Fabric intercepts and discards the write. The attribute value in the USD stage changes, but the running simulation is unaffected. USD attribute writes are silently ignored once Fabric is active. This constraint only applies to runtime modification of USD attributes after simulation has started. It does not prevent you from setting robot configuration in USD files. The correct workflow for robot properties update is:

| Phase | Mechanism | USD file values |
|---|---|---|
| Before `sim.reset()` | USD stage parsed by PhysX | Fully effective — this is how you configure the robot |
| Prestartup events window | `EventManager.apply(mode="prestartup")` (line 161–162) | USD prim attributes can be written safely here |
| After `sim.reset()` | Fabric / PhysX Tensor API | USD attributes ignored; use `write_*_to_sim()` |
| After stop + new `sim.reset()` | USD re-parsed | USD values effective again |


The pre-Fabric prestartup window (created specifically for USD-level domain randomisation, as noted in the code comment at `manager_based_env.py:155–157`) allows event functions to modify USD prim attributes before simulation starts — for example, randomising mesh scale or initial joint drive parameters at environment launch. After Fabric activation, physics parameters that must change between episodes must be updated via the PhysX Tensor API methods exposed by `Articulation.root_physx_view`. These are documented in §5.6.

#### 4.2.4 TRON1A USD Sub-layer Architecture

The TRON1A robot assets (located under `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/usd/`) follow a sub-layer composition pattern:

```
SF_TRON1A.usd          ← Root layer (composition entry point)
├── SF_TRON1A_base.usd      ← Geometry: meshes, joints, joint limits
├── SF_TRON1A_physics.usd   ← Physics: MassAPI (mass, inertia, CoM),
│                               CollisionAPI, DriveAPI (stiffness, damping)
└── SF_TRON1A_sensor.usd    ← Sensors: contact geometry, camera rigs
```

`UsdFileCfg.usd_path` (e.g., `solefoot_cfg.py:8`) always points to the root layer `SF_TRON1A.usd`. When this file is referenced into the stage via `_spawn_from_usd_file()`, USD's composition rules automatically load and merge all sub-layers. 

---

### 4.3 Configuration Hierarchy

The path from a Python configuration object to a simulated robot involves several distinct stages of resolution. The following steps lead to the spawning of simulation entities til they are ready for usage:

Step 1 — Task configuration (`tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py:121`):

```python
env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
    task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
)
```

`parse_env_cfg()` resolves the registered task name to a concrete `SFEnvCfg` instance ([`tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:963`](tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py)) containing `decimation=4`, `episode_length_s=20.0`, `sim.dt=0.005`.

Step 2 — Scene configuration (`limx_base_env_cfg.py:34`):

`SFEnvCfg` holds a `SFSceneCfg(InteractiveSceneCfg)` as `self.cfg.scene`. `SFSceneCfg` declares `robot: ArticulationCfg = MISSING` at `limx_base_env_cfg.py:71` — a placeholder that task-specific configurations replace with a concrete instance such as `SOLEFOOT_IDENTIFIED_CFG` from `solefoot_identified_cfg.py:101`.

Step 3 — Environment construction (`gym.make()` → `ManagerBasedEnv.__init__()` → `InteractiveScene.__init__()`):

`InteractiveScene` iterates over all entities declared in `SFSceneCfg`. For the robot entity, it instantiates `Articulation(cfg=robot_cfg)`, which inherits from `AssetBase`.

Step 4 — Prim spawning (`asset_base.py:84`):

```python
self.cfg.spawn.func(
    self.cfg.prim_path,
    self.cfg.spawn,
    translation=self.cfg.init_state.pos,
    orientation=self.cfg.init_state.rot,
)
```

`cfg.spawn.func` resolves to `spawn_from_usd()` ([`IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:38`](IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py)), which calls `_spawn_from_usd_file(prim_path, cfg.usd_path, ...)` (`from_files.py:268`). For URDF sources, `spawn_from_urdf()` (`from_files.py:80`) first converts the URDF to USD via `UrdfConverter(cfg)` then calls `_spawn_from_usd_file(prim_path, urdf_loader.usd_path, ...)`. In both cases, `_spawn_from_usd_file()` calls `create_prim(prim_path, usd_path=usd_path, ...)` (`from_files.py:314`), inserting a reference arc into the live USD stage. This is the moment the robot exists in the scene graph.

Step 5 — Physics initialisation (`sim.reset()` → timeline PLAY → `_initialize_callback()`):

When `sim.reset()` fires, PhysX parses the USD stage. All `MassAPI`, `DriveAPI`, `CollisionAPI`, and `ArticulationRootAPI` values — including link lengths encoded in geometry, mass, inertia, stiffness, and damping — are read from the USD stage and become the initial physics state. `Articulation._initialize_impl()` (`articulation.py:1506`) then creates `root_physx_view = self._physics_sim_view.create_articulation_view(...)` (line ~1547), the PhysX Tensor API handle through which all subsequent runtime physics updates are issued.

---

### 4.4 Environment Lifecycle

The complete lifecycle of a `ManagerBasedRLEnv` environment from construction to teardown proceeds through the following stages.

#### Construction Phase

All code executes within `ManagerBasedEnv.__init__()` ([`IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:77`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py)):

1. `SimulationContext` created (`manager_based_env.py:104`): Initialises the Isaac Sim application context, rendering pipeline, and physics engine parameters (`sim.dt`, `sim.gravity`, etc.).

2. `InteractiveScene` created (`manager_based_env.py:142`): Iterates over all entities in `SFSceneCfg`. For each asset, calls `cfg.spawn.func()` to insert reference arcs into the USD stage. `GridCloner` runs during this phase, producing prim paths (`/World/envs/env_{i}/Robot`) for all `num_envs` parallel environments. The `_articulations` dictionary ([`interactive_scene.py:125`](IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py)) is populated with all `Articulation` instances.

3. `EventManager` created (`manager_based_env.py:158`): Created before `sim.reset()` specifically to allow USD-level randomisation events. The Events Manager is created before the simulation starts to allow USD-related randomization events that must happen before the simulation starts. Example: randomizing mesh scale".

4. Prestartup events applied (`manager_based_env.py:161–162`): `event_manager.apply(mode="prestartup")` fires all `EventTermCfg` instances registered under the `"prestartup"` mode. These event functions can safely write to USD prim attributes at this point, before Fabric activates.

5. `sim.reset()` called (`manager_based_env.py:173`): The timeline transitions to PLAY. This triggers timeline callbacks. `AssetBase._initialize_callback()` ([`asset_base.py:304`](IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py)) fires for every asset, which calls `_initialize_impl()` if `not self._is_initialized`. For the robot articulation, this creates `root_physx_view`. Fabric activates after which USD prim attribute writes are ignored.

6. `scene.update()` (`manager_based_env.py:178`): Pre-populates sensor and asset data buffers so that the observation manager can initialise with valid tensors.

7. `load_managers()` called (`manager_based_env.py:180`): Creates the six MDP managers in order (Refer to [`manager_based_rl_env.py:109`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py) for more details):
   - `CommandManager` (`manager_based_rl_env.py:113`)
   - `ActionManager` and `ObservationManager` via `manager_based_env.py317:320`
   - `TerminationManager` (`manager_based_rl_env.py:121`)
   - `RewardManager` (`manager_based_rl_env.py:124`)
   - `CurriculumManager` (`manager_based_rl_env.py:127`)
   - Startup events applied: `event_manager.apply(mode="startup")` (`manager_based_rl_env.py:134–135`)

#### Episode Loop

Once construction is complete, `OnPolicyRunner.learn()` ([`rsl_rl/rsl_rl/runners/on_policy_runner.py:62`](rsl_rl/rsl_rl/runners/on_policy_runner.py)) drives the training loop. A high-level overview follows here; §4.5 covers the step loop in detail.

```
for iteration in range(num_iterations):
    collect rollout (num_steps_per_env × env.step())
    compute returns
    loss_dict = alg.update()       ← PPO gradient step
    ← EA hook point here (§4.9)
    save checkpoint
```

Within each `env.step()` call, `_reset_idx()` ([`manager_based_rl_env.py:349`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)) triggers `"reset"` events (domain randomisation) for all environments whose episodes have terminated.

#### End-of-Life Sequence

When `env.close()` is called (e.g., after `runner.learn()` returns at `train.py:214`):

1. `sim.stop()` fires timeline STOP event.
2. `AssetBase._invalidate_initialize_callback()` ([`asset_base.py:324`](IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py)) fires for every asset, setting `self._is_initialized = False`. `root_physx_view` is invalidated.
3. If prims are explicitly deleted via `delete_prim()` ([`prims.py:189`](IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py)), `AssetBase._on_prim_deletion()` (line 331) clears all callbacks.
4. `simulation_app.close()` (`train.py:221`) shuts down the Isaac Sim application.

---

### 4.5 Step Loop and MDP Cycle

This section details the inner mechanics of a single training iteration.

#### `OnPolicyRunner.learn()` outer loop

```python
# rsl_rl/rsl_rl/runners/on_policy_runner.py:98
for it in range(start_iter, tot_iter):
    # --- Rollout collection (inference mode) ---
    with torch.inference_mode():
        for _ in range(self.num_steps_per_env):        # line 102
            actions = self.alg.act(obs)                # line 104
            obs, rewards, dones, extras = self.env.step(actions)  # line 106
            self.alg.process_env_step(obs, rewards, dones, extras)# line 110
            cur_reward_sum += rewards                  # line 125
            new_ids = (dones > 0).nonzero(as_tuple=False) # line 129
            # per-env fitness: cur_reward_sum[new_ids] at episode end
            cur_reward_sum[new_ids] = 0                # line 132

        self.alg.compute_returns(obs)                  # line 145

    # --- Policy update ---
    loss_dict = self.alg.update()                      # line 148
    # ← EA hook point: inject generation update here

    if it % self.save_interval == 0:
        self.save(...)                                 # line 158-159
```

#### `ManagerBasedRLEnv.step()` inner mechanics

Each call to `env.step(actions)` ([`manager_based_rl_env.py:153`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)) executes:

1. Decimation loop (`decimation=4` times): applies action → calls `sim.step()` → advances physics by `dt=0.005 s`. Net policy step: 4 × 0.005 = 0.02 s = 50 Hz policy rate.
2. Scene update: sensors and asset data buffers refreshed.
3. Termination check: environments exceeding `episode_length_s=20.0` or hitting fall conditions are flagged.
4. `_reset_idx(env_ids)` (line 349): for flagged environments — fires `"reset"` domain randomisation events, resets scene buffers, resets all six MDP managers.
5. Reward computation: `RewardManager` accumulates scalar reward.
6. Observation computation: `ObservationManager` constructs the `TensorDict` returned to the runner.

---

### 4.6 Physics Parameter Modification

Upon `env.reset()` invocation, the physics state of the simulation is handed over to PhysX from the USD files. Updates to physics properties must be made through PhysX APIs. Therefoer, once the simulation is running, physics parameters can be modified without restarting the simulation provided the changes do not involve geometry or joint topology. All modifications are issued via `Articulation.root_physx_view` using CPU tensors. It must be noted that the requirement of CPU tensors constraint realtime updage of physics paramters. Consequently, the physics parameters can only be updated at environment reset. If, however, the environment is stopped again, for robot morphology updates, USD objects may be directly modified to update the parameters required.

#### 4.6.1 Joint Control Parameters Update through PhysX

These can be applied to a subset of environments at any time step:

```python
# Stiffness — ImplicitActuator only; writes DriveAPI physics:stiffness via PhysX
robot.write_joint_stiffness_to_sim(
    stiffness=new_kp,          # shape: (len(env_ids), num_joints)
    joint_ids=joint_indices,
    env_ids=env_ids,
)  # articulation.py:652 → root_physx_view.set_dof_stiffnesses()

# Damping — ImplicitActuator only
robot.write_joint_damping_to_sim(
    damping=new_kd,
    joint_ids=joint_indices,
    env_ids=env_ids,
)  # articulation.py:681 → root_physx_view.set_dof_dampings()

# For IdentifiedActuator (explicit): update software-side gains directly
for name, actuator in robot.actuators.items():
    if isinstance(actuator, IdentifiedActuator):
        actuator.stiffness[env_ids] = new_kp_values
        actuator.damping[env_ids] = new_kd_values

# Joint position limits
robot.write_joint_position_limit_to_sim(
    limits=new_limits,  # shape: (len(env_ids), num_joints, 2)  [lower, upper]
    joint_ids=joint_indices,
    env_ids=env_ids,
)  # articulation.py:710

# Armature (reflected rotor inertia)
robot.write_joint_armature_to_sim(
    armature=new_armature,
    joint_ids=joint_indices,
    env_ids=env_ids,
)  # articulation.py:840 → root_physx_view.set_dof_armatures()

# Joint friction coefficient
robot.write_joint_friction_coefficient_to_sim(
    joint_friction_coeff=new_friction,
    joint_ids=joint_indices,
    env_ids=env_ids,
)  # articulation.py:871
```

#### 4.6.2 Body Inertial Parameters Update through PhysX

The `randomize_rigid_body_mass` event function (events.py:286) demonstrates the pattern:

```python
# Get current masses (shape: (num_envs, num_bodies))
masses = robot.root_physx_view.get_masses()     # returns CPU tensor
masses[env_ids, body_ids] *= mass_scale_factors

# Write back to PhysX
robot.root_physx_view.set_masses(masses, env_ids.cpu())

# For inertia (shape: (num_envs, num_bodies, 9) — flattened 3×3 tensor)
inertias = robot.root_physx_view.get_inertias()
inertias[env_ids] = new_inertia_values
robot.root_physx_view.set_inertias(inertias, env_ids.cpu())
```

#### 4.6.3 Surface Properties Update through PhysX

```python
# Contact material properties (friction, restitution)
robot.root_physx_view.set_material_properties(materials, env_ids)
```

#### 4.6.4 USD Attribute Modification During Stop

While the simulation is stopped (between STOP and the next PLAY event), the Fabric interface is inactive and USD prim attribute writes become effective. This allows modifying physics parameters (mass, stiffness, joint limits) in the existing USD stage without replacing prims.

```python
env.sim.stop()  # Fabric deactivates

# Safe to write USD attributes here
stage = omni.usd.get_context().get_stage()
for i in range(num_envs):
    joint_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/knee_L_Joint")
    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    drive_api.GetStiffnessAttr().Set(new_stiffness)

env.sim.reset()  # PhysX re-parses; new values take effect
```

The writes above are inert until the next PLAY event. When `sim.reset()` is invoked after a stop, the timeline transitions to PLAY and `SimulationManager.initialize_physics()` calls `force_load_physics_from_usd()` ([`IsaacSim/source/extensions/isaacsim.core.simulation_manager/python/impl/simulation_manager.py:237`](IsaacSim/source/extensions/isaacsim.core.simulation_manager/python/impl/simulation_manager.py)). This routine re-reads every physics-bearing attribute, such as `MassAPI`, `DriveAPI`, `CollisionAPI`, `ArticulationRootAPI`, and the geometry of every collider from the composed USD stage and rebuilds the PhysX scene and `root_physx_view` from scratch (the same parse described in [Section 4.3](#43-configuration-hierarchy), Step 5). Whatever opinion was authored on a prim during the stop window therefore becomes the new initial physics state once Fabric re-activates. Because the re-parse walks the whole stage, the prims themselves do not need to be deleted and re-spawned — only their attribute opinions are overwritten in place. This is the mechanism that allows a morphology update to reuse the existing prim tree (and the existing `Articulation` object) rather than incurring the full delete/spawn/rebuild sequence of [Section 4.7](#47-robot-morphology-update-during-training).

The following table enumerates the attributes that can be safely authored during the stop window and re-parsed on restart. All are read by PhysX at `force_load_physics_from_usd()` time; the schema definitions are catalogued in [Section 4.2.2](#422-physics-schemas).

| Modifiable Item | USD API Schema → Attribute | Affected Prim | Description |
|---|---|---|---|
| Link mass | `UsdPhysics.MassAPI` → `physics:mass` | Rigid body link | Overrides PhysX's density-derived mass. Read at parse; mirrors the runtime path `root_physx_view.set_masses()`. |
| Centre of mass | `UsdPhysics.MassAPI` → `physics:centerOfMass` | Rigid body link | Local-frame CoM offset of the link. |
| Inertia tensor | `UsdPhysics.MassAPI` → `physics:diagonalInertia`, `physics:principalAxes` | Rigid body link | Principal moments and their orientation; recomputed when link geometry changes (see `_get_box_inertia`, [`usd_generator.py:694`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). |
| Drive stiffness / damping | `UsdPhysics.DriveAPI` → `drive:angular:physics:stiffness`, `drive:angular:physics:damping` | Joint | PD gains for `ImplicitActuator` joints (`Kp`, `Kd` of the force law in §4.2.2). Read by `extract_usd.py` via `DriveAPI(prim, "angular")`. |
| Drive target / max force | `UsdPhysics.DriveAPI` → `drive:angular:physics:targetPosition`, `physics:maxForce` | Joint | Equilibrium pose and torque ceiling of the drive. |
| Joint limits | `UsdPhysics.RevoluteJoint` → `physics:lowerLimit`, `physics:upperLimit` | Joint | Per-DOF range of motion. Read by `extract_usd.py` via `GetLowerLimitAttr()` / `GetUpperLimitAttr()`. |
| Joint attachment frame | `UsdPhysics.Joint` → `physics:localPos0`, `physics:localRot0`, `physics:localPos1`, `physics:localRot1` | Joint | Parent-side (`*0`) and child-side (`*1`) anchor frames. Moving `localPos0` shifts the entire sub-tree below the joint — this is how a longer parent link relocates its child joint (cf. `_update_joint_position`, [`usd_generator.py:729`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). |
| Primitive collider size | `UsdGeom.Cube` → `size`; `UsdGeom.Capsule` / `UsdGeom.Cylinder` → `height`, `radius` | Collision geometry | Intrinsic dimensions of an analytic collider. Editing these resizes the collision shape directly (cf. the box-size edit in `_update_link_length`, [`usd_generator.py:779`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). |
| Prim transform | `UsdGeom.Xformable` → `xformOp:translate`, `xformOp:scale`, `xformOp:orient` | Any geometry prim | Per-prim placement and scale. Non-uniform `xformOp:scale` is the standard encoding for a non-cubic box (cf. the origin recentring in `_update_link_length`, [`usd_generator.py:784`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). |
| Collision enable / approximation | `UsdPhysics.CollisionAPI`, `UsdPhysics.MeshCollisionAPI` → `physics:approximation` | Collision geometry | Toggles collision and selects the mesh approximation (e.g. `convexHull`, `none` for primitives). |

> Note. Every entry above carries a geometric or inertial opinion that PhysX consumes only at parse time, so it must be authored while stopped. Purely dynamic state that PhysX already exposes through the Tensor API (joint stiffness/damping, mass, inertia, friction, joint limits, armature) can alternatively be updated at runtime without a stop via `root_physx_view.set_*` — see the methods listed in [Section 4.6](#46-physics-parameter-modification) above. The stop-window path is mandatory only for the attributes that have no Tensor-API equivalent, namely collider geometry and joint attachment frames.

#### 4.6.5 Primitive-Shape Colliders and Efficient Size Updates

The cost of re-parsing a resized collider depends entirely on whether the collider is an analytic primitive or a cooked mesh. PhysX represents `UsdGeom` `Cube`, `Sphere`, `Capsule`, `Cylinder`, and `Cone` exactly, ensuring the resulting collision representations precisely map to these geometries, so no preprocessing is required. Mesh colliders, by contrast, must be cooked (the process that generates collision approximations from mesh data is commonly referred to as cooking) ([Omni Physics — Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html)). Cooking runs whenever a triangle mesh, convex hull, or convex decomposition is parsed, and cooking data for all simulated objects must be available before starting a simulation. Consequently, a primitive collider whose dimensions change between a stop and the next reset incurs essentially zero re-cook cost, whereas a mesh collider must be re-cooked (the result is cached in the stage, but the first parse of each new shape still pays the cost).

This distinction is what makes link-length updates expressible as in-place attribute edits. The four TRON1A scalable links (`hip_{L,R}_thigh_Link`, `knee_{L,R}_Link`) are authored as boxes (`_parse_scalable_links_from_urdf` reads `visual/geometry/box`, [`usd_generator.py:200`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)), so each link's length is a primitive dimension rather than baked mesh data. A box accepts non-uniform scale — PhysX multiplies the cube size by the world scale — whereas spheres and capsules do not (a non-uniform scale around the radius triggers a warning and PhysX falls back to the maximum-axis scale). The practical rule is therefore: lengthen a box link with a non-uniform `xformOp:scale` (or by editing `UsdGeom.Cube.size`), and lengthen a capsule link by editing its intrinsic `height` attribute — never by a non-uniform scale.

The in-place equivalent of the generator's `_update_link_length` ([`usd_generator.py:751`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)) — which today edits the URDF and re-converts to USD — authors the same four quantities directly onto the live prims during the stop window:

```python
from pxr import UsdGeom, UsdPhysics, Gf

env.sim.stop()  # Fabric deactivates; USD stage becomes authoritative
stage = omni.usd.get_context().get_stage()

s_thigh = scales["thigh_length_scale"]          # design vector from the generator
x, y, z0 = scalable_links["hip_R_thigh_Link"]["size"]   # base URDF box extents
z_new = z0 * s_thigh
density = link_densities["hip_R_thigh_Link"]
CHILD_OFFSET = 0.05                              # SCALABLE_LINK_CHILD_OFFSET, usd_generator.py:162

for i in range(num_envs):
    link_path = f"/World/envs/env_{i}/Robot/hip_R_thigh_Link"

    # 1) Resize the primitive box collider/visual along z (non-uniform scale is
    #    valid for a Cube; for a Capsule edit GetHeightAttr() instead).
    geom = UsdGeom.Xformable(stage.GetPrimAtPath(f"{link_path}/collisions"))
    geom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, float(s_thigh)))

    # 2) Recentre the box so its origin sits at -z_new/2 (mirrors usd_generator.py:784)
    geom.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -z_new / 2.0))

    # 3) Recompute mass + diagonal inertia from the new box volume
    #    (mirrors _get_box_mass / _get_box_inertia, usd_generator.py:690-702)
    m_new = density * x * y * z_new
    ixx = m_new * (y * y + z_new * z_new) / 12.0
    iyy = m_new * (x * x + z_new * z_new) / 12.0
    izz = m_new * (x * x + y * y) / 12.0
    mass_api = UsdPhysics.MassAPI.Get(stage, link_path)
    mass_api.GetMassAttr().Set(m_new)
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))

    # 4) Move the child joint's parent-side frame down by the new length
    #    (mirrors _update_joint_position, usd_generator.py:729-749)
    knee = UsdPhysics.Joint(stage.GetPrimAtPath(f"{link_path}/../knee_R_Joint"))
    knee.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -(z_new + CHILD_OFFSET)))

env.sim.reset()   # force_load_physics_from_usd re-reads geometry; new lengths take effect
```

The four edits correspond one-to-one to the URDF mutations performed by `_update_link_length` (box `size` → step 1; element origin → step 2; mass/inertia → step 3; child-joint origin → step 4). Because the boxes are primitives, step 1 is a pure attribute write with no cooking, and steps 2–4 are scalar attribute writes; the only unavoidable cost remaining is the whole-stage re-parse inside `sim.reset()`. If the converter is configured to emit primitive box colliders rather than the current `collider_type="convex_hull"` ([`usd_generator.py:501`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)), a single `UsdGeom.Cube` then serves visual, collision, and physics simultaneously, and no convex hull is ever re-cooked across generations.

#### 4.6.6 What Cannot Be Changed at Runtime

The following require a full stop/play cycle with a new USD file:

- Collision geometry (mesh vertices, convex hull shapes)
- Link lengths (URDF/USD geometry transforms)
- Joint attachment points (the physical connection between links)
- Number of DOFs (joint topology)
- Addition or removal of joints or links

---

### 4.7 Robot Morphology Update During Training

This section investigates how the robot's physical morphology can be updated during training.  The full configuration path from `train.py` to the simulation were investigated and five approaches were discovered.

#### 4.7.1 Configuration Trace: From `train.py` to Physics

```
train.py:121
  parse_env_cfg(task_name) → SFEnvCfg
    └── cfg.scene.robot = SOLEFOOT_IDENTIFIED_CFG  (set by task-specific config)
          └── spawn = UsdFileCfg(usd_path="...SF_TRON1A/SF_TRON1A.usd")

train.py:150
  gym.make(task_id, cfg=env_cfg)
    └── ManagerBasedEnv.__init__()
          └── InteractiveScene(cfg.scene)
                └── Articulation(cfg=robot_cfg)
                      └── AssetBase.__init__()  (asset_base.py:84)
                            └── cfg.spawn.func(prim_path, cfg.spawn, ...)
                                  └── spawn_from_usd()  (from_files.py:38)
                                        └── _spawn_from_usd_file()  (from_files.py:268)
                                              └── create_prim(prim_path, usd_path=...)
                                                    └── reference arc inserted into USD stage

train.py (inside gym.make)
  sim.reset()  (manager_based_env.py:173)
    └── PhysX parses USD stage:
          MassAPI → body masses, inertias
          CollisionAPI → collision mesh cooking (IRREVOCABLE)
          DriveAPI → joint stiffness, damping
          ArticulationRootAPI → articulation anchor
    └── _initialize_impl()  (articulation.py:1506)
          └── root_physx_view created  (~line 1547)
```

The intervention points are therefore:
- USD file content — defines all initial physics values; modify before `sim.reset()`
- `root_physx_view` tensor writes — modify runtime physics after `sim.reset()`; no geometry changes possible

#### 4.7.2 `MultiUsdFileCfg` at Startup

To start an anvironment with multiple USD files, `MultiUsdFileCfg` is used. A list of USD files is passed to `MultiUsdFileCfg` ([`wrappers_cfg.py:46`](IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers_cfg.py)). Each environment receives a different robot variant at initialisation time according to distribution configuration set. `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG` in `solefoot_identified_cfg.py:113` uses this approach with the three TRON1A hardware variants. It must be noted that the joint topology for each unique agent must be the same. The following code snippet is only for illustration.

```python
# solefoot_identified_cfg.py:113
SOLEFOOT_IDENTIFIED_MULTIUSD_CFG = ArticulationCfg(
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[usd_path_sf, usd_path_pf, usd_path_wf],
        random_choice=True,  # random_choice=False for deterministic assignment
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=True,
    ),
    ...
)
```

Using `MultiUsdFileCfg` at startup enable use of any geometry and full morphological diversity across environments at init time. However, the robot morphology is fixed after initialisation and cannot be update between generations without stop/play and all USD variants must share the same joint topology (identical DOF count and joint names, since `ArticulationView` indexes joints by position). It must be noted that using `MultiUsdFileCfg` requires `replicate_physics=False` and this entails serious init overhead (required for different physics per env). This leads to 15+ minutes initialisation time for ≥4000 environments ([IsaacLab #4434](https://github.com/isaac-sim/IsaacLab/issues/4434)). Optimisation strategies will need to be investigated here.

#### 4.7.3 Stop → Delete → Spawn → Play

Robot morphological parameter updates require explicitly stopping the simulation, deletion of all robot prims, spawning of new prims from the new USD files, and a simulation restart. This is the only approach capable of changing collision geometry and link lengths during a training run.

For co-optimisation with `m` design candidates and `k` rollout copies per design, the total environment count is `num_envs = m × k`. `spawn_multi_usd_file` assigns design `i` to environments `[i*k … i*k+k-1]` via round-robin (`index % m`), batching all `m×k` prim copies into a single `Sdf.ChangeBlock()` transaction (`wrappers.py:106`), which is the efficiency equivalent of running `m` cloners simultaneously.

After the swap, three categories of manager state require attention — `InteractiveScene._articulations`, the action manager, and the event manager. Observation, reward, and termination managers are entirely dynamic (`env.scene[name]` called per step) and require no update.

##### Asset Reference Map

| Manager / Term | Attribute cached at `__init__` | File:Line | Action required |
|---|---|---|---|
| `ActionTerm` (all subclasses) | `self._asset` | `action_manager.py:55` | Re-bind to new articulation |
| `JointPositionAction` | `self._offset` (if `use_default_offset=True`) | `joint_actions.py:195` | Re-read from `new_articulation.data.default_joint_pos` |
| `JointAction` | `self._joint_ids`, `self._num_joints` | `joint_actions.py:65–68` | No update — same joint topology means same indices |
| `randomize_rigid_body_material` | `self.asset` | `events.py:197` | Re-bind to new articulation |
| `randomize_rigid_body_mass` | `self.asset` | `events.py:320` | Re-bind to new articulation |
| `randomize_joint_parameters` | `self.asset` | `events.py:685` | Re-bind to new articulation |
| All MDP observation functions | `env.scene[name]` per call | `observations.py` | No update — dynamic |
| All MDP reward functions | `env.scene[name]` per call | `rewards.py` | No update — dynamic |
| All MDP termination functions | `env.scene[name]` per call | `terminations.py` | No update — dynamic |
| Function-based event terms | `env.scene[name]` per call | `events.py` | No update — dynamic |

Sensor behaviour: sensors stored in `scene._sensors` (e.g. `ContactSensor`) are `AssetBase` subclasses with their own prim paths. When `delete_prim(robot_path)` fires, `_on_prim_deletion` in `asset_base.py:331–347` matches the parent path and calls `_clear_callbacks()` — deregistering the PLAY subscription. When `sim.reset()` then fires the PLAY event, the sensor no longer responds and its `root_physx_view` is never rebuilt. To fix this, `sensor._register_callbacks()` must be called after spawning new prims but before `sim.reset()`, so that the sensor re-subscribes and reinitialises when PLAY fires.

##### Full Respawn Sequence

```python
def respawn_robots(env: ManagerBasedRLEnv, new_usd_paths: list[str]):
    """
    Generation-boundary respawn for m design candidates × k rollout copies.
    new_usd_paths: list of m USD file paths (one per design candidate).
    Assumes same joint topology across all USD variants (same joint names and count).
    """
    m = len(new_usd_paths)

    # ── Step 1: Stop simulation ───────────────────────────────────────────────
    # Fires STOP event → AssetBase._invalidate_initialize_callback on all assets
    # → sets _is_initialized=False on current Articulation and all sensors.
    # source: asset_base.py:324–326
    env.sim.stop()

    # ── Step 2: Delete all robot prims ────────────────────────────────────────
    # _on_prim_deletion fires for each deleted root path:
    #   - Clears old Articulation callbacks (asset_base.py:331–347)
    #   - Matches parent path of child sensor prims → clears sensor PLAY subscriptions
    for i in range(env.scene.cfg.num_envs):
        sim_utils.delete_prim(f"/World/envs/env_{i}/Robot")

    # ── Step 3: Spawn m×k new prims in one Sdf.ChangeBlock transaction ────────
    # spawn_multi_usd_file → spawn_multi_asset:
    #   1. Builds m template prims under /World/Template/Asset_XXXX (wrappers.py:88)
    #   2. Sdf.ChangeBlock() + Sdf.CopySpec() copies design (i // k % m) to env_i
    #      (wrappers.py:106–116) — single atomic USD edit for all m×k envs
    #   3. Deletes /World/Template after (wrappers.py:119)
    #   4. Sets carb flag /isaaclab/spawn/multi_assets=True (wrappers.py:125)
    new_spawn_cfg = sim_utils.MultiUsdFileCfg(
        usd_path=new_usd_paths,     # m paths; round-robin across m×k envs
        random_choice=False,
        rigid_props=env.scene["robot"].cfg.spawn.rigid_props,
        articulation_props=env.scene["robot"].cfg.spawn.articulation_props,
        activate_contact_sensors=True,
    )
    sim_utils.spawn_multi_usd_file(
        prim_path="/World/envs/env_.*/Robot",   # regex resolved to all num_envs paths
        cfg=new_spawn_cfg,
        replicate_physics=False,                # required: heterogeneous envs
    )
    # source: wrappers.py:131–200; wrappers_cfg.py:46–66

    # ── Step 4: Re-register sensor callbacks BEFORE sim.reset() ──────────────
    # _clear_callbacks() in step 2 deregistered sensor PLAY subscriptions.
    # New robot prims now exist in the stage; calling _register_callbacks() here
    # re-subscribes each sensor so _initialize_callback fires when PLAY is raised.
    # source: asset_base.py:280–302 (_register_callbacks), 349–360 (_clear_callbacks)
    for sensor in env.scene._sensors.values():
        sensor._register_callbacks()

    # ── Step 5: Reset simulation ──────────────────────────────────────────────
    # Calls force_load_physics_from_usd() → PhysX re-parses all m×k new prims.
    # Fires PLAY event:
    #   - New Articulation's _initialize_callback → root_physx_view rebuilt
    #   - Sensors' _initialize_callback → sensor views rebuilt (step 4 enabled this)
    # source: simulation_context.py:637–642 (reset_async), 512–531 (reset)
    env.sim.reset()

    # ── Step 6: Create new Articulation wrapping all m×k envs ─────────────────
    # Single ArticulationView batches all m×k prims — valid because same joint topology.
    # PLAY already fired in step 5, so _initialize_impl() runs immediately here.
    # source: asset_base.py:304–322 (_initialize_callback)
    new_robot_cfg = env.scene["robot"].cfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=new_spawn_cfg,
    )
    new_articulation = Articulation(cfg=new_robot_cfg)

    # ── Step 7: Patch InteractiveScene._articulations ─────────────────────────
    # scene.reset(), write_data_to_sim(), and update() all iterate
    # _articulations.values() — must point to new object.
    # source: interactive_scene.py:125, 447–448, 464–465, 482–483
    env.scene._articulations["robot"] = new_articulation

    # ── Step 8: Re-bind ActionTerm._asset (all action terms) ──────────────────
    # ActionTerm.__init__ caches self._asset at construction (action_manager.py:55).
    # apply_actions() calls self._asset.set_joint_position_target() — stale after swap.
    # _joint_ids and _num_joints are unchanged (same topology) and need no update.
    # source: joint_actions.py:65–68, 199
    for term in env.action_manager._terms.values():
        term._asset = new_articulation
        # JointPositionAction: if use_default_offset=True, _offset was set from old
        # asset's default_joint_pos at init. Re-read from new articulation in case
        # link length / actuator position changes alter the equilibrium pose.
        # source: joint_actions.py:194–195
        if hasattr(term, "_offset") and getattr(term.cfg, "use_default_offset", False):
            term._offset = new_articulation.data.default_joint_pos[
                :, term._joint_ids
            ].clone()

    # ── Step 9: Re-bind class-based EventManager terms ────────────────────────
    # Class-based ManagerTermBase subclasses cache self.asset at __init__.
    # Function-based event terms call env.scene[name] per invocation — skip those.
    # source: events.py:197 (randomize_rigid_body_material)
    # source: events.py:320 (randomize_rigid_body_mass)
    # source: events.py:685 (randomize_joint_parameters)
    for mode_terms in env.event_manager._terms.values():
        for term in mode_terms:
            if hasattr(term, "asset"):          # class-based ManagerTermBase term
                term.asset = new_articulation

    # ── Step 10: Clear episode state and sync to PhysX ────────────────────────
    env.reset()
```


Critical Considerations to Note:
- `InteractiveScene._articulations` (`interactive_scene.py:125`) is a plain `dict` with no public add/remove API — must be patched directly.
- Sensor callbacks must be re-registered before `sim.reset()`, not after — PLAY fires inside `reset()` and sensors must be subscribed before it fires.
- `replicate_physics=False` is required in `InteractiveSceneCfg` for heterogeneous envs; the carb flag `/isaaclab/spawn/multi_assets` (set by `spawn_multi_asset` at `wrappers.py:125`) triggers a warning if `replicate_physics=True` is detected (`interactive_scene.py:225`).
- All m USD variants must share the same joint topology. `ActionTerm._joint_ids` and `_num_joints` are not updated — if joint names or counts differ, `load_managers()` (`manager_based_env.py:294`) must be called instead, which rebuilds all managers from scratch.
- Complete Deletion and Respawn Sequence is computationally expensive

The aforementioned re-spawn sequence allows for any geometry change and complete morphological freedom with any number of design candidates per generation. However, the computational overhead of simulating heterogeneous robots must be taken into account. Seconds of overhead may be expected per generation dominated by `force_load_physics_from_usd()` (`simulation_context.py:640`), which scales with `m×k` prim count and USD mesh complexity.

#### 4.7.4 Procedural In-Memory USD Generation

Generate USD content programmatically at runtime using `Usd.Stage.CreateInMemory()`. No pre-existing USD files are required; the stage is constructed entirely from Python. This approach provides maximum flexibility for the EA — any topology, geometry, or physics configuration can be generated on the fly. However, the computational overhead of procedural in momory USD generation is significant, making the system unviable for immediate use. The method has been documented though for future references.

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf

def create_robot_usd_in_memory(copt_params: dict) -> str:
    """Generate a robot USD stage from coptlogical parameters."""
    stage = Usd.Stage.CreateInMemory()

    # Define root xform
    root = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # Define links with parameterised geometry
    for i, (name, length, mass) in enumerate(copt_params["links"]):
        link = UsdGeom.Cylinder.Define(stage, f"/Robot/{name}")
        link.GetHeightAttr().Set(length)

        # Apply mass
        mass_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass_api.GetMassAttr().Set(mass)

        # Apply collision
        UsdPhysics.CollisionAPI.Apply(link.GetPrim())

    # Define joints with parameterised drive properties
    for joint_name, stiffness, damping in copt_params["joints"]:
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{joint_name}")
        drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive_api.GetStiffnessAttr().Set(stiffness)
        drive_api.GetDampingAttr().Set(damping)

    # Export to temporary file
    tmp_path = f"/tmp/ea_gen_{uuid.uuid4()}.usd"
    stage.Export(tmp_path)
    return tmp_path
```

The generated USD file path can then be passed to `UsdFileCfg` or `MultiUsdFileCfg` for spawning. This USD update method provides maximum flexibility allowing any topology, geometry, and physics to be generated from EA parameters. However, this method is the most complex to implement with the same stop/play overhead as for reloading and requires careful prim hierarchy design to match joint topology constraints.

#### 4.7.5 Primitive Shape Geometry In-Place Update

The respawn pathway of Section 4.7.3, `respawn_robots` ([`respawn.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/respawn.py)), pays the full `stop → delete → spawn → play` cost every generation. The entire robot prim tree is deleted, re-spawned from new USD files, and the whole stage is re-parsed and re-cooked by `force_load_physics_from_usd()` ([Section 4.3](#43-configuration-hierarchy), Step 5). For a population that varies in only link lengths, almost all of the aforementioned steps are redundant. Because the parent URDF (`base_robot.urdf`) models every structural link with primitive box geometry, and the URDF to USD conversion preserves those primitives as analytic `UsdGeomCube` prims rather than baking them into cooked meshes (Section 4.1.1), a link's size is carried entirely in editable USD attributes. This makes an in-place articulation update possible. Thus, a new morphology can be applied by overwriting those attributes on the existing prims during the stop window ([Section 4.6](#46-physics-parameter-modification)) and triggering a single `sim.reset()`, with no prim deleted, re-spawned, or re-cooked. This sub-section describes such a method, to be added to a new file [`update.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/update.py). The method resizes a set of named links and relocates their child joints, so that the rescaled leg remains a kinematically feasible linkage rather than a set of overlapping or detached segments.

The URDF importer authors every link as a flat sibling under the robot root, `<robot>/<link_name>` ([`UrdfImporter.cpp:340, 349-350`](IsaacSim/source/extensions/isaacsim.asset.importer.urdf/plugins/isaacsim.asset.importer.urdf/UrdfImporter.cpp)) within the prim tree, and gives each link prim an identity transform stack ending in `xformOp:scale = (1, 1, 1)` (`UrdfImporter.cpp:355-361`), denoting scale with respect to an identity geometrical shape. The box dimensions and the URDF `<origin>` offset are carried on the according to the `visuals`/`collisions` child Xform wrappers beneath the link (`UrdfImporter.cpp:255-266, 443, 534`). Joint prims live in a separate `<robot>/joints` scope and bind their two links through `Body0`/`Body1` relationships with explicitly authored `localPos0`/`localPos1` anchor frames (`UrdfImporter.cpp:1056-1069, 1474`). As a result, a link's length is changed by editing the geometry wrapper's scale and centred-origin translate, so the geometry resize is independent of the joint frames. Furthermore, because the joints are separate prims outside the link's subtree, resizing a link does not move its joints and a scaled link's child joint must be relocated explicitly, otherwise the next segment stays attached at the old length and the leg is no longer a valid linkage.

The offline generator handles this in `_update_link_length` ([`usd_generator.py:751-801`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). Lengthening the thigh to `z_new = z0 · s_thigh` and recentring its origin to `-z_new/2` would otherwise leave the knee joint at the original height, so the shank would attach partway up the thigh. The generator therefore moves the scaled link's child joint to the new segment end (step 4) — `joint_z = -(z_new + CHILD_OFFSET)`, where `CHILD_OFFSET = 0.05m` is the fixed gap between the link's far edge and its child joint (`usd_generator.py:799-801, 162`). The in-place method reproduces the same four-step edit catalogued in [Section 4.6.5](#465-primitive-shape-colliders-and-efficient-size-updates) — (1) resize the box, (2) recentre its origin, (3) recompute mass and inertia, (4) move the child joint — but authors it directly onto the live prims of every environment.

`_update_link_length` looks the child joint up through a hand-maintained name map (`SCALABLE_LINK_CHILD_JOINTS = {hip_R_thigh_Link: knee_R_Joint, knee_R_Link: ankle_R_actuator_Joint, …}`, [`usd_generator.py:166-171`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). The USD stage already encodes the same relationship explicitly, so the method recovers it from the joint space rather than hard-coding names: every joint under the `<robot>/joints` scope carries a `physics:body0` relationship to its parent link and `physics:body1` to its child (`UrdfImporter.cpp:1056-1061`). A scaled link's child joint is therefore the joint whose `Body0` targets that link; iterating the joints scope and matching `Body0` returns it without any external map and stays correct if joints are renamed or added.

Run while the simulation is stopped, the method resolves the (optionally wildcard) robot prim path with `find_matching_prim_paths` and, for each matched articulation and each `(link_name, scale)` pair, reads the link's current box length from its `collisions` wrapper, resizes both geometry wrappers and recentres their origin, scales the authored mass, then scans the robot's `joints` scope for the joint whose `Body0` targets the link and lowers its `localPos0` to the new segment end. Because the link prim itself is left at identity scale, the joint anchor is authored as an exact absolute value, with no risk of the link scale being applied to it a second time. The edits take effect on the next `sim.reset()`.

```python
"""In-place primitive-geometry updates for articulation links."""

from __future__ import annotations

from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.envs import ManagerBasedRLEnv


def _scale_op(xformable: UsdGeom.Xformable) -> UsdGeom.XformOp:
    return next(op for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale)


def update_articulation_links(
    prim_path: str,
    link_extents: list[tuple[str, list[float]]],
    stage=None,
) -> None:
    """Author ABSOLUTE box extents (idempotent) onto matching robot prims.

    Must be called while the simulation is stopped (Fabric deactivated).

    link_extents: list of (link_name, [x_abs, y_abs, z_abs]) — target box dims (m).
    """
    stage = stage or get_current_stage()
    for robot_path in sim_utils.find_matching_prim_paths(prim_path):
        joints = stage.GetPrimAtPath(f"{robot_path}/joints")
        for link_name, extents in link_extents:
            x_new, y_new, z_new = (float(v) for v in extents)
            link_path = f"{robot_path}/{link_name}"

            coll_xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"{link_path}/collisions"))
            x0, y0, z0 = _scale_op(coll_xf).Get()          # current (prev-gen) extents

            for purpose in ("collisions", "visuals"):
                w = UsdGeom.Xformable(stage.GetPrimAtPath(f"{link_path}/{purpose}"))
                _scale_op(w).Set(Gf.Vec3d(x_new, y_new, z_new))    # SET absolute
                for op in w.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        t = op.Get()
                        op.Set(Gf.Vec3d(t[0], t[1], -z_new / 2.0))

            mass_api = UsdPhysics.MassAPI.Get(stage, link_path)
            density = mass_api.GetMassAttr().Get() / (x0 * y0 * z0)    # invariant
            m_new = density * x_new * y_new * z_new
            mass_api.GetMassAttr().Set(m_new)
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(
                m_new * (y_new * y_new + z_new * z_new) / 12.0,
                m_new * (x_new * x_new + z_new * z_new) / 12.0,
                m_new * (x_new * x_new + y_new * y_new) / 12.0,
            ))

            for joint_prim in joints.GetChildren():
                joint = UsdPhysics.Joint(joint_prim)
                targets = joint.GetBody0Rel().GetTargets()
                if targets and str(targets[0]) == link_path:
                    p0 = joint.GetLocalPos0Attr().Get()
                    offset = abs(p0[2]) - z0                     # recover constant gap
                    joint.GetLocalPos0Attr().Set(Gf.Vec3d(p0[0], p0[1], -(z_new + offset)))
                    break
```

The method takes the robot articulation's prim path, which may be a regular expression such as `/World/envs/env_.*/Robot`, so one call updates every matching environment, together with `link_extents`, a list of `(link_name, [x_abs, y_abs, z_abs])` tuples expressing the absolute target box dimensions in metres. For each matched robot it reads the currently authored extents `(x0, y0, z0)` from the link's `collisions` wrapper scale — used only to recover the invariant density and the constant child-joint attachment gap — then sets (not multiplies) both the `collisions` and `visuals` wrappers' scale to the requested absolute values and resets their translate so the box stays centred at `-z_new/2`. It then recomputes the link's `MassAPI` mass and diagonal inertia from the new extents. It recovers the constant density from the original mass and box volume, sets `m_new = density · x_new · y_new · z_new`, and writes the solid-box diagonal inertia `(m·(y²+z²)/12, m·(x²+z²)/12, m·(x²+y²)/12)` — the same formulae used by `_get_box_mass`/`_get_box_inertia` in the offline generator ([`usd_generator.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). It then walks the robot's `joints` scope, identifies the scaled link's child joint by the `Body0` relationship, recovers the constant attachment gap as `offset = |localPos0.z| − z0`, and lowers the joint's `localPos0` to `-(z_new + offset)`, moving the knee to the end of the lengthened thigh (or the ankle to the end of the lengthened shank) exactly as `_update_joint_position` does in the offline generator. Because absolute extents are set rather than multiplied, re-applying the same generation is a no-op and successive generations overwrite rather than compound; the cumulative-tracking caveat of earlier drafts does not apply. After the loop, a single `sim.reset()` re-parses the stage and the rebuilt leg is both geometrically and kinematically consistent. The full per-environment round-robin wrapper `apply_link_length_params` ([`update.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/update.py)) batches all edits in a `Sdf.ChangeBlock` and logs the `sim.reset()` time.

#### 4.7.6 Instantiation and Prototypes

The in-place method of [Section 4.7.5](#475-primitive-shape-geometry-in-place-update) was written against a flattened model of the stage wherein  `<robot>/<link>/collisions` carries an editable `xformOp:scale` that can be set per environment. However, such a model does not apply to IsaacLab as the urdf converter sets assets to be instanceable for memory optimisation. When the converter emits an instanceable asset and the multi-USD spawner loads a heterogeneous population, the geometry the method in Section 4.7.5 tries to edit no longer lives within the prim at `<robot>/<link>/collisions`, and the per-environment write it attempts is rejected by OpenUSD. Thus, scenegraph instancing and the prototypes it creates needs to be understand and investigates how the URDF converter and the multi-asset spawner produce them in the context of IsaacLab, and derives the corrected update logic.

Scenegraph instancing is OpenUSD's mechanism for sharing the read-only scene description of repeated content across many call-sites without duplicating it in memory. A composition arc (a reference or internal reference) whose source prim is tagged `instanceable = true` becomes an instance. USD recognises that all instances composed from the same source with the same arcs are interchangeable and represents their shared sub-tree exactly once as an implicit prototype prim, conventionally named `/__Prototype_<N>` ([OpenUSD: Scenegraph Instancing](https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html)). Prototypes are generated and owned by `UsdStage`. They do not exist in any layer's scene description, cannot be named in advance, and cannot be edited. Neither the prototype prims themselves nor any descendant beneath an instance may carry an authored override, thus making a large parallel scene tractable on the GPU. PhysX/Fabric cook and upload the geometry of a prototype once and reuse it for every instance, so memory and cooking cost scale with the number of distinct assets, not with the number of environments.

The co-optimisation scene is the worst case for naïve duplication and the best case for instancing. `num_envs = 4096` parallel robots are drawn from a population of `num_individuals` distinct designs (the captured run used `256` [`copt_on_policy_runner.py:79`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)). The converter marks every link's geometry instanceable, so each individual contributes one prototype per geometry prim. We have 35 geometry prims for the biped robot under development (18 visual + 17 collision; `limx_imu` has no collider). In 4096 environments, we have 8960 prototypes backing 143360 instances (`8960 = 256 × 35`, `143360 = 4096 × 35`), each prototype serving `4096 / 256 = 16` instances. Editing a prototype therefore edits 16 robots at once — which is the leverage the corrected method exploits, and the constraint it must respect.

As described in Section 4.1, the URDF importer maps each link to a flat prim `<robot>/<link>` and lowers a `<box>` to a unit `UsdGeomCube` (`size = 1.0`) whose parent `mesh_0` Xform carries `xformOp:scale = (size_x, size_y, size_z)` in double precision (`UrdfImporter.cpp:200-214` `addBox`, `255-266` `addMesh`/`AddScaleOp(PrecisionDouble)`); a `<cylinder>` becomes a native `UsdGeomCylinder` with `mesh_0` scale `(1,1,1)` (`UrdfImporter.cpp:216-227`). What Section 4.1 glossed over is the instancing wrapper the importer wraps around that geometry. The importer first defines two top-level scopes, `/visuals` and `/colliders` (`UrdfImporter.cpp:1481-1482`), and authors each link's actual geometry there — at `/visuals/<link>/mesh_0` and `/colliders/<link>/mesh_0`. The per-link `<robot>/<link>/visuals` and `<robot>/<link>/collisions` prims that appear in the tree are then made internal references to those scopes and tagged instanceable:
```text
meshes_base.GetPrim().GetReferences().AddInternalReference(SdfPath(source_name));  # UrdfImporter.cpp:530, 568
meshes_base.GetPrim().SetInstanceable(true);                                       # UrdfImporter.cpp:531, 569
#   visuals:    source_name = "/visuals/"   + link.name     (UrdfImporter.cpp:445)
#   collisions: source_name = "/colliders/" + link.name     (UrdfImporter.cpp:535)
```
So the editable box dimension is not on the `<link>/collisions` wrapper that Section 4.7.5 reads. It is the `xformOp:scale` on `mesh_0` inside the referenced scope, and the wrapper itself is an instanceable internal-reference with no transform ops of its own.

Because `<link>/visuals` and `<link>/collisions` are instanceable, the composed stage hides their contents behind a prototype. Consequently, visuals and collisions are separate prototypes with `/visuals/<link>` and `/colliders/<link>` as different sources, so a single link contributes two instanced geometry prototypes, and a scalable link must be updated in both. Furthermore, the `/__Prototype_<N>` names are assigned by `UsdStage` in internal creation order and follow no derivable pattern from the link name. The only stable, derivable handles are (a) the instance path `/World/envs/env_{i}/Robot/<link>/{visuals|collisions}`, which is fully deterministic, and (b) the source path `/visuals/<link>/mesh_0` or `/colliders/<link>/mesh_0` inside the individual's layer. The prototype that connects them may be resolved at runtime with `GetPrototype()`.

The multi-USD spawner round-robins the population into the scene by copying, into each `/World/envs/env_{i}/Robot`, a proto prim that merely references `biped_{i mod N}.usd`. This produces two distinct sharing mechanisms that the update logic must treat differently:
- Geometry scale is shared by instancing. The instanceable `visuals`/`collisions` wrappers compose `/visuals|/colliders/<link>`; all 16 environments of an individual collapse onto one prototype. The geometry cannot be edited through any instance, but it can be edited at its source `mesh_0`, and the change propagates to the prototype and hence to all 16 instances.
- Mass, inertia, and joint frames are shared by referencing. `MassAPI` lives directly on the link prim `<robot>/<link>` and the joint anchors on `<robot>/joints/<joint>` — neither is instanceable. Each environment obtains them by reference composition from `biped_k.usd`; the 16 environments of individual `k` share that one layer, so an opinion authored on `biped_k.usd` recomposes into all 16.

Both mechanisms therefore converge on the same edit target, the individual's source layer, and both commit on the next `sim.reset()`, when `force_load_physics_from_usd()` re-parses the recomposed stage ([Section 4.6.4](#464-usd-attribute-modification-during-stop)). A single edit per individual, authored on `biped_k.usd` through the `Sdf` layer API, fans out to 16 environments: geometry via the prototype, dynamics via the reference. This is the mechanism that lets the corrected method touch the first `num_individuals` environments only rather than all 4096.

Against the real structure, the Section 4.7.5 worker fails in two ways:
- `_scale_op` searches `<link>/collisions` for a `TypeScale` op, but that wrapper is an instanceable internal-reference and carries no transform ops. The generator in `_scale_op` returns no scale op and `next(...)` raises `StopIteration`. The scale it wants is one level down, on `mesh_0`, inside the prototype.
- Authoring `xformOp:scale` on `<link>/collisions/mesh_0` through an environment path is forbidden, because that prim is a descendant of an instance proxy and "you can't edit any prim beneath the instanceable-marked prim" ([Omniverse: Instancing](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent/modularity-guide/instancing.html)) as USD ignores or rejects the opinion. By contrast the method's mass/inertia/joint writes are legal as those prims are not instanced but are authored per-environment as strong local opinions on all 4096 composed prims, which is `O(num_envs)` work that the reference mechanism makes unnecessary.

Thus, the USD update logic for IsaacLab implementation must be changed to accommodate aforementioned findings. `apply_link_length_params` must iterate over the first `num_individuals` environments (round-robin guarantees this set covers every design exactly once, and the spawner uses the identical `index mod N` map, [`wrappers.py:114`](IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers.py)) and for each individual scalable link it must resolve the two geometry prototypes from the deterministic instance paths via `GetPrototype()` and hand each to `update_articulation_links`. `update_articulation_links` must no longer iterate over links. Given a single prototype it needs to resolve the strongest authored spec of the prototype's `mesh_0` with `UsdPrim.GetPrimStack()`, which points at `/visuals|/colliders/<link>/mesh_0` in the live `biped_k.usd` layer, not the read-only `/__Prototype_<N>` path. This needs to then set the existing `double3` `xformOp:scale` through the `Sdf` layer API and author the same link's `MassAPI` mass/inertia and its child joint's `localPos0` on that same source layer. Because all edits are absolute and authored on the source layer, re-applying a generation is idempotent, the population's 16-way fan-out is automatic on `sim.reset()`.

---

### 4.8 Recommended Strategy: Hybrid Two-Tier Co-optimisation

The methods for robot morphology update in §4.7 reveal a clean architectural split for design and policy co-optimisation: within-generation policy optimisation (learning policy parameters while the simulation runs for `N` iterations) pairs naturally with between-generation morphology updates (reloading geometry between EA cycles every `N` iteration). The following sub-sections describe the proposed architecture.

#### 4.8.1 Architecture Overview

```
EA Generation k
│
├── Generate population USDs
│     generate_population_usds(ea_population[k]) → /tmp/gen_k/variant_{0..N-1}.usd
│
├── Update Morphology
│     env.sim.stop()
│     Update Articulation through either of the methods discussed above
│     env.sim.reset()
│
├── Train N PPO iterations
│     for it in range(K):
│         rollout → alg.update()
│         collect per-env fitness from cur_reward_sum[new_ids]
│
└── EA Update
      fitness_per_individual = aggregate_fitness(env_id_to_individual_map)
      ea_population[k+1] = ea.evolve(ea_population[k], fitness_per_individual)
```

It must be noted that domain randomisation to be applied during PPO training must be adjusted accordingly to ensure that the optimised robot morphology and design paramters are utilised for learning gait generation.

The key design decisions to be noted are:
1. `random_choice=False`: ensures `env_i` always receives `variant_i.usd`, enabling clean fitness-to-individual mapping. With `random_choice=True`, the environment-to-individual assignment would be non-deterministic.
2. Joint topology constraint: all USD variants in the list must declare the same joints with the same names. Actuator gains, mass, inertia, link lengths, and collision geometry can vary freely; the joint graph cannot.
3. Population size vs. environment count: the number of USD variants equals `num_envs`. If the EA population is smaller than `num_envs`, assign multiple environments per individual and average their fitness scores.
4. `replicate_physics=False` overhead mitigation: the 15+ minute init time for large environment counts is avoided by keeping population size ≤ 512 per generation. Pure geometry differences (link lengths, collision shapes) always require `replicate_physics=False` and a correspondingly smaller population or pre-generation strategy.

#### 4.8.2  Implementation Requirements

This section describes the requirement set and design documentation for the implemented Robot Design and Policy Co-optimisation. The Design optimiser, the EA is hooked immediately after `alg.update()` in the runner's `learn()` loop. A custom `CoptOnPolicyRunner` (following the guidelines in §3) overrides `learn()` to inject generational logic as follows:

```python
# runners/copt_on_policy_runner.py
class CoptOnPolicyRunner(OnPolicyRunner):

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # ... standard initialisation ...
        episode_fitnesses = torch.zeros(self.env.num_envs, device=self.device)
        env_to_individual = self._assign_individuals_to_envs()  # env_i → individual_i

        for it in range(start_iter, tot_iter):
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    cur_reward_sum += rewards
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    # Accumulate fitness per individual
                    if new_ids.numel() > 0:
                        for env_id in new_ids[:, 0]:
                            ind_id = env_to_individual[env_id]
                            episode_fitnesses[ind_id] = cur_reward_sum[env_id].item()
                    cur_reward_sum[new_ids] = 0
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            # ← EA update every ea_update_interval PPO iterations
            if (it + 1) % self.ea_update_interval == 0:
                fitness_per_ind = episode_fitnesses.cpu().numpy()
                new_population = self.ea.evolve(self.population, fitness_per_ind)
                self.population = new_population
                self._reload_morphology(new_population)
                episode_fitnesses.zero_()
                env_to_individual = self._assign_individuals_to_envs()

    def _reload_morphology(self, population):
        """Stop simulation, generate new USDs, reload, restart."""
        # 1. Respawn in case Stop -> Delete -> Respawn -> Play method is used
        respawn_robots(self.env, polulation.get_usd_paths())
        # If in place updates are used corresponding methods for the update method are to be used here
        self.generation += 1
```

#### 4.8.3 Generation of USD Files from EA Parameters

For geometry-varying morphology, a minimal USD generation function using the OpenUSD Python API:

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import tempfile, os

def generate_robot_usd(individual: dict, output_path: str) -> str:
    """Generate a TRON1A-topology robot USD from EA parameters.

    individual keys: link_lengths, link_masses, joint_stiffness, joint_damping
    All joint names must match the original topology for ArticulationView compatibility.
    """
    # Load the base USD as a template (preserves joint topology and collision geometry)
    base_stage = Usd.Stage.Open(BASE_USD_PATH)

    # Override MassAPI values for each link
    for link_name, mass in zip(LINK_NAMES, individual["link_masses"]):
        prim = base_stage.GetPrimAtPath(f"/Robot/{link_name}")
        mass_api = UsdPhysics.MassAPI.Get(prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(float(mass))

    # Override DriveAPI values for each joint
    for joint_name, kp, kd in zip(JOINT_NAMES, individual["joint_stiffness"], individual["joint_damping"]):
        prim = base_stage.GetPrimAtPath(f"/Robot/{joint_name}")
        drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive_api.GetStiffnessAttr().Set(float(kp))
        drive_api.GetDampingAttr().Set(float(kd))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_stage.Export(output_path)
    return output_path
```

Since current objective is to integrate a placeholder usd generator that generates usd files with random design parameters, we need not implement more features in the cooptimsation runner.

### 4.9 DesignGeneratorBase API

The respawn pathway of [Section 4.7.3](#473-stop--delete--spawn--play), driven by `respawn_robots` ([`respawn.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/respawn.py)), pays the full `stop → delete → spawn → play` cost — as much as 200 s for the configured population — on every generation. With a design update scheduled every 10 PPO iterations (≈3 s each), this cost is untenable. [Section 4.7.5](#475-primitive-shape-geometry-in-place-update) established that, because every structural link of `base_robot.urdf` is primitive box geometry, a new morphology can be authored directly onto the live prims during the stop window and committed with a single `sim.reset()`, with no prim deleted, re-spawned, or re-cooked. This section specifies the implementation of that in-place route as a second morphology-update pathway, `CoptOnPolicyRunner._update_morphology`. The full respawn pathway `CoptOnPolicyRunner._reload_morphology` is still required to load the first population (and any late-start random designs) through the complete respawn sequence.

The implementation stores all link-scaling logic in `DesignGeneratorBase` which shared by its children `RandomDesignGenerator` and `CMAESDesignGenerator`, yielding one standard API across the three classes. The following four factors are taken into account:
1. Actuator parameters are applied after `sim.reset()`. A stopped articulation re-runs `_initialize_impl()` on the next play as the STOP event sets `_is_initialized = False` and the subsequent PLAY re-initialises ([`asset_base.py:294, 311, 322, 326`](IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py)) — which rebuilds the `IdentifiedActuator` tensors from cfg. A patch written before the reset is therefore overwritten, so `apply_actuator_params` runs last, mirroring the order already used in `_reload_morphology`.
2. Link lengths are authored as absolute box extents relative to the base URDF. The first `_reload_morphology` call ([`copt_on_policy_runner.py:112`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)) is left unchanged, so every environment starts at the base geometry; each later generation is authored as an absolute target computed from the cached base sizes (`self.scalable_links`). Updates overwrite rather than compound, so the cumulative-tracking caveat of Section 4.7.5 no longer applies.
3. All link-scaling machinery moves to `DesignGeneratorBase`. `RandomDesignGenerator`'s additional non-length mutations (link-mass scale, actuator-cylinder geometry, joint limits) are retained and applied after the link-length edits through an override hook.
4. `sim.reset()` time is logged inside `apply_link_length_params`. Independent `sim.reset()` makes latency logging easier.

Because the in-place pathway never destroys the `Articulation` Python object — `scene._articulations["robot"]` is never swapped — all cached asset references in the action, event, and command managers, the `ActionTerm._offset` (joint topology is unchanged), and the sensor callbacks remain valid. This is why `_update_morphology` omits the manager-rebinding and sensor-re-registration steps that `respawn_robots` performs (Section 4.7.3).

#### 4.9.1 `class DesignGeneratorBase`

`DesignGeneratorBase`, currently a pure ABC, becomes the home of every link-scaling attribute and method. It gains a concrete `__init__` that caches the base box sizes and densities once, the shared population pipeline, the absolute-extent computation, the shared URDF/USD authoring, and the absolute link-extent application that supersedes both `RandomDesignGenerator._scale_joint_z_origin` and `CMAESDesignGenerator._update_link_length`.

A module-level constant maps each scalable link to the sampled scale that drives its length:
```python
# Which sampled scale drives each scalable link's length.
SCALABLE_LINK_LENGTH_SCALE: dict[str, str] = {
    "hip_R_thigh_Link": "thigh_length_scale",
    "hip_L_thigh_Link": "thigh_length_scale",
    "knee_R_Link":      "shank_length_scale",
    "knee_L_Link":      "shank_length_scale",
}
```

The `Population` ABC exposes an accessor so every population exposes its absolute link extents to the in-place pathway:
```python
@abstractmethod
def get_link_length_params(self) -> list[dict[str, dict[str, float]]]:
    """Return absolute link-extent dicts, one per individual."""
    ...
```

The base class implements all shared machinery:
```python
class DesignGeneratorBase(ABC):

    def __init__(
        self,
        base_urdf_path: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
    ) -> None:
        self.base_urdf_path = base_urdf_path
        self.num_individuals = num_individuals
        self.output_dir = output_dir
        self.param_ranges: dict[str, tuple[float, float]] = {**DEFAULT_PARAM_RANGES}
        if param_ranges is not None:
            self.param_ranges.update(param_ranges)
        # Cached base box sizes + densities (used to emit ABSOLUTE extents).
        self.scalable_links, self.link_densities = _parse_scalable_links_from_urdf(base_urdf_path)
        # Transient per-individual state during the generate_population loop.
        self._current_root: ET.Element | None = None
        self._current_scales: dict[str, float] | None = None

    # ---- abstract design hooks ------------------------------------------------
    @abstractmethod
    def _generate_individual(
        self, generation: int, idx: int
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
        """Return (link_length_params, actuator_params) for one individual."""
        ...

    @abstractmethod
    def generate_population(self, generation: int) -> Population: ...

    def update_with_fitness(self, fitness: list[float]) -> None:
        pass

    def sample_batch(self) -> None:           # fixed: was missing self
        pass

    # ---- shared population pipeline -------------------------------------------
    def _build_population(self, generation: int, indices) -> "RandomPopulation":
        usd_files: list[str] = []
        actuator_params: list[dict[str, dict]] = []
        link_length_params: list[dict[str, dict[str, float]]] = []
        for idx in indices:
            llp, act = self._generate_individual(generation, idx)
            urdf_path = self._generate_individual_urdf(generation, idx, llp)
            usd_path = self._generate_individual_usd(urdf_path, idx)
            usd_files.append(usd_path)
            actuator_params.append(act)
            link_length_params.append(llp)
        return RandomPopulation(usd_files, actuator_params, link_length_params)

    # ---- scale sampling + absolute-extent computation -------------------------
    def _sample_scales(self, rng: np.random.Generator) -> dict[str, float]:
        return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.param_ranges.items()}

    def _compute_link_extents(self, scales: dict[str, float]) -> dict[str, dict[str, float]]:
        """Absolute box extents (metres) = cached base size × sampled length scale."""
        extents: dict[str, dict[str, float]] = {}
        for link, scale_key in SCALABLE_LINK_LENGTH_SCALE.items():
            x, y, z0 = self.scalable_links[link]["size"]
            extents[link] = {"x": x, "y": y, "z": z0 * scales.get(scale_key, 1.0)}
        return extents

    # ---- URDF authoring (shared) ----------------------------------------------
    def _generate_individual_urdf(
        self, generation: int, idx: int, link_length_params: dict[str, dict[str, float]]
    ) -> str:
        tree = ET.parse(self.base_urdf_path)
        self._current_root = tree.getroot()
        for link_name, ext in link_length_params.items():           # link lengths FIRST
            self._apply_link_extents(link_name, ext["x"], ext["y"], ext["z"])
        self._apply_extra_urdf_mutations()                          # subclass extras AFTER
        self._current_root = None

        gen_dir = os.path.join(self.output_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        urdf_path = os.path.join(gen_dir, f"individual_{idx:04d}.urdf")
        tree.write(urdf_path, xml_declaration=True, encoding="utf-8")
        return urdf_path

    def _apply_extra_urdf_mutations(self) -> None:
        """Hook: extra non-length URDF edits. Base is a no-op."""
        pass

    def _generate_individual_usd(self, urdf_path: str, idx: int) -> str:
        usd_out_dir = os.path.join(os.path.dirname(urdf_path), f"individual_{idx:04d}_usd")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path, usd_dir=usd_out_dir, usd_file_name=f"biped_{idx}.usd",
            link_density=0.0, merge_fixed_joints=False, fix_base=False, self_collision=False,
            collider_type="convex_hull", joint_drive=None, force_usd_conversion=True,
        )
        return UrdfConverter(cfg).usd_path

    # ---- absolute link-extent application (supersedes _scale_joint_z_origin
    #      and _update_link_length) ---------------------------------------------
    def _apply_link_extents(self, link_name: str, x: float, y: float, z: float) -> None:
        if link_name not in self.scalable_links:
            raise ValueError(f"{link_name!r} is not a scalable link.")
        density = self.link_densities[link_name]
        child_joint = self.scalable_links[link_name]["child_joint"]
        new_origin_z = -z / 2.0
        link = self._find_link(link_name)

        for el_tag in ("visual", "collision"):                       # 1) box size (absolute)
            el = link.find(el_tag)
            if el is None:
                continue
            box = el.find("geometry/box")
            if box is None:
                raise ValueError(f"{link_name}/{el_tag} has no <box> geometry.")
            box.set("size", f"{x} {y} {z}")

        for el_tag in ("visual", "collision", "inertial"):           # 2) recentre origin
            el = link.find(el_tag)
            if el is None:
                continue
            origin = el.find("origin")
            if origin is None:
                continue
            ox, oy, _ = (float(v) for v in origin.get("xyz", "0 0 0").split())
            origin.set("xyz", f"{ox} {oy} {new_origin_z}")

        m_new = self._get_box_mass(density, x, y, z)                 # 3) mass + inertia
        self._update_inertial(link_name, m_new, self._get_box_inertia(m_new, x, y, z))

        self._update_joint_position(                                 # 4) child joint
            link_name, child_joint, -(z + SCALABLE_LINK_CHILD_OFFSET)
        )

    # ---- helpers moved verbatim from CMAESDesignGenerator ---------------------
    #   _get_box_mass, _get_box_inertia, _update_inertial,
    #   _update_joint_position, _find_link, _find_joint   (unchanged bodies)
```

`_get_box_mass`, `_get_box_inertia`, `_update_inertial`, `_update_joint_position`, `_find_link`, and `_find_joint` move up from `CMAESDesignGenerator` with unchanged bodies ([`usd_generator.py:690-821`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)).

#### 4.9.2 `class RandomDesignGenerator`

`RandomDesignGenerator` is reduced to the random-search specifics: a thin `__init__`, a `_generate_individual` that returns parameters only, the actuator-parameter builder, and the override hook that re-applies its non-length URDF mutations after the link-length edits. Its length handling now flows through the shared `_apply_link_extents`, so the thigh/shank boxes are resized (and their masses recomputed) rather than only their child joints being nudged.

```python
class RandomDesignGenerator(DesignGeneratorBase):
    def __init__(self, base_urdf_path, num_individuals, param_ranges=None, output_dir="/tmp/copt_usds"):
        super().__init__(base_urdf_path, num_individuals, param_ranges, output_dir)

    def generate_population(self, generation: int) -> Population:
        return self._build_population(generation, range(self.num_individuals))

    def _generate_individual(self, generation, idx):
        rng = np.random.default_rng(seed=generation * 10000 + idx)
        scales = self._sample_scales(rng)
        self._current_scales = scales
        return self._compute_link_extents(scales), self._build_actuator_params(scales)

    def _build_actuator_params(self, scales):
        act_params: dict[str, dict] = {}
        for group, baseline in ACTUATOR_BASELINES.items():
            act_params[group] = {
                "effort_limit":      baseline["effort_limit"]      * scales["joint_effort_scale"],
                "velocity_limit":    baseline["velocity_limit"]    * scales["velocity_limit_scale"],
                "saturation_effort": baseline["saturation_effort"] * scales["saturation_effort_scale"],
                "armature":          baseline["armature"]          * scales["armature_scale"],
                "friction_static":   baseline["friction_static"]   * scales["friction_static_scale"],
                "friction_dynamic":  baseline["friction_dynamic"]  * scales["friction_dynamic_scale"],
                "stiffness":         baseline["stiffness"]         * scales[f"{group}_stiffness_scale"],
                "damping":           baseline["damping"]           * scales[f"{group}_damping_scale"],
            }
        return act_params

    def _apply_extra_urdf_mutations(self):
        scales, root = self._current_scales, self._current_root
        s_mass = scales["link_mass_scale"]                                   # C: mass & inertia
        for link in root.iter("link"):
            if IMU_LINK_NAME in link.get("name", ""):
                continue
            for mass_el in link.iter("mass"):
                mass_el.set("value", str(float(mass_el.get("value", 0.0)) * s_mass))
            for inertia_el in link.iter("inertia"):
                for attr in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                    val = inertia_el.get(attr)
                    if val is not None:
                        inertia_el.set(attr, str(float(val) * s_mass))
        s_r, s_l = scales["actuator_radius_scale"], scales["actuator_length_scale"]   # D: cylinders
        for link in root.iter("link"):
            if any(al in link.get("name", "") for al in ABAD_HIP_LINKS):
                for cyl in link.iter("cylinder"):
                    cyl.set("radius", str(float(cyl.get("radius", 0.0)) * s_r))
                    cyl.set("length", str(float(cyl.get("length", 0.0)) * s_l))
        s_eff, s_vel = scales["joint_effort_scale"], scales["velocity_limit_scale"]   # E: joint limits
        for joint in root.iter("joint"):
            grp = JOINT_TO_ACTUATOR.get(joint.get("name", ""))
            if grp is None:
                continue
            for limit_el in joint.iter("limit"):
                limit_el.set("effort",   str(ACTUATOR_BASELINES[grp]["effort_limit"]   * s_eff))
                limit_el.set("velocity", str(ACTUATOR_BASELINES[grp]["velocity_limit"] * s_vel))
```

Removed from `RandomDesignGenerator`: `_scale_joint_z_origin`, `_convert_urdf_to_usd` (now the base `_generate_individual_usd`), the old monolithic `_generate_individual` body, and the two-argument `RandomPopulation(...)` construction.

#### 4.9.3 `class CMAESDesignGenerator`

`CMAESDesignGenerator` keeps its CMA-ES state machine but sheds the duplicated link-scaling machinery now provided by the base class. Its `_generate_individual` returns absolute extents and an empty actuator dict, and `generate_population` reuses the shared pipeline while retaining the `ask`/`tell` solution bookkeeping.

```python
        super().__init__(base_urdf_path, num_individuals, param_ranges, output_dir)
        self.param_ranges = {**CMAES_PARAM_RANGES}
        if param_ranges is not None:
            for key, rng in param_ranges.items():
                if key in self.param_ranges:
                    self.param_ranges[key] = rng
        # ... unchanged CMA-ES state (param_keys, _es, _pending/_last_solutions, late_start) ...

    def _generate_individual(self, generation, idx):
        if self.late_start:
            rng = np.random.default_rng(seed=generation * 10000 + idx)
            scales = self._sample_scales(rng)
        else:
            scales = self._denormalise(self._pending_solutions[idx])
        self._current_scales = scales
        return self._compute_link_extents(scales), {}      # actuator overrides empty (as today)

    def generate_population(self, generation: int) -> Population:
        if self._terminated:
            return None
        assert self._pending_solutions is not None, "generate_population called before sample_batch"
        pop = self._build_population(generation, range(len(self._pending_solutions)))
        self._last_solutions = self._pending_solutions
        self._pending_solutions = None
        return pop
```

Removed from `CMAESDesignGenerator`: `_update_link_length`, `_get_box_mass`, `_get_box_inertia`, `_update_inertial`, `_update_joint_position`, `_find_link`, `_find_joint`, the `_parse_scalable_links_from_urdf` call in `__init__`, and the redundant `_sample_scales` override. `update_with_fitness`, `sample_batch`, `_denormalise`, `_sanitise_cost`, and `save_state` are unchanged.

#### 4.9.4 `class CoptOnPolicyRunner`

`CoptOnPolicyRunner` gains the in-place pathway `_update_morphology` next to the existing `_reload_morphology`. It stops the simulation, advances the design distribution with the last population's fitness, generates the next population, authors the new link lengths in place (the timed `sim.reset()` happens inside `apply_link_length_params`), resets the environments, then re-applies actuator parameters and clears the fitness accumulators.

```python
from co_optimisation.utils.update import apply_link_length_params   # NEW import

    def _update_morphology(self) -> None:
        """In-place EA cycle: no delete/respawn, no manager re-binding."""
        unwrapped_env = self.env.unwrapped
        sim = unwrapped_env.sim

        if sim.is_playing():                                   # 1. stop (Fabric off)
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        fitness = self._compute_individual_fitness()           # 2. update distribution
        print("Updating Design Population (in-place)")
        self._design_generator.update_with_fitness(fitness)

        self.current_population = self._design_generator.generate_population(self.generation)  # 3.
        self.generation += 1

        print("Applying link length parameters in place")      # 4. author + sim.reset() (inside)
        apply_link_length_params(
            unwrapped_env, self.current_population.get_link_length_params()
        )

        unwrapped_env.reset()                                  # 5. reset episodes

        print("Applying Sampled Actuator Parameters")          # 6. actuators AFTER reset
        apply_actuator_params(
            unwrapped_env, self.current_population.get_actuator_params()
        )

        self._individual_fitness.zero_()                       # 7. zero accumulators
        self._individual_episode_counts.zero_()
```

The initial spawn in `learn()` stays `_reload_morphology` ([`copt_on_policy_runner.py:112`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)); only the in-loop EA-update block ([`copt_on_policy_runner.py:253-254`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)) switches to the in-place route:
```python
                with torch.inference_mode():
                    self._update_morphology()        # was self._reload_morphology()
                obs = self.env.get_observations().to(self.device)
```

##### The Committed `_update_morphology`

The method as it now stands in the source ([`copt_on_policy_runner.py:587-633`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)) departs from the sketch above in six respects. The following are the differences:
1. The first departure is the signature, which takes the absolute iteration `it`. The method serves both entry points, the initial spawn passing zero ([`copt_on_policy_runner.py:246`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)) and the in-loop evolutionary update passing `it + 1` ([`copt_on_policy_runner.py:393`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)), which supersedes the arrangement described immediately above, since the first population is now authored in place rather than respawned. The argument exists because the late-start boundary must be tested inside the method, and the schedule that gates the in-loop call is itself now evaluated against the absolute iteration rather than the resume-relative `(it - start_iter + 1)` of the original ([`copt_on_policy_runner.py:289-292`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)), so a run resumed at iteration twelve thousand enters the evolutionary phase at once instead of repeating the random phase from its start. On a fresh run `start_iter` is zero and the two forms agree.

2. The second departure is the ordering, which closes the off-by-one fitness evaluation the sketch would otherwise have carried. `update_with_fitness` now precedes `generate_population`, and it performs the `es.tell` against `_last_solutions`, the genotypes that earned the fitness just accumulated instead of their successor ([`usd_generator.py:751-764`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). `generate_population` then materialises that pending batch into individuals and rotates it into `_last_solutions` ([`usd_generator.py:766-778`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)), leaving the generator holding precisely the genotypes whose fitness the next visit will report. That rotation is why the order is significant, for calling `generate_population` first would overwrite `_last_solutions` with the fresh batch and rank it by the preceding generation's returns.

3. The third departure is the late-start guard, which separates three cases the single branch of the sketch could not. Below the `ea_late_start` threshold the generator is still drawing random designs, no fitness is reported, and only the learning rate is reinitialised. On the first visit at or above the threshold, `_copt_started` being false, the method asks the opening CMA-ES batch through `sample_batch` and reports nothing, since `_last_solutions` is still empty and `update_with_fitness` asserts against exactly that state ([`usd_generator.py:754-756`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). Every later visit takes the reporting branch. The flag is carried in the checkpoint ([`copt_on_policy_runner.py:161, 208`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py)), so a resumed run does not ask a second opening batch over the one it already holds.

4. The fourth departure is the reinitialisation of the learning rate on both branches, `reinit_learning_rate` restoring the configured value from `init_learning_rate` ([`copt_ppo.py:347-348`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/algorithms/copt_ppo.py)). The adaptive KL schedule drives the rate down across a generation, and were it left alone the rate reached at the close of one morphology would be inherited by the next, which is a different optimisation problem and merits the configured rate afresh.

5. The fifth departure is the guard against a null population, `generate_population` returning `None` once the CMA-ES instance declares a stop ([`usd_generator.py:766-771`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py)). Every step that follows, the authoring of the extents, the reset, the actuator application, and the zeroing of the accumulators, is therefore conditional, so a terminated search leaves the standing morphology untouched and the loop degenerates into ordinary PPO fine-tuning of the parent policy on the selected design.

6. The sixth departure is slight in code and material in effect. The reset is bracketed by `_suppress_terrain_curriculum`, a flag the delayed terrain curriculum consults before promoting or demoting and which makes it return the mean terrain level unaltered ([`curriculums.py:470-477`](tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/curriculums.py)). The episodes are truncated by the morphology swap rather than concluded, so the distance walked says nothing about the policy's competence on the terrain, and admitting it would demote every environment at once at every generation boundary.

```python
    def _update_morphology(self, it: int) -> None:
        """In-place EA cycle: no delete/respawn, no manager re-binding."""
        unwrapped_env = self.env.unwrapped
        sim = unwrapped_env.sim

        if sim.is_playing():  # 1. stop (Fabric off)
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        if self._ea_late_start <= it:
            if not self._copt_started:
                self.alg.reinit_learning_rate()
                self._design_generator.sample_batch()
                self._copt_started = True
            else:
                fitness = self._compute_individual_fitness()
                print("Updating Design Population")
                self._design_generator.update_with_fitness(fitness)
        else:
            self.alg.reinit_learning_rate()
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
        self.generation += 1
        if self.current_population is not None:
            print(
                "Applying link length parameters in place"
            )  # 4. author + sim.reset() (inside)
            apply_link_length_params(
                unwrapped_env, self.current_population.get_link_length_params()
            )

            # 5. reset episodes — suppress terrain promotion/demotion because the
            # episodes are truncated by the morphology swap and the walked-distance
            # signal is meaningless at this boundary.
            unwrapped_env._suppress_terrain_curriculum = True
            unwrapped_env.reset()
            unwrapped_env._suppress_terrain_curriculum = False

            print("Applying Sampled Actuator Parameters")  # 6. actuators AFTER reset
            apply_actuator_params(
                unwrapped_env, self.current_population.get_actuator_params()
            )

            self._individual_fitness.zero_()  # 7. zero accumulators
            self._individual_episode_counts.zero_()
```

#### 4.9.5 Additional Major Changes

`RandomPopulation` carries the third payload and exposes it:

```python
class RandomPopulation(Population):
    def __init__(self, usd_files, actuator_params, link_length_params):
        self._usd_files = usd_files
        self._actuator_params = actuator_params
        self._link_length_params = link_length_params

    def get_usd_files(self): return self._usd_files
    def get_actuator_params(self): return self._actuator_params
    def get_link_length_params(self): return self._link_length_params
```

A new file `utils/update.py` provides `update_articulation_links` (the per-prototype, absolute-set in-place worker) and `apply_link_length_params` (per-individual prototype authoring with a single timed `sim.reset()`). As established in [Section 4.7.6](#476-instantiation-and-prototypes), the URDF importer marks each link's geometry instanceable, so the box size that earlier drafts tried to set on the `<link>/collisions` wrapper is not editable through any environment prim. Instead, the editable transform ops live on the prototype's source `mesh_0` inside the per-individual USD layer. `update_articulation_links` therefore takes one geometry prototype path, resolves the strongest authored spec of its `mesh_0` with `UsdPrim.GetPropertyStack()` (which points at the live source layer, never the read-only `/__Prototype_<N>` path), and sets the existing `double3` `xformOp:scale` through the `Sdf` layer API.

A subtlety governs the physics quantities: the importer emits visuals, collisions, and physics into separate sub-stages, so the geometry meshes and the rigid-body data generally live in different layers. Assuming the link's `MassAPI` shares the geometry layer raises `AttributeError: 'NoneType' object has no attribute 'default'`, because `GetAttributeAtPath` returns `None` for a spec absent from that layer. The worker therefore resolves the link's `physics:mass`/`physics:diagonalInertia` and the child joint's `physics:localPos0` each from its own authoring layer via that prim's `GetPropertyStack()`, given a representative environment's composed link prim (`link_prim_path`). The link prim and joints scope are not instanced; they are shared across the design's environments by the reference to the design's USD, so editing each once propagates to every instance by reference composition, exactly as editing the prototype propagates the geometry by instancing. All current values are read back as invariants (density from `mass/(x0·y0·z0)`, attachment gap from `|localPos0.z| − z0`) so absolute targets are idempotent. Thus, re-applying a generation is a no-op and successive generations overwrite rather than compound. `apply_link_length_params` drives this by iterating only the first `num_individuals` environments, resolving each scalable link's two geometry prototypes from the deterministic instance paths via `GetPrototype()`, and a single timed `sim.reset()` commits every edit.

```python
"""In-place primitive-geometry updates for *instanced* articulation links.

The URDF importer authors each link's geometry as an instanceable internal
reference: ``<robot>/<link>/{visuals,collisions}`` are instance prims whose box
lives at ``/visuals/<link>/mesh_0`` and ``/colliders/<link>/mesh_0`` (a unit
``UsdGeomCube`` scaled by ``mesh_0``'s ``xformOp:scale``) inside the per-individual
USD layer (Section 4.7.6).  Instanced geometry cannot be overridden through an
environment prim, so the box size is set on the prototype's *source* ``mesh_0`` via
the ``Sdf`` layer API.  Mass/inertia (link prim) and the child-joint anchor (joints
scope) are NOT instanced; they reach each environment by the *reference* to the
design's USD and -- importantly -- generally live in a DIFFERENT layer than the
geometry meshes, so each is resolved from its own authoring layer via the prim's
property stack.  Editing one design's prototype/source updates all of its instances
on the next ``sim.reset()``.
"""
from __future__ import annotations

import time

from pxr import Gf, Sdf, Usd

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage

# Child joint relocated when a scalable link's length changes (its Body0 == link).
SCALABLE_LINK_CHILD_JOINTS = {
    "hip_R_thigh_Link": "knee_R_Joint",
    "hip_L_thigh_Link": "knee_L_Joint",
    "knee_R_Link":      "ankle_R_actuator_Joint",
    "knee_L_Link":      "ankle_L_actuator_Joint",
}
GEOM_MESH = "mesh_0"


def _source_spec(attr):
    """(layer, Sdf.Path) of the strongest authored opinion for *attr* (or None)."""
    if not attr:
        return None, None
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    return (stack[0].layer, stack[0].path) if stack else (None, None)


def update_articulation_links(prototype_path, link_prim_path, extents, stage=None) -> None:
    """Author ABSOLUTE box extents for ONE geometry prototype on its source layer.

    Sets the prototype ``mesh_0``'s scale/translate and — idempotently — the owning
    link's mass/inertia and the child joint's ``localPos0``.  Each quantity is
    authored on whichever layer actually defines it (resolved per-attribute), so
    geometry and physics may live in separate layers.  ``link_prim_path`` is a
    representative environment's composed (non-instanced) link prim.  Must run while
    the simulation is stopped.
    """
    stage = stage or get_current_stage()
    x_new, y_new, z_new = (float(v) for v in extents)
    link_name = link_prim_path.rsplit("/", 1)[1]
    robot_path = link_prim_path.rsplit("/", 1)[0]
    child_joint = SCALABLE_LINK_CHILD_JOINTS[link_name]

    # geometry source (instanced): via the prototype's own mesh_0
    mesh = stage.GetPrimAtPath(prototype_path).GetChild(GEOM_MESH)
    if not mesh:
        return
    g_layer, g_scale = _source_spec(mesh.GetAttribute("xformOp:scale"))
    if g_layer is None or g_scale is None:
        return
    g_trans = g_scale.GetPrimPath().AppendProperty("xformOp:translate")
    x0, y0, z0 = mesh.GetAttribute("xformOp:scale").Get()
    t0 = mesh.GetAttribute("xformOp:translate").Get()

    # physics sources (reference-shared, NOT instanced): each on its OWN layer
    link_prim = stage.GetPrimAtPath(link_prim_path)
    joint_prim = stage.GetPrimAtPath(f"{robot_path}/joints/{child_joint}")
    m_layer, m_path = _source_spec(link_prim.GetAttribute("physics:mass"))
    i_layer, i_path = _source_spec(link_prim.GetAttribute("physics:diagonalInertia"))
    j_layer, j_path = _source_spec(joint_prim.GetAttribute("physics:localPos0"))

    mass0 = link_prim.GetAttribute("physics:mass").Get()
    p0 = joint_prim.GetAttribute("physics:localPos0").Get()
    density = mass0 / (x0 * y0 * z0)
    offset = abs(p0[2]) - z0
    m_new = density * x_new * y_new * z_new

    with Sdf.ChangeBlock():
        # (1) geometry — prototype source; the only legal resize of instanced geometry
        g_layer.GetAttributeAtPath(g_scale).default = Gf.Vec3d(x_new, y_new, z_new)
        if g_layer.GetAttributeAtPath(g_trans) is not None:
            g_layer.GetAttributeAtPath(g_trans).default = Gf.Vec3d(t0[0], t0[1], -z_new / 2.0)
        # (2) mass + solid-box inertia — link prim, reference-shared (idempotent)
        if m_layer is not None and m_path is not None:
            m_layer.GetAttributeAtPath(m_path).default = m_new
        if i_layer is not None and i_path is not None:
            i_layer.GetAttributeAtPath(i_path).default = Gf.Vec3f(
                m_new * (y_new * y_new + z_new * z_new) / 12.0,
                m_new * (x_new * x_new + z_new * z_new) / 12.0,
                m_new * (x_new * x_new + y_new * y_new) / 12.0)
        # (3) child-joint anchor — joints scope, reference-shared (idempotent)
        if j_layer is not None and j_path is not None:
            j_layer.GetAttributeAtPath(j_path).default = Gf.Vec3f(p0[0], p0[1], -(z_new + offset))


def apply_link_length_params(env: ManagerBasedRLEnv, link_length_params_list) -> None:
    """Author absolute link extents on every design's source layer, then one timed
    sim.reset().  Iterating the first ``num_individuals`` environments visits each
    design once (round-robin), and editing a prototype updates all its instances."""
    sim, scene = env.sim, env.scene
    num_individuals = len(link_length_params_list)
    env_paths = scene.env_prim_paths

    if sim.is_playing():                                            # safety net
        sim._disable_app_control_on_stop_handle = True
        sim.stop()
        sim._disable_app_control_on_stop_handle = False

    stage = get_current_stage()
    for individual in range(num_individuals):                      # first N envs = each design once
        env_path = env_paths[individual]
        params = link_length_params_list[individual]
        for link_name, d in params.items():
            ext = [d["x"], d["y"], d["z"]]
            link_prim_path = f"{env_path}/Robot/{link_name}"
            for purpose in ("visuals", "collisions"):
                inst = stage.GetPrimAtPath(f"{link_prim_path}/{purpose}")
                proto = inst.GetPrototype() if inst else None
                if proto:
                    update_articulation_links(proto.GetPath().pathString, link_prim_path, ext, stage)

    print("Reactivating Physics (in-place prototype update)")
    start = time.perf_counter()
    sim.reset()
    print(f"Total time taken for sim.reset(): {time.perf_counter() - start:.4f} seconds")
```

`utils/__init__.py` re-exports the helpers:

```python
from co_optimisation.utils.update import update_articulation_links, apply_link_length_params
```

#### 4.9.6 Architecture Conclusion

This sub-section describes the complete architecture implemented across §4.9.1–4.9.5, with diagrams of the class and function hierarchies, and an end-to-end trace of both morphology-update pathways. It serves as a reference for understanding how the pieces compose and why the ordering and layering choices were made.

##### Class and Population Hierarchy

The implementation introduces one concrete population type and a three-level generator class tree:

```
ABC
├── Population (abstract: get_usd_files, get_actuator_params, get_link_length_params)
│   └── RandomPopulation ──────────────────────── concrete payload
│       ├── _usd_files        : list[str]           ← for _reload_morphology (respawn)
│       ├── _actuator_params  : list[dict[str,dict]] ← for apply_actuator_params
│       └── _link_length_params: list[dict[str,      ← for _update_morphology (in-place)
│                                        dict[str,float]]]
│
└── DesignGeneratorBase (ABC + concrete machinery)   [usd_generator.py]
    │   __init__                 cache scalable_links / link_densities from base URDF
    │   _build_population        shared pipeline: individual → urdf → usd → RandomPopulation
    │   _sample_scales           uniform draw from self.param_ranges
    │   _compute_link_extents    absolute extents = base_size × scale (for each SCALABLE link)
    │   _generate_individual_urdf parse base URDF, apply extents, hook, write file
    │   _apply_extra_urdf_mutations  ← hook, no-op in base
    │   _generate_individual_usd    UrdfConverter → .usd file
    │   _apply_link_extents      set box size, recentre origin, recompute mass/inertia, move child joint
    │   _get_box_mass / _get_box_inertia / _update_inertial  ← physics helpers
    │   _update_joint_position / _find_link / _find_joint    ← URDF element helpers
    │
    └── RandomDesignGenerator                        [usd_generator.py]
        │   __init__             super().__init__(...)  — no extra state
        │   generate_population  → self._build_population(generation, range(N))
        │   _generate_individual → (_compute_link_extents(scales), _build_actuator_params(scales))
        │   _build_actuator_params  scale all IdentifiedActuator attrs from ACTUATOR_BASELINES
        │   _apply_extra_urdf_mutations  ← override: C) mass scale, D) cylinder scale, E) joint limits
        │
        └── CMAESDesignGenerator                     [usd_generator.py]
                __init__         super().__init__(); restrict param_ranges to CMAES_PARAM_RANGES;
                │                initialise CMA-ES (_es), _pending_solutions, _last_solutions
                sample_batch     _pending_solutions = _es.ask()
                update_with_fitness  _es.tell(solutions, costs); sample_batch()
                generate_population  → self._build_population(...) + CMA-ES bookkeeping
                _generate_individual → (_compute_link_extents(scales), {})
                                       scales from _denormalise(_pending_solutions[idx])
                                       or _sample_scales (late_start mode)
                _denormalise     map unit-hypercube CMA-ES solution → physical scale dict
                _sanitise_cost   guard non-finite fitness values
                save_state       pickle CMA-ES state for resume
```

`RandomPopulation.get_link_length_params()` returns a list of dicts — one per individual — where each dict maps link name to `{"x": …, "y": …, "z": …}` absolute box dimensions (metres):

```
link_length_params[individual_idx] = {
    "hip_R_thigh_Link": {"x": 0.05,  "y": 0.032, "z": 0.2750},  # thigh at scale 1.10
    "hip_L_thigh_Link": {"x": 0.05,  "y": 0.032, "z": 0.2750},
    "knee_R_Link":      {"x": 0.025, "y": 0.032, "z": 0.3120},  # shank at scale 1.04
    "knee_L_Link":      {"x": 0.025, "y": 0.032, "z": 0.3120},
}
```

The `x` and `y` dimensions are always equal to the base URDF values (`_compute_link_extents` keeps them fixed); only `z` varies with the sampled scale.

##### `generate_population()` Call Tree

The two generators share the entire URDF/USD pipeline through `_build_population`. Their only difference is in how `_generate_individual` sources the scale vector and whether `_apply_extra_urdf_mutations` is a no-op or performs additional edits.

`RandomDesignGenerator.generate_population(generation)`

```
generate_population(generation)
└── _build_population(generation, range(N))          [base]
    └── for idx in 0..N-1:
        ├── _generate_individual(generation, idx)     [RandomDesignGenerator]
        │   ├── np.random.default_rng(generation*10000+idx)
        │   ├── _sample_scales(rng)                   [base] ← all DEFAULT_PARAM_RANGES keys
        │   ├── self._current_scales = scales
        │   ├── _compute_link_extents(scales)         [base] ← absolute x,y,z per scalable link
        │   └── _build_actuator_params(scales)        [RandomDesignGenerator]
        │
        ├── _generate_individual_urdf(gen,idx,llp)   [base]
        │   ├── ET.parse(base_urdf_path)              ← always starts from the base URDF
        │   ├── self._current_root = root
        │   ├── for link_name, ext in llp.items():
        │   │   └── _apply_link_extents(link,x,y,z)  [base]
        │   │       ├── _find_link(link_name)
        │   │       ├── box.set("size", f"{x} {y} {z}")
        │   │       ├── origin.set("xyz", f"ox oy {-z/2}")  [visual, collision, inertial]
        │   │       ├── _get_box_mass(density, x, y, z)
        │   │       ├── _get_box_inertia(m, x, y, z)
        │   │       ├── _update_inertial(link, m, moi)
        │   │       └── _update_joint_position(link, child_joint, -(z+0.05))
        │   │           └── _find_joint(child_joint)
        │   ├── _apply_extra_urdf_mutations()         [RandomDesignGenerator override]
        │   │   ├── C) ×s_mass  on ALL link masses and inertias  (skips limx_imu)
        │   │   ├── D) ×s_r, ×s_l  on cylinder radius/length    (abad/hip links only)
        │   │   └── E) set <limit effort/velocity> from ACTUATOR_BASELINES×scale  (per joint)
        │   ├── self._current_root = None
        │   └── tree.write(gen_dir/individual_idx.urdf)
        │
        └── _generate_individual_usd(urdf_path, idx) [base]
            └── UrdfConverter(UrdfConverterCfg(...)).usd_path
                → gen_dir/individual_idx_usd/biped_idx.usd
    └── RandomPopulation(usd_files, actuator_params, link_length_params)
```

`CMAESDesignGenerator.generate_population(generation)`

```
generate_population(generation)
├── assert _pending_solutions is not None            ← set by prior sample_batch()
├── _build_population(generation, range(len(solutions))) [base]
│   └── for idx in 0..N-1:
│       ├── _generate_individual(generation, idx)    [CMAESDesignGenerator]
│       │   ├── if late_start: _sample_scales(rng)
│       │   │   else: _denormalise(_pending_solutions[idx])
│       │   │         ← map [0,1]^2 → {"thigh_length_scale": v, "shank_length_scale": v}
│       │   ├── self._current_scales = scales
│       │   └── _compute_link_extents(scales), {}    ← empty actuator overrides
│       │
│       ├── _generate_individual_urdf(gen,idx,llp)  [base]  ← identical to Random path
│       │   ├── _apply_link_extents(×4 links)        [base]
│       │   └── _apply_extra_urdf_mutations()        [base - no-op]
│       │
│       └── _generate_individual_usd(urdf_path,idx) [base]  ← identical to Random path
│
├── self._last_solutions = _pending_solutions
├── self._pending_solutions = None
└── RandomPopulation(usd_files, [{},...], link_length_params)
```

The critical difference: `CMAESDesignGenerator` restricts `param_ranges` to `{"thigh_length_scale", "shank_length_scale"}` (set in `__init__` after `super().__init__()`), so `_compute_link_extents` only varies the z dimension of thigh and shank links. It does not override `_apply_extra_urdf_mutations`, so the no-op base hook is used — no mass, cylinder, or joint-limit mutations.

##### Dual Morphology-Update Pathway in `CoptOnPolicyRunner`

`learn()` uses two morphology-update methods. They differ in cost (200 s vs. single `sim.reset()`), what changes on the stage (entire prim tree vs. box attributes), and what manager state requires rebuilding:

```
CoptOnPolicyRunner.learn()
│
├── [iteration 0, initial spawn]
│   └── _reload_morphology()                         [full respawn, ~200 s]
│       ├── design_generator.sample_batch()          ← CMA-ES ask()
│       ├── design_generator.generate_population()   ← URDF+USD for every individual
│       ├── respawn_robots(env, usd_files)           [respawn.py]
│       │   ├── sim.stop()
│       │   ├── delete_prim(env_path/Robot)          ← destroy old articulation prims
│       │   ├── spawn_multi_usd_file(...)            ← re-create from new USD files
│       │   ├── sensor._register_callbacks()         ← re-subscribe PLAY events
│       │   ├── Articulation(new_robot_cfg)          ← new Python object
│       │   ├── sim.reset()                          ← PLAY fires _initialize_impl()
│       │   ├── scene._articulations["robot"] = new  ← swap Python reference
│       │   ├── term._asset = new_articulation       ← rebind action terms
│       │   ├── event_manager term rebinding
│       │   ├── command_manager term rebinding
│       │   └── env.reset()
│       └── apply_actuator_params(env, actuator_params)
│
└── [every ea_update_interval iterations thereafter]
    └── _update_morphology()                         [in-place update, ~sim.reset() only]
        ├── sim.stop()                               [1] deactivate Fabric
        ├── _compute_individual_fitness()            [2] mean return per individual
        ├── design_generator.update_with_fitness()   [3] CMA-ES tell + ask
        ├── design_generator.generate_population()   [4] URDF+USD (offline, file I/O only)
        ├── apply_link_length_params(env, llp)       [5] author prims + sim.reset()
        ├── env.reset()                              [6] reset all episodes
        ├── apply_actuator_params(env, act_params)   [7] patch IdentifiedActuator tensors
        └── zero _individual_fitness, _episode_counts [8]
```

The in-place pathway omits all of `respawn_robots`'s steps 2–9 (delete, spawn, sensor re-registration, new `Articulation` object, `scene._articulations` swap, action/event/command manager rebinding) because the `Articulation` Python object is never replaced. `scene._articulations["robot"]` continues to point to the same object, so all cached `.robot`, `._asset`, and `._offset` references in the managers remain valid throughout. The only structural change on the USD stage is to the box geometry attributes on the existing prims — no prim is created or destroyed.

##### `apply_link_length_params` and `update_articulation_links` Flow

The function pair in [`update.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/update.py) bridges the per-individual `link_length_params` dict from `RandomPopulation` to the USD stage. Because the geometry is instanced ([Section 4.7.6](#476-instantiation-and-prototypes)), authoring goes to each design's prototype source layer rather than to per-environment prims, and a single timed `sim.reset()` commits everything:

```
apply_link_length_params(env, link_length_params_list)     [update.py]
├── sim.stop()  (safety net if not already stopped)
├── stage = get_current_stage()
├── for individual in range(num_individuals):              ← first N envs = each design once
│   ├── env_path = scene.env_prim_paths[individual]        ← round-robin: env i ↔ individual i
│   ├── params   = link_length_params_list[individual]
│   └── for link_name, {x,y,z} in params.items():          ← scalable links only
│       └── for purpose in ("visuals", "collisions"):
│           ├── inst  = stage.GetPrimAtPath(env_path/Robot/link_name/purpose)
│           ├── proto = inst.GetPrototype()                ← implicit /__Prototype_<N> (read-only)
│           └── update_articulation_links(proto.path, env_path/Robot/link_name, [x,y,z], stage)
│               ├── mesh = proto.GetChild("mesh_0")
│               ├── (g_layer, g_scale) = GetPropertyStack(mesh.xformOp:scale)[0]   ← geometry layer
│               ├── (m_layer, m_path)  = GetPropertyStack(link.physics:mass)[0]    ← physics layer
│               │        ← link/joint prims (NOT instanced) resolved on their OWN layers,
│               │          which generally differ from the geometry layer
│               ├── read invariants: x0,y0,z0 (mesh) ; mass0 (link) ; localPos0 (joint)
│               │        density = mass0/(x0·y0·z0) ;  offset = |localPos0.z| − z0
│               └── Sdf.ChangeBlock():                      ← writes batched per prototype
│                   ├── g_layer[g_scale].default      = Vec3d(x,y,z)          ← resize (instanced)
│                   ├── g_layer[g_translate].default  = Vec3d(t0x,t0y,-z/2)   ← recentre
│                   ├── m_layer[link.mass]            = m_new                 ← reference-shared
│                   ├── i_layer[link.diagInertia]     = solid-box(...)        ← reference-shared
│                   └── j_layer[joint.localPos0]      = (px,py,-(z+offset))   ← move joint
└── sim.reset()   (timed, logged)                          ← PLAY re-parses recomposed stage
```

Only the first `num_individuals` environments are visited: round-robin (`env_idx mod N`, the spawner's own assignment) makes that set cover every design exactly once, and editing a design's prototype fans the geometry change out to all `num_envs / num_individuals` instances, while editing its (referenced, non-instanced) link/joint prims fans mass/inertia/joint out to the same environments by reference composition. The per-prototype `Sdf.ChangeBlock()` batches that prototype's writes; the composition queries (`GetPrototype`, `GetPropertyStack`) run before the block, since deferred-notification blocks must not wrap composition reads. The same `individual` index orders authoring identically to `apply_actuator_params` and the runner's `_env_to_individual` map, keeping design identity consistent across all three. Re-applying a generation is idempotent because every value is set absolutely and the invariants (density, attachment gap) are recovered from the current source values.

##### Key Invariants and Design Rationale

| Property | Mechanism | Why it matters |
|---|---|---|
| No manager re-binding | `Articulation` object identity preserved; `scene._articulations["robot"]` never swapped | All `._asset`, `.robot`, `._offset` caches in action/event/command managers remain valid |
| No cross-generation compounding | Absolute extents SET, not multiplied; density and joint-gap recovered from current source layer as invariants | Every generation is authored relative to the base URDF, regardless of what the stage currently holds |
| One edit, N instances | Geometry is set on the shared `/__Prototype_<N>` source `mesh_0` (instancing); mass/inertia/joints on the link/joints prims referenced by every env of the design (reference composition) | Authoring only the first `num_individuals` environments updates all `num_envs` robots on `sim.reset()` — instanced geometry cannot be edited per-environment in any case |
| Domain randomisation unaffected | Mass randomisation writes PhysX tensors via `root_physx_view.set_masses()` over `default_mass` ([`events.py:363-397`](IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py)), authoring no USD opinion | The source-layer mass edit becomes the new parsed `default_mass`; randomisation re-applies its per-env scale on top, neither shadowing nor shadowed |
| Actuators applied last | `apply_actuator_params` called after `sim.reset()` and `env.reset()` | `sim.reset()` fires `_initialize_impl()` which rebuilds `IdentifiedActuator` tensors from cfg, overwriting any earlier patch |
| Generation 0 = base URDF | Initial `_reload_morphology` path is unchanged; first population is loaded via full respawn from gen-0 USD files | Provides a clean, known-good base from which all absolute extents are computed |
| URDF/USD files always generated | `_build_population` is called on every `generate_population` for both pathways | USD files are still used by `_reload_morphology` for the initial population; generating them also provides a persistent record of the morphology tested each generation |
| Deterministic per individual | `RandomDesignGenerator._generate_individual` seeds `np.random.default_rng(generation * 10000 + idx)` | Allows exact reproduction of any individual from `(generation, idx)` alone |


### 4.10 Key Classes and Interfaces

All classes mentioned throughout §5 are documented here with full API details. Shorthand names are used freely in subsequent sections.

#### `SimulationContext`
File: `IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py`
- Role: Singleton controller for the Isaac Sim simulation. Extends `isaacsim.core.api.simulation_context.SimulationContext` to add configuration-driven setup, physics/render settings, fabric interface management, and callback orchestration. Provides the canonical play/pause/stop/step/render interface for Isaac Lab environments.
- Constructor Args:
  - `cfg` (`SimulationCfg | None`): Simulation configuration (physics dt, gravity, device, render interval, etc.). If `None`, a default `SimulationCfg()` is used.
- Key Variables:
  - `cfg` (`SimulationCfg`): The active simulation configuration.
  - `render_mode` (`RenderMode`): Enum controlling what gets rendered: `NO_GUI_OR_RENDERING (-1)`, `NO_RENDERING (0)`, `PARTIAL_RENDERING (1)`, `FULL_RENDERING (2)`.
  - `_initial_stage` (`Usd.Stage`): The USD stage active during scene construction; returned by `get_initial_stage()`.
  - `_fabric_iface`: Fabric (FlatCache) interface handle; `None` if fabric is disabled.
  - `_has_gui` (bool): Whether any GUI (local, livestream, or XR) is active.
- Key Methods:
  - `reset(soft=False)`: Re-raises any pending callback exception, calls parent `reset()`, forces the correct CUDA device, initializes kinematic bodies in fabric, and performs warm-up render passes.
  - `step(render=True)`: Advances physics one step. Blocks while the timeline is paused. Optionally renders after stepping.
  - `forward()`: Updates articulation kinematics and pushes data through the fabric interface without stepping physics.
  - `get_initial_stage() -> Usd.Stage`: Returns `_initial_stage`, the stage active during scene construction.
  - `set_render_mode(mode)`: Switches between `FULL_RENDERING` and `PARTIAL_RENDERING` at runtime when a GUI is present.

#### `InteractiveScene`
File: `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py`
- Role: Parses an `InteractiveSceneCfg` to build the simulation world. Spawns all assets into the USD stage and clones them across `num_envs` environments using `GridCloner`. Provides a unified interface to reset and update all scene entities.
- Constructor Args:
  - `cfg` (`InteractiveSceneCfg`): Configuration specifying number of environments, spacing, `replicate_physics`, `filter_collisions`, and all entity configurations declared as attributes.
- Key Variables:
  - `_articulations` (`dict[str, Articulation]`): Name-keyed dictionary of all articulation assets in the scene; exposed via the `articulations` property (line 381).
  - `_rigid_objects`, `_sensors`, `_extras` (dicts): Rigid body, sensor, and miscellaneous prim containers.
  - `_terrain` (`TerrainImporter | None`): The terrain importer, or `None` if no terrain configured.
  - `_default_env_origins` (`torch.Tensor`): Shape `(num_envs, 3)` tensor of environment origin positions.
  - `sim` (`SimulationContext`): Reference to the active simulation context singleton.
  - `cloner` (`GridCloner`): Used to replicate the `env_0` source prim across all environments.
- Key Properties:
  - `articulations -> dict[str, Articulation]`: Returns `_articulations`.
  - `env_origins -> torch.Tensor`: Terrain origins if a terrain is present, otherwise `_default_env_origins`.
  - `num_envs -> int`: `cfg.num_envs`.
- Key Methods:
  - `reset(env_ids=None)`: Iterates over all assets, sensors, and surface grippers and calls their individual `reset(env_ids)`.
  - `clone_environments(copy_from_source=False)`: Calls `GridCloner.clone(...)` to replicate `/World/envs/env_0` to all environment paths and sets `_default_env_origins`.
  - `filter_collisions(global_prim_paths=None)`: Prevents inter-environment physics interactions, excluding global paths such as the ground plane.

#### `ManagerBasedEnv`
File: `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`
- Role: Base environment class for the manager-based workflow. Creates and owns the `SimulationContext`, `InteractiveScene`, `EventManager`, `ActionManager`, `ObservationManager`, and `RecorderManager`. Does not define rewards or terminations (those are in the RL subclass).
- Constructor Args:
  - `cfg` (`ManagerBasedEnvCfg`): Full environment configuration including scene, sim, actions, observations, events, recorders, decimation, and seed.
- Key Variables:
  - `sim` (`SimulationContext`): Created or retrieved singleton simulation context.
  - `scene` (`InteractiveScene`): The scene built from `cfg.scene`.
  - `event_manager` (`EventManager`): Constructed before `sim.reset()` to allow `"prestartup"` USD-level randomization events.
  - `action_manager` (`ActionManager`): Processes raw policy actions.
  - `observation_manager` (`ObservationManager`): Computes observations.
  - `extras` (dict): Dictionary for passing auxiliary info alongside observations.
  - `_sim_step_counter` (int): Counts raw physics steps since env creation.
- Key Properties:
  - `num_envs -> int`: `scene.num_envs`.
  - `physics_dt -> float`: `cfg.sim.dt`.
  - `step_dt -> float`: `cfg.sim.dt * cfg.decimation`.
  - `device -> str`: `sim.device`.
- Key Methods:
  - `load_managers()` (line 180): Instantiates `RecorderManager`, `ActionManager`, `ObservationManager` (in that order) and fires the `"startup"` event mode. Called automatically in standalone mode after `sim.reset()`; in extension mode the user must call this manually.
  - `reset(seed, env_ids, options) -> (obs, extras)`: Calls `_reset_idx(env_ids)`, writes data to sim, optionally re-renders for RTX sensors, computes and returns observations.
  - `close()`: Cleans up managers and simulation resources.

#### `ManagerBasedRLEnv`
File: `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- Role: RL-specific subclass of `ManagerBasedEnv` that adds `CommandManager`, `TerminationManager`, `RewardManager`, and `CurriculumManager`. Implements the full MDP `step()` loop and `_reset_idx()` with per-episode bookkeeping. Implements `gym.Env` for compatibility with RL libraries.
- Constructor Args:
  - `cfg` (`ManagerBasedRLEnvCfg`): Extends `ManagerBasedEnvCfg` with episode length, rewards, terminations, commands, and curriculum configs.
  - `render_mode` (`str | None`): Gymnasium render mode. Defaults to `None`.
- Key Variables:
  - `episode_length_buf` (`torch.Tensor`): Shape `(num_envs,)` integer tensor tracking the current step count within each environment's episode. Incremented by 1 on each `step()` call; reset to 0 for terminated envs in `_reset_idx()`. Created before `super().__init__()` so MDP functions can reference it during manager construction.
  - `reward_buf` (`torch.Tensor`): Shape `(num_envs,)` scalar reward for the current step.
  - `reset_buf`, `reset_terminated`, `reset_time_outs` (`torch.Tensor`): Boolean masks indicating which envs need reset, which terminated naturally, and which timed out.
  - `common_step_counter` (int): Global step counter across all envs, used by curriculum.
  - `command_manager` (`CommandManager`): Generates and updates locomotion velocity commands.
  - `termination_manager` (`TerminationManager`): Computes per-env termination conditions.
  - `reward_manager` (`RewardManager`): Computes scalar rewards.
  - `curriculum_manager` (`CurriculumManager`): Adjusts difficulty over training.
- Key Methods:
  - `load_managers()` (line 109): Extends parent by first constructing `CommandManager` (observations depend on commands), then calls `super().load_managers()`, then constructs `TerminationManager`, `RewardManager`, `CurriculumManager`, and configures Gym spaces.
  - `step(action) -> (obs, reward, terminated, time_out, extras)` (line 153): Full MDP step. Processes actions, runs `cfg.decimation` physics sub-steps, increments `episode_length_buf` and `common_step_counter`, computes terminations and rewards, resets terminated envs via `_reset_idx()`, updates commands and interval events, then returns observations.
  - `_reset_idx(env_ids)` (line 349): Runs curriculum update, resets scene, fires `"reset"` events, resets all managers in order, and zeros `episode_length_buf[env_ids]`.
  - `max_episode_length -> int`: `ceil(episode_length_s / step_dt)`.

#### `AssetBase`
File: `IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py`
- Role: Abstract base class for all physics-enabled assets (rigid objects, articulations, deformable objects). Manages the lifecycle of USD prim spawning, PhysX handle initialization/invalidation via timeline callbacks, and debug visualization. Subclasses must implement `_initialize_impl()`, `reset()`, `write_data_to_sim()`, and `update()`.
- Constructor Args:
  - `cfg` (`AssetBaseCfg`): Configuration with `prim_path`, `spawn` (spawner config), `init_state`, and `debug_vis`.
- Key Variables:
  - `_is_initialized` (bool): `False` at construction (line 72); set to `True` by `_initialize_callback()` after the timeline PLAY event fires. Reset to `False` on timeline STOP.
  - `stage` (`Usd.Stage`): Handle to the current USD stage at construction time.
  - `_initialize_handle`: Subscription to the timeline PLAY event.
  - `_invalidate_initialize_handle`: Subscription to the timeline STOP event.
  - `_prim_deletion_callback_id`: Callback ID registered with `SimulationManager` for prim deletion events.
  - `_device` (str): Set during `_initialize_callback` from `SimulationManager.get_physics_sim_device()`.
- Key Methods:
  - `_register_callbacks()` (line 267): Subscribes `_initialize_callback` to timeline PLAY (order=10), `_invalidate_initialize_callback` to timeline STOP (order=10), and `_on_prim_deletion` to `IsaacEvents.PRIM_DELETION`. Uses weak references for safe garbage collection.
  - `_initialize_callback(event)` (line 304): Triggered on timeline PLAY. Retrieves backend/device from `SimulationManager` and calls `_initialize_impl()` (overridden by subclasses to create PhysX views and populate data buffers). Sets `_is_initialized = True`.
  - `_invalidate_initialize_callback(event)` (line 324): Triggered on timeline STOP. Sets `_is_initialized = False` and unsubscribes the debug visualization handle.
  - `_on_prim_deletion(prim_path)` (line 331): If the deleted path is "/" or a prefix of this asset's prim path, calls `_clear_callbacks()` to fully unsubscribe all event handles.
  - `is_initialized -> bool`: Property returning `_is_initialized`.
- Important Logic: Spawning happens in `__init__` immediately if `cfg.spawn is not None`. Exceptions raised inside `_initialize_impl()` are stored in `builtins.ISAACLAB_CALLBACK_EXCEPTION` to be re-raised on the next `sim.step()` or `sim.reset()` call, since Omniverse silently swallows exceptions in callbacks.

#### `Articulation`
File: `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py`
- Role: Concrete asset class for articulated rigid-body systems (robots with joints). Manages PhysX `ArticulationView` handles, per-joint actuator models, data buffers for joint/body state, and methods for writing simulation parameters (stiffness, damping, limits, armature, friction) directly into the physics engine.
- Constructor Args:
  - `cfg` (`ArticulationCfg`): Includes `spawn`, `init_state`, `actuators` dict, `soft_joint_pos_limit_factor`, and optionally `articulation_root_prim_path`.
- Key Variables:
  - `actuators` (`dict[str, ActuatorBase]`): Maps actuator group names to actuator model instances; built during `_process_actuators_cfg()`.
  - `_root_physx_view` (`physx.ArticulationView`): The PhysX tensor API view for the articulation, created in `_initialize_impl()`. Exposed via the `root_physx_view` property.
  - `_data` (`ArticulationData`): Holds all joint/body state tensors and their defaults (positions, velocities, accelerations, limits, stiffness, damping, armature, friction, mass, inertia).
  - `_ALL_INDICES` (`torch.Tensor`): Index tensor `[0, ..., num_instances-1]` for broadcast indexing.
- Key Properties:
  - `root_physx_view -> physx.ArticulationView`: Low-level PhysX handle. Setters require CPU tensors.
  - `num_joints -> int`, `num_bodies -> int`, `joint_names -> list[str]`, `body_names -> list[str]`, `is_fixed_base -> bool`.
  - `data -> ArticulationData`: The data buffer container.
- Key Methods:
  - `_initialize_impl()` (line 1506): Resolves the articulation root prim path, creates `_root_physx_view`, initializes `_data = ArticulationData(...)`, calls `_create_buffers()` (reads default joint stiffness/damping/armature/friction/limits/mass/inertia from PhysX), then `_process_actuators_cfg()` (instantiates actuator models).
  - `write_joint_stiffness_to_sim(stiffness, joint_ids, env_ids)` (line 652): Updates `_data.joint_stiffness` buffer and calls `root_physx_view.set_dof_stiffnesses(...)` (requires CPU tensor).
  - `write_joint_damping_to_sim(damping, joint_ids, env_ids)` (line 681): Updates `_data.joint_damping` and calls `root_physx_view.set_dof_dampings(...)`.
  - `write_joint_position_limit_to_sim(limits, joint_ids, env_ids, warn_limit_violation)` (line 710): Updates `_data.joint_pos_limits`, clamps `default_joint_pos` to the new limits if needed, calls `set_dof_limits(...)`, and recomputes `soft_joint_pos_limits` using `soft_joint_pos_limit_factor`.
  - `write_joint_armature_to_sim(armature, joint_ids, env_ids)` (line 840): Updates `_data.joint_armature` and calls `set_dof_armatures(...)`.
  - `write_joint_friction_coefficient_to_sim(joint_friction_coeff, ..., joint_ids, env_ids)` (line 871): For Isaac Sim < 5.0 sets static friction via `set_dof_friction_coefficients`; for >= 5.0 patches static/dynamic/viscous components and writes back via `set_dof_friction_properties`.

#### `ArticulationCfg`
File: `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py` (line 16)
- Role: Configuration dataclass for an `Articulation` asset. Inherits `AssetBaseCfg` (providing `prim_path`, `spawn`, `init_state`, `debug_vis`) and adds articulation-specific settings.
- Key Fields:
  - `class_type: type = Articulation`: Tells the asset factory which class to instantiate.
  - `articulation_root_prim_path: str | None = None`: Optional explicit relative path from `prim_path` to the prim carrying `UsdPhysics.ArticulationRootAPI`. If `None`, the class auto-discovers it.
  - `init_state` (`InitialStateCfg`): Nested config with `pos`, `rot` (initial root pose), `lin_vel`, `ang_vel` (initial root velocity), `joint_pos` and `joint_vel` (regex-keyed initial values, default `{".*": 0.0}`).
  - `soft_joint_pos_limit_factor: float = 1.0` (line 56): Fraction of the full joint range to use as "soft" limits in `ArticulationData.soft_joint_pos_limits`. Value of 0.9 means soft limits cover 90% of hardware range. Used for termination and reward shaping; does not affect physics.
  - `actuators: dict[str, ActuatorBaseCfg] = MISSING` (line 66): Required. Maps actuator group names to their configuration objects (`ImplicitActuatorCfg`, `IdentifiedActuatorCfg`, etc.). Each entry specifies `joint_names_expr`, stiffness, damping, and limits.

#### `UsdFileCfg`
File: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py` (line 74)
- Role: Configuration for spawning an asset by referencing a USD file. Inherits from `FileCfg` which itself inherits `RigidObjectSpawnerCfg` and `DeformableObjectSpawnerCfg`.
- Key Fields:
  - `func: Callable = spawn_from_usd` (line 99): The spawner function called when this config is used.
  - `usd_path: str = MISSING` (line 101): Required. Filesystem or Nucleus URL path to the `.usd` / `.usda` / `.usdc` file.
  - `variants: object | dict[str, str] | None = None`: USD variant selections to apply after spawning.
  - `rigid_props` (`RigidBodyPropertiesCfg | None`, inherited): Rigid body physics properties (damping, gravity, velocity limits, etc.).
  - `articulation_props` (`ArticulationRootPropertiesCfg | None`, inherited): Articulation root properties (self-collision, solver iterations).
  - `joint_drive_props` (`JointDrivePropertiesCfg | None`, inherited): Overrides drive properties on all joints. Prefer `ArticulationCfg.actuators` for per-group control.
  - `activate_contact_sensors: bool` (inherited from `RigidObjectSpawnerCfg`): Whether to enable contact reporting on the prim's collision shapes.
  - `scale: tuple[float,float,float] | None = None`: Uniform or anisotropic scale override.

#### `MultiUsdFileCfg`
File: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers_cfg.py` (line 46)
- Role: Variant of `UsdFileCfg` that accepts a list of USD file paths and either randomly selects one per spawn or cycles through them in order. Used to introduce morphological diversity across environments at spawn time.
- Key Fields (inherits all of `UsdFileCfg`):
  - `func = spawn_multi_usd_file` (line 57): The spawner function (not `spawn_from_usd`).
  - `usd_path: str | list[str] = MISSING` (line 59): A single path string or a list of USD file paths.
  - `random_choice: bool = True` (line 37): If `True`, each spawn randomly selects from the list. If `False`, spawns cycle through the list in order.

#### `UrdfFileCfg`
File: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py` (line 114)
- Role: Configuration for spawning an asset from a URDF file. Combines `FileCfg` with `UrdfConverterCfg` (URDF-to-USD conversion settings). The converter produces a cached USD file on first use; subsequent calls use the cached version.
- Key Fields:
  - `func: Callable = spawn_from_urdf` (line 132): Dispatches to `spawn_from_urdf()`, which calls `UrdfConverter(cfg)` to produce a USD, then delegates to `_spawn_from_usd_file(...)`.
  - Inherited from `UrdfConverterCfg`: `asset_path: str = MISSING` (path to the `.urdf` file), `fix_base: bool = False` (whether to fix the root link), `merge_fixed_joints: bool = False` (collapse fixed joints to reduce the articulation tree), `make_instanceable: bool = True` (enables GPU instancing), `convex_decompose_mesh: bool = False` (use V-HACD for collision meshes).
  - All `FileCfg` properties (`rigid_props`, `articulation_props`, `joint_drive_props`, `scale`, `visual_material`) also apply.

#### `ImplicitActuatorCfg`
File: `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_cfg.py`
- Role: Configuration for an implicit PD actuator where PD control is handled entirely inside the PhysX physics engine (not computed by Python). Stiffness and damping values are written directly into the simulation's joint drive properties.
- Key Fields:
  - `class_type: type = ImplicitActuator`: Marker for the actuator factory.
  - `joint_names_expr: list[str] = MISSING`: List of joint name patterns (strings or regex) this actuator group controls.
  - `stiffness: dict[str,float] | float | None = MISSING`: PD proportional gain (Kp). Written to PhysX via `set_dof_stiffnesses`. If `None`, value from USD is used.
  - `damping: dict[str,float] | float | None = MISSING`: PD derivative gain (Kd). Written to PhysX via `set_dof_dampings`. If `None`, value from USD is used.
  - `effort_limit_sim: dict[str,float] | float | None = None`: Maximum effort enforced by the physics solver. If `None`, the value from USD is used.
  - `velocity_limit_sim: dict[str,float] | float | None = None`: Maximum joint velocity enforced by the solver.
  - `friction: dict[str,float] | float | None = None`: Static friction coefficient of the joint. If `None`, value from USD is used.
  - `armature: dict[str,float] | float | None = None`: Added to joint-space inertia diagonal to improve simulation stability.

#### `IdentifiedActuatorCfg`
File: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/actuators/actuator_cfg.py` (line 15)
- Role: Configuration for the `IdentifiedActuator`, an explicit DC motor model extended with identified friction parameters. Inherits `DCMotorCfg` → `IdealPDActuatorCfg` → `ActuatorBaseCfg`, adding static, dynamic, and viscous friction identification from real hardware. Unlike `ImplicitActuatorCfg`, the USD `DriveAPI` stiffness and damping are set to zero; torque is computed entirely in Python and applied as a feed-forward force command.
- Key Fields:
  - `class_type: type = IdentifiedActuator`: Instantiates the `IdentifiedActuator` Python actuator model (not handled by PhysX).
  - `stiffness` (inherited, `MISSING`): PD Kp gain (Nm/rad).
  - `damping` (inherited, `MISSING`): PD Kd gain (Nm·s/rad).
  - `effort_limit` (inherited): Maximum output torque for clipping in the explicit actuator model.
  - `velocity_limit` (inherited): Maximum velocity used by the actuator model.
  - `saturation_effort: float = MISSING` (from `DCMotorCfg`): Peak motor torque (Nm). Torque is clipped to this before the PD computation.
  - `friction_static: float = MISSING`: Constant static friction torque (Nm) applied when joint velocity is below `activation_vel`.
  - `activation_vel: float = MISSING`: Velocity threshold (rad/s) below which static friction is active; above which dynamic friction takes over.
  - `friction_dynamic: float = MISSING`: Dynamic friction coefficient (Nm·s/rad). Friction torque = `friction_dynamic * |velocity|`, opposing motion.
- Identified per-joint-group values used in `SOLEFOOT_IDENTIFIED_CFG`: Abad: Kp=55, Kd=13.5, sat=402 Nm; Hip: Kp=80, Kd=13, sat=443 Nm; Knee: Kp=60, Kd=4, sat=560 Nm; Ankle: Kp=10, Kd=0.5, sat=402 Nm.

#### `EventManager`
File: `IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py`
- Role: Manages a collection of event terms grouped by mode. Applies the appropriate terms when triggered by the environment at different simulation lifecycle points.
- Constructor Args:
  - `cfg` (object or `dict[str, EventTermCfg]`): Configuration object or dictionary of event terms.
  - `env` (`ManagerBasedEnv`): The environment instance.
- Key Variables:
  - `_mode_term_names` (`dict[str, list[str]]`): Maps mode name to list of term names in that mode.
  - `_mode_term_cfgs` (`dict[str, list[EventTermCfg]]`): Maps mode name to list of `EventTermCfg` objects.
  - `_interval_term_time_left` (list of tensors): Per-environment time remaining before next trigger for each `"interval"` term.
  - `_reset_term_last_triggered_step_id` (list of tensors): Tracks the last global env step at which each `"reset"` term was triggered per environment.
- Event Modes:
  - `"prestartup"`: Applied once before `sim.reset()`. Used for USD-level changes (mesh scale, etc.).
  - `"startup"`: Applied once after all managers are created and `sim.reset()` has been called.
  - `"reset"`: Applied on environment reset. Supports `min_step_count_between_reset` to prevent over-frequent triggering.
  - `"interval"`: Applied when per-environment timers expire. The manager handles timing logic internally.
- Key Methods:
  - `apply(mode, env_ids=None, dt=None, global_env_step_count=None)`: Main dispatch. For `"interval"` mode: decrements per-env timers by `dt`, triggers terms where timers have expired (≤ 1e-6 s), and samples new random intervals. For `"reset"` mode: checks `min_step_count_between_reset`, triggers only environments that have waited long enough. For all other modes: applies terms to all given `env_ids` unconditionally.
  - `reset(env_ids=None) -> dict`: Resets class-based terms and re-samples interval timers for newly-reset environments.
  - `available_modes -> list[str]`: Returns all configured modes.

#### `OnPolicyRunner`
File: `rsl_rl/rsl_rl/runners/on_policy_runner.py` (line 23)
- Role: Orchestrates on-policy RL training (PPO). Handles the rollout collection loop, policy update, logging to Tensorboard/W&B/Neptune, checkpoint saving and loading, and optional multi-GPU distributed training.
- Constructor Args:
  - `env` (`VecEnv`): The vectorized environment (typically a `ManagerBasedRLEnv` wrapped for RSL-RL).
  - `train_cfg` (dict): Training configuration dict with keys: `algorithm`, `policy`, `num_steps_per_env`, `save_interval`, `obs_groups`, `logger`, and optionally `rnd_cfg`.
  - `log_dir` (`str | None`): Directory for checkpoints and logs.
  - `device` (str): Compute device string (e.g. `"cuda:0"`).
- Key Variables:
  - `env` (`VecEnv`): Reference to the environment.
  - `alg` (`PPO`): The algorithm instance created by `_construct_algorithm()`. Contains `alg.policy` (the actor-critic network) and `alg.optimizer`.
  - `num_steps_per_env` (int): Number of environment steps collected per rollout before a policy update.
  - `save_interval` (int): Save a checkpoint every this many learning iterations.
  - `current_learning_iteration` (int): Index of the last completed update iteration.
  - `cur_reward_sum` (`torch.Tensor`, line 80): Shape `(num_envs,)` tensor accumulating rewards step-by-step within the rollout. When an environment's `done` flag fires, its accumulated sum is pushed into the `rewbuffer` deque and reset to 0. Used to compute `"Train/mean_reward"`.
  - `log_dir`, `writer`: Logging directory and summary writer (Tensorboard / W&B / Neptune).
- Key Methods:
  - `learn(num_learning_iterations, init_at_random_ep_len=False)` (line 62): Main training loop. For each iteration: runs `num_steps_per_env` rollout steps (inference mode) calling `alg.act(obs)`, `env.step(actions)`, `alg.process_env_step(...)`; accumulates and clears `cur_reward_sum` on episode boundaries; calls `alg.compute_returns(obs)` and `alg.update()`; calls `log()` and `save()` at configured intervals.
  - `_construct_algorithm(obs)` (line 395): Resolves RND and symmetry configs, instantiates the actor-critic policy class (from `policy_cfg["class_name"]`), instantiates the PPO algorithm class (from `alg_cfg["class_name"]`), initializes the rollout storage buffer, and returns the algorithm.
  - `save(path, infos=None)` (line 291): Saves `model_state_dict`, `optimizer_state_dict`, current iteration, and optional infos to a `.pt` file.
  - `load(path, load_optimizer=True, map_location=None)` (line 309): Loads `model_state_dict` and conditionally restores optimizer state and `current_learning_iteration`.
  - `get_inference_policy(device=None) -> callable`: Switches to eval mode and returns `alg.policy.act_inference`.

#### `spawn_from_usd()`
File: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` (line 38)
- Signature: `spawn_from_usd(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None) -> Usd.Prim`
- Decorator: `@clone` — resolves a prim path regex pattern into a list of concrete paths and handles cloning when multiple paths match.
- What it does: Validates that the configured USD file path exists, then delegates to `_spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)` to create the prim, apply all properties from `cfg` (rigid body props, collision props, articulation root props, joint drive props, visual material, scale, semantic tags, variants), and return the prim.

#### `spawn_from_urdf()`
File: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` (line 80)
- Signature: `spawn_from_urdf(prim_path: str, cfg: UrdfFileCfg, translation=None, orientation=None) -> Usd.Prim`
- Decorator: `@clone`
- What it does: Instantiates `converters.UrdfConverter(cfg)` which converts the URDF to a cached USD file, then delegates to `_spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)` to spawn and configure the prim exactly as `spawn_from_usd` would.

#### `_spawn_from_usd_file()`
File: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` (line 268)
- Signature: `_spawn_from_usd_file(prim_path: str, usd_path: str, cfg: FileCfg, translation=None, orientation=None) -> Usd.Prim`
- What it does: Internal helper shared by both `spawn_from_usd` and `spawn_from_urdf`. Checks `usd_path` exists (with timeout, including a Nucleus `/4.5` → `/5.0` fallback), calls `create_prim(prim_path, usd_path=usd_path, translation, orientation, scale=cfg.scale)` (line 314) if no prim exists at `prim_path`, then applies all `cfg` properties sequentially. If the prim already exists (e.g. in heterogeneous cloning where the env grid is pre-cloned), it skips prim creation and only applies config properties.

#### `create_prim()`
File: `IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py` (line 44)
- Signature: `create_prim(prim_path, prim_type="Xform", position=None, translation=None, orientation=None, scale=None, usd_path=None, semantic_label=None, semantic_type="class", attributes=None, stage=None) -> Usd.Prim`
- What it does: Validates that `position` and `translation` are not both provided. Raises `ValueError` if a prim already exists at `prim_path`. Calls `stage.DefinePrim(prim_path, prim_type)`, sets any `attributes` key-value pairs, adds a USD reference arc via `add_usd_reference(...)` if `usd_path` is provided, adds semantic labels if requested, and sets the XForm pose (translate → orient → scale) in USD canonical format. Returns the created prim.

#### `delete_prim()`
File: `IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py` (line 189)
- Signature: `delete_prim(prim_path: str | Sequence[str], stage: Usd.Stage | None = None) -> bool`
- What it does: Accepts a single path string or a list of paths. Executes the Omniverse Kit `"DeletePrimsCommand"` to remove the prims and all their descendants from the stage. Returns `True` on success. Active `Articulation` assets watching the deleted prim path will have their `root_physx_view` invalidated via the `_on_prim_deletion` callback.

#### `randomize_actuator_gains`
File: `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` (line 541)
- Signature (`__call__`): `(env, env_ids, asset_cfg: SceneEntityCfg, stiffness_distribution_params: tuple|None, damping_distribution_params: tuple|None, operation: Literal["add","scale","abs"] = "abs", distribution: Literal["uniform","log_uniform","gaussian"] = "uniform")`
- What it does: A `ManagerTermBase` callable event term. For each actuator group in the asset, computes the intersection of the actuator's joint indices with `asset_cfg.joint_ids`. If `stiffness_distribution_params` is not `None`: resets stiffness to `default_joint_stiffness` (to avoid compounding randomizations), applies `_randomize_prop_by_op(stiffness, params, ..., operation, distribution)`, stores the result, and if the actuator is an `ImplicitActuator` calls `write_joint_stiffness_to_sim(...)` to push to PhysX (line 648–649). Repeats for damping. Always randomizes from default values to ensure idempotent behavior across resets.

#### `randomize_rigid_body_mass`
File: `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` (line 286)
- Signature (`__call__`): `(env, env_ids, asset_cfg: SceneEntityCfg, mass_distribution_params: tuple[float,float], operation: Literal["add","scale","abs"], distribution: Literal["uniform","log_uniform","gaussian"] = "uniform", recompute_inertia: bool = True, min_mass: float = 1e-6)`
- What it does: Gets current body masses from `root_physx_view.get_masses()`, resets selected body masses for `env_ids` to `default_mass`, applies `_randomize_prop_by_op(...)`, clamps to `min_mass`, then calls `root_physx_view.set_masses(masses, env_ids)`. If `recompute_inertia=True`, recomputes the inertia tensor assuming uniform density (scales inertia by the mass ratio) and calls `root_physx_view.set_inertias(...)`. Uses CPU tensors throughout.

#### `randomize_joint_parameters`
File: `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` (line 652)
- Signature (`__call__`): `(env, env_ids, asset_cfg: SceneEntityCfg, friction_distribution_params: tuple|None, armature_distribution_params: tuple|None, lower_limit_distribution_params: tuple|None, upper_limit_distribution_params: tuple|None, operation: Literal["add","scale","abs"] = "abs", distribution: Literal["uniform","log_uniform","gaussian"] = "uniform")`
- What it does: A `ManagerTermBase` callable event term that randomizes low-level PhysX joint properties. For each non-`None` parameter set, applies `_randomize_prop_by_op(default_value.clone(), params, env_ids, joint_ids, operation, distribution)` then calls the corresponding `Articulation.write_*_to_sim(...)` method: `write_joint_friction_coefficient_to_sim`, `write_joint_armature_to_sim`, or `write_joint_position_limit_to_sim`. Always randomizes from default values. Requires CPU tensors; recommended to apply at startup/reset rather than as interval events.

#### `SFSceneCfg`
File: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py` (line 34)
- Role: Scene configuration for the Sole-Foot (SF) biped. Inherits `InteractiveSceneCfg`.
- Key Fields:
  - `robot: ArticulationCfg = MISSING` (line 71): Required. Must be set by the concrete task class (e.g. assigned to `SOLEFOOT_IDENTIFIED_CFG`).
  - `terrain: TerrainImporterCfg`: A flat plane terrain at `/World/ground` with `static_friction=1.0`, `dynamic_friction=1.0`, marble tile visual material. `collision_group=-1`, `max_init_terrain_level=0`.
  - `light: AssetBaseCfg`: A dome light at `/World/skyLight` with `intensity=750.0` and an HDR sky texture from Isaac Nucleus.
  - `height_scanner: RayCasterCfg`: Ray-cast height sensor attached to `{ENV_REGEX_NS}/Robot/base_Link`. Offset 20 m above, `ray_alignment="yaw"`, 1.6 m × 1.0 m grid at 0.1 m resolution, sampling against `/World/ground`.
  - `contact_forces: ContactSensorCfg`: Contact sensor on all robot links (`{ENV_REGEX_NS}/Robot/.*`), `history_length=4`, `track_air_time=True`, `update_period=0.0` (every physics step).

#### `SFEnvCfg`
File: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py` (line 963)
- Role: Full RL environment configuration for the SF biped task. Inherits `ManagerBasedRLEnvCfg`.
- Key Fields:
  - `scene: SFSceneCfg = SFSceneCfg(num_envs=4096, env_spacing=2.5)`: Scene with 4096 parallel environments at 2.5 m spacing.
  - `observations: ObservationsCfg`: Base lin/ang velocity, projected gravity, joint pos/vel, last action, height scan, velocity commands, etc.
  - `actions: ActionsCfg`: Joint position action on `".*"` joints, `scale=0.25`, `use_default_offset=True`.
  - `commands: CommandsCfg`: Uniform velocity commands for x, y (±1.0 m/s), yaw (±1.0 rad/s), heading (±π), resampled every 12–18 s.
  - `rewards: RewardsCfg`: Full reward configuration (tracking velocity, base height, gait symmetry, energy penalties, contact forces, etc.).
  - `terminations: TerminationsCfg`: Termination conditions (base contact, joint limits, etc.).
  - `events: EventsCfg`: Domain randomization events (mass, friction, gains, push forces, etc.).
  - `curriculum: CurriculumCfg`: Terrain level curriculum, push force curriculum, command velocity curriculum.
  - `__post_init__` settings (line 980): `decimation=4` (4 physics steps per env step), `episode_length_s=20.0`, `sim.dt=0.005` (200 Hz physics), `sim.render_interval=8` (render every 8 physics steps), `seed=42`.

#### `SOLEFOOT_CFG`
File: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_cfg.py` (line 10)
- Role: Baseline articulation configuration for the SF (Sole-Foot) TRON1A robot using simple implicit PD actuators. Loads from the local USD at `../usd/SF_TRON1A/SF_TRON1A.usd`.
- What's configured:
  - Spawn: `UsdFileCfg` with `rigid_props` (gravity enabled, no linear/angular damping, `max_depenetration_velocity=1.0`), `articulation_props` (self-collision enabled, 4 solver position + velocity iterations), `activate_contact_sensors=True`.
  - Init state: Root at z=0.8 m, all joints at 0.0 rad with 0.0 rad/s velocity.
  - `soft_joint_pos_limit_factor = 0.9`.
  - Actuators (2 groups, all `ImplicitActuatorCfg`): `"legs"` (abad L/R, hip L/R, knee L/R): `effort_limit_sim=80 Nm`, `velocity_limit_sim=25 rad/s`, `stiffness=50`, `damping=2.2`. `"ankles"` (ankle L/R): `effort_limit_sim=80 Nm`, `velocity_limit_sim=25 rad/s`, `stiffness=15`, `damping=0.8`.

#### `SOLEFOOT_IDENTIFIED_CFG`
File: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_identified_cfg.py` (line 101)
- Role: Physics-identified articulation configuration for the SF TRON1A robot. Uses `IdentifiedActuator` (explicit DC motor + friction model) with parameters identified from real hardware. Same USD and rigid/articulation props as `SOLEFOOT_CFG`.
- What's configured:
  - Spawn: `UsdFileCfg` pointing to `SF_TRON1A.usd`, same `rigid_props` and `articulation_props` as `SOLEFOOT_CFG`.
  - `soft_joint_pos_limit_factor = 0.9`.
  - Actuators (4 groups, all `IdentifiedActuatorCfg`): `"abad"` (abad L/R): Kp=55, Kd=13.5, sat=402 Nm, `friction_static=0.3 Nm`, `activation_vel=0.1 rad/s`, `friction_dynamic=0.02 Nm·s/rad`. `"hip"` (hip L/R): Kp=80, Kd=13, sat=443 Nm. `"knee"` (knee L/R): Kp=60, Kd=4, sat=560 Nm, `friction_static=0.8 Nm`. `"ankle"` (ankle L/R): Kp=10, Kd=0.5, sat=402 Nm, `friction_static=0.1 Nm`.

#### `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG`
File: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_identified_cfg.py` (line 113)
- Role: Variant of `SOLEFOOT_IDENTIFIED_CFG` that randomizes the robot's foot morphology across environments by randomly selecting from three USD files (sole-foot, point-foot, wheeled-foot variants). Uses identical identified actuators and init state.
- What's configured:
  - Spawn: `MultiUsdFileCfg` with `usd_path=[usd_path_sf, usd_path_pf, usd_path_wf]` (`SF_TRON1A`, `PF_TRON1A`, `WF_TRON1A`), `random_choice=True`. Same `rigid_props` and `articulation_props`.
  - Actuators: Identical 4-group `IdentifiedActuatorCfg` setup as `SOLEFOOT_IDENTIFIED_CFG`. The foot morphology changes but joint layout remains the same across all three USD variants.

---

## 5. Robot Design and Controller co-optimisation

This section records the co-optimisation algorithm as implemented in the committed codebase, so that it can be read alongside the design discussion in [Section 4.8](#48-recommended-strategy-hybrid-two-tier-co-optimisation) and the API reference in [Section 4.9](#49-designgeneratorbase-api). The full investigation of its behaviour, with a prioritised implementation plan, lives in [plans/COPT_INVESTIGATION_PLAN.md](plans/COPT_INVESTIGATION_PLAN.md).

### 5.1 Implemented Loop

The entry point `scripts/rsl_rl/train.py` selects the COPT path when invoked with `--policy-type COPT` (task `Isaac-Limx-SF-Copt-Rough-v0`). It constructs a `GrowingDesignDistCMAESDesignGenerator`, sets `agent_cfg.policy.class_name = "CoptActorCritic"`, and instantiates a `CoptOnPolicyRunner` ([`train.py:194-235`](tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py)). The runner subclasses `OnPolicyRunner` and overrides `learn` to interleave an outer evolutionary update with the inner PPO update.

The inner loop is unchanged RSL-RL PPO. Every `ea_update_interval` iterations the runner calls `_update_morphology`, which advances the design generator, authors the new link extents into the per-environment USD prototypes via `apply_link_length_params`, resets all environments, re-applies actuator parameters, and zeroes the per-individual fitness accumulators.

Configuration (committed values):

| Setting | Value | Source |
|---|---|---|
| `ea_update_interval` | 120 | `train.py:204` |
| `ea_late_start` | 8000 | `train.py:205` |
| `num_individuals` | 64 | `train.py:201` |
| `randomise_before_late_start` | True | `train.py:227` |
| optimised parameters | `thigh_length_scale`, `shank_length_scale` | `usd_generator.py` |
| parameter bounds | `(0.85, 1.15)` each | `DEFAULT_PARAM_RANGES` |
| CMA-ES `sigma0` | 0.75 | `train.py:214` |
| CMA-ES domain | unit square `[0,1]²`, mean `0.5` | `usd_generator.py:720-731` |
| link-extent rounding | 2 decimals | `usd_generator.py:372` |

For the first `ea_late_start` iterations the generator samples random designs from a distribution whose spread grows from 5 % to the full range; thereafter it samples from the CMA-ES search distribution and denormalises each unit-square coordinate to the physical scale range. The morphology window spans `120 × 25 = 3000` control steps, which exceeds the `1000`-step episode length, so episodes complete within a window.

The policy `CoptActorCritic` ([`co_optimisation/modules/copt_actor_critic.py`](tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/modules/copt_actor_critic.py)) gives the actor the policy observations concatenated with a 16-dim latent from an estimator over the privileged observations. The latent is not detached, so the estimator trains jointly with the actor. The inherited critic observes the `critic` group, which carries the true per-environment link lengths, masses, and inertias, so the value function can condition on the morphology it evaluates.

### 5.2 Known Issues and Investigation Findings

Snapping link extents to a 1 cm grid erases sub-centimetre variety and flattens the fitness, tripping the `tolfun`/`tolflatfitness` stop criteria. Rounding to four decimals preserves the selection gradient, however, it degrades PPO learning. The rounding therefore remains the one live defect in the current implementation, and the one centimetre resolution floor on link extents stands due to further increase in resolution being detrimental to PPO learning as observed in some experiments. It has been mitigated rather than removed, the search ranges having been widened from `(0.85, 1.15)` to `(0.75, 1.25)` expressly so that the lattice surviving the quantisation is finer in relative terms.

### 5.3 Learned-Model Extension

A learned-model variant of this pipeline, policy type `COPT-LEARNED`, extends `CoptActorCritic` with an encoder-decoder estimator (`CoptEstimator`) trained jointly with PPO under a single optimiser, adding an MSE model-estimation loss that regresses the robot's dynamic state (torques, accelerations, inertia, contact forces, foot velocities) from the same latent that conditions the actor. The full design, literature grounding, and step-by-step implementation plan are documented in [plans/COPT_LEARNED_MODEL.md](plans/COPT_LEARNED_MODEL.md).

---
