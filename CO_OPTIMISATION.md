# Design and Controller Co-optimisation: RSL-RL Architecture and Implementation Guide

## 1. Introduction

This document provides a comprehensive overview of the software architecture of the `rsl_rl` reinforcement learning package and serves as a detailed guideline for implementing custom learning algorithms within this framework. 

The primary context for this documentation is the broader goal of design and controller co-optimization using IsaacLab. While IsaacLab provides a robust, GPU-accelerated simulation environment for complex robotic systems, `rsl_rl` acts as the learning engine—a fast, PyTorch-based RL library designed for high-performance training of locomotion policies. Standard algorithms like PPO are often insufficient for advanced tasks requiring memory, state estimation, or morphological adaptations. Therefore, understanding how to extend `rsl_rl` to implement custom training loops, auxiliary losses, and specialized neural network architectures is crucial.

Please note that this document focuses on the implementation of the learning algorithm using RSL-RL and IsaacLab. The specific application to design and controller co-optimization will be detailed in subsequent sections. Refer to [ARCHITECTURE.md](ARCHITECTURE.md) for more information regarding the configuration of the algorithms and environments implemented within this workspace.

## 2. RSL-RL Framework Overview

### a. Architecture

The `rsl_rl` package is designed with a highly modular, object-oriented architecture that cleanly separates the environment interaction loop, the mathematical algorithm logic, and the neural network definitions. This separation of concerns allows researchers to swap components without rewriting the entire training pipeline.

The flow of data during a standard training iteration involves four main components:

1.  **Runner (`OnPolicyRunner`)**: The orchestrator. It manages the training loop, initializing the environment, algorithm, and policy based on configuration. It is responsible for executing rollouts (interacting with the environment) and triggering the algorithm's update phase.
2.  **Algorithm (`PPO`)**: The core learning algorithm. It holds references to the policy and the storage buffer. After the runner completes a rollout, the algorithm computes advantages and returns, and then executes the optimization steps (e.g., computing surrogate, value, and entropy losses and applying gradient descent).
3.  **Modules / Networks (`ActorCritic`)**: The neural network definitions (subclasses of `torch.nn.Module`). These modules take observations (typically as a `TensorDict`), route them through the appropriate sub-networks (Actor and Critic MLPs), and handle the sampling of action distributions.
4.  **Storage (`RolloutStorage`)**: A specialized buffer that accumulates transitions during rollouts. It stores observations, actions, rewards, values, and environment termination signals, providing mini-batches to the algorithm during the update phase.

### b. Key Classes and Interfaces

Below is the API documentation for the core classes, detailing their constructor signatures, key member variables, and essential methods.

#### `runners.OnPolicyRunner`
- **Role**: Orchestrates the training process, logging, and checkpointing.
- **Constructor Args**:
  - `env`: The RL environment (wrapped to provide the expected interface).
  - `train_cfg` (dict): Training configuration parameters.
  - `log_dir` (str): Directory for saving logs and checkpoints.
  - `device` (str): Compute device (e.g., 'cuda:0').
- **Key Methods**:
  - `learn(num_learning_iterations, init_at_random_ep_len=False)`: The main training loop. Alternates between collecting data (rollouts) and updating the policy.
  - `_construct_algorithm()`: Internal method to instantiate the RL algorithm and policy based on `train_cfg`.
  - `save(path, infos=None)`: Saves the model state, including the policy and optimizer.
  - `load(path, load_optimizer=True)`: Loads a saved model state.

#### `algorithms.PPO`
- **Role**: Implements the Proximal Policy Optimization algorithm.
- **Constructor Args**:
  - `actor_critic` (nn.Module): The policy and value network module.
  - `device` (str): Compute device.
  - `**kwargs`: Hyperparameters (e.g., `clip_param`, `gamma`, `lam`, `learning_rate`).
- **Key Variables**:
  - `transition`: A specialized dictionary to hold the current step's data before it is added to storage.
  - `storage`: Instance of `RolloutStorage`.
- **Key Methods**:
  - `act(obs, critic_obs)`: Queries the policy for actions and values during rollouts.
  - `process_env_step(rewards, dones, infos)`: Adds the current transition to the storage buffer.
  - `compute_returns(last_critic_obs)`: Calculates generalized advantage estimations (GAE) at the end of a rollout.
  - `update()`: Iterates over mini-batches, computes losses (surrogate, value, entropy), and performs backpropagation to update the `actor_critic` network.

#### `modules.ActorCritic`
- **Role**: Defines the neural network architecture for the policy (actor) and value function (critic).
- **Constructor Args**:
  - `num_actor_obs` (int): Dimension of the actor's observation space.
  - `num_critic_obs` (int): Dimension of the critic's observation space.
  - `num_actions` (int): Dimension of the action space.
  - `actor_hidden_dims` (list): Architecture of the actor MLP.
  - `critic_hidden_dims` (list): Architecture of the critic MLP.
- **Key Methods**:
  - `update_distribution(observations)`: Passes observations through the actor network and instantiates a `torch.distributions.Normal` distribution.
  - `act(observations, **kwargs)`: Samples actions from the distribution and returns them.
  - `evaluate(critic_observations, **kwargs)`: Passes observations through the critic network to estimate the value.
  - `act_inference(observations)`: Used during deployment; returns the mean of the action distribution deterministically.

#### `storage.RolloutStorage`
- **Role**: Manages the collection and mini-batch generation of rollout data.
- **Key Methods**:
  - `add_transitions(...)`: Inserts a step's data into the buffer.
  - `compute_returns(last_values, gamma, lam)`: Computes GAE.
  - `mini_batch_generator(num_mini_batches, num_epochs)`: Yields randomized mini-batches for the algorithm's update loop.

### c. Sample Usage

A standard training setup using IsaacLab and RSL-RL can be found in `@tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py`. The typical workflow is as follows:

1. **Configuration Loading**: The script parses arguments to load the environment configuration (e.g., `LimxBaseEnvCfg`) and the RSL-RL agent configuration (e.g., `LimxBothPpoRunnerCfg`) from the task registry.
2. **Environment Creation**: The IsaacLab simulation is launched, and the environment is created using `gym.make(task_name, cfg=env_cfg)`.
3. **Environment Wrapping**: The core IsaacLab environment is wrapped using `RslRlVecEnvWrapper`. This crucial step bridges the gap between IsaacLab's output format and the `TensorDict` structure expected by `rsl_rl`.
4. **Runner Initialization**: An instance of `OnPolicyRunner` (or a custom derivative) is created, passing the wrapped environment and configuration.
5. **Execution**: `runner.learn()` is invoked to begin the training process. Logging (via TensorBoard) and model checkpoints are automatically handled within the runner's log directory.

## 3. Implementing Custom RL Algorithm

To implement advanced architectures (like state estimators or morphological co-optimization), one must extend the base classes provided by `rsl_rl`.

### a. Guidelines

The process of implementing a new algorithm involves the following structured steps. Let us consider the example of a custom learning algorithm whose package name is `sample_algorithm` All custom code for any RL algorithm should be placed in a self-contained package directory under the workspace (e.g., `tron1-rl-isaaclab-cozum/sample_algorithm/`), mirroring the sub-package layout of the base `rsl_rl` library:

```
sample_algorithm/
├── sample_algorithm/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── morpho_ppo.py              # Custom PPO subclass
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── morpho_actor_critic.py     # Custom ActorCritic subclass
│   │   └── morpho_embedding.py        # Standalone nn.Module networks
│   ├── runners/
│   │   ├── __init__.py
│   │   └── morpho_on_policy_runner.py # Custom OnPolicyRunner subclass
│   └── storage/
│       ├── __init__.py
│       └── morpho_rollout_storage.py  # Custom RolloutStorage subclass
└── setup.py
```

---

1. **Runner Implementation** (`runners/morpho_on_policy_runner.py`):

   - **Create** `MorphoOnPolicyRunner(OnPolicyRunner)` inheriting from `rsl_rl.runners.OnPolicyRunner`
   - **Override** `_construct_algorithm(self, obs: TensorDict) -> MorphoPPO`:
     - Replicate the body of `OnPolicyRunner._construct_algorithm()` (`rsl_rl/rsl_rl/runners/on_policy_runner.py`), which calls `resolve_rnd_config`, `resolve_symmetry_config`, instantiates the policy via `eval(self.policy_cfg.pop("class_name"))`, then instantiates the algorithm via `eval(self.alg_cfg.pop("class_name"))`, and finally calls `alg.init_storage(...)`. Add any additional steps for algorithm construction here. 
     - In the custom override, pass any additional constructor arguments required by the custom policy (e.g., `morpho_cfg`) alongside the standard `(obs, obs_groups, num_actions, **policy_cfg)` signature. `HIMOnPolicyRunner._construct_algorithm()` in `himloco/himloco/runners/him_on_policy_runner.py` follows this exact pattern, forwarding `encoder_cfg` as an extra positional argument to `HIMActorCritic`.
   - **Override `save(self, path: str, infos: dict | None = None) -> None`**: 
     - Extend the base `save()`, which writes `model_state_dict` and `optimizer_state_dict` into a `torch.save` dict. 
     - Add an entry for every auxiliary optimizer or sub-module whose state must survive a checkpoint, e.g. `"morpho_optimizer_state_dict": self.alg.morpho_optimizer.state_dict()`. `HIMOnPolicyRunner.save()` demonstrates this by persisting `"estimator_optimizer_state_dict"` alongside the main optimizer.
   - **Override `load(self, path: str, load_optimizer: bool = True, map_location=None) -> dict`**: After restoring `model_state_dict` and `optimizer_state_dict` from the loaded dict, additionally call `load_state_dict` for each auxiliary component keyed in the checkpoint, e.g. `self.alg.morpho_optimizer.load_state_dict(loaded_dict["morpho_optimizer_state_dict"])`. `HIMOnPolicyRunner.load()` follows this pattern for the estimator optimizer.

2. **Learning Implementation** (`runners/morpho_on_policy_runner.py`):

   - **Override `learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None`** only when the learning and data collection logic differs from the standard rollout-update loop
   - Inject **pre-loop** logic (e.g., warm-up epochs for an auxiliary network using a supervised signal) before `for it in range(start_iter, tot_iter):`.
   - Inject **intra-rollout** logic (e.g., refreshing morphological parameters each episode reset) inside the `with torch.inference_mode():` block after each `self.env.step()` call.
   - Inject **post-update** logic (e.g., annealing a regularisation coefficient based on a metric in `loss_dict`) immediately after `loss_dict = self.alg.update()`.
   - If none of these extension points are required, do **not** override `learn()`.

3. **Update Logging** (`runners/morpho_on_policy_runner.py` or `algorithms/morpho_ppo.py`):

   - The base `OnPolicyRunner.log()` already iterates over every key in `loss_dict` returned by `update()` and writes each to TensorBoard via `self.writer.add_scalar(f"Loss/{key}", value, locs["it"])`. It is therefore sufficient to ensure every auxiliary loss or metric is included in the `dict` returned by the custom `update()` (e.g., `loss_dict["morpho_reg"] = mean_morpho_reg_loss`). No override of `log()` is needed for scalar metrics.
   - **Override `log(self, locs: dict, width: int = 80, pad: int = 35) -> None`** only when non-scalar summaries are needed, such as writing parameter histograms via `self.writer.add_histogram("Morpho/params", params_tensor, locs["it"])` or custom console output is required. Always call `super().log(locs, width, pad)` first to preserve the standard metrics.

4. **Implement Custom Storage** (`storage/morpho_rollout_storage.py`):
    - **Create** `MorphoRolloutStorage(RolloutStorage)` inheriting from `rsl_rl.storage.RolloutStorage`
    - **Extend the inner `Transition` class**: Add the new field (e.g., `self.morpho_params: torch.Tensor | None = None`) and clear it in `clear()`. This inner class is not inherited — it must be redefined in full, as shown in `HIMRolloutStorage.Transition`.
    - **Override `__init__`**: Call `super().__init__(...)` then allocate the additional buffer (e.g., `self.morpho_params = torch.zeros(num_transitions_per_env, num_envs, num_morpho_params, device=device)`). `HIMRolloutStorage.__init__()` shows this pattern, allocating `self.next_observations` as an additional `TensorDict` buffer.
    - **Override `add_transitions(self, transition)`**: Call `super().add_transitions(transition)`, which writes all standard fields and **increments `self.step`**. Then copy the extra field at index `self.step - 1` (the slot just written), e.g. `self.morpho_params[self.step - 1].copy_(transition.morpho_params)`. `HIMRolloutStorage.add_transitions()` demonstrates this pattern.
    - **Override `mini_batch_generator(self, num_mini_batches, num_epochs)`**: Flatten the extra buffer alongside the standard ones (`self.morpho_params.flatten(0, 1)`) and yield it as an additional element in the mini-batch tuple. The consuming `MorphoPPO.update()` loop must unpack it in the same order. `HIMRolloutStorage.mini_batch_generator()` illustrates this by yielding `next_obs_batch` as the second element of the tuple.
    - Implement only when the algorithm needs to buffer additional per-step data beyond the standard fields (`observations`, `actions`, `rewards`, `dones`, `values`, `log_probs`, `mu`, `sigma`)

5. **Implement Custom Algorithm** (`algorithms/morpho_ppo.py`):

   - **Create** `MorphoPPO(PPO)` inheriting from `rsl_rl.algorithms.PPO`.
   - **`__init__(self, policy, morpho_lr: float = 1e-4, morpho_reg_coef: float = 0.01, **kwargs)`**: 
     - Call `super().__init__(policy, **kwargs)` to initialise all standard PPO components (the main `self.optimizer` over `policy.parameters()`, `clip_param`, `gamma`, etc.).
     - Instantiate a **separate** `self.morpho_optimizer = optim.Adam(self.policy.morpho_embedding.parameters(), lr=morpho_lr)` so the auxiliary network's learning rate is controlled independently. Store `self.morpho_reg_coef`.
     - Reassign `self.transition = MorphoRolloutStorage.Transition()` so the extended fields are available during rollouts. `HIMPPO.__init__()` in `himloco/himloco/algorithms/him_ppo.py` follows this same pattern.
   - **Override `init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape)`**:
     - Instantiate `MorphoRolloutStorage` instead of the base `RolloutStorage`, preserving the same call signature `(training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device)`. `HIMPPO.init_storage()` demonstrates this substitution with `HIMRolloutStorage`.
   - **Override `process_env_step(self, obs, rewards, dones, extras)`**:
     - Capture any additional per-step data **before** delegating to `super().process_env_step(obs, rewards, dones, extras)` (which calls `self.storage.add_transitions(self.transition)` and clears the transition).
   - **Override `update(self) -> dict[str, float]`**:
     - Replicate the mini-batch update loop from `ppo.py`.
    - After computing the standard PPO losses, compute the auxiliary loss (e.g., `morpho_reg_loss = self.morpho_reg_coef * embed.pow(2).mean()`). Call `self.optimizer.zero_grad()` and `self.morpho_optimizer.zero_grad()`, call `.backward()` on the combined loss, clip gradients via `nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)`, then call `.step()` on both optimizers.
    - Include the auxiliary loss scalar in the returned dict: `loss_dict["morpho_reg"] = mean_morpho_reg`. `HIMPPO.update()` in `himloco/himloco/algorithms/him_ppo.py` illustrates this pattern with `estimation_loss` and `swap_loss`.

6. **Implement Policy Module** (`modules/morpho_actor_critic.py`):

   - **Create** `MorphoActorCritic(ActorCritic)` inheriting from `rsl_rl.modules.ActorCritic`.
   - **`__init__(self, obs, obs_groups, num_actions, morpho_cfg, **kwargs)`**:
     - Call `super().__init__(obs, obs_groups, num_actions, **kwargs)` to build the standard actor/critic MLPs and observation normalisers.
     - Instantiate the `MorphoEmbedding` sub-module (see step 6). Then **rebuild** `self.actor` as an `MLP` (from `rsl_rl.networks`) whose input dimension is `num_actor_obs + embed_dim`, since the parent constructed `self.actor` sized for `num_actor_obs` alone.
   - **Override `_update_distribution(self, obs: TensorDict) -> None`**:
     - Retrieve the auxiliary input from the `TensorDict` (e.g., `morpho_params = obs["morpho"]`), pass it through `self.morpho_embedding` to obtain `embed`, concatenate with the normalised standard actor observation via `torch.cat([actor_obs, embed], dim=-1)`, then call `self.actor(augmented_obs)` and construct `self.distribution = Normal(mean, std)`
     - The base `act()` calls `_update_distribution()` then `self.distribution.sample()`, so overriding this single method is the minimal and preferred change.
   - **Override `act_inference(self, obs: TensorDict) -> torch.Tensor`**:
     - Mirror the `_update_distribution` augmentation logic but return the actor output deterministically (mean of distribution), used during deployment.
     - The base implementation in `actor_critic.py` returns `self.actor(obs)` directly without any embedding, so this must be overridden.
   - **Override `evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor`**:
     - Override only if the critic should also receive the morphological embedding.
     - If the critic relies solely on standard privileged observations, fall back to `return super().evaluate(obs, **kwargs)`.

7. **Implement Network Elements** (`modules/morpho_embedding.py`):

   - **Create** `MorphoEmbedding(nn.Module)`: a standalone `torch.nn.Module` responsible for encoding the raw morphological parameter vector into a fixed-size embedding.
   - **`__init__(self, num_morpho_params: int, embed_dim: int, hidden_dims: list[int], activation: str)`**:
     - Build the encoder using `rsl_rl.networks.MLP` or a custom `nn.Sequential` that maps inputs of shape `[batch, num_morpho_params]` to `[batch, embed_dim]`
   - **`forward(self, morpho_params: torch.Tensor) -> torch.Tensor`**:
     - Pass the input through the encoder and return the embedding tensor of shape `[batch, embed_dim]`
   - Instantiate this inside `MorphoActorCritic.__init__()` as `self.morpho_embedding = MorphoEmbedding(...)`. Because it is a named sub-module, its parameters are automatically included in `policy.state_dict()` and `policy.parameters()`, and are therefore covered by the base runner's `save()` / `load()` at no extra cost — unless a separate optimizer is used for it (see step 4).

---


### b. Implementing HIMLoco

The implementation of HIMLoco (History Information Model) at `@tron1-rl-isaaclab-cozum/himloco/himloco/` serves as an excellent case study applying these guidelines:

*   **Runner (`HIMOnPolicyRunner`)**: Inherits from `OnPolicyRunner`. It ensures the `HIMPPO` algorithm and `HIMActorCritic` policy are used. Crucially, it overrides `save()` and `load()` to handle the state of the `HIMEstimator`'s separate optimizer, which is trained alongside the PPO actor-critic.
*   **Algorithm (`HIMPPO`)**: Inherits from `PPO`. 
    *   It uses a custom `HIMRolloutStorage` and overrides `process_env_step()` to store `next_observations`, which are required for temporal self-supervised learning.
    *   The `update()` method is significantly modified. It calls `self.policy.estimator.update()` to calculate two new auxiliary losses: `estimation_loss` (an MSE loss for predicting ground truth velocity) and `swap_loss` (a self-supervised contrastive loss using the Sinkhorn-Knopp algorithm, similar to SwAV).
*   **Policy (`HIMActorCritic`)**: Inherits from `ActorCritic`. It acts as a wrapper. Before passing data to the standard actor MLP, it extracts the `obsHistory` from the environment's `TensorDict` and passes it through the `HIMEstimator`. The resulting estimated velocity and latent history vector are concatenated with the raw observations to form the augmented input for the actor.
*   **Network Elements (`HIMEstimator`)**: A custom `nn.Module` representing the core of the HIMLoco approach. It processes observation history using convolutional or dense layers to generate a latent representation, implementing the specific forward passes needed for both the velocity prediction and the contrastive clustering loss.

## 4. Design and Controller co-optimisation

<!-- *(This section is a placeholder for the implementation details of the RL algorithm dedicated to the co-optimization problem, including how morphological parameters and the control policy will be jointly updated.)* -->

---

## 5. Environment Lifecycle in IsaacLab

This section provides the relevant background required for robot design and controller co-optimisation using IsaacLab. The full lifecycle of a simulated environment from configuration through initialisation to runtime is introduced and the procedure for physics parameters and robot geometry modification is discussed. The section concludes with a recommendation for a hybrid strategy for jointly optimising robot morphology and locomotion policy using Evolutionary Algorithms (EA) and PPO.

---

### 5.1 OpenUSD Theory: A Working Overview

IsaacLab uses **OpenUSD** (Universal Scene Description) as its native scene representation format. OpenUSD is a language agnostic scene and object description format used extensively in the animation and robotics industries. IsaacLab leverages OpenUSD as a universal scene description format before PhysX takes over simulation.

#### 5.1.1 Stage, Layer, and Scene Composition

**`UsdStage`** is the single authoritative view of the scene. It is a composed, in-memory object assembled from one or more file-backed layers. The stage is not a file itself — it is the result of composing multiple files according to USD's composition rules.

**`SdfLayer`** is the file-level container. USD supports two formats: `.usda` (human-readable ASCII) and `.usdc` (binary crate format). A single USD asset such as `SF_TRON1A.usd` is typically the *root layer* of a composition that includes several sub-layers — for example, separate layers for geometry, physics, and sensors. When the stage is opened, all contributing layers are parsed and their values are composed according to the **opinion strength** rules. An **opinion** is USD's term for a single layer's assertion about the value of a property or metadata field. Since OpenUSD composes multiple layers together to create the stage, multiple layers may attempt to override the value of the same attribute. Each layer possess an opinion and opinion resolution is performed using a fixed **opinion strength** hierarchy (from strongest to weakest): **local opinions → reference opinions → payload opinions → inherited opinions → built-in fallbacks**.

**Composition Arcs** are the mechanism by which layers and prims reference one another. The most common arc is a **reference** — a statement that the content of an external USD file should be inserted at a given prim path. This is how `_spawn_from_usd_file()` ([`IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:314`](IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py)) adds a robot to the scene: it calls `create_prim(prim_path, usd_path=usd_path, ...)` ([`IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py:44`](IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py)), which inserts a reference arc to the robot's USD file at the path returned by `GridCloner`.

**`SdfPath`** is the hierarchical addressing scheme for all scene objects, following a Unix-style path notation (e.g., `/World/envs/env_0/Robot/base_Link`). The `GridCloner` used by `InteractiveScene` ([`IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:125`](IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py)) produces environment-scoped paths of the form `/World/envs/env_{i}/Robot`, giving each parallel environment its own isolated prim subtree.

**`UsdPrim`** is the fundamental addressable node in the scene graph. Every robot link, joint, and sensor is a prim. Prims carry typed schemas and API schemas that describe their physical behaviour. The `Articulation._initialize_impl()` method ([`IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:1506`](IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py)) searches the prim subtree for a prim bearing `UsdPhysics.ArticulationRootAPI` to anchor the PhysX articulation.

**`UsdAttribute`** is a typed, namespaced property on a prim. Attributes carry opinions from layers and are composed according to USD's value resolution rules. Physics attributes such as `drive:angular:physics:stiffness` etc. are USD attributes that PhysX reads at `sim.reset()` time. Refer to [Section 5.1.2](5.1.2-physics-schema) for more information regarding physics simulation.

Within an IsaacLab session, the current stage is accessed via `omni.usd.get_context().get_stage()` or through the `use_stage(self.sim.get_initial_stage())` context manager used during scene construction (line 141 of `manager_based_env.py`). The `AssetBase.__init__()` method ([`IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py:74`](IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py)) also holds a direct reference as `self.stage = get_current_stage()`. The reference to the stage variable in IsaacLab may be used to access specific prims, update prim or stage properties, perform runtime composition etc before the simulation is handled over to PhysX.

#### 5.1.2 Physics Schemas

USD physics is defined by a set of **API schemas** applied to prims. In object-oriented terms, a **typed schema** defines *what a prim is* (e.g. `Mesh`, `Xform`, `Capsule` — analogous to a class). An **API schema** defines *capabilities a prim has*, independently of its type — analogous to an interface or mixin. API schemas can, thus, be applied to any prim without changing its type. The following schemas govern articulation behaviour in PhysX:

**`UsdPhysics.ArticulationRootAPI`**: Applied to the root prim of a robot (e.g., `base_Link`). Marks the entire prim subtree as a reduced-coordinate articulation, enabling PhysX to use the Featherstone algorithm for efficient dynamics. Detected by `Articulation._initialize_impl()` at line 1525 of `articulation.py`. Configured in IsaacLab via `ArticulationRootPropertiesCfg` ([`IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py`](IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py)), which is passed as `articulation_props` in `UsdFileCfg` (e.g., `solefoot_cfg.py:23`).

**`UsdPhysics.RigidBodyAPI`**: Applied to each link, enabling mass and inertia simulation. Every robot link that contributes to dynamics must have this schema.

**`UsdPhysics.MassAPI`**: Provides explicit mass (`physics:mass`), centre of mass (`physics:centerOfMass`), and inertia tensor (`physics:diagonalInertia`, `physics:principalAxes`) overrides on a rigid body prim. If absent, PhysX computes these from geometry density. At runtime, mass and inertia are updated via `root_physx_view.set_masses()` and `root_physx_view.set_inertias()` (used in `randomize_rigid_body_mass` at [`IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:286`](IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py)).

**`UsdPhysics.CollisionAPI`**: Applied to geometry prims to mark them as collision shapes. Collision geometry is **cooked** (pre-processed for narrow-phase collision) by PhysX at load time (during `sim.reset()`). This cooking is irrevocable for the lifetime of the simulation — collision geometry, and by extension link lengths and joint attachment points, **cannot be modified at runtime**. This is the fundamental reason why geometric morphology changes require a new USD load.

**`UsdPhysics.DriveAPI`**: Applied to joint prims, defining the PD drive for each degree of freedom. The force law is:

```
F = Kp × (q* − q) − Kd × (q̇* − q̇)
```

where `Kp` is `drive:angular:physics:stiffness`, `Kd` is `drive:angular:physics:damping`, `q*` is `drive:angular:physics:targetPosition`, and `q̇*` is `drive:angular:physics:targetVelocity`.

**IsaacLab manifestations**:
- For `ImplicitActuatorCfg` ([`isaaclab/actuators/actuator_cfg.py`](IsaacLab/source/isaaclab/isaaclab/actuators/actuator_cfg.py)): the `stiffness` and `damping` fields are written directly as `DriveAPI` attributes on the joint prim at spawn time. PhysX then applies the force law natively.
- For `IdentifiedActuatorCfg` ([`tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/actuators/actuator_cfg.py:15`](tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/actuators/actuator_cfg.py)): the USD `DriveAPI` stiffness and damping are set to zero; torque is computed in the actuator model by `IdentifiedActuator.compute()` and applied as a feed-forward force command. The `saturation_effort`, `friction_static`, `activation_vel`, and `friction_dynamic` fields model physical actuator non-linearities absent from the standard DriveAPI.

#### 5.1.3 The Fabric Interface and Runtime Constraints

The **Fabric interface** is a USD/PhysX integration layer that activates when `sim.reset()` is called ([`IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:173`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py)). Fabric takes ownership of all live physics state. It maintains a high-performance, GPU-resident data store that mirrors the PhysX simulation state. Rather than reading from the USD stage on every step, the simulation reads from and writes to the Fabric store directly via the PhysX Tensor API. This is what makes GPU-accelerated parallel simulation possible.

After `sim.reset()` fires, if you attempt to modify a USD prim attribute at runtime — for example, `prim.GetAttribute("drive:angular:physics:stiffness").Set(25.0)` — Fabric intercepts and discards the write. The attribute value in the USD stage changes, but the running simulation is unaffected. USD attribute writes are silently ignored once Fabric is active. This constraint only applies to *runtime modification* of USD attributes after simulation has started. It does **not** prevent you from setting robot configuration in USD files. The correct workflow for robot properties update is:

| Phase | Mechanism | USD file values |
|---|---|---|
| Before `sim.reset()` | USD stage parsed by PhysX | **Fully effective — this is how you configure the robot** |
| Prestartup events window | `EventManager.apply(mode="prestartup")` (line 161–162) | USD prim attributes can be written safely here |
| After `sim.reset()` | Fabric / PhysX Tensor API | USD attributes ignored; use `write_*_to_sim()` |
| After stop + new `sim.reset()` | USD re-parsed | USD values effective again |


The pre-Fabric prestartup window (created specifically for USD-level domain randomisation, as noted in the code comment at `manager_based_env.py:155–157`) allows event functions to modify USD prim attributes before simulation starts — for example, randomising mesh scale or initial joint drive parameters at environment launch. After Fabric activation, physics parameters that must change between episodes must be updated via the PhysX Tensor API methods exposed by `Articulation.root_physx_view`. These are documented in §5.6.

#### 5.1.4 TRON1A USD Sub-layer Architecture

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

### 5.2 Configuration Hierarchy

The path from a Python configuration object to a simulated robot involves several distinct stages of resolution. The following steps lead to the spawning of simulation entities til they are ready for usage:

**Step 1 — Task configuration** (`tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py:121`):

```python
env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
    task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
)
```

`parse_env_cfg()` resolves the registered task name to a concrete `SFEnvCfg` instance ([`tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:963`](tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py)) containing `decimation=4`, `episode_length_s=40.0`, `sim.dt=0.005`.

**Step 2 — Scene configuration** (`limx_base_env_cfg.py:34`):

`SFEnvCfg` holds a `SFSceneCfg(InteractiveSceneCfg)` as `self.cfg.scene`. `SFSceneCfg` declares `robot: ArticulationCfg = MISSING` at `limx_base_env_cfg.py:71` — a placeholder that task-specific configurations replace with a concrete instance such as `SOLEFOOT_IDENTIFIED_CFG` from `solefoot_identified_cfg.py:101`.

**Step 3 — Environment construction** (`gym.make()` → `ManagerBasedEnv.__init__()` → `InteractiveScene.__init__()`):

`InteractiveScene` iterates over all entities declared in `SFSceneCfg`. For the robot entity, it instantiates `Articulation(cfg=robot_cfg)`, which inherits from `AssetBase`.

**Step 4 — Prim spawning** (`asset_base.py:84`):

```python
self.cfg.spawn.func(
    self.cfg.prim_path,
    self.cfg.spawn,
    translation=self.cfg.init_state.pos,
    orientation=self.cfg.init_state.rot,
)
```

`cfg.spawn.func` resolves to `spawn_from_usd()` ([`IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:38`](IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py)), which calls `_spawn_from_usd_file(prim_path, cfg.usd_path, ...)` (`from_files.py:268`). For URDF sources, `spawn_from_urdf()` (`from_files.py:80`) first converts the URDF to USD via `UrdfConverter(cfg)` then calls `_spawn_from_usd_file(prim_path, urdf_loader.usd_path, ...)`. In both cases, `_spawn_from_usd_file()` calls `create_prim(prim_path, usd_path=usd_path, ...)` (`from_files.py:314`), inserting a **reference arc** into the live USD stage. This is the moment the robot exists in the scene graph.

**Step 5 — Physics initialisation** (`sim.reset()` → timeline PLAY → `_initialize_callback()`):

When `sim.reset()` fires, PhysX parses the USD stage. All `MassAPI`, `DriveAPI`, `CollisionAPI`, and `ArticulationRootAPI` values — including link lengths encoded in geometry, mass, inertia, stiffness, and damping — are read from the USD stage and become the initial physics state. `Articulation._initialize_impl()` (`articulation.py:1506`) then creates `root_physx_view = self._physics_sim_view.create_articulation_view(...)` (line ~1547), the PhysX Tensor API handle through which all subsequent runtime physics updates are issued.

---

### 5.3 Environment Lifecycle

The complete lifecycle of a `ManagerBasedRLEnv` environment from construction to teardown proceeds through the following stages.

#### Construction Phase

All code executes within `ManagerBasedEnv.__init__()` ([`IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:77`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py)):

1. **`SimulationContext` created** (`manager_based_env.py:104`): Initialises the Isaac Sim application context, rendering pipeline, and physics engine parameters (`sim.dt`, `sim.gravity`, etc.).

2. **`InteractiveScene` created** (`manager_based_env.py:142`): Iterates over all entities in `SFSceneCfg`. For each asset, calls `cfg.spawn.func()` to insert reference arcs into the USD stage. `GridCloner` runs during this phase, producing prim paths (`/World/envs/env_{i}/Robot`) for all `num_envs` parallel environments. The `_articulations` dictionary ([`interactive_scene.py:125`](IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py)) is populated with all `Articulation` instances.

3. **`EventManager` created** (`manager_based_env.py:158`): Created *before* `sim.reset()` specifically to allow USD-level randomisation events. The Events Manager is created before the simulation starts to allow USD-related randomization events that must happen before the simulation starts. Example: randomizing mesh scale"*.

4. **Prestartup events applied** (`manager_based_env.py:161–162`): `event_manager.apply(mode="prestartup")` fires all `EventTermCfg` instances registered under the `"prestartup"` mode. These event functions can safely write to USD prim attributes at this point, before Fabric activates.

5. **`sim.reset()` called** (`manager_based_env.py:173`): The timeline transitions to PLAY. This triggers timeline callbacks. `AssetBase._initialize_callback()` ([`asset_base.py:304`](IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py)) fires for every asset, which calls `_initialize_impl()` if `not self._is_initialized`. For the robot articulation, this creates `root_physx_view`. Fabric activates after which USD prim attribute writes are ignored.

6. **`scene.update()`** (`manager_based_env.py:178`): Pre-populates sensor and asset data buffers so that the observation manager can initialise with valid tensors.

7. **`load_managers()` called** (`manager_based_env.py:180`): Creates the six MDP managers in order (Refer to [`manager_based_rl_env.py:109`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py) for more details):
   - `CommandManager` (`manager_based_rl_env.py:113`)
   - `ActionManager` and `ObservationManager` via `manager_based_env.py317:320`
   - `TerminationManager` (`manager_based_rl_env.py:121`)
   - `RewardManager` (`manager_based_rl_env.py:124`)
   - `CurriculumManager` (`manager_based_rl_env.py:127`)
   - Startup events applied: `event_manager.apply(mode="startup")` (`manager_based_rl_env.py:134–135`)

#### Episode Loop

Once construction is complete, `OnPolicyRunner.learn()` ([`rsl_rl/rsl_rl/runners/on_policy_runner.py:62`](rsl_rl/rsl_rl/runners/on_policy_runner.py)) drives the training loop. A high-level overview follows here; §5.4 covers the step loop in detail.

```
for iteration in range(num_iterations):
    collect rollout (num_steps_per_env × env.step())
    compute returns
    loss_dict = alg.update()       ← PPO gradient step
    ← EA hook point here (§5.8)
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

### 5.4 Step Loop and MDP Cycle

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

1. **Decimation loop** (`decimation=4` times): applies action → calls `sim.step()` → advances physics by `dt=0.005 s`. Net policy step: 4 × 0.005 = 0.02 s = 50 Hz policy rate.
2. **Scene update**: sensors and asset data buffers refreshed.
3. **Termination check**: environments exceeding `episode_length_s=40.0` or hitting fall conditions are flagged.
4. **`_reset_idx(env_ids)`** (line 349): for flagged environments — fires `"reset"` domain randomisation events, resets scene buffers, resets all six MDP managers.
5. **Reward computation**: `RewardManager` accumulates scalar reward.
6. **Observation computation**: `ObservationManager` constructs the `TensorDict` returned to the runner.

---

### 5.5 Physics Parameter Modification

Upon `env.reset()` invocation, the physics state of the simulation is handed over to PhysX from the USD files. Updates to physics properties must be made through PhysX APIs. Therefoer, once the simulation is running, physics parameters can be modified without restarting the simulation provided the changes do not involve geometry or joint topology. All modifications are issued via `Articulation.root_physx_view` using CPU tensors. It must be noted that the requirement of CPU tensors constraint realtime updage of physics paramters. Consequently, the physics parameters can only be updated at environment reset. If, however, the environment is stopped again, for robot morphology updates, USD objects may be directly modified to update the parameters required.

#### Joint Control Parameters Update through PhysX

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

#### Body Inertial Parameters Update through PhysX

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

#### Surface Properties Update through PhysX

```python
# Contact material properties (friction, restitution)
robot.root_physx_view.set_material_properties(materials, env_ids)
```

#### USD Attribute Modification During Stop

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

#### What Cannot Be Changed at Runtime

The following require a full stop/play cycle with a new USD file:

- **Collision geometry** (mesh vertices, convex hull shapes)
- **Link lengths** (URDF/USD geometry transforms)
- **Joint attachment points** (the physical connection between links)
- **Number of DOFs** (joint topology)
- **Addition or removal of joints or links**

---

### 5.6 Robot Morphology Update During Training

This section investigates how the robot's physical morphology can be updated during training.  The full configuration path from `train.py` to the simulation were investigated and five approaches were discovered.

#### Configuration Trace: From `train.py` to Physics

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
- **USD file content** — defines all initial physics values; modify before `sim.reset()`
- **`root_physx_view` tensor writes** — modify runtime physics after `sim.reset()`; no geometry changes possible

#### `MultiUsdFileCfg` at Startup

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

#### Stop → Delete → Spawn → Play

Robot morphological parameter updates require explicitly stopping the simulation, deletion of all robot prims, spawning of new prims from the new USD files, and a simulation restart. This is the only approach capable of changing collision geometry and link lengths during a training run.

For co-optimisation with `m` design candidates and `k` rollout copies per design, the total environment count is `num_envs = m × k`. `spawn_multi_usd_file` assigns design `i` to environments `[i*k … i*k+k-1]` via round-robin (`index % m`), batching all `m×k` prim copies into a single `Sdf.ChangeBlock()` transaction (`wrappers.py:106`), which is the efficiency equivalent of running `m` cloners simultaneously.

After the swap, **three categories of manager state** require attention — `InteractiveScene._articulations`, the action manager, and the event manager. Observation, reward, and termination managers are entirely dynamic (`env.scene[name]` called per step) and require no update.

##### Asset Reference Map

| Manager / Term | Attribute cached at `__init__` | File:Line | Action required |
|---|---|---|---|
| `ActionTerm` (all subclasses) | `self._asset` | `action_manager.py:55` | Re-bind to new articulation |
| `JointPositionAction` | `self._offset` (if `use_default_offset=True`) | `joint_actions.py:195` | Re-read from `new_articulation.data.default_joint_pos` |
| `JointAction` | `self._joint_ids`, `self._num_joints` | `joint_actions.py:65–68` | **No update** — same joint topology means same indices |
| `randomize_rigid_body_material` | `self.asset` | `events.py:197` | Re-bind to new articulation |
| `randomize_rigid_body_mass` | `self.asset` | `events.py:320` | Re-bind to new articulation |
| `randomize_joint_parameters` | `self.asset` | `events.py:685` | Re-bind to new articulation |
| All MDP observation functions | `env.scene[name]` per call | `observations.py` | **No update** — dynamic |
| All MDP reward functions | `env.scene[name]` per call | `rewards.py` | **No update** — dynamic |
| All MDP termination functions | `env.scene[name]` per call | `terminations.py` | **No update** — dynamic |
| Function-based event terms | `env.scene[name]` per call | `events.py` | **No update** — dynamic |

**Sensor behaviour**: sensors stored in `scene._sensors` (e.g. `ContactSensor`) are `AssetBase` subclasses with their own prim paths. When `delete_prim(robot_path)` fires, `_on_prim_deletion` in `asset_base.py:331–347` matches the parent path and calls `_clear_callbacks()` — deregistering the PLAY subscription. When `sim.reset()` then fires the PLAY event, the sensor no longer responds and its `root_physx_view` is never rebuilt. To fix this, `sensor._register_callbacks()` must be called after spawning new prims but before `sim.reset()`, so that the sensor re-subscribes and reinitialises when PLAY fires.

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


**Critical Considerations to Note**:
- `InteractiveScene._articulations` (`interactive_scene.py:125`) is a plain `dict` with no public add/remove API — must be patched directly.
- Sensor callbacks must be re-registered before `sim.reset()`, not after — PLAY fires inside `reset()` and sensors must be subscribed before it fires.
- `replicate_physics=False` is required in `InteractiveSceneCfg` for heterogeneous envs; the carb flag `/isaaclab/spawn/multi_assets` (set by `spawn_multi_asset` at `wrappers.py:125`) triggers a warning if `replicate_physics=True` is detected (`interactive_scene.py:225`).
- All m USD variants must share the same joint topology. `ActionTerm._joint_ids` and `_num_joints` are not updated — if joint names or counts differ, `load_managers()` (`manager_based_env.py:294`) must be called instead, which rebuilds all managers from scratch.

The aforementioned re-spawn sequence allows for any geometry change and complete morphological freedom with any number of design candidates per generation. However, the computational overhead of simulating heterogeneous robots must be taken into account. Seconds of overhead may be expected per generation dominated by `force_load_physics_from_usd()` (`simulation_context.py:640`), which scales with `m×k` prim count and USD mesh complexity.

#### Procedural In-Memory USD Generation

Generate USD content programmatically at runtime using `Usd.Stage.CreateInMemory()`. No pre-existing USD files are required; the stage is constructed entirely from Python. This approach provides maximum flexibility for the EA — any topology, geometry, or physics configuration can be generated on the fly. However, the computational overhead of procedural in momory USD generation is significant, making the system unviable for immediate use. The method has been documented though for future references.

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf

def create_robot_usd_in_memory(morpho_params: dict) -> str:
    """Generate a robot USD stage from morphological parameters."""
    stage = Usd.Stage.CreateInMemory()

    # Define root xform
    root = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # Define links with parameterised geometry
    for i, (name, length, mass) in enumerate(morpho_params["links"]):
        link = UsdGeom.Cylinder.Define(stage, f"/Robot/{name}")
        link.GetHeightAttr().Set(length)

        # Apply mass
        mass_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass_api.GetMassAttr().Set(mass)

        # Apply collision
        UsdPhysics.CollisionAPI.Apply(link.GetPrim())

    # Define joints with parameterised drive properties
    for joint_name, stiffness, damping in morpho_params["joints"]:
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

---

### 5.7 Recommended Strategy: Hybrid Two-Tier Co-optimisation

The methods for robot morphology update in §5.6 reveal a clean architectural split for design and policy co-optimisation: within-generation policy optimisation (learning policy parameters while the simulation runs for `N` iterations) pairs naturally with between-generation morphology updates (reloading geometry between EA cycles every `N` iteration). The following sub-sections describe the proposed architecture.

#### Architecture Overview

```
EA Generation k
│
├── Generate population USDs
│     generate_population_usds(ea_population[k]) → /tmp/gen_k/variant_{0..N-1}.usd
│
├── Load generated USDs into simulation
│     env.sim.stop()
│     delete_prims("/World/envs/env_*/Robot")
│     reconstruct InteractiveScene with MultiUsdFileCfg(
│         usd_path=["/tmp/gen_k/variant_0.usd", ..., "/tmp/gen_k/variant_{N-1}.usd"],
│         random_choice=False  ← deterministic: env_i → variant_i
│     )
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
1. **`random_choice=False`**: ensures `env_i` always receives `variant_i.usd`, enabling clean fitness-to-individual mapping. With `random_choice=True`, the environment-to-individual assignment would be non-deterministic.
2. **Joint topology constraint**: all USD variants in the list must declare the same joints with the same names. Actuator gains, mass, inertia, link lengths, and collision geometry can vary freely; the joint graph cannot.
3. **Population size vs. environment count**: the number of USD variants equals `num_envs`. If the EA population is smaller than `num_envs`, assign multiple environments per individual and average their fitness scores.
4. **`replicate_physics=False` overhead mitigation**: the 15+ minute init time for large environment counts is avoided by keeping population size ≤ 512 per generation, or by using `replicate_physics=True` and applying per-individual physics overrides post-spawn via `write_joint_stiffness_to_sim()`, `root_physx_view.set_masses()`, etc., for parameters that differ between individuals. Pure geometry differences (link lengths, collision shapes) always require `replicate_physics=False` and a correspondingly smaller population or pre-generation strategy.

#### Sample Implementation

The EA hook point is immediately after `alg.update()` in the runner's `learn()` loop. A custom `MorphoOnPolicyRunner` (following the guidelines in §3) overrides `learn()` to inject generational logic as follows:

```python
# runners/morpho_on_policy_runner.py
class MorphoOnPolicyRunner(OnPolicyRunner):

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
        # 1. Stop
        self.env.unwrapped.sim.stop()

        # 2. Generate USD files for this generation
        usd_paths = []
        for i, individual in enumerate(population):
            usd_path = generate_robot_usd(individual, f"/tmp/gen_{self.generation}/variant_{i}.usd")
            usd_paths.append(usd_path)

        # 3. Delete existing prims
        scene = self.env.unwrapped.scene
        for env_path in scene.env_prim_paths:
            sim_utils.delete_prim(f"{env_path}/Robot")

        # 4. Build new ArticulationCfg with MultiUsdFileCfg
        new_robot_cfg = SOLEFOOT_IDENTIFIED_CFG.replace(
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_paths,
                random_choice=False,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                activate_contact_sensors=True,
            )
        )

        # 5. Spawn new articulation and re-register
        new_articulation = Articulation(cfg=new_robot_cfg)
        self.env.unwrapped.sim.reset()  # _initialize_impl() fires → new root_physx_view
        scene._articulations["robot"] = new_articulation

        # 6. Reset all MDP managers and episode buffers
        self.env.unwrapped.episode_length_buf.zero_()
        self.generation += 1
```

#### Generation of USD Files from EA Parameters

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

### 5.8 Key Classes and Interfaces

All classes mentioned throughout §5 are documented here with full API details. Shorthand names are used freely in subsequent sections.

#### `SimulationContext`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py`
- **Role**: Singleton controller for the Isaac Sim simulation. Extends `isaacsim.core.api.simulation_context.SimulationContext` to add configuration-driven setup, physics/render settings, fabric interface management, and callback orchestration. Provides the canonical play/pause/stop/step/render interface for Isaac Lab environments.
- **Constructor Args**:
  - `cfg` (`SimulationCfg | None`): Simulation configuration (physics dt, gravity, device, render interval, etc.). If `None`, a default `SimulationCfg()` is used.
- **Key Variables**:
  - `cfg` (`SimulationCfg`): The active simulation configuration.
  - `render_mode` (`RenderMode`): Enum controlling what gets rendered: `NO_GUI_OR_RENDERING (-1)`, `NO_RENDERING (0)`, `PARTIAL_RENDERING (1)`, `FULL_RENDERING (2)`.
  - `_initial_stage` (`Usd.Stage`): The USD stage active during scene construction; returned by `get_initial_stage()`.
  - `_fabric_iface`: Fabric (FlatCache) interface handle; `None` if fabric is disabled.
  - `_has_gui` (bool): Whether any GUI (local, livestream, or XR) is active.
- **Key Methods**:
  - `reset(soft=False)`: Re-raises any pending callback exception, calls parent `reset()`, forces the correct CUDA device, initializes kinematic bodies in fabric, and performs warm-up render passes.
  - `step(render=True)`: Advances physics one step. Blocks while the timeline is paused. Optionally renders after stepping.
  - `forward()`: Updates articulation kinematics and pushes data through the fabric interface without stepping physics.
  - `get_initial_stage() -> Usd.Stage`: Returns `_initial_stage`, the stage active during scene construction.
  - `set_render_mode(mode)`: Switches between `FULL_RENDERING` and `PARTIAL_RENDERING` at runtime when a GUI is present.

#### `InteractiveScene`
**File**: `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py`
- **Role**: Parses an `InteractiveSceneCfg` to build the simulation world. Spawns all assets into the USD stage and clones them across `num_envs` environments using `GridCloner`. Provides a unified interface to reset and update all scene entities.
- **Constructor Args**:
  - `cfg` (`InteractiveSceneCfg`): Configuration specifying number of environments, spacing, `replicate_physics`, `filter_collisions`, and all entity configurations declared as attributes.
- **Key Variables**:
  - `_articulations` (`dict[str, Articulation]`): Name-keyed dictionary of all articulation assets in the scene; exposed via the `articulations` property (line 381).
  - `_rigid_objects`, `_sensors`, `_extras` (dicts): Rigid body, sensor, and miscellaneous prim containers.
  - `_terrain` (`TerrainImporter | None`): The terrain importer, or `None` if no terrain configured.
  - `_default_env_origins` (`torch.Tensor`): Shape `(num_envs, 3)` tensor of environment origin positions.
  - `sim` (`SimulationContext`): Reference to the active simulation context singleton.
  - `cloner` (`GridCloner`): Used to replicate the `env_0` source prim across all environments.
- **Key Properties**:
  - `articulations -> dict[str, Articulation]`: Returns `_articulations`.
  - `env_origins -> torch.Tensor`: Terrain origins if a terrain is present, otherwise `_default_env_origins`.
  - `num_envs -> int`: `cfg.num_envs`.
- **Key Methods**:
  - `reset(env_ids=None)`: Iterates over all assets, sensors, and surface grippers and calls their individual `reset(env_ids)`.
  - `clone_environments(copy_from_source=False)`: Calls `GridCloner.clone(...)` to replicate `/World/envs/env_0` to all environment paths and sets `_default_env_origins`.
  - `filter_collisions(global_prim_paths=None)`: Prevents inter-environment physics interactions, excluding global paths such as the ground plane.

#### `ManagerBasedEnv`
**File**: `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`
- **Role**: Base environment class for the manager-based workflow. Creates and owns the `SimulationContext`, `InteractiveScene`, `EventManager`, `ActionManager`, `ObservationManager`, and `RecorderManager`. Does not define rewards or terminations (those are in the RL subclass).
- **Constructor Args**:
  - `cfg` (`ManagerBasedEnvCfg`): Full environment configuration including scene, sim, actions, observations, events, recorders, decimation, and seed.
- **Key Variables**:
  - `sim` (`SimulationContext`): Created or retrieved singleton simulation context.
  - `scene` (`InteractiveScene`): The scene built from `cfg.scene`.
  - `event_manager` (`EventManager`): Constructed before `sim.reset()` to allow `"prestartup"` USD-level randomization events.
  - `action_manager` (`ActionManager`): Processes raw policy actions.
  - `observation_manager` (`ObservationManager`): Computes observations.
  - `extras` (dict): Dictionary for passing auxiliary info alongside observations.
  - `_sim_step_counter` (int): Counts raw physics steps since env creation.
- **Key Properties**:
  - `num_envs -> int`: `scene.num_envs`.
  - `physics_dt -> float`: `cfg.sim.dt`.
  - `step_dt -> float`: `cfg.sim.dt * cfg.decimation`.
  - `device -> str`: `sim.device`.
- **Key Methods**:
  - `load_managers()` (line 180): Instantiates `RecorderManager`, `ActionManager`, `ObservationManager` (in that order) and fires the `"startup"` event mode. Called automatically in standalone mode after `sim.reset()`; in extension mode the user must call this manually.
  - `reset(seed, env_ids, options) -> (obs, extras)`: Calls `_reset_idx(env_ids)`, writes data to sim, optionally re-renders for RTX sensors, computes and returns observations.
  - `close()`: Cleans up managers and simulation resources.

#### `ManagerBasedRLEnv`
**File**: `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- **Role**: RL-specific subclass of `ManagerBasedEnv` that adds `CommandManager`, `TerminationManager`, `RewardManager`, and `CurriculumManager`. Implements the full MDP `step()` loop and `_reset_idx()` with per-episode bookkeeping. Implements `gym.Env` for compatibility with RL libraries.
- **Constructor Args**:
  - `cfg` (`ManagerBasedRLEnvCfg`): Extends `ManagerBasedEnvCfg` with episode length, rewards, terminations, commands, and curriculum configs.
  - `render_mode` (`str | None`): Gymnasium render mode. Defaults to `None`.
- **Key Variables**:
  - `episode_length_buf` (`torch.Tensor`): Shape `(num_envs,)` integer tensor tracking the current step count within each environment's episode. Incremented by 1 on each `step()` call; reset to 0 for terminated envs in `_reset_idx()`. Created before `super().__init__()` so MDP functions can reference it during manager construction.
  - `reward_buf` (`torch.Tensor`): Shape `(num_envs,)` scalar reward for the current step.
  - `reset_buf`, `reset_terminated`, `reset_time_outs` (`torch.Tensor`): Boolean masks indicating which envs need reset, which terminated naturally, and which timed out.
  - `common_step_counter` (int): Global step counter across all envs, used by curriculum.
  - `command_manager` (`CommandManager`): Generates and updates locomotion velocity commands.
  - `termination_manager` (`TerminationManager`): Computes per-env termination conditions.
  - `reward_manager` (`RewardManager`): Computes scalar rewards.
  - `curriculum_manager` (`CurriculumManager`): Adjusts difficulty over training.
- **Key Methods**:
  - `load_managers()` (line 109): Extends parent by first constructing `CommandManager` (observations depend on commands), then calls `super().load_managers()`, then constructs `TerminationManager`, `RewardManager`, `CurriculumManager`, and configures Gym spaces.
  - `step(action) -> (obs, reward, terminated, time_out, extras)` (line 153): Full MDP step. Processes actions, runs `cfg.decimation` physics sub-steps, increments `episode_length_buf` and `common_step_counter`, computes terminations and rewards, resets terminated envs via `_reset_idx()`, updates commands and interval events, then returns observations.
  - `_reset_idx(env_ids)` (line 349): Runs curriculum update, resets scene, fires `"reset"` events, resets all managers in order, and zeros `episode_length_buf[env_ids]`.
  - `max_episode_length -> int`: `ceil(episode_length_s / step_dt)`.

#### `AssetBase`
**File**: `IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py`
- **Role**: Abstract base class for all physics-enabled assets (rigid objects, articulations, deformable objects). Manages the lifecycle of USD prim spawning, PhysX handle initialization/invalidation via timeline callbacks, and debug visualization. Subclasses must implement `_initialize_impl()`, `reset()`, `write_data_to_sim()`, and `update()`.
- **Constructor Args**:
  - `cfg` (`AssetBaseCfg`): Configuration with `prim_path`, `spawn` (spawner config), `init_state`, and `debug_vis`.
- **Key Variables**:
  - `_is_initialized` (bool): `False` at construction (line 72); set to `True` by `_initialize_callback()` after the timeline PLAY event fires. Reset to `False` on timeline STOP.
  - `stage` (`Usd.Stage`): Handle to the current USD stage at construction time.
  - `_initialize_handle`: Subscription to the timeline PLAY event.
  - `_invalidate_initialize_handle`: Subscription to the timeline STOP event.
  - `_prim_deletion_callback_id`: Callback ID registered with `SimulationManager` for prim deletion events.
  - `_device` (str): Set during `_initialize_callback` from `SimulationManager.get_physics_sim_device()`.
- **Key Methods**:
  - `_register_callbacks()` (line 267): Subscribes `_initialize_callback` to timeline PLAY (order=10), `_invalidate_initialize_callback` to timeline STOP (order=10), and `_on_prim_deletion` to `IsaacEvents.PRIM_DELETION`. Uses weak references for safe garbage collection.
  - `_initialize_callback(event)` (line 304): Triggered on timeline PLAY. Retrieves backend/device from `SimulationManager` and calls `_initialize_impl()` (overridden by subclasses to create PhysX views and populate data buffers). Sets `_is_initialized = True`.
  - `_invalidate_initialize_callback(event)` (line 324): Triggered on timeline STOP. Sets `_is_initialized = False` and unsubscribes the debug visualization handle.
  - `_on_prim_deletion(prim_path)` (line 331): If the deleted path is "/" or a prefix of this asset's prim path, calls `_clear_callbacks()` to fully unsubscribe all event handles.
  - `is_initialized -> bool`: Property returning `_is_initialized`.
- **Important Logic**: Spawning happens in `__init__` immediately if `cfg.spawn is not None`. Exceptions raised inside `_initialize_impl()` are stored in `builtins.ISAACLAB_CALLBACK_EXCEPTION` to be re-raised on the next `sim.step()` or `sim.reset()` call, since Omniverse silently swallows exceptions in callbacks.

#### `Articulation`
**File**: `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py`
- **Role**: Concrete asset class for articulated rigid-body systems (robots with joints). Manages PhysX `ArticulationView` handles, per-joint actuator models, data buffers for joint/body state, and methods for writing simulation parameters (stiffness, damping, limits, armature, friction) directly into the physics engine.
- **Constructor Args**:
  - `cfg` (`ArticulationCfg`): Includes `spawn`, `init_state`, `actuators` dict, `soft_joint_pos_limit_factor`, and optionally `articulation_root_prim_path`.
- **Key Variables**:
  - `actuators` (`dict[str, ActuatorBase]`): Maps actuator group names to actuator model instances; built during `_process_actuators_cfg()`.
  - `_root_physx_view` (`physx.ArticulationView`): The PhysX tensor API view for the articulation, created in `_initialize_impl()`. Exposed via the `root_physx_view` property.
  - `_data` (`ArticulationData`): Holds all joint/body state tensors and their defaults (positions, velocities, accelerations, limits, stiffness, damping, armature, friction, mass, inertia).
  - `_ALL_INDICES` (`torch.Tensor`): Index tensor `[0, ..., num_instances-1]` for broadcast indexing.
- **Key Properties**:
  - `root_physx_view -> physx.ArticulationView`: Low-level PhysX handle. Setters require CPU tensors.
  - `num_joints -> int`, `num_bodies -> int`, `joint_names -> list[str]`, `body_names -> list[str]`, `is_fixed_base -> bool`.
  - `data -> ArticulationData`: The data buffer container.
- **Key Methods**:
  - `_initialize_impl()` (line 1506): Resolves the articulation root prim path, creates `_root_physx_view`, initializes `_data = ArticulationData(...)`, calls `_create_buffers()` (reads default joint stiffness/damping/armature/friction/limits/mass/inertia from PhysX), then `_process_actuators_cfg()` (instantiates actuator models).
  - `write_joint_stiffness_to_sim(stiffness, joint_ids, env_ids)` (line 652): Updates `_data.joint_stiffness` buffer and calls `root_physx_view.set_dof_stiffnesses(...)` (requires CPU tensor).
  - `write_joint_damping_to_sim(damping, joint_ids, env_ids)` (line 681): Updates `_data.joint_damping` and calls `root_physx_view.set_dof_dampings(...)`.
  - `write_joint_position_limit_to_sim(limits, joint_ids, env_ids, warn_limit_violation)` (line 710): Updates `_data.joint_pos_limits`, clamps `default_joint_pos` to the new limits if needed, calls `set_dof_limits(...)`, and recomputes `soft_joint_pos_limits` using `soft_joint_pos_limit_factor`.
  - `write_joint_armature_to_sim(armature, joint_ids, env_ids)` (line 840): Updates `_data.joint_armature` and calls `set_dof_armatures(...)`.
  - `write_joint_friction_coefficient_to_sim(joint_friction_coeff, ..., joint_ids, env_ids)` (line 871): For Isaac Sim < 5.0 sets static friction via `set_dof_friction_coefficients`; for >= 5.0 patches static/dynamic/viscous components and writes back via `set_dof_friction_properties`.

#### `ArticulationCfg`
**File**: `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py` (line 16)
- **Role**: Configuration dataclass for an `Articulation` asset. Inherits `AssetBaseCfg` (providing `prim_path`, `spawn`, `init_state`, `debug_vis`) and adds articulation-specific settings.
- **Key Fields**:
  - `class_type: type = Articulation`: Tells the asset factory which class to instantiate.
  - `articulation_root_prim_path: str | None = None`: Optional explicit relative path from `prim_path` to the prim carrying `UsdPhysics.ArticulationRootAPI`. If `None`, the class auto-discovers it.
  - `init_state` (`InitialStateCfg`): Nested config with `pos`, `rot` (initial root pose), `lin_vel`, `ang_vel` (initial root velocity), `joint_pos` and `joint_vel` (regex-keyed initial values, default `{".*": 0.0}`).
  - `soft_joint_pos_limit_factor: float = 1.0` (line 56): Fraction of the full joint range to use as "soft" limits in `ArticulationData.soft_joint_pos_limits`. Value of 0.9 means soft limits cover 90% of hardware range. Used for termination and reward shaping; does not affect physics.
  - `actuators: dict[str, ActuatorBaseCfg] = MISSING` (line 66): **Required.** Maps actuator group names to their configuration objects (`ImplicitActuatorCfg`, `IdentifiedActuatorCfg`, etc.). Each entry specifies `joint_names_expr`, stiffness, damping, and limits.

#### `UsdFileCfg`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py` (line 74)
- **Role**: Configuration for spawning an asset by referencing a USD file. Inherits from `FileCfg` which itself inherits `RigidObjectSpawnerCfg` and `DeformableObjectSpawnerCfg`.
- **Key Fields**:
  - `func: Callable = spawn_from_usd` (line 99): The spawner function called when this config is used.
  - `usd_path: str = MISSING` (line 101): **Required.** Filesystem or Nucleus URL path to the `.usd` / `.usda` / `.usdc` file.
  - `variants: object | dict[str, str] | None = None`: USD variant selections to apply after spawning.
  - `rigid_props` (`RigidBodyPropertiesCfg | None`, inherited): Rigid body physics properties (damping, gravity, velocity limits, etc.).
  - `articulation_props` (`ArticulationRootPropertiesCfg | None`, inherited): Articulation root properties (self-collision, solver iterations).
  - `joint_drive_props` (`JointDrivePropertiesCfg | None`, inherited): Overrides drive properties on all joints. Prefer `ArticulationCfg.actuators` for per-group control.
  - `activate_contact_sensors: bool` (inherited from `RigidObjectSpawnerCfg`): Whether to enable contact reporting on the prim's collision shapes.
  - `scale: tuple[float,float,float] | None = None`: Uniform or anisotropic scale override.

#### `MultiUsdFileCfg`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers_cfg.py` (line 46)
- **Role**: Variant of `UsdFileCfg` that accepts a list of USD file paths and either randomly selects one per spawn or cycles through them in order. Used to introduce morphological diversity across environments at spawn time.
- **Key Fields** (inherits all of `UsdFileCfg`):
  - `func = spawn_multi_usd_file` (line 57): The spawner function (not `spawn_from_usd`).
  - `usd_path: str | list[str] = MISSING` (line 59): A single path string or a list of USD file paths.
  - `random_choice: bool = True` (line 37): If `True`, each spawn randomly selects from the list. If `False`, spawns cycle through the list in order.

#### `UrdfFileCfg`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py` (line 114)
- **Role**: Configuration for spawning an asset from a URDF file. Combines `FileCfg` with `UrdfConverterCfg` (URDF-to-USD conversion settings). The converter produces a cached USD file on first use; subsequent calls use the cached version.
- **Key Fields**:
  - `func: Callable = spawn_from_urdf` (line 132): Dispatches to `spawn_from_urdf()`, which calls `UrdfConverter(cfg)` to produce a USD, then delegates to `_spawn_from_usd_file(...)`.
  - **Inherited from `UrdfConverterCfg`**: `asset_path: str = MISSING` (path to the `.urdf` file), `fix_base: bool = False` (whether to fix the root link), `merge_fixed_joints: bool = False` (collapse fixed joints to reduce the articulation tree), `make_instanceable: bool = True` (enables GPU instancing), `convex_decompose_mesh: bool = False` (use V-HACD for collision meshes).
  - All `FileCfg` properties (`rigid_props`, `articulation_props`, `joint_drive_props`, `scale`, `visual_material`) also apply.

#### `ImplicitActuatorCfg`
**File**: `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_cfg.py`
- **Role**: Configuration for an implicit PD actuator where PD control is handled entirely inside the PhysX physics engine (not computed by Python). Stiffness and damping values are written directly into the simulation's joint drive properties.
- **Key Fields**:
  - `class_type: type = ImplicitActuator`: Marker for the actuator factory.
  - `joint_names_expr: list[str] = MISSING`: List of joint name patterns (strings or regex) this actuator group controls.
  - `stiffness: dict[str,float] | float | None = MISSING`: PD proportional gain (Kp). Written to PhysX via `set_dof_stiffnesses`. If `None`, value from USD is used.
  - `damping: dict[str,float] | float | None = MISSING`: PD derivative gain (Kd). Written to PhysX via `set_dof_dampings`. If `None`, value from USD is used.
  - `effort_limit_sim: dict[str,float] | float | None = None`: Maximum effort enforced by the physics solver. If `None`, the value from USD is used.
  - `velocity_limit_sim: dict[str,float] | float | None = None`: Maximum joint velocity enforced by the solver.
  - `friction: dict[str,float] | float | None = None`: Static friction coefficient of the joint. If `None`, value from USD is used.
  - `armature: dict[str,float] | float | None = None`: Added to joint-space inertia diagonal to improve simulation stability.

#### `IdentifiedActuatorCfg`
**File**: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/actuators/actuator_cfg.py` (line 15)
- **Role**: Configuration for the `IdentifiedActuator`, an explicit DC motor model extended with identified friction parameters. Inherits `DCMotorCfg` → `IdealPDActuatorCfg` → `ActuatorBaseCfg`, adding static, dynamic, and viscous friction identification from real hardware. Unlike `ImplicitActuatorCfg`, the USD `DriveAPI` stiffness and damping are set to zero; torque is computed entirely in Python and applied as a feed-forward force command.
- **Key Fields**:
  - `class_type: type = IdentifiedActuator`: Instantiates the `IdentifiedActuator` Python actuator model (not handled by PhysX).
  - `stiffness` (inherited, `MISSING`): PD Kp gain (Nm/rad).
  - `damping` (inherited, `MISSING`): PD Kd gain (Nm·s/rad).
  - `effort_limit` (inherited): Maximum output torque for clipping in the explicit actuator model.
  - `velocity_limit` (inherited): Maximum velocity used by the actuator model.
  - `saturation_effort: float = MISSING` (from `DCMotorCfg`): Peak motor torque (Nm). Torque is clipped to this before the PD computation.
  - `friction_static: float = MISSING`: Constant static friction torque (Nm) applied when joint velocity is below `activation_vel`.
  - `activation_vel: float = MISSING`: Velocity threshold (rad/s) below which static friction is active; above which dynamic friction takes over.
  - `friction_dynamic: float = MISSING`: Dynamic friction coefficient (Nm·s/rad). Friction torque = `friction_dynamic * |velocity|`, opposing motion.
- **Identified per-joint-group values used in `SOLEFOOT_IDENTIFIED_CFG`**: Abad: Kp=55, Kd=13.5, sat=402 Nm; Hip: Kp=80, Kd=13, sat=443 Nm; Knee: Kp=60, Kd=4, sat=560 Nm; Ankle: Kp=10, Kd=0.5, sat=402 Nm.

#### `EventManager`
**File**: `IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py`
- **Role**: Manages a collection of event terms grouped by mode. Applies the appropriate terms when triggered by the environment at different simulation lifecycle points.
- **Constructor Args**:
  - `cfg` (object or `dict[str, EventTermCfg]`): Configuration object or dictionary of event terms.
  - `env` (`ManagerBasedEnv`): The environment instance.
- **Key Variables**:
  - `_mode_term_names` (`dict[str, list[str]]`): Maps mode name to list of term names in that mode.
  - `_mode_term_cfgs` (`dict[str, list[EventTermCfg]]`): Maps mode name to list of `EventTermCfg` objects.
  - `_interval_term_time_left` (list of tensors): Per-environment time remaining before next trigger for each `"interval"` term.
  - `_reset_term_last_triggered_step_id` (list of tensors): Tracks the last global env step at which each `"reset"` term was triggered per environment.
- **Event Modes**:
  - `"prestartup"`: Applied once before `sim.reset()`. Used for USD-level changes (mesh scale, etc.).
  - `"startup"`: Applied once after all managers are created and `sim.reset()` has been called.
  - `"reset"`: Applied on environment reset. Supports `min_step_count_between_reset` to prevent over-frequent triggering.
  - `"interval"`: Applied when per-environment timers expire. The manager handles timing logic internally.
- **Key Methods**:
  - `apply(mode, env_ids=None, dt=None, global_env_step_count=None)`: Main dispatch. For `"interval"` mode: decrements per-env timers by `dt`, triggers terms where timers have expired (≤ 1e-6 s), and samples new random intervals. For `"reset"` mode: checks `min_step_count_between_reset`, triggers only environments that have waited long enough. For all other modes: applies terms to all given `env_ids` unconditionally.
  - `reset(env_ids=None) -> dict`: Resets class-based terms and re-samples interval timers for newly-reset environments.
  - `available_modes -> list[str]`: Returns all configured modes.

#### `OnPolicyRunner`
**File**: `rsl_rl/rsl_rl/runners/on_policy_runner.py` (line 23)
- **Role**: Orchestrates on-policy RL training (PPO). Handles the rollout collection loop, policy update, logging to Tensorboard/W&B/Neptune, checkpoint saving and loading, and optional multi-GPU distributed training.
- **Constructor Args**:
  - `env` (`VecEnv`): The vectorized environment (typically a `ManagerBasedRLEnv` wrapped for RSL-RL).
  - `train_cfg` (dict): Training configuration dict with keys: `algorithm`, `policy`, `num_steps_per_env`, `save_interval`, `obs_groups`, `logger`, and optionally `rnd_cfg`.
  - `log_dir` (`str | None`): Directory for checkpoints and logs.
  - `device` (str): Compute device string (e.g. `"cuda:0"`).
- **Key Variables**:
  - `env` (`VecEnv`): Reference to the environment.
  - `alg` (`PPO`): The algorithm instance created by `_construct_algorithm()`. Contains `alg.policy` (the actor-critic network) and `alg.optimizer`.
  - `num_steps_per_env` (int): Number of environment steps collected per rollout before a policy update.
  - `save_interval` (int): Save a checkpoint every this many learning iterations.
  - `current_learning_iteration` (int): Index of the last completed update iteration.
  - `cur_reward_sum` (`torch.Tensor`, line 80): Shape `(num_envs,)` tensor accumulating rewards step-by-step within the rollout. When an environment's `done` flag fires, its accumulated sum is pushed into the `rewbuffer` deque and reset to 0. Used to compute `"Train/mean_reward"`.
  - `log_dir`, `writer`: Logging directory and summary writer (Tensorboard / W&B / Neptune).
- **Key Methods**:
  - `learn(num_learning_iterations, init_at_random_ep_len=False)` (line 62): Main training loop. For each iteration: runs `num_steps_per_env` rollout steps (inference mode) calling `alg.act(obs)`, `env.step(actions)`, `alg.process_env_step(...)`; accumulates and clears `cur_reward_sum` on episode boundaries; calls `alg.compute_returns(obs)` and `alg.update()`; calls `log()` and `save()` at configured intervals.
  - `_construct_algorithm(obs)` (line 395): Resolves RND and symmetry configs, instantiates the actor-critic policy class (from `policy_cfg["class_name"]`), instantiates the PPO algorithm class (from `alg_cfg["class_name"]`), initializes the rollout storage buffer, and returns the algorithm.
  - `save(path, infos=None)` (line 291): Saves `model_state_dict`, `optimizer_state_dict`, current iteration, and optional infos to a `.pt` file.
  - `load(path, load_optimizer=True, map_location=None)` (line 309): Loads `model_state_dict` and conditionally restores optimizer state and `current_learning_iteration`.
  - `get_inference_policy(device=None) -> callable`: Switches to eval mode and returns `alg.policy.act_inference`.

#### `spawn_from_usd()`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` (line 38)
- **Signature**: `spawn_from_usd(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None) -> Usd.Prim`
- **Decorator**: `@clone` — resolves a prim path regex pattern into a list of concrete paths and handles cloning when multiple paths match.
- **What it does**: Validates that the configured USD file path exists, then delegates to `_spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)` to create the prim, apply all properties from `cfg` (rigid body props, collision props, articulation root props, joint drive props, visual material, scale, semantic tags, variants), and return the prim.

#### `spawn_from_urdf()`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` (line 80)
- **Signature**: `spawn_from_urdf(prim_path: str, cfg: UrdfFileCfg, translation=None, orientation=None) -> Usd.Prim`
- **Decorator**: `@clone`
- **What it does**: Instantiates `converters.UrdfConverter(cfg)` which converts the URDF to a cached USD file, then delegates to `_spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)` to spawn and configure the prim exactly as `spawn_from_usd` would.

#### `_spawn_from_usd_file()`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` (line 268)
- **Signature**: `_spawn_from_usd_file(prim_path: str, usd_path: str, cfg: FileCfg, translation=None, orientation=None) -> Usd.Prim`
- **What it does**: Internal helper shared by both `spawn_from_usd` and `spawn_from_urdf`. Checks `usd_path` exists (with timeout, including a Nucleus `/4.5` → `/5.0` fallback), calls `create_prim(prim_path, usd_path=usd_path, translation, orientation, scale=cfg.scale)` (line 314) if no prim exists at `prim_path`, then applies all `cfg` properties sequentially. If the prim already exists (e.g. in heterogeneous cloning where the env grid is pre-cloned), it skips prim creation and only applies config properties.

#### `create_prim()`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py` (line 44)
- **Signature**: `create_prim(prim_path, prim_type="Xform", position=None, translation=None, orientation=None, scale=None, usd_path=None, semantic_label=None, semantic_type="class", attributes=None, stage=None) -> Usd.Prim`
- **What it does**: Validates that `position` and `translation` are not both provided. Raises `ValueError` if a prim already exists at `prim_path`. Calls `stage.DefinePrim(prim_path, prim_type)`, sets any `attributes` key-value pairs, adds a USD reference arc via `add_usd_reference(...)` if `usd_path` is provided, adds semantic labels if requested, and sets the XForm pose (translate → orient → scale) in USD canonical format. Returns the created prim.

#### `delete_prim()`
**File**: `IsaacLab/source/isaaclab/isaaclab/sim/utils/prims.py` (line 189)
- **Signature**: `delete_prim(prim_path: str | Sequence[str], stage: Usd.Stage | None = None) -> bool`
- **What it does**: Accepts a single path string or a list of paths. Executes the Omniverse Kit `"DeletePrimsCommand"` to remove the prims and all their descendants from the stage. Returns `True` on success. Active `Articulation` assets watching the deleted prim path will have their `root_physx_view` invalidated via the `_on_prim_deletion` callback.

#### `randomize_actuator_gains`
**File**: `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` (line 541)
- **Signature** (`__call__`): `(env, env_ids, asset_cfg: SceneEntityCfg, stiffness_distribution_params: tuple|None, damping_distribution_params: tuple|None, operation: Literal["add","scale","abs"] = "abs", distribution: Literal["uniform","log_uniform","gaussian"] = "uniform")`
- **What it does**: A `ManagerTermBase` callable event term. For each actuator group in the asset, computes the intersection of the actuator's joint indices with `asset_cfg.joint_ids`. If `stiffness_distribution_params` is not `None`: resets stiffness to `default_joint_stiffness` (to avoid compounding randomizations), applies `_randomize_prop_by_op(stiffness, params, ..., operation, distribution)`, stores the result, and if the actuator is an `ImplicitActuator` calls `write_joint_stiffness_to_sim(...)` to push to PhysX (line 648–649). Repeats for damping. Always randomizes from **default** values to ensure idempotent behavior across resets.

#### `randomize_rigid_body_mass`
**File**: `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` (line 286)
- **Signature** (`__call__`): `(env, env_ids, asset_cfg: SceneEntityCfg, mass_distribution_params: tuple[float,float], operation: Literal["add","scale","abs"], distribution: Literal["uniform","log_uniform","gaussian"] = "uniform", recompute_inertia: bool = True, min_mass: float = 1e-6)`
- **What it does**: Gets current body masses from `root_physx_view.get_masses()`, resets selected body masses for `env_ids` to `default_mass`, applies `_randomize_prop_by_op(...)`, clamps to `min_mass`, then calls `root_physx_view.set_masses(masses, env_ids)`. If `recompute_inertia=True`, recomputes the inertia tensor assuming uniform density (scales inertia by the mass ratio) and calls `root_physx_view.set_inertias(...)`. Uses CPU tensors throughout.

#### `randomize_joint_parameters`
**File**: `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` (line 652)
- **Signature** (`__call__`): `(env, env_ids, asset_cfg: SceneEntityCfg, friction_distribution_params: tuple|None, armature_distribution_params: tuple|None, lower_limit_distribution_params: tuple|None, upper_limit_distribution_params: tuple|None, operation: Literal["add","scale","abs"] = "abs", distribution: Literal["uniform","log_uniform","gaussian"] = "uniform")`
- **What it does**: A `ManagerTermBase` callable event term that randomizes low-level PhysX joint properties. For each non-`None` parameter set, applies `_randomize_prop_by_op(default_value.clone(), params, env_ids, joint_ids, operation, distribution)` then calls the corresponding `Articulation.write_*_to_sim(...)` method: `write_joint_friction_coefficient_to_sim`, `write_joint_armature_to_sim`, or `write_joint_position_limit_to_sim`. Always randomizes from **default** values. Requires CPU tensors; recommended to apply at startup/reset rather than as interval events.

#### `SFSceneCfg`
**File**: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py` (line 34)
- **Role**: Scene configuration for the Sole-Foot (SF) biped. Inherits `InteractiveSceneCfg`.
- **Key Fields**:
  - `robot: ArticulationCfg = MISSING` (line 71): **Required.** Must be set by the concrete task class (e.g. assigned to `SOLEFOOT_IDENTIFIED_CFG`).
  - `terrain: TerrainImporterCfg`: A flat plane terrain at `/World/ground` with `static_friction=1.0`, `dynamic_friction=1.0`, marble tile visual material. `collision_group=-1`, `max_init_terrain_level=0`.
  - `light: AssetBaseCfg`: A dome light at `/World/skyLight` with `intensity=750.0` and an HDR sky texture from Isaac Nucleus.
  - `height_scanner: RayCasterCfg`: Ray-cast height sensor attached to `{ENV_REGEX_NS}/Robot/base_Link`. Offset 20 m above, `ray_alignment="yaw"`, 1.6 m × 1.0 m grid at 0.1 m resolution, sampling against `/World/ground`.
  - `contact_forces: ContactSensorCfg`: Contact sensor on all robot links (`{ENV_REGEX_NS}/Robot/.*`), `history_length=4`, `track_air_time=True`, `update_period=0.0` (every physics step).

#### `SFEnvCfg`
**File**: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py` (line 963)
- **Role**: Full RL environment configuration for the SF biped task. Inherits `ManagerBasedRLEnvCfg`.
- **Key Fields**:
  - `scene: SFSceneCfg = SFSceneCfg(num_envs=4096, env_spacing=2.5)`: Scene with 4096 parallel environments at 2.5 m spacing.
  - `observations: ObservationsCfg`: Base lin/ang velocity, projected gravity, joint pos/vel, last action, height scan, velocity commands, etc.
  - `actions: ActionsCfg`: Joint position action on `".*"` joints, `scale=0.25`, `use_default_offset=True`.
  - `commands: CommandsCfg`: Uniform velocity commands for x, y (±1.0 m/s), yaw (±1.0 rad/s), heading (±π), resampled every 12–18 s.
  - `rewards: RewardsCfg`: Full reward configuration (tracking velocity, base height, gait symmetry, energy penalties, contact forces, etc.).
  - `terminations: TerminationsCfg`: Termination conditions (base contact, joint limits, etc.).
  - `events: EventsCfg`: Domain randomization events (mass, friction, gains, push forces, etc.).
  - `curriculum: CurriculumCfg`: Terrain level curriculum, push force curriculum, command velocity curriculum.
  - **`__post_init__` settings** (line 980): `decimation=4` (4 physics steps per env step), `episode_length_s=40.0`, `sim.dt=0.005` (200 Hz physics), `sim.render_interval=8` (render every 8 physics steps), `seed=42`.

#### `SOLEFOOT_CFG`
**File**: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_cfg.py` (line 10)
- **Role**: Baseline articulation configuration for the SF (Sole-Foot) TRON1A robot using simple implicit PD actuators. Loads from the local USD at `../usd/SF_TRON1A/SF_TRON1A.usd`.
- **What's configured**:
  - **Spawn**: `UsdFileCfg` with `rigid_props` (gravity enabled, no linear/angular damping, `max_depenetration_velocity=1.0`), `articulation_props` (self-collision enabled, 4 solver position + velocity iterations), `activate_contact_sensors=True`.
  - **Init state**: Root at z=0.8 m, all joints at 0.0 rad with 0.0 rad/s velocity.
  - **`soft_joint_pos_limit_factor = 0.9`**.
  - **Actuators** (2 groups, all `ImplicitActuatorCfg`): `"legs"` (abad L/R, hip L/R, knee L/R): `effort_limit_sim=80 Nm`, `velocity_limit_sim=25 rad/s`, `stiffness=50`, `damping=2.2`. `"ankles"` (ankle L/R): `effort_limit_sim=80 Nm`, `velocity_limit_sim=25 rad/s`, `stiffness=15`, `damping=0.8`.

#### `SOLEFOOT_IDENTIFIED_CFG`
**File**: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_identified_cfg.py` (line 101)
- **Role**: Physics-identified articulation configuration for the SF TRON1A robot. Uses `IdentifiedActuator` (explicit DC motor + friction model) with parameters identified from real hardware. Same USD and rigid/articulation props as `SOLEFOOT_CFG`.
- **What's configured**:
  - **Spawn**: `UsdFileCfg` pointing to `SF_TRON1A.usd`, same `rigid_props` and `articulation_props` as `SOLEFOOT_CFG`.
  - **`soft_joint_pos_limit_factor = 0.9`**.
  - **Actuators** (4 groups, all `IdentifiedActuatorCfg`): `"abad"` (abad L/R): Kp=55, Kd=13.5, sat=402 Nm, `friction_static=0.3 Nm`, `activation_vel=0.1 rad/s`, `friction_dynamic=0.02 Nm·s/rad`. `"hip"` (hip L/R): Kp=80, Kd=13, sat=443 Nm. `"knee"` (knee L/R): Kp=60, Kd=4, sat=560 Nm, `friction_static=0.8 Nm`. `"ankle"` (ankle L/R): Kp=10, Kd=0.5, sat=402 Nm, `friction_static=0.1 Nm`.

#### `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG`
**File**: `tron1-rl-isaaclab-cozum/exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_identified_cfg.py` (line 113)
- **Role**: Variant of `SOLEFOOT_IDENTIFIED_CFG` that randomizes the robot's foot morphology across environments by randomly selecting from three USD files (sole-foot, point-foot, wheeled-foot variants). Uses identical identified actuators and init state.
- **What's configured**:
  - **Spawn**: `MultiUsdFileCfg` with `usd_path=[usd_path_sf, usd_path_pf, usd_path_wf]` (`SF_TRON1A`, `PF_TRON1A`, `WF_TRON1A`), `random_choice=True`. Same `rigid_props` and `articulation_props`.
  - **Actuators**: Identical 4-group `IdentifiedActuatorCfg` setup as `SOLEFOOT_IDENTIFIED_CFG`. The foot morphology changes but joint layout remains the same across all three USD variants.

---
