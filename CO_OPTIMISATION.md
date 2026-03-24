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
