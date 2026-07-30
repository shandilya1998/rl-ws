# HIMLoco Integration, Compatibility Analysis and Implementation Plan

> Status, verified against the live sources on 2026-07-30. Implemented, by a different mechanism than the one proposed here. This plan called for a `HIMVecEnvWrapper` inside the `himloco` package, wrapping the environment to supply the members the runner expects. The implementation instead subclasses the environment itself, as `HIMManagerBasedRLEnv` in `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/envs/him_env.py`, and registers that class as the gym entry point of every hybrid internal model task, so the missing members are provided by the environment rather than by a wrapper around it. No wrapper of the proposed name exists anywhere in the workspace. Read the interface analysis of sections 1 and 2, which remains accurate and is the only record of these requirements, and disregard the mechanism of section 3. This document was formerly named `TODO_himloco.md`. See [README.md](README.md) for the full register.

This document serves as the master checklist for achieving compatibility between `himloco`, `IsaacLab`, and `rsl_rl` using an inheritance-based approach within the `himloco` package.

## 1. Interface Analysis: Missing & Extra Variables

### A. Expected by HIMOnPolicyRunner (but missing from VecEnv/RslRlVecEnvWrapper)
The following members are accessed by `HIMOnPolicyRunner` but are not present in the standard `IsaacLab` wrapper or `rsl_rl` base class:
- **`num_obs`**: Expected as an integer attribute representing policy observation dimension.
- **`num_privileged_obs`**: Expected as an integer attribute representing critic observation dimension.
- **`num_one_step_obs`**: Expected for history-based observation processing in `HIMActorCritic`.
- **`get_privileged_observations()`**: Method expected to return the critic/privileged observation tensor.
- **`reset()`**: Method call used in `__init__` (not explicitly in `VecEnv` ABC).
- **`unwrapped.episode_length_buf`**: Accessed directly for random initialization.
- **`unwrapped.max_episode_length`**: Accessed directly for random initialization.

### B. Interface Signature Mismatches
- **`step(actions)` Return Signature**:
    - **HIM Runner Expects (5 values):** `obs, privileged_obs, rewards, dones, infos`
    - **IsaacLab Provides (4 values):** `obs_dict (TensorDict), rewards, dones, extras`
- **Observation Format**:
    - **HIM Runner Expects:** Raw `torch.Tensor` for observations.
    - **IsaacLab Provides:** `TensorDict` containing groups like `"policy"` and `"critic"`.

---

## 2. Identified Bugs & Required Fixes in HIMLoco

### HIMOnPolicyRunner (`tron1-rl-isaaclab-cozum/himloco/himloco/runners/him_on_policy_runner.py`)
- [ ] **Fix Typos:** Line 64: `self.self.policy_cfg` -> `self.policy_cfg`.
- [ ] **Fix Dict Pop Logic:** Use `self.policy_cfg.pop("class_name")` and `self.alg_cfg.pop("class_name")` to prevent passing the string to constructors.
- [ ] **Redundant Code:** Remove the duplicated save/log block at the end of the `learn()` method (lines 173-190).
- [ ] **Observation Mapping:** Update `learn()` to extract `"policy"` and `"critic"` from the environment's output.

### HIMEstimator (`tron1-rl-isaaclab-cozum/himloco/himloco/modules/him_estimator.py`)
- [ ] **Slicing Logic:** `HIMEstimator.update` uses hardcoded slices:
    - Velocity: `next_critic_obs[:, num_one_step_obs : num_one_step_obs + 3]`
    - Next Obs: `next_critic_obs[:, 3 : num_one_step_obs + 3]`
    - **Action:** Ensure the Task Config (MDP) observation order matches these slices.

---

## 3. Implementation Plan: Inheritance-Based Bridge

### Step 1: Create `HIMVecEnvWrapper` in `himloco`
Instead of modifying `IsaacLab`, create `himloco/himloco/utils/wrappers.py`.
- [ ] **Inherit from `RslRlVecEnvWrapper`**.
- [ ] **Constructor:**
    - Call `super().__init__(env, clip_actions)`.
    - Set `self.num_obs = self.get_observations()["policy"].shape[1]`.
    - Set `self.num_privileged_obs = self.get_observations()["critic"].shape[1]`.
    - (Optional) Set `self.num_one_step_obs` from config.
- [ ] **Implement `get_privileged_observations()`**: Return `self.get_observations()["critic"]`.
- [ ] **Override `step(actions)`**: (Optional) Bridge the 4-value to 5-value gap to minimize runner changes, OR update the runner to handle the 4-value return.

### Step 2: Update Runner Logic
- [ ] **TensorDict Compatibility:** Ensure the runner can handle the `TensorDict` returned by the wrapper if not flattened/processed by the custom wrapper.

### Step 3: Entry Point Integration (`scripts/rsl_rl/train.py`)
- [ ] **Dynamic Wrapping:**
    ```python
    if args_cli.policy_type == "HIMPPO":
        from himloco.utils.wrappers import HIMVecEnvWrapper
        env = HIMVecEnvWrapper(env)
        runner_cls = HIMOnPolicyRunner
    ```

---

## 4. Configuration Requirements
- [ ] **`agent_cfg`**: Must include `obs_history_len` and `num_one_step_obs`.
- [ ] **Task Config (`PFBaseEnvCfg`)**: 
    - Verify `"critic"` observation group content.
    - Content order: `[One-Step Observations] + [Linear Velocity (3)] + [Other State]`.

## 5. Summary of Analysis Context
- `IsaacLab` uses `ManagerBasedRLEnv` which manages observations via `ObservationManager`.
- Observations are grouped into sets (policy, critic, etc.) defined in `RslRlOnPolicyRunnerCfg.obs_groups`.
- `RslRlVecEnvWrapper` acts as the bridge but lacks the "legacy" attributes (`num_obs`, etc.) that `HIMOnPolicyRunner` was built for.
- `himloco`'s `HIMPPO` and `HIMActorCritic` are tightly coupled to specific observation shapes and indexing, requiring careful alignment in the Task MDP configuration.
