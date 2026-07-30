# RL Environment Comparison: Root Cause Analysis of Velocity Tracking

## 1. Introduction

This report provides a comprehensive, component-by-component comparison between two reinforcement learning (RL) implementations for bipedal locomotion using Proximal Policy Optimization (PPO):
1. **Tron1 (`tron1-rl-isaaclab-cozum`)**: An implementation leveraging the History Information Model (HIM) architecture, aiming to train a robust policy using historical sensor data and implicit state estimation.
2. **Berkeley Humanoid (`isaac_berkeley_humanoid`)**: A highly successful, validated implementation that yields policies with excellent real-world velocity tracking capabilities.

**The Problem:** The policy trained using the Berkeley environment demonstrates good velocity tracking, whereas the Tron1 environment struggles to learn a policy capable of tracking commanded velocities accurately. This document systematically dissects the architecture, observations, actions, curriculums, rewards, and agent configurations to isolate the root causes of this performance discrepancy.

---

## 2. Implementation and Architecture

Both projects are built on top of the **Isaac Lab** and **RSL-RL** frameworks. They share a similar manager-based MDP design pattern but differ significantly in their configuration and architectural paradigms.

*   **Tron1 Implementation**: 
    *   Environment configuration: `SFHIMEnvCfg` within `cfg/SF/limx_base_env_cfg.py`.
    *   Architecture: **HIM (History Information Model)**.
*   **Berkeley Implementation**: 
    *   Environment configuration: `LocomotionVelocityRoughEnvCfg` within `velocity_env_cfg.py`.
    *   Architecture: Standard Markovian Actor-Critic.

---

## 3. Observations

### 3.1 Individual Observations List

#### **Tron1 Implementation (`HIMObservationsCfg`)**

**Actor (Policy & History Groups):**
*   `base_ang_vel`: Angular velocity of the base.
*   `proj_gravity`: Gravity vector projected into the base frame.
*   `joint_pos`: Relative joint positions.
*   `joint_vel`: Relative joint velocities.
*   `last_action`: The action output from the previous control step.
*   `velocity_commands`: Linear (XY) and Angular (Z) velocity targets.
*   *(Note: `base_lin_vel` is explicitly commented out in the Actor config to force HIM inference)*.

**Critic (Privileged Group):**
*   `base_lin_vel`: Ground truth linear velocity.
*   `base_ang_vel`: Angular velocity.
*   `proj_gravity`: Projected gravity.
*   `joint_pos`: Relative joint positions.
*   `joint_vel`: Relative joint velocities.
*   `last_action`: Last action output.
*   `velocity_commands`: Velocity targets.
*   `heights`: Terrain height scan (from `height_scanner`).
*   `robot_joint_torque`: Instantaneous joint torques.
*   `robot_joint_acc`: Instantaneous joint accelerations.
*   `feet_lin_vel`: Linear velocity of the feet (ankles).
*   `robot_mass`: Total mass of the robot.
*   `robot_inertia`: Moments of inertia of the chassis.
*   `robot_joint_pos`: Raw joint positions.
*   `robot_joint_stiffness`: Actuator $K_p$ gains.
*   `robot_joint_damping`: Actuator $K_d$ gains.
*   `robot_pos`: World position.
*   `robot_vel`: World velocity.
*   `robot_material_properties`: Friction and restitution.
*   `feet_contact_force`: 3D contact forces on the feet.

#### **Berkeley Implementation (`ObservationsCfg`)**

**Actor (Policy Group):**
*   `base_lin_vel`: Ground truth linear velocity.
*   `base_ang_vel`: Angular velocity.
*   `projected_gravity`: Projected gravity.
*   `velocity_commands`: Velocity targets.
*   `hip_pos`: Relative joint positions for Hip joints.
*   `kfe_pos`: Relative joint positions for Knee joints.
*   `ffe_pos`: Relative joint positions for Ankle pitch joints.
*   `faa_pos`: Relative joint positions for Ankle roll joints.
*   `joint_vel`: Relative joint velocities (all joints).
*   `actions`: The last action output.
*   `height_scan`: Terrain height scan.

**Critic:**
*   In the Berkeley implementation, the Critic utilizes the same observations as the Actor.


---

### 3.2 Observation Differences Analysis

| Observation Term | Tron1 (`HIM`) | Berkeley | Impact on Velocity Tracking |
| :--- | :--- | :--- | :--- |
| **Base Lin Vel** | **Hidden (Commented)** | **Visible (Raw + Noise)** | **Critical**: Direct signal vs. difficult inference. |
| **Base Ang Vel** | Scale: 0.25, $\sigma: 0.05$ | Scale: 1.0, $\pm 0.2$ (U) | Tron1 muffles the rotation signal. |
| **Proj. Gravity** | Scale: 1.0, $\sigma: 0.025$ | Scale: 1.0, $\pm 0.05$ (U) | Berkeley is more robust to orientation noise. |
| **Joint Velocities**| **Scale: 0.05**, $\sigma: 0.01$ | **Scale: 1.0**, **$\pm 1.5$ (U)** | **Extreme**: 150x noise difference & 20x scale difference. |
| **Height Scan** | **NONE (Blind)** | **Included ($\pm 0.1$)** | Berkeley sees terrain; Tron1 is reactive only. |
| **History Length** | **25 steps (0.5s)** | **1 step (No history)** | Tron1 introduces temporal lag/latency. |
| **Last Action** | Included | Included |

### 3.3 Root Cause Analysis (Observations)
1.  **Direct Velocity Sensing**: Berkeley provides the policy with its own velocity. Tron1 relies on an inference task using 25 steps of history. If the HIM encoder is not perfectly converged, the policy literally does not know how fast it is going.
2.  **Signal vs. Noise**: Tron1 down-scales joint velocity by **20x** (0.05) relative to Berkeley. This squashes the signal into a range where the network might ignore it. Berkeley uses raw values and massive noise ($\pm 1.5$), forcing the network to learn robust, large-scale motor features.
3.  **Spatial Awareness**: Berkeley's Actor sees the terrain heights; Tron1's Actor is blind. This makes velocity tracking on non-flat surfaces reactive rather than proactive for Tron1.

---

## 4. Actions

### 4.1 Detail of Actions
Both implementations utilize **joint position control**, where the RL policy outputs a target joint angle offset relative to a nominal posture. This is managed by the `mdp.JointPositionActionCfg` class.

*   **Action Mechanism**: The policy's raw outputs (typically in the range $[-1, 1]$) are multiplied by the `action_scale`, added to the default joint positions (`use_default_offset=True`), and then passed to the simulation's PD controller.
*   **Control Frequency**: Both projects set `decimation = 4` and `sim.dt = 0.005s`, resulting in a control frequency of **50 Hz**. The robot's state is updated at 200 Hz, but the RL policy only issues new commands every 4 simulation steps.
*   **Noise Models for Actions**:
    *   **Exploration Noise**: During training, PPO uses a Gaussian distribution for actions. Both projects initialize this with `init_noise_std=1.0`.
    *   **Actuator Execution Noise (Tron1)**: Uses `randomize_actuator_gains` (Events) to scale joint stiffness ($K_p$) and damping ($K_d$) by **0.5x to 2.0x**. This introduces significant variance in how a command is physically executed.
    *   **Actuator Execution Noise (Berkeley)**: Uses a more targeted hardware noise model, randomizing **Joint Armature** (rotor inertia) and **Joint Friction** ($\pm 10\%$). It also adds a static **Joint Offset** ($\pm 0.05$ rad) at startup to simulate miscalibrated encoders.

### 4.2 Action Differences Comparison

| Parameter | Tron1 | Berkeley | Impact |
| :--- | :--- | :--- | :--- |
| **Action Scale** | **0.25** | **0.50** | Tron1 restricts the maximum possible stride length. |
| **Control Frequency** | 50 Hz | 50 Hz | Latency is identical. |
| **Actuator Noise** | Aggressive Gain Scaling | Hardware-centric (Armature/Offsets) | Berkeley focuses on sensor/motor bias; Tron1 on strength variance. |

### 4.3 Root Cause Analysis (Actions)
**Restricted Control Authority**: The primary failure point in the action configuration is the **0.25 Scale**. 
1.  **Stride Length Constraint**: High-speed tracking requires the robot to take long, fast steps. A scale of 0.25 limits the joint deviation from the nominal pose to 1/4 of the network's output range. This likely prevents the robot from reaching the physical stride extension needed for the 1.2 m/s target.
2.  **Stiffness Interaction**: Tron1 randomizes actuator gains up to 2.0x. If the robot's joints are randomized to be very stiff, a small action scale (0.25) results in extremely small, rigid movements that cannot compensate for the momentum required for rapid velocity changes.
3.  **Lack of Encoder Bias**: Unlike Berkeley, Tron1 does not randomize joint offsets. This makes the policy less robust to the "zero-point" errors typical of real-world hardware, which can manifest as drift or tracking bias.

---

## 5. Curriculum

### 5.1 Detail of Curriculum
*   **Tron1**: Uses `terrain_levels_vel` (difficulty increases with distance). Velocity curriculum is commented out.
*   **Berkeley**: Uses `modify_command_velocity` (starts small, ramps up at step 5000) and `modify_push_force`.

### 5.2 Curriculum Differences

| Term | Tron1 | Berkeley | Root Cause Analysis |
| :--- | :--- | :--- | :--- |
| **Velocity Ramp** | **None (Fixed 1.2 m/s)** | **Active (0 to 3.0 m/s)** | **Trial by Fire**: Tron1 tries to run before it can walk, leading to constant failure early on. |
| **Push Ramp** | Constant | Active (starts gentle) | Berkeley protects the learning process from early bullying. |
### 5.3 Root Cause Analysis (Curriculum)
**The Speed Gap**: By commanding 1.2 m/s from iteration zero, Tron1 creates a "failure state" where the robot falls immediately. Without a ramp, it never receives the positive reinforcement needed to learn high-speed gaits.

---

## 6. Rewards

### 6.1 Individual Rewards List and Significance

#### **Tron1 Implementation (`RewardsCfg`)**
*   **`keep_balance` (Weight: 0.05)**: Reward for staying alive. Provides a baseline signal.
*   **`rew_lin_vel_xy` (Weight: 15.0, Std: 0.2)**: Primary tracking reward. Very narrow "peak".
*   **`rew_ang_vel_z` (Weight: 5.0, Std: 0.25)**: Turning tracking reward.
*   **`rew_keep_ankle_pitch_zero_in_air` (Weight: 1.0)**: Aesthetic reward to keep feet flat in air.
*   **`rew_no_fly` (Weight: 0.5)**: Penalizes flight/jumping to enforce a walking gait.
*   **`pen_base_height` (Weight: -5.0)**: Penalizes deviation from 0.75m target height.
*   **`pen_lin_vel_z` (Weight: -0.5)**: Penalizes vertical bouncing.
*   **`pen_ang_vel_xy` (Weight: -0.05)**: Penalizes torso pitching and rolling.
*   **`pen_joint_torque` (Weight: -0.00008)**: Penalizes high motor effort.
*   **`pen_joint_accel` (Weight: -5e-7)**: Penalizes jerky joint movements.
*   **`pen_action_rate` (Weight: -0.01)**: Penalizes large changes between consecutive actions.
*   **`pen_joint_pos_limits` (Weight: -2.0)**: Heavy penalty for hitting joint end-stops.
*   **`pen_undesired_contacts` (Weight: -0.5)**: Penalizes knees, hips, or base hitting the ground.
*   **`pen_action_smoothness` (Weight: -0.075)**: Second-order action penalty for fluid motion.
*   **`pen_flat_orientation` (Weight: -1.0)**: Penalizes the base not being level.
*   **`pen_feet_distance` (Weight: -100.0)**: Extremely heavy penalty for feet crossing (<0.115m).
*   **`pen_feet_regulation` (Weight: -0.2)**: Penalizes foot velocity near the ground.
*   **`pen_joint_vel_l2` (Weight: -5e-5)**: Penalizes high joint speeds.
*   **`feet_air_time` (Weight: 2.0)**: Rewards keeping feet in air for a specific duration range.
*   **`feet_slide` (Weight: -0.25)**: Penalizes the foot moving while in contact with ground.

#### **Berkeley Implementation (`RewardsCfg`)**
*   **`track_lin_vel_xy_exp` (Weight: 1.0, Std: 0.5)**: Broad tracking reward peak.
*   **`track_ang_vel_z_exp` (Weight: 0.5, Std: 0.5)**: Broad turning reward.
*   **`lin_vel_z_l2` (Weight: -2.0)**: Heavy penalty on vertical movement to encourage efficiency.
*   **`ang_vel_xy_l2` (Weight: -0.05)**: Standard orientation penalty.
*   **`joint_torques_l2` (Weight: -1e-5)**: Light effort penalty.
*   **`action_rate_l2` (Weight: -0.01)**: Standard jitter penalty.
*   **`feet_air_time` (Weight: 2.0)**: Encourages steps.
*   **`feet_slide` (Weight: -0.25)**: Penalizes skating.
*   **`undesired_contacts` (Weight: -1.0)**: Stricter penalty for falling.
*   **`joint_deviation_hip/knee` (Weight: -0.1)**: Encourages nominal posture without rigid height constraints.

---

### 6.2 Reward Differences Analysis

| Reward Function | Tron1 (Failing) | Berkeley (Success) | Significance of Inclusion/Parameters |
| :--- | :--- | :--- | :--- |
| **Lin Vel XY Track** | **Included (W: 15.0, $\sigma: 0.2$)** | **Included (W: 1.0, $\sigma: 0.5$)** | **Critical**: Tron1 peak is 2.5x narrower; Berkeley provides broad gradient. |
| **Ang Vel Z Track** | Included (W: 5.0, $\sigma: 0.25$) | Included (W: 0.5, $\sigma: 0.5$) | Tron1 is twice as restrictive on turning precision. |
| **Feet Distance Pen** | **Included (Weight: -100.0)** | **None** | **Minefield**: -100.0 penalty scares policy into not moving. |
| **Base Height Pen** | **Included (Weight: -5.0)** | **None** | Forces rigid vertical pose; robot can't "lean" to accelerate. |
| **Ankle Pitch (Air)** | **Included (Weight: 1.0)** | **None** | **Unique/Aesthetic**: Adds task complexity distracting from tracking. |
| **No Fly Reward** | **Included (Weight: 0.5)** | **None** | Enforces walking; restricts high-speed running gaits. |
| **Feet Regulation** | **Included (Weight: -0.2)** | **None** | Penalizes foot speed; directly fights tracking commands. |
| **Stay Alive / Bal.** | **Included (Weight: 0.05)** | **None** | Provides signal for "not dying" regardless of tracking. |
| **Flat Orientation** | **Included (Weight: -1.0)** | **None** | Prevents the lean needed for efficient forward momentum. |
| **Joint Pos Limits** | **Included (Weight: -2.0)** | **None** | Restricts dynamic range needed for large strides. |
| **Action Smoothness** | **Included (Weight: -0.075)** | **None** | Encourages fluid motion but adds another competing objective. |
| **Joint Deviation** | **None** | **Included (Weight: -0.1)** | Berkeley uses "soft nudges" for pose instead of rigid penalties. |
| **Lin Vel Z Penalty** | Included (Weight: -0.5) | **Included (Weight: -2.0)** | Berkeley is 4x stricter on vertical bouncing (efficiency). |
| **Undesired Contact** | Included (Weight: -0.5) | **Included (Weight: -1.0)** | Berkeley is 2x stricter on falls and knees hitting the ground. |

### 6.3 Root Cause Analysis (Rewards)
1.  **The Sparse Reward Trap ($\sigma=0.2$)**: Because Tron1's tracking reward peak is so narrow, a robot that is slightly off-speed (e.g., 0.5 m/s error) receives effectively zero reward. Without a curriculum to guide it into this narrow peak, the robot never "discovers" the tracking task. Berkeley's broad $\sigma=0.5$ peak ensures the robot is rewarded for even "vaguely correct" movement, providing a dense gradient for optimization.
2.  **Penalty Minefield (Feet Distance)**: The massive `-100.0` feet distance penalty in Tron1 creates a "Binary Failure" state. For a new policy, standing still is the only guaranteed way to avoid this penalty. The PPO algorithm finds this local minimum (stationary robot) more stable than exploring the "dangerous" task of running, which might trigger the -100.0 penalty or the -5.0 height penalty.
3.  **Posture vs. Performance**: Unique Tron1 rewards like `pen_base_height`, `pen_flat_orientation`, and `rew_keep_ankle_pitch_zero_in_air` force the robot into a specific aesthetic and geometric posture. However, high-speed locomotion requires dynamic leaning and height changes. By over-constraining the pose, Tron1 physically prevents the robot from reaching the kinematics required for velocity tracking.
4.  **Damping Forces (`feet_regulation`)**: By penalizing foot speed when close to the ground, Tron1 effectively adds "numerical friction" to the gait. To track 1.2 m/s, the feet *must* move fast. This reward directly competes with the tracking command, leading to a sluggish and unresponsive policy.
5.  **Lack of Effective Regularization**: Berkeley omits the rigid penalties but includes a heavy `lin_vel_z` penalty (-2.0) and `joint_deviation` (-0.1). This forces the robot to move efficiently (less bouncing) and stay near its nominal pose without making the pose "more important" than the tracking command itself.

---

## 7. Agents

### 7.1 Agent Architecture and Configuration
The RL agents are managed by the RSL-RL library, using Proximal Policy Optimization (PPO).

#### **Tron1 Implementation (`SF_TRON1AFlatPPORunnerCfg`)**
*   **Actor/Critic Architecture**: Both networks use an MLP with hidden dimensions of `[512, 256, 128]` and **ELU** activation functions.
*   **HIM Encoder**: A specialized MLP encoder processes the 25-step observation history. It has hidden dimensions of `[256, 128, 64, 16]` and outputs a 19-dimensional latent embedding to the Actor.
*   **Gradient Flow**: `output_detach=True`. The encoder is trained only via a separate estimator loss, and is "frozen" relative to the PPO policy gradients.
*   **Algorithm Params**: Uses an `adaptive` learning rate schedule starting at `1e-3`, with an `entropy_coef` of **0.01** to maintain exploration.

#### **Berkeley Implementation (`BerkeleyHumanoidFlatPPORunnerCfg`)**
*   **Actor/Critic Architecture**: For flat/velocity tasks, Berkeley uses a much leaner MLP with hidden dimensions of `[128, 128, 128]` and **ELU** activation.
*   **Simplicity**: It does not use an encoder or history; the Actor maps the current state directly to joint targets.
*   **Algorithm Params**: Uses an `adaptive` learning rate schedule starting at `1e-3`, with a lower `entropy_coef` of **0.005**, prioritizing convergence and precision over excessive exploration.

### 7.2 Agent Differences and Root Cause Analysis
1.  **Network Over-Complexity**: Tron1 uses a very large network `[512, 256, 128]` for a task that Berkeley successfully solves with `[128, 128, 128]`. Combined with the 25-step flattened history input, Tron1 has orders of magnitude more parameters to optimize, which slows down the discovery of the sparse tracking reward.
2.  **The Information Bottleneck (`output_detach`)**: The primary root cause is the **detached encoder**. Because the Actor cannot backpropagate gradients into the encoder, it cannot "fine-tune" its own state estimation specifically for the velocity tracking objective. It is at the mercy of the pre-trained or independently trained estimator.
3.  **High Entropy Variance**: The **0.01 entropy** in Tron1 encourages a "noisy" gait. While good for stability, it fights the precision required to track a specific commanded velocity exactly.

---

## 8. Conclusion

### 8.1 Combined Table of All Differences

| Feature | Tron1 Implementation | Berkeley Implementation | Impact on Tracking |
| :--- | :--- | :--- | :--- |
| **Scene** | Flat Plane, no scanner | Rough Terrain, Torso Scanner | Berkeley is robust; Tron1 is reactive/blind. |
| **Lin Vel Obs** | Hidden (HIM Inference) | **Visible (Ground Truth)** | **Critical**: Direct signal vs. "guessing." |
| **Joint Vel Scale**| **0.05** | **1.0 (Raw)** | Tron1 muffles the limb dynamics signal. |
| **Obs Noise** | Light Gaussian ($\sigma=0.01$) | Heavy Uniform ($\pm 1.5$) | Berkeley forces robustness via noise. |
| **History Length** | **25 Steps** | 1 Step | Tron1 has significant temporal latency. |
| **Action Scale** | **0.25** | **0.50** | Tron1 limits physical stride authority. |
| **Curriculum** | None (Fixed max speed) | **Active Velocity Ramp** | Tron1 skips the "walk before run" phase. |
| **Reward Peak** | **Sharp ($\sigma=0.2$)** | **Broad ($\sigma=0.5$)** | Tron1 reward is too sparse to find. |
| **Reward Minefield**| **-100.0 (Feet Dist)** | None | Tron1 policy is "scared" to move. |
| **Network Size** | [512, 256, 128] | [128, 128, 128] | Tron1 has higher convergence difficulty. |
| **Encoder Grads** | **Detached** | N/A | Actor can't improve its own estimation. |

### 8.2 Summary of Root Causes
The failure of Tron1 to track velocity is the result of a **"Gradient Blackout"**:
1.  **Information Deprivation**: The Actor doesn't see its velocity, and its primary sensor signals (joint velocities) are scaled to near-zero.
2.  **Reward Sparsity**: The mathematical target is a narrow needle ($\sigma=0.2$) that is never reached because there is no velocity curriculum.
3.  **Constraint Overpowering**: The massive penalty for crossing feet (-100.0) makes the local minimum of "standing still" more attractive than the difficult task of running.
4.  **Architectural Isolation**: The detached encoder prevents the Actor from learning the state features it needs to track velocity correctly.

### 8.3 Actionable Recommendations
1.  **Broaden the Signal**: Increase `std` for tracking rewards to **0.5** and `joint_vel` observation scale to **0.5**.
2.  **Implement Education**: Enable the **Velocity Curriculum** to ramp up speed targets over the first 5000-7500 iterations.
3.  **Clear the Minefield**: Reduce the `feet_distance` penalty to **-1.0** and the `base_height` penalty to **-1.0** to encourage exploration of the gait.
4.  **Boost Authority**: Increase `action_scale` to **0.5** to allow the robot the physical freedom to take long strides.
5.  **Connect Gradients**: Set `output_detach=False` in the agent configuration to allow end-to-end training of the velocity estimation and tracking tasks.
