# Architecture: TRON1A Bipedal Locomotion RL Framework

This document describes the high-level software architecture of the bipedal locomotion reinforcement learning framework for the [Limx Dynamics](https://www.limxdynamics.com/en) TRON1A robot family. It is intended as a technical reference for developers and researchers working within or extending this codebase. For installation and usage instructions, see [README.md](README.md).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture](#3-system-architecture)
4. [Simulation Architecture](#4-simulation-architecture)
   - [4.1 Asset Layer](#41-asset-layer)
   - [4.2 MDP Components](#42-mdp-components)
   - [4.3 Environment Configuration](#43-environment-configuration)
   - [4.4 Task Registration](#44-task-registration)
   - [4.5 Policy Description](#45-policy-description)
5. [RL Training Architecture](#5-rl-training-architecture)
6. [Adding a New Robot](#6-adding-a-new-robot)
7. [djinn Automation Interface](#7-djinn-automation-interface)
8. [Complete Task Registry](#8-complete-task-registry)
9. [Developer Reference: Key Classes & Files](#9-developer-reference-key-classes--files)

---

## 1. Project Overview

This repository implements a reinforcement learning training and evaluation framework for bipedal locomotion, targeting the TRON1A robot family from Limx Dynamics. The framework is built on top of [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (NVIDIA's robot learning framework) and uses [RSL-RL](https://github.com/leggedrobotics/rsl_rl) for Proximal Policy Optimization (PPO)-based training. Specific versions of the prerequisite software have been used to build docker images to ensure dependency consistency across experiment devices.

**Robot Variants:**

Three hardware configurations of the TRON1A are supported, each with its own USD model and environment configuration:

| Variant | Identifier | Description |
|---------|------------|-------------|
| **SoleFoot** (SF)  | `SF_TRON1A` | Flat-soled feet; 8 DOF (6 leg + 2 ankle). Primary development platform — the focus of this document. |
| **PointFoot** (PF)  | `PF_TRON1A` | Point-contact feet |
| **WheelFoot** (WF) | `WF_TRON1A` | Wheeled feet; combines leg and wheel actuator groups. |

This document focuses on the SoleFoot (SF) configuration as the primary reference implementation. The patterns described here apply directly to PF and WF, differing only in asset models, actuator parameters, and registered task IDs. A guide for adding a new robot variant is provided in [Section 6](#6-adding-a-new-robot).

A fourth robot, the SD_BRS1 biped, has since been added and is registered under the `SDBRS1` task prefix. It is not a TRON1A variant, it is a separate machine built on a simplified primitive URDF, and it carries its own environment configuration in `cfg/SF/brs_base_env_cfg.py` rather than inheriting the SF template. Its geometry is specified in [context/sd_brs_urdf.md](context/sd_brs_urdf.md) and its actuator gains are derived in [tron1-rl-isaaclab-cozum/context/BRS.md](tron1-rl-isaaclab-cozum/context/BRS.md).

**Documentation map:**

The workspace keeps its two architecture documents at the root and organises everything else into two indexed directories.

| Location | Contents |
|---|---|
| `ARCHITECTURE.md` | This document, the software architecture, task registry, and training pipelines |
| `CO_OPTIMISATION.md` | The design and controller co-optimisation architecture in full |
| [context/](context/README.md) | Established facts, codebase investigations, experiment records, and the literature survey |
| [context/artefacts/](context/artefacts/README.md) | Debugging artefacts, the exported plots and dashboards the documents cite as evidence |
| [plans/](plans/README.md) | Designs and implementation plans, each carrying its implementation status |
| [tron1-rl-isaaclab-cozum/context/](tron1-rl-isaaclab-cozum/context/README.md) | Facts specific to the simulation repository, robot parameterisation and algorithm mathematics |
| [tron1-rl-isaaclab-cozum/plans/](tron1-rl-isaaclab-cozum/plans/README.md) | Plans confined to the simulation repository, presently empty |

Every one of those directories carries a `README.md` that registers its contents with a summary, a revision date, and an implementation status. The file [context/README.md](context/README.md) is an entry point for all project level facts and history, whereas [plans/README.md](plans/README.md) is a detailed history of proposed or completed work describing implementation nuances and design decisions. The conventions governing all the project documentation are recorded in [CLAUDE.md](CLAUDE.md).

**Framework Stack:**

```
Isaac Sim (physics simulation, USD rendering)
    └── Isaac Lab (ManagerBasedRLEnv, sensors, terrain, MDP managers)
            └── bipedal_locomotion (extension: assets, configs, tasks, MDP)
                    └── RSL-RL (PPO algorithm, OnPolicyRunner, HIMOnPolicyRunner, CoptOnPolicyRunner)
                            └── djinn (workspace CLI: container + training automation)
```

---

## 2. Repository Structure

```
/ws/                                         # Workspace root (PROJECT_WS)
├── djinn                                    # Unified CLI entry point (see Section 7)
├── ARCHITECTURE.md                          # This document
├── CO_OPTIMISATION.md                       # Co-optimisation architecture and implementation guide
├── README.md                                # Installation and usage guide for the workspace
├── CLAUDE.md                                # Code change and writing conventions for this workspace
├── context/                                 # Established facts and investigation records
│   ├── README.md                            # Index, one summary per document (read this first)
│   │      .
│   │      .
│   │      .
│   └── artefacts/                           # Debugging artefacts, plots and dashboards
│       └── README.md                        # Register attributing each artefact to its run
├── plans/                                   # Designs and implementation plans
│   ├── README.md                            # Index, one summary and status per document
│   │      .
│   │      .
│   │      .
├── docker/
│   └── scripts/
│       └── start.sh                         # Docker container launch helper
├── IsaacLab/                                # Isaac Lab (git submodule, reference)
├── rsl_rl/                                  # RSL-RL library (git submodule, reference)
└── tron1-rl-isaaclab-cozum/                 # Main project directory
    ├── ARCHITECTURE.md                      # Architecture Document detailing the simulation architecture
    ├── README.md                            # Installation and usage guide for independent usage of the simulation
    ├── pyproject.toml                       # Linting and type-check configuration
    ├── exts/
    │   └── bipedal_locomotion/              # Isaac Lab extension package
    │       ├── setup.py                     # Package installation metadata
    │       └── bipedal_locomotion/          # Simulation directory (primary codebase)
    │           ├── __init__.py
    │           ├── assets/
    │           │   ├── config/              # Robot ArticulationCfg definitions
    │           │   └── usd/                 # USD robot model files
    │           │       ├── PF_TRON1A/
    │           │       ├── SF_TRON1A/
    │           │       └── WF_TRON1A/
    │           ├── actuators/               # Custom actuator models
    │           ├── tasks/
    │           │   └── locomotion/
    │           │       ├── agents/          # PPO runner configurations
    │           │       ├── cfg/             # Common (shared) environment configurations
    │           │       │   ├── SF/          # SoleFoot MDP template + terrain configs
    │           │       │   ├── PF/          # PointFoot MDP template
    │           │       │   └── WF/          # WheelFoot MDP template
    │           │       ├── mdp/             # MDP functions: rewards, observations, events
    │           │       └── robots/          # Task-specific environment configs + registry
    │           └── utils/
    │               └── wrappers/rsl_rl/     # RSL-RL compatibility wrappers and export utils
    ├── scripts/
    │   └── rsl_rl/                          # Training and evaluation scripts
    │       ├── train.py                     # Training entry point
    │       ├── play.py                      # Evaluation and data logging
    │       ├── visualise.py                 # Interactive Plotly visualization
    │       └── cli_args.py                  # Shared CLI argument definitions
    └── himloco/                             # HIM (History Information Model) runner and modules
```

---

## 3. System Architecture

### Runtime Flow

At runtime, the system composes as follows:

```
User
  └── djinn start train <mode> <gpu>
        └── docker exec → isaac-lab-base container
              └── ./isaaclab.sh -p scripts/rsl_rl/train.py --task <task_id>
                    ├── AppLauncher (Isaac Sim)
                    ├── gym.make(task_id)
                    │     ├── Loads env_cfg (from robots/__init__.py registry)
                    │     └── Instantiates ManagerBasedRLEnv
                    │           ├── SceneManager     (terrain, robot, sensors)
                    │           ├── ObservationManager (policy + critic groups)
                    │           ├── RewardManager    (weighted reward terms)
                    │           ├── ActionManager    (joint position targets)
                    │           ├── EventManager     (domain randomization)
                    │           ├── TerminationManager
                    │           └── CurriculumManager
                    ├── RslRlVecEnvWrapper  (Isaac Lab → RSL-RL interface)
                    └── OnPolicyRunner / HIMOnPolicyRunner / CoptOnPolicyRunner
                          └── runner.learn()  (PPO training loop)
```

### Data Flow Per Step

```
ObservationManager → policy obs (noisy, 1-step)      → critic obs (privileged, no noise)  
                                          ↓
                              Policy (Actor network)
                                          ↓
                      Actions (joint position deltas, scaled)
                                          ↓
                  ActionManager → Articulation (PD control at 250 Hz)
                                          ↓
  Physics step (Isaac Sim, dt=0.005s → 200 Hz physics update , decimation=4 → 50 Hz control)
                                          ↓
                              RewardManager → scalar reward
                              TerminationManager → done flags
                          CurriculumManager → curriculum updates
```

---

## 4. Simulation Architecture

All simulation code lives under the **simulation directory**:
`exts/bipedal_locomotion/bipedal_locomotion/`

This section describes the SF (SoleFoot) implementation in full. The same patterns applies to PF and WF.

---

### 4.1 Asset Layer

**Location:** `assets/`

The assets describe in details the physical robot hardware for the Isaac Sim physics engine to simulate via Universal Scene Description (USD) models and Python configuration objects.

**USD Models** are stored in `assets/usd/SF_TRON1A/` as `SF_TRON1A.usd`. The USD file defines the robot's visual geometry, collision meshes, joint structure, and inertial properties. Isaac Sim loads this file at simulation start.

**Asset Configurations** in `assets/config/` define `ArticulationCfg` objects that Isaac Lab uses to instantiate and control the robot. Two variants exist for SF:

| Config Object | File | Description |
|---|---|---|
| `SOLEFOOT_CFG` | `solefoot_cfg.py` | Standard robot with `ImplicitActuatorCfg`. The physics engine computes joint torques internally from PD targets. |
| `SOLEFOOT_IDENTIFIED_CFG` | `solefoot_identified_cfg.py` | Uses `IdentifiedActuator` — a learned motor model that more accurately reflects real hardware dynamics, improving sim-to-real transfer. |
| `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG` | `solefoot_identified_cfg.py` | Uses `IdentifiedActuator` — a learned motor model that more accurately reflects real hardware dynamics, improving sim-to-real transfer.  Supports multiple USD files for spawning different articulations|

Each config specifies:
- **Initial state**: spawn position `(0.0, 0.0, 0.8)` m, all joints at `0.0` rad
- **Rigid body properties**: gravity enabled, low damping, depenetration velocity capped
- **Actuator groups**:
  - Two groups with separate PD parameters for `SOLEFOOT_CFG`
    - `legs`: abad, hip, knee joints — higher stiffness for load-bearing
    - `ankles`: ankle joints — lower damping for compliant foot Contact
  - Identified actuator articulations contain unique actuators for each joint

---

### 4.2 MDP Components

**Location:** `tasks/locomotion/mdp/` and `tasks/locomotion/cfg/SF/limx_base_env_cfg.py`

The environment is structured as a Markov Decision Process (MDP) managed by Isaac Lab's `ManagerBasedRLEnv`. Each component of the MDP is defined as a configuration dataclass and executed by a corresponding manager at runtime.

#### Commands

The command space defines the task the robot is asked to perform. SF uses a single velocity command term: `base_velocity`, configured via `UniformVelocityCommandCfg`.

At each episode, a target velocity is sampled uniformly, for instance, from:
- Linear velocity x: `[-1.0, 1.0]` m/s
- Linear velocity y: `[-1.0, 1.0]` m/s
- Angular velocity z (yaw rate): `[-1.0, 1.0]` rad/s
The command velocity sampling ranges are updated based on the curriculum updates configured for the task.

The heading control mode (`heading_command=True`) is also enabled, using a stiffness gain of `0.5` to convert heading error into yaw rate commands. Commands are resampled every 12–18 seconds. A small fraction of environments (`rel_standing_envs=0.02`) receive zero velocity commands to train standing behavior. The current velocity command is included as an observation in both the policy and critic groups.

#### Observations

Two observation groups are defined in `ObservationsCfg`:

**Policy group** (`PolicyCfg`) — the input to the actor network. Contains only on-robot-measurable quantities, with additive Gaussian noise applied to simulate sensor imperfection:

| Term | Description | Scale |
|---|---|---|
| `base_lin_vel` | Base linear velocity (3D) | 1.0 |
| `base_ang_vel` | Base angular velocity (3D) | 0.25 |
| `proj_gravity` | Projected gravity vector (3D) | 1.0 |
| `joint_pos` | Joint positions relative to default (8D) | 1.0 |
| `joint_vel` | Joint velocities relative to default (8D) | 0.25 |
| `last_action` | Previous action (8D) | 1.0 |
| `velocity_commands` | Current velocity command (3D) | 1.0 |

**Critic group** (`CriticCfg`) — privileged input available only to the critic during training (not available on hardware). The scale of all terms is the same. Includes all policy terms (without noise) plus:

| Term | Description |
|---|---|
| `robot_joint_torque` | Applied joint torques |
| `robot_joint_acc` | Joint accelerations |
| `feet_lin_vel` | Linear velocity of ankle links |
| `robot_mass` | Robot body masses |
| `robot_inertia` | Robot body inertia tensors |
| `robot_joint_stiffness` / `robot_joint_damping` | Randomized actuator gains |
| `robot_material_properties` | Surface friction coefficients |
| `feet_contact_force` | Contact forces at ankle links |
| `heights` | Height scan map (ray-cast grid, 1.6 m × 1.0 m, 0.1 m resolution) |


For the HIM architecture, a third group `obsHistory` is added — a 25-step rolling buffer of the policy observations. See §4.5 for details.

#### Rewards

The total reward is a weighted sum of terms defined in `RewardsCfg`. Terms are grouped by purpose:

**Velocity Tracking** — the primary task objective:
- `rew_lin_vel_xy` (weight: +15): exponential reward for matching commanded linear velocity
- `rew_ang_vel_z` (weight: +5): exponential reward for matching commanded yaw rate

**Stability and Balance:**
- `keep_balance` (weight: +0.05): constant survival reward per step
- `rew_no_fly` (weight: +0.5): reward for maintaining at least one foot contact
- `pen_base_height` (weight: −5.0): L2 penalty for deviating from target base height (0.75 m)
- `pen_lin_vel_z` (weight: −0.5): penalizes vertical base velocity
- `pen_ang_vel_xy` (weight: −0.05): penalizes roll/pitch angular velocity
- `pen_flat_orientation` (weight: −1.0): penalizes tilting from upright pose

**Foot Movement Quality:**
- `feet_air_time` (weight: +2.0): rewards foot air time within a target range [0.2, 0.5] s when moving
- `feet_slide` (weight: −0.25): penalizes foot sliding during ground contact
- `pen_feet_distance` (weight: −100): heavily penalizes feet coming too close together (< 0.115 m)
- `pen_feet_regulation` (weight: −0.2): penalizes feet deviating from nominal position relative to base
- `rew_keep_ankle_pitch_zero_in_air` (weight: +1.0): rewards level ankle orientation during swing

**Actuation Quality:**
- `pen_action_rate` (weight: −0.01): penalizes large changes in action between timesteps
- `pen_action_smoothness` (weight: −0.075): penalizes second-order action derivative
- `pen_joint_torque` (weight: −8e-5): penalizes high joint torques
- `pen_joint_accel` (weight: −5e-7): penalizes high joint accelerations
- `pen_joint_vel_l2` (weight: −5e-5): penalizes high joint velocities
- `pen_joint_pos_limits` (weight: −2.0): penalizes joint limit violations
- `pen_undesired_contacts` (weight: −0.5): penalizes contact on non-foot links (abad, hip, knee, base)

#### Events (Domain Randomization)

Events apply perturbations and randomizations that train the policy to be robust to sim-to-real gaps. They are organized by when they fire:

**Startup** (once per training run):
- Mass randomization: base link ±5 kg additive; limb links ×[0.8, 1.2] scale
- Actuator gain randomization: stiffness and damping per joint group (abad, hip/knee, ankle) **(EXPERIMENTAL)**
- Joint default position offsets: ±0.05 rad (simulates calibration error)
- Surface friction randomization: [0.2, 1.25] for all robot links
- Joint friction model scaling: ×[0.9, 1.1]
- Joint armature scaling: ×[1.0, 1.05]

**Reset** (at each episode start):
- Root state: random position offset ±0.5 m, random yaw, small random velocity
- Joint state: random offset ±0.2 rad, small random velocity

**Interval** (periodically during training):
- Push disturbance: random base velocity impulse (±0.5 m/s x/y), applied every 10–15 s

#### Curriculum

Curriculum learning progressively increases task difficulty as the agent improves:

- **Terrain levels** (`terrain_levels_vel`): advances the robot to harder terrain tiles as average episode return improves
- **Push force** (`modify_push_force`): gradually increases the maximum push velocity from 0 to 3.0 m/s, starting after iteration 1500 in increments every 200 iterations
- **Command velocity range** (`modify_command_velocity_x`): expands the forward velocity range from [−1.0, 1.0] to [−1.5, 3.0] m/s, starting after iteration 1500

Curriculum terms are disabled in play/evaluation variants to hold task difficulty constant.

#### Terminations

Episodes end when:
- `time_out`: episode duration exceeds 20 seconds
- `base_contact`: the robot's `base_Link` makes contact with the ground (fall detection)

---

### 4.3 Environment Configuration

**Location:** `tasks/locomotion/cfg/SF/` (common configurations) and `tasks/locomotion/robots/` (task-specific configurations)

The environment configuration system uses a **two-layered architecture** that separates shared MDP logic from task-specific scenario customization.

#### Layer 1 — Common Configuration (`cfg/SF/`)

`cfg/SF/limx_base_env_cfg.py` defines the **complete MDP template** shared across all SF tasks. It contains:

- `SFSceneCfg`: scene layout — terrain, lighting, robot placeholder (`MISSING`), height scanner, contact sensor
- `CommandsCfg`, `ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, `EventsCfg`, `TerminationsCfg`, `CurriculumCfg`: all reward weights, observation terms, event parameters, and their default values
- `SFEnvCfg`: the root environment config that assembles all the above; sets simulation parameters (`dt=0.005`, `decimation=4`, `episode_length=20s`, `num_envs=4096`)
- `SFHIMEnvCfg`: an alternate root config for the HIM architecture, using `HIMObservationsCfg` instead. Rest of the environment configuration is the same as that for the base environment implementation. 

A separate file `cfg/SF/limx_berkeley_env_cfg.py` defines `SFBerkeleyEnvCfg`, which inherits from `SFEnvCfg` and overrides reward weights to match a Berkeley-style training regime with tighter contact constraints. We have exposed the berkeley implementation as independent tasks. However,**we are observing a crash in the berkeley implementation due to `NaN` values received during training.**

`cfg/SF/terrains_cfg.py` defines terrain generator configurations referenced by name in scenario classes:
- `BERKELEY_MIMIC_TERRAINS_CFG`: rough terrain heightfield used for robust locomotion training. Implements a terrain with a slightly reduced difficulty compared to the berkeley environment. This was done as the most difficult terrain generated was deemed too hard to learn biped locomotion for Tron1.
- `BLIND_ROUGH_TERRAINS_CFG` / `BLIND_ROUGH_TERRAINS_PLAY_CFG`
- `STAIRS_TERRAINS_CFG` / `STAIRS_TERRAINS_PLAY_CFG`

**Changes to the common layer affect every registered SF task.** This is the right place to modify shared reward weights, add a new observation term, or tune domain randomization parameters.

#### Layer 2 — Task-Specific Configuration (`robots/`)

`robots/limx_solefoot_env_cfg.py` defines concrete scenario classes by inheriting from the common configs and using `__post_init__` to apply overrides. This is where the robot asset is bound and where each scenario's distinguishing properties are set.

**Inheritance chain (example — Identified Blind Rough):**

```
ManagerBasedRLEnvCfg          (Isaac Lab — external base class)
    └── SFEnvCfg              (cfg/SF/limx_base_env_cfg.py — full MDP template)
          └── SFBaseEnvCfg    (robots/limx_solefoot_env_cfg.py — binds SOLEFOOT_CFG asset,
          │                    sets default joint positions, configures termination body names)
          └── SFBlindRoughEnvCfg   (sets terrain_type="generator", assigns terrain config)
                └── SFIdentifiedBlindRoughEnvCfg  (swaps robot to SOLEFOOT_IDENTIFIED_CFG)
```

**Scenario dimensions and what each layer controls:**

| Dimension | Set in | Options |
|---|---|---|
| Robot asset | `robots/` (Base class) | `SOLEFOOT_CFG`, `SOLEFOOT_IDENTIFIED_CFG` |
| Actuator model | `assets/config/` | `ImplicitActuator`, `IdentifiedActuator` |
| Terrain type | `robots/` (Scenario class) | Flat plane, rough heightfield, stairs |
| Height scanner | `robots/` (Scenario class) | Enabled (privileged) / disabled (blind) |
| Env count | `robots/` (Play class) | 4096 (train), 32 (play) |
| Randomization | `robots/` (Play class) | Full (train), disabled (play) |
| Curriculum | `robots/` (Play class/Flat class) | Enabled (train), disabled (play/flat) |

**Play variants** (`*_PLAY` suffix) reduce `num_envs` to 32, disable observation noise, remove push events and mass randomization, and disable curriculum stepping. They are used exclusively with `play.py`.

---

### 4.4 Task Registration

**Location:** `tasks/locomotion/robots/__init__.py`

All environments are registered with OpenAI Gymnasium via `gym.register()`. This file is the **central task registry** — it maps human-readable task IDs to `(env_cfg, agent_cfg)` pairs. When `train.py` calls `gym.make(task_id)`, Gymnasium looks up this registry and instantiates the appropriate environment.

**Registration pattern:**

```python
gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SFIdentifiedBlindRoughEnvCfg,    # from robots/
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg, # from agents/
    },
)
```

**Task ID naming convention:** `Isaac-Limx-{Robot}-[Identified]-[Perception/Architecture]-[Terrain]-[Play]-v0`

- `Robot`: `SF`, `PF`, or `WF`
- `Identified` (optional): robot uses identified actuator model
- `Perception/Architecture`: `Blind` (no height scan) or omitted (height scan available to critic), `Berkeley` or `HI<`
- `Terrain`: `Flat`, `Stairs`, or `Rough`
- `Play` (optional): evaluation variant

The full SF task registry:

| Task ID | Env Config Class | Terrain | Blind | Identified | Play |
|---|---|---|---|---|---|
| `Isaac-Limx-SF-Blind-Flat-v0` | `SFBlindFlatEnvCfg` | Flat | Yes | No | No |
| `Isaac-Limx-SF-Blind-Flat-Play-v0` | `SFBlindFlatEnvCfg_PLAY` | Flat | Yes | No | Yes |
| `Isaac-Limx-SF-Blind-Rough-v0` | `SFBlindRoughEnvCfg` | Rough | Yes | No | No |
| `Isaac-Limx-SF-Blind-Rough-Play-v0` | `SFBlindRoughEnvCfg_PLAY` | Rough | Yes | No | Yes |
| `Isaac-Limx-SF-HIM-Blind-Flat-v0` | `SFHIMBlindFlatEnvCfg` | Flat | Yes | No | No |
| `Isaac-Limx-SF-HIM-Blind-Flat-Play-v0` | `SFHIMBlindFlatEnvCfg_PLAY` | Flat | Yes | No | Yes |
| `Isaac-Limx-SF-HIM-Blind-Rough-v0` | `SFHIMBlindRoughEnvCfg` | Rough | Yes | No | No |
| `Isaac-Limx-SF-HIM-Blind-Rough-Play-v0` | `SFHIMBlindRoughEnvCfg_PLAY` | Rough | Yes | No | Yes |
| `Isaac-Limx-SF-Berkeley-v0` | `SFBerkeleyRoughEnvCfg` | Rough | No | No | No |
| `Isaac-Limx-SF-Berkeley-Play-v0` | `SFBerkeleyRoughEnvCfg_PLAY` | Rough | No | No | Yes |
| `Isaac-Limx-SF-Identified-Blind-Flat-v0` | `SFIdentifiedBlindFlatEnvCfg` | Flat | Yes | Yes | No |
| `Isaac-Limx-SF-Identified-Blind-Flat-Play-v0` | `SFIdentifiedBlindFlatEnvCfg_PLAY` | Flat | Yes | Yes | Yes |
| `Isaac-Limx-SF-HIM-Identified-Blind-Flat-v0` | `SFHIMIdentifiedBlindFlatEnvCfg` | Flat | Yes | Yes | No |
| `Isaac-Limx-SF-HIM-Identified-Blind-Flat-Play-v0` | `SFHIMIdentifiedBlindFlatEnvCfg_PLAY` | Flat | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Blind-Flat-Urdf-v0` | `SFIdentifiedBlindFlatEnvUrdfCfg` | Flat | Yes | Yes | No |
| `Isaac-Limx-SF-Identified-Blind-Flat-Play-Urdf-v0` | `SFIdentifiedBlindFlatEnvUrdfCfg_PLAY` | Flat | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Blind-Rough-v0` | `SFIdentifiedBlindRoughEnvCfg` | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-Identified-Blind-Rough-Play-v0` | `SFIdentifiedBlindRoughEnvCfg_PLAY` | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-v0` | `SFHIMIdentifiedBlindRoughEnvCfg` | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Play-v0` | `SFHIMIdentifiedBlindRoughEnvCfg_PLAY` | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0` | `SFIdentifiedBlindRoughEnvUrdfCfg` | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0` | `SFIdentifiedBlindRoughEnvUrdfCfg_PLAY` | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0` | `SFHIMIdentifiedBlindRoughEnvUrdfCfg` | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Play-Urdf-v0` | `SFHIMIdentifiedBlindRoughEnvUrdfCfg_PLAY` | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Berkeley-v0` | `SFIdentifiedBerkeleyRoughEnvCfg` | Rough | No | Yes | No |
| `Isaac-Limx-SF-Identified-Berkeley-Play-v0` | `SFIdentifiedBerkeleyRoughEnvCfg_PLAY` | Rough | No | Yes | Yes |
| `Isaac-Limx-SF-Copt-Flat-v0` | `SFCoptBlindFlatEnvCfg` | Flat | Yes | Yes | No |
| `Isaac-Limx-SF-Copt-Rough-v0` | `SFCoptBlindRoughEnvCfg` | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-Copt-Rough-Play-v0` | `SFCoptBlindRoughEnvCfg_PLAY` | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-Copt-Learned-Flat-v0` | `SFCoptBlindFlatEnvCfg` | Flat | Yes | Yes | No |
| `Isaac-Limx-SF-Copt-Learned-Flat-Play-v0` | `SFCoptBlindFlatEnvCfg_PLAY` | Flat | Yes | Yes | Yes |
| `Isaac-Limx-SF-Copt-Learned-Rough-v0` | `SFCoptBlindRoughEnvCfg` | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-Copt-Learned-Rough-Play-v0` | `SFCoptBlindRoughEnvCfg_PLAY` | Rough | Yes | Yes | Yes |

The `HIM` tasks register `HIMManagerBasedRLEnv` (`tasks/locomotion/envs/him_env.py`) as their Gymnasium entry point rather than the stock `ManagerBasedRLEnv`, because the HIM runner expects members the stock environment does not expose. The `Urdf` tasks spawn the robot from the authored URDF rather than from a pre-converted USD file. Both families are additions made after this section was first written.

The three `Copt` tasks drive the co-optimisation pipeline of [Section 5.1](#51-co-optimisation-training-copt) and are launched with `--policy-type COPT`. They use the multi-USD identified articulation so that a distinct morphology can be spawned per environment. The four `Copt-Learned` tasks drive the learned-model extension of that pipeline, launched with `--policy-type COPT-LEARNED` and documented in full in [plans/COPT_LEARNED_MODEL.md](plans/COPT_LEARNED_MODEL.md).

Note that the `Copt-Learned` tasks bind the same environment configuration classes as the plain `Copt` tasks and differ only in their agent configuration, which is `SFCoptLearnedModelPPORunnerCfg`. The distinct `SFCoptLearnedModel*EnvCfg` classes that the plan proposed were never created, the additional observation groups the learned model requires were folded into the shared `CoptObservationsCfg` instead, so both policy types read the same environment and the choice of estimator architecture rests entirely with the agent configuration and the `--policy-type` flag.

**SD_BRS1 task registry:**

| Task ID | Env Config Class | Terrain | HIM |
|---|---|---|---|
| `Isaac-Limx-SDBRS1-Blind-Flat-v0` | `SDBRS1BlindFlatEnvCfg` | Flat | No |
| `Isaac-Limx-SDBRS1-Blind-Flat-Play-v0` | `SDBRS1BlindFlatEnvCfg_PLAY` | Flat | No |
| `Isaac-Limx-SDBRS1-Blind-Flat2-v0` | `SDBRS1BlindFlatEnv2Cfg` | Flat (simplified URDF) | No |
| `Isaac-Limx-SDBRS1-Blind-Flat2-Play-v0` | `SDBRS1BlindFlatEnv2Cfg_PLAY` | Flat (simplified URDF) | No |
| `Isaac-Limx-SDBRS1-Blind-Rough-v0` | `SDBRS1BlindRoughEnvCfg` | Rough | No |
| `Isaac-Limx-SDBRS1-Blind-Rough-Play-v0` | `SDBRS1BlindRoughEnvCfg_PLAY` | Rough | No |
| `Isaac-Limx-SDBRS1-HIM-Blind-Flat-v0` | `SDBRS1HIMBlindFlatEnvCfg` | Flat | Yes |
| `Isaac-Limx-SDBRS1-HIM-Blind-Flat-Play-v0` | `SDBRS1HIMBlindFlatEnvCfg_PLAY` | Flat | Yes |
| `Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0` | `SDBRS1HIMBlindRoughEnvCfg` | Rough | Yes |
| `Isaac-Limx-SDBRS1-HIM-Blind-Rough-Play-v0` | `SDBRS1HIMBlindRoughEnvCfg_PLAY` | Rough | Yes |

All ten are defined in `robots/brs_solefoot_env_cfg.py` against the environment template `cfg/SF/brs_base_env_cfg.py`, and all share the agent configuration `SD_BRS1FlatPPORunnerCfg`. The `Flat2` pair uses the simplified primitive URDF that replaced the mesh assembly after a permanent shank into foot self penetration was traced as the root cause of a degenerate gait, an investigation recorded in [context/brs_gait.md](context/brs_gait.md). The SD_BRS1 agent configuration is the only one in the framework that enables symmetry data augmentation, through the mirror in `mdp/symmetry/brs.py`.

---

### 4.5 Policy Description

**Location:** `tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`, `utils/wrappers/rsl_rl/`

The policy network architecture and training hyperparameters are defined as `RslRlOnPolicyRunnerCfg` subclasses.

#### Actor-Critic Network

Both the actor (policy) and critic (value function) share the same feedforward MLP topology:

- **Hidden layers**: `[512, 256, 128]`
- **Activation**: ELU
- **Initial action noise std**: `1.0`

The actor takes the **policy observation** as input and outputs joint position delta targets. The critic takes the **critic observation** (privileged) as input during training but is discarded at deployment.

#### Proprioceptive Encoder

An additional encoder MLP compresses part of the proprioceptive observation into a latent embedding before it is passed to the actor-critic. This helps the network learn compact representations of the robot's state in the `HIM` implementation. This is unused for the base and berkeley implementatins:

- **Hidden dims**: `[128, 64, 16]` for the standard flat configs (PF, SF, WF) and the COPT config; `[512, 256, 128]` for the Berkeley config
- **Output dim** (`num_output_dim`): `19` for all configs
- **Activation**: ELU
- **Output detach** (`output_detach=True`): for the standard `ActorCritic` path the encoder gradient is not propagated into the main actor-critic during the PPO update. Note that the co-optimisation policy `CoptActorCritic` builds its own estimator from `hidden_dims`, emitting a latent of width `hidden_dims[-1]` (16 for SF), and concatenates that latent to the actor input **without** applying the detach, so in COPT the estimator is trained jointly with the actor (`co_optimisation/modules/copt_actor_critic.py:93-105,127-137`)

#### Standard PPO Policy vs. HIM Policy

Three policy architectures are available, selected via `--policy-type` at runtime:

| Mode | `--policy-type` | Network | Observation Groups Used |
|---|---|---|---|
| Standard PPO | `PPO` *(default)*    | `ActorCritic` + encoder | `policy`, `critic` |
| HIM (History Information Model) | `HIMPPO` | `HIMActorCritic` | `policy`, `obsHistory`, `critic`, `estimatorGT` |
| COPT (design and policy co-optimisation) | `COPT` | `CoptActorCritic` + estimator | `policy`, `critic`, `predictedPrivilegedObs` |
| COPT-LEARNED (co-optimisation with a learned dynamics model) | `COPT-LEARNED` | `CoptLearnedModelActorCritic` + `CoptEstimator` | `policy`, `critic`, `morphologyObs`, `obsHistory`, `predictedPrivilegedObs` |
| COPT-LEARNED-2 (as above, estimator conditioned on history alone) | `COPT-LEARNED-2` | `CoptLearnedModelV2ActorCritic` + `CoptEstimator` | `policy`, `critic`, `obsHistory`, `predictedMorphologyObs` |

The observation groups available to the co-optimisation policies are those declared on `CoptObservationsCfg` (`cfg/SF/limx_base_env_cfg.py:360`), namely `policy`, `critic`, `commands`, `morphologyObs`, `predictedMorphologyObs`, `predictedPrivilegedObs`, and `obsHistory`. Earlier revisions of this document named these groups `privilegedObs` and `criticOnly`, which never existed in the committed configuration.

The COPT mode runs the design and policy co-optimisation pipeline described in [Section 5.1](#51-co-optimisation-training-copt). It pairs a `CoptOnPolicyRunner` with a `CoptActorCritic` whose estimator encodes the privileged observations into a latent that conditions the actor on the current morphology. See [CO_OPTIMISATION.md](CO_OPTIMISATION.md) for the full design.

The COPT-LEARNED mode extends COPT with an encoder-decoder estimator (`CoptEstimator`) trained jointly with PPO under a single optimiser: the encoder consumes the privileged morphology/terrain observations plus a flattened proprioceptive history and produces a 16-dim latent that conditions the actor, while a decoder regresses the robot's ground-truth dynamic state (torques, accelerations, inertia, contact forces, foot velocities) from that latent, adding an MSE model-estimation loss to the PPO objective. See [plans/COPT_LEARNED_MODEL.md](plans/COPT_LEARNED_MODEL.md) for the full design and implementation plan.

The HIM architecture addresses the state estimation problem: rather than using the noisy single-step base velocity observation, HIM uses an estimator network trained on a 25-step history buffer (`obsHistory`) to infer a latent representation of hidden states (e.g., base velocity, terrain properties). The estimator is supervised against ground-truth values (`estimatorGT`). The history buffer has `history_length=25` and `flatten_history_dim=False`.

To use HIM, the task must use `SFHIMEnvCfg` as its base (which activates `HIMObservationsCfg` including the `obsHistory` group), and `train.py` must be called with `--policy_type HIMPPO`.

#### Policy Export

After evaluation, `play.py` can export the trained actor for hardware deployment:
- **TorchScript (JIT)**: `export_policy_as_jit()` — full policy as a traced module
- **ONNX**: `export_mlp_as_onnx()` — actor MLP only

Export is triggered by setting `EXPORT_POLICY = True` in `play.py` in before the invocation of the main method of the evaluation script.

---

## 5. RL Training Architecture

The RL training system is built on RSL-RL's PPO implementation, wrapped and extended by this framework. This section covers the algorithm configuration, runner selection, training loop, and evaluation pipeline.

### PPO Configuration

Runner configurations in `agents/limx_rsl_rl_ppo_cfg.py` define all PPO hyperparameters. Two configurations are relevant for SF:

| Parameter | `SF_TRON1AFlatPPORunnerCfg` | `SF_Berkeley_PPORunnerCfg` | `SFCoptPPORunnerCfg` |
|---|---|---|---|
| `max_iterations` | 30,000 | 30,000 | 30,000 |
| `save_interval` | 500 | 500 | 500 |
| `num_steps_per_env` | 25 | 24 | 25 (inherited) |
| `learning_rate` | 1e-3 | 1e-3 | 1e-3 |
| `schedule` | adaptive | adaptive | adaptive |
| `gamma` (discount) | 0.99 | 0.99 | 0.99 |
| `lam` (GAE λ) | 0.95 | 0.95 | 0.95 |
| `clip_param` (ε) | 0.2 | 0.2 | 0.2 |
| `entropy_coef` | 0.005 | 0.005 | 0.005 |
| `num_learning_epochs` | 5 | 5 | 5 |
| `num_mini_batches` | 4 | 4 | 4 |

All three configurations share `entropy_coef = 0.005`. `SFCoptPPORunnerCfg` (`agents/limx_rsl_rl_ppo_cfg.py:131`) is a thin subclass of `SF_TRON1AFlatPPORunnerCfg` that overrides only `experiment_name` to `sf_copt`, so it inherits `num_steps_per_env = 25` and the encoder dimensions of the flat config. The Berkeley config differs from the flat config only in `num_steps_per_env = 24`, its larger encoder, and its experiment name.

The effective rollout batch size per update is `num_steps_per_env × num_envs`, which is `25 × 4096 = 102,400` transitions for the flat and COPT configs and `24 × 4096 = 98,304` for the Berkeley config. Each PPO update runs `5` epochs over `4` mini-batches.

The adaptive learning rate schedule adjusts the learning rate based on KL divergence, targeting `desired_kl=0.01`.

### Runner Selection

Three runners are available:

- **`OnPolicyRunner`** (default, from `rsl_rl`): standard PPO with `ActorCritic` network
- **`HIMOnPolicyRunner`** (from `himloco/`): PPO with `HIMActorCritic` + history estimator
- **`CoptOnPolicyRunner`** (from `co_optimisation/`): subclasses `OnPolicyRunner` and interleaves an outer evolutionary design update with the policy update, using `CoptActorCritic`

The runner is selected in `train.py` based on `--policy-type`:

```bash
# Standard PPO
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Limx-SF-Identified-Blind-Rough-v0

# HIM PPO
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Limx-SF-HIM-v0 --policy-type HIMPPO

# COPT (design and policy co-optimisation)
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Limx-SF-Copt-Rough-v0 --policy-type COPT
```

When `HIMPPO` is selected, `train.py` sets `agent_cfg.policy.class_name = "HIMActorCritic"` and `agent_cfg.algorithm.class_name = "HIMPPO"` before instantiating the runner. When `COPT` is selected, `train.py` sets `agent_cfg.policy.class_name = "CoptActorCritic"`, constructs a `GrowingDesignDistCMAESDesignGenerator`, and instantiates `CoptOnPolicyRunner` with that generator (`scripts/rsl_rl/train.py:194-235`).

### Training Loop

1. `train.py` calls `parse_env_cfg()` and `parse_rsl_rl_cfg()` to load configs from the registry
2. `gym.make(task_id)` instantiates `ManagerBasedRLEnv`
3. `RslRlVecEnvWrapper` adapts the Isaac Lab environment to RSL-RL's expected interface
4. `runner.learn(num_learning_iterations)` executes the training loop:
   - **Rollout collection**: 25 steps per environment at 50 Hz control (24 for the Berkeley config)
   - **Advantage estimation**: GAE with γ=0.99, λ=0.95
   - **PPO update**: 5 epochs × 4 mini-batches, clipped surrogate objective
   - **Checkpoint save**: every 500 iterations to `logs/rsl_rl/<experiment_name>/<timestamp>/`

**Log directory structure:**
```
logs/rsl_rl/<experiment_name>/<timestamp>/
    ├── params/
    │   ├── env.yaml       # Full environment config dump
    │   ├── agent.yaml     # Full agent config dump
    │   ├── env.pkl        # Pickled env config (for exact reload)
    │   └── agent.pkl
    ├── videos/train/      
    │   ├── train/         # Training rollout videos (if --video)
    │   ├── play/          # Evaluation rollout videos (if --video)
    └── model_<iter>.pt    # Checkpoint files
```

### Evaluation Pipeline

`play.py` loads a checkpoint and runs inference for a fixed number of steps, collecting data via `DataLogger`:

- **Joint metrics**: velocities, torques, powers (torque × velocity), positions, accelerations
- **Base metrics**: linear and angular velocity (actual vs. commanded)
- **Link properties**: mass and geometry logged from USD file to `link_properties.csv`

After the run, `DataLogger.plot()` calls `visualise.visualise()` to generate an interactive Plotly HTML dashboard with per-joint subplots organized into tabs. Raw data is also saved as `.npy` for further analysis.

### 5.1 Co-optimisation Training (COPT)

The COPT mode jointly optimises the robot **design** and the locomotion **policy**. The inner loop is the standard RSL-RL PPO update, and an outer evolutionary loop replaces the morphology of the parallel environments at a fixed cadence so that the policy is trained across a distribution of designs while the design is driven toward higher fitness. The pipeline lives in the `co_optimisation/` package and is documented in full in [CO_OPTIMISATION.md](CO_OPTIMISATION.md).

**Runner — `CoptOnPolicyRunner`** (`co_optimisation/runners/copt_on_policy_runner.py`) subclasses `OnPolicyRunner` and overrides `learn` to interleave the evolutionary update with the policy update, and `_construct_algorithm` to build a `CoptActorCritic`. Its co-optimisation configuration (set in `train.py:200-227`) is:

| Setting | Value | Previous value | Meaning |
|---|---|---|---|
| `ea_update_interval` | 240 | 120 | iterations between morphology updates |
| `ea_late_start` | 12000 | 8000 | iterations of random design sampling before CMA-ES begins |
| `num_individuals` | 256 | 64 | designs evaluated in parallel (round-robin over the 4096 envs) |
| `randomise_before_late_start` | True | True | sample random designs during the late-start window |

The previous-value column records the configuration that every run before 2026-06-30 used, and which the analyses in [context/task_plots.md](context/task_plots.md) and [context/cmaes.md](context/cmaes.md) therefore describe. The retuning followed the investigation recorded in [plans/COPT_INVESTIGATION_PLAN.md](plans/COPT_INVESTIGATION_PLAN.md).

**Design generator — `GrowingDesignDistCMAESDesignGenerator`** (`co_optimisation/runners/usd_generator.py`) optimises the `thigh_length_scale` and `shank_length_scale`, each bounded to `(0.75, 1.25)`. CMA-ES operates on the unit square `[0,1]²` with `sigma0=0.25` and the box centre as the initial mean, denormalising each coordinate to the physical scale range. For the first 12000 iterations the generator samples random designs from a distribution whose spread grows from 5 % to the full range, thereafter it samples from the CMA-ES search distribution. The bounds and the initial step size were both widened and reduced respectively after the original values, `(0.85, 1.15)` and `sigma0=0.75`, were found to pin the search against the upper bound of the box, an analysis given in [context/cmaes.md](context/cmaes.md).

**Morphology update** — every 240 iterations the runner stops the simulation, advances the design generator, authors the new link extents into the per-environment USD prototypes via `apply_link_length_params`, calls `env.reset()` for all environments, reapplies actuator parameters, and zeroes the per-individual fitness accumulators (`_update_morphology`). The COPT env uses `SOLEFOOT_IDENTIFIED_MULTIUSD_CFG` with `replicate_physics=False` so that distinct articulations can be spawned per environment. The link extents are rounded to two decimal places when authored, which imposes a design resolution floor of roughly one centimetre, and that rounding is the one recommendation of the investigation plan that has not been applied.

**Checkpointing** — `CoptOnPolicyRunner` overrides `save` and `load` so that a resume restores the full pipeline state rather than the network weights alone. The design generator serialises its own CMA-ES state, and the environment and manager state is captured and restored by `capture_env_state` and `restore_env_state` in `co_optimisation/utils/env_state.py`. This is what permits a run to resume without repeating the random design phase, which was the second objective of the investigation.

**Policy — `CoptActorCritic`** (`co_optimisation/modules/copt_actor_critic.py`) gives the actor the proprioceptive policy observations concatenated with a 16-dim latent produced by an estimator network from the privileged observations. The inherited critic observes the `critic` group, which includes the true per-environment link lengths, body masses, and inertias, so the value function can condition on the morphology it evaluates.

> **Investigation note, updated 2026-07-30.** The investigation into the training variance, checkpoint completeness, and CMA-ES sampling behaviour of this pipeline is recorded in [plans/COPT_INVESTIGATION_PLAN.md](plans/COPT_INVESTIGATION_PLAN.md), and its remedies have since been implemented. The defects it identified, namely an off-by-one pairing in the CMA-ES `tell` and `ask` order, an oversized `sigma0` for the unit box, a resume-relative morphology schedule, a reward-window flood at the morphology update, an unguarded terrain curriculum during a morphology reset, and globally rather than per-design normalised advantages, are all corrected in the live sources. One recommendation was not adopted, the two-decimal rounding of link extents that erases design variety remains in `usd_generator.py`. Consult that document and [context/copt.md](context/copt.md) sections 8 to 10 before modifying the COPT runner or generator, the latter carrying the current statement of code state.

---

## 6. Adding a New Robot

This section describes the complete process for introducing a new robot variant to the framework, using SF as the reference at each step.

### Step 1 — USD Asset

Prepare the robot's USD model and place it at:
```
assets/usd/<ROBOT_NAME>/<ROBOT_NAME>.usd
```
The USD file should define rigid body physics, joint structure, collision geometry, and inertial properties. Ensure joint names are consistent — they will be referenced by name in configuration files.

### Step 2 — Asset Configuration

Create `assets/config/<robot>_cfg.py` and define an `ArticulationCfg` object:

```python
MY_ROBOT_CFG = ArticulationCfg(
    spawn=UsdFileCfg(usd_path=f"{BIPEDAL_LOCOMOTION_EXT_DIR}/assets/usd/MY_ROBOT/MY_ROBOT.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, <spawn_height>),
        joint_pos={<joint_name>: 0.0, ...},
    ),
    actuators={
        "<group_name>": ImplicitActuatorCfg(
            joint_names_expr=["<regex>"],
            stiffness=<kp>, damping=<kd>,
            effort_limit_sim=<N*m>, velocity_limit_sim=<rad/s>,
        ),
    },
)
```

Define an identified actuator variant (`MY_ROBOT_IDENTIFIED_CFG`) if hardware motor characterization data is available.

### Step 3 — Common Environment Configuration

Create `tasks/locomotion/cfg/<ROBOT>/limx_base_env_cfg.py`. This file defines the MDP template shared by all tasks for the new robot. Inherit from `ManagerBasedRLEnvCfg` and define all configuration inner classes:

- `<ROBOT>SceneCfg(InteractiveSceneCfg)`: scene with terrain, sensors, robot placeholder
- `CommandsCfg`, `ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, `EventsCfg`, `TerminationsCfg`, `CurriculumCfg`
- `<ROBOT>EnvCfg(ManagerBasedRLEnvCfg)`: root config assembling all components, setting `dt`, `decimation`, `num_envs`

Refer to `cfg/SF/limx_base_env_cfg.py` for the complete SF reference. Observation dimensions must match the robot's DOF count. Reward weights should be tuned for the new robot's dynamics.

Create `cfg/<ROBOT>/terrains_cfg.py` with terrain generator configurations appropriate for the robot's locomotion mode.

### Step 4 — Task-Specific Environment Configuration

Create `tasks/locomotion/robots/limx_<robot>_env_cfg.py`. Define:

1. `<ROBOT>BaseEnvCfg` — inherits from `<ROBOT>EnvCfg`, binds the asset in `__post_init__`, sets default joint positions, configures contact/termination body names
2. `<ROBOT>BaseEnvCfg_PLAY` — inherits from base, reduces `num_envs=32`, disables randomization and curriculum
3. Scenario leaf classes (`<ROBOT>BlindFlatEnvCfg`, `<ROBOT>BlindRoughEnvCfg`, etc.) — each overrides terrain, height scanner, or command ranges as needed

### Step 5 — Agent Configuration

Add a `RunnerCfg` class in `tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`:

```python
@configclass
class MY_ROBOT_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "my_robot_flat"
    max_iterations = 30000
    save_interval = 500
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(...)
    encoder = EncoderCfg(
        num_output_dim=<latent_dim>,
        hidden_dims=[256, 128, 64, 16],
    )
```

Set `encoder.num_output_dim` and `encoder.hidden_dims` based on the observation space size of the new robot in the aforementioned example. Any additional algorithm specific configuration must be performed here.

### Step 6 — Task Registration

Add `gym.register()` calls for all scenario variants in `tasks/locomotion/robots/__init__.py`:

```python
gym.register(
    id="Isaac-Limx-<ROBOT>-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_<robot>_env_cfg.<ROBOT>BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": MY_ROBOT_PPORunnerCfg(),
    },
)
```

Follow the task ID naming convention: `Isaac-Limx-{Robot}-[Identified]-[Blind]-{Terrain}-[Play]-v0`.

### Step 7 — djinn Integration

Add the new robot's task IDs to the `djinn` script's `start train` and `start play` branches:

```bash
elif [[ "$3" == "<new_mode>" ]]; then
    task="Isaac-Limx-<Task>-Blind-<Rough/Flat>-v0"
fi
```

---

## 7. djinn Automation Interface

`djinn` (located at `$PROJECT_WS/djinn`) is the unified CLI entry point for the workspace. It manages the Docker container lifecycle and wraps training and evaluation commands so users do not need to interact with Docker or Isaac Lab internals directly.

### Command Grammar

All `djinn` commands follow this grammar:

```
djinn <verb> <subject> [objects...]
```

- **`verb`**: the action class — what kind of operation to perform
- **`subject`**: the target entity — what the verb acts on
- **`objects`**: ordered positional parameters whose meaning depends on the verb and subject combination

### Container Management

#### `djinn up <subject>`

Start a container. If the container exists but is stopped, it is restarted. If it does not exist, it is created from the appropriate image.

| Subject | Container Name | Description |
|---|---|---|
| `dev` | `dev_container` | Development container (`shandilya1998/nrt:dev`), CPU only |
| `lab` | `isaac-lab-base` | Isaac Lab training container with GPU access |

```bash
djinn up lab      # Start the Isaac Lab training container
djinn up dev      # Start the development container
```

#### `djinn down <subject>`

Stop and remove a container.

```bash
djinn down lab    # Stop and remove isaac-lab-base
djinn down dev    # Stop and remove dev_container
```

#### `djinn ps [-a]`

List running containers. Pass `-a` to show all containers including stopped ones.

```bash
djinn ps          # Show running containers
djinn ps -a       # Show all containers
```

#### `djinn exec <subject> <command>`

Execute a shell command inside a running container.

| Subject | Resolves to container |
|---|---|
| `lab` | `isaac-lab-base` |
| `dev` | `dev_container` |

```bash
djinn exec lab "pip install -e biped/exts/bipedal_locomotion"
djinn exec lab "ls /workspace/isaaclab/logs/rsl_rl"
```

#### `djinn build docker <subject>`

Build a Docker image.

```bash
djinn build docker isaaclab    # Build the Isaac Lab container image
```

---

### Training & Evaluation

All training and evaluation commands execute inside the `isaac-lab-base` container via `iexec`.

#### `djinn start train <mode> [gpu_id]`

Start a training run. Installs `bipedal_locomotion` before training.

| Object | Position | Description |
|---|---|---|
| `mode` | 1st | Training mode: `base`, `base-urdf`, `berkeley`, `him`, `him-urdf`, `copt`, `copt-learned`, or `brs` |
| `gpu_id` | 2nd (optional) | CUDA device index (default: `0`) |

| Mode | Task ID | `--policy-type` |
|---|---|---|
| `base` | `Isaac-Limx-SF-Identified-Blind-Flat-v0` | `PPO` |
| `base-urdf` | `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0` | `PPO` |
| `berkeley` | `Isaac-Limx-SF-Identified-Berkeley-v0` | `PPO` |
| `him` | `Isaac-Limx-SF-HIM-Identified-Blind-Rough-v0` | `HIMPPO` |
| `him-urdf` | `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0` | `HIMPPO` |
| `copt` | `Isaac-Limx-SF-Copt-Rough-v0` | `COPT` |
| `copt-learned` | `Isaac-Limx-SF-Copt-Learned-Rough-v0` | `COPT-LEARNED` |
| `copt-learned-2` | `Isaac-Limx-SF-Copt-Learned-Rough-v0` | `COPT-LEARNED-2` (not reachable, see §4.5) |
| `brs` | `Isaac-Limx-SDBRS1-Blind-Flat-v0` | `PPO` |
| `brs-simplified` | `Isaac-Limx-SDBRS1-Blind-Flat2-v0` | `PPO` |

The `djinn start train` branch installs `bipedal_locomotion`, `co_optimisation`, and `himloco` before launching, and passes `--save_interval 1000` on the command line, which overrides the `save_interval` of `500` set in the runner config. Note that `brs` selects the flat task rather than the rough one, contrary to what earlier revisions of this document stated, and that `copt-learned-2` will not produce a co-optimisation run until the policy-type guard in `train.py` is widened.

```bash
djinn start train base          # Train base mode on GPU 0
djinn start train berkeley 1    # Train Berkeley mode on GPU 1
djinn start train him 0         # Train HIM mode on GPU 0
djinn start train copt 0        # Train design and policy co-optimisation on GPU 0
djinn start train copt-learned 0  # Train co-optimisation with a learned dynamics model on GPU 0
djinn start train brs 0         # Train the SD_BRS1 biped on flat terrain on GPU 0
djinn start train brs-simplified 0  # Train SD_BRS1 on the simplified primitive URDF
```

#### `djinn start play <mode> <checkpoint_path> <seed> [gpu_id]`

Load a checkpoint and run policy evaluation.

| Object | Position | Description |
|---|---|---|
| `mode` | 1st | Evaluation mode: `base`, `base-urdf`, `berkeley`, `him`, `copt`, `copt-learned`, `brs`, or `brs-simplified` |
| `checkpoint_path` | 2nd | Path to checkpoint **relative to `/workspace/isaaclab/logs/rsl_rl/`** |
| `seed` | 3rd | Random seed for reproducible evaluation |
| `gpu_id` | 4th (optional) | CUDA device index (default: `0`) |

```bash
djinn start play base sf_tron_1a_flat/2024-01-01_12-00-00/model_15000.pt 42 0
```

#### `djinn start evaluation [seed] [gpu_id]`

Batch evaluation over all experiment directories under `logs/rsl_rl/test/`. Evaluates `model_14999.pt` in each subdirectory on the `Isaac-Limx-SF-Blind-Flat-v0` task.

| Object | Position | Description |
|---|---|---|
| `seed` | 1st (optional) | Random seed (default: `42`) |
| `gpu_id` | 2nd (optional) | CUDA device index (default: `0`) |

```bash
djinn start evaluation 42 0
```

#### `djinn start visualise-evaluation`

Launch the interactive visualization dashboard using the data collected by `play.py`.

```bash
djinn start visualise-evaluation
```

---

## 8. Complete Task Registry

All registered Gymnasium task IDs across all robots, generated from the registry source rather than transcribed. The env config classes named here live in `robots/limx_pointfoot_env_cfg.py`, `robots/limx_wheelfoot_env_cfg.py`, `robots/limx_solefoot_env_cfg.py`, and `robots/brs_solefoot_env_cfg.py` according to their robot. The HIM env column records whether the task registers `HIMManagerBasedRLEnv` as its Gymnasium entry point in place of the stock `ManagerBasedRLEnv`, which is required by the HIM runner and is independent of the `--policy-type` selected at launch.

| Task ID | Robot | Env Config Class | Terrain | Identified | HIM env | Play |
|---|---|---|---|---|---|---|
| `Isaac-Limx-PF-Blind-Flat-v0` | PointFoot | PFBlindFlatEnvCfg | Flat | No | No | No |
| `Isaac-Limx-PF-Blind-Flat-Play-v0` | PointFoot | PFBlindFlatEnvCfg_PLAY | Flat | No | No | Yes |
| `Isaac-Limx-WF-Blind-Flat-v0` | WheelFoot | WFBlindFlatEnvCfg | Flat | No | No | No |
| `Isaac-Limx-WF-Blind-Flat-Play-v0` | WheelFoot | WFBlindFlatEnvCfg_PLAY | Flat | No | No | Yes |
| `Isaac-Limx-SF-Blind-Flat-v0` | SoleFoot | SFBlindFlatEnvCfg | Flat | No | No | No |
| `Isaac-Limx-SF-Blind-Flat-Play-v0` | SoleFoot | SFBlindFlatEnvCfg_PLAY | Flat | No | No | Yes |
| `Isaac-Limx-SF-Blind-Rough-v0` | SoleFoot | SFBlindRoughEnvCfg | Rough | No | No | No |
| `Isaac-Limx-SF-Blind-Rough-Play-v0` | SoleFoot | SFBlindRoughEnvCfg_PLAY | Rough | No | No | Yes |
| `Isaac-Limx-SF-HIM-Blind-Flat-v0` | SoleFoot | SFHIMBlindFlatEnvCfg | Flat | No | Yes | No |
| `Isaac-Limx-SF-HIM-Blind-Flat-Play-v0` | SoleFoot | SFHIMBlindFlatEnvCfg_PLAY | Flat | No | Yes | Yes |
| `Isaac-Limx-SF-HIM-Blind-Rough-v0` | SoleFoot | SFHIMBlindRoughEnvCfg | Rough | No | Yes | No |
| `Isaac-Limx-SF-HIM-Blind-Rough-Play-v0` | SoleFoot | SFHIMBlindRoughEnvCfg_PLAY | Rough | No | Yes | Yes |
| `Isaac-Limx-SF-Berkeley-v0` | SoleFoot | SFBerkeleyRoughEnvCfg | Flat | No | No | No |
| `Isaac-Limx-SF-Berkeley-Play-v0` | SoleFoot | SFBerkeleyRoughEnvCfg_PLAY | Flat | No | No | Yes |
| `Isaac-Limx-SF-Identified-Blind-Flat-v0` | SoleFoot | SFIdentifiedBlindFlatEnvCfg | Flat | Yes | No | No |
| `Isaac-Limx-SF-Identified-Blind-Flat-Play-v0` | SoleFoot | SFIdentifiedBlindFlatEnvCfg_PLAY | Flat | Yes | No | Yes |
| `Isaac-Limx-SF-HIM-Identified-Blind-Flat-v0` | SoleFoot | SFHIMIdentifiedBlindFlatEnvCfg | Flat | Yes | Yes | No |
| `Isaac-Limx-SF-HIM-Identified-Blind-Flat-Play-v0` | SoleFoot | SFHIMIdentifiedBlindFlatEnvCfg_PLAY | Flat | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Blind-Flat-Urdf-v0` | SoleFoot | SFIdentifiedBlindFlatEnvUrdfCfg | Flat | Yes | No | No |
| `Isaac-Limx-SF-Identified-Blind-Flat-Play-Urdf-v0` | SoleFoot | SFIdentifiedBlindFlatEnvUrdfCfg_PLAY | Flat | Yes | No | Yes |
| `Isaac-Limx-SF-Identified-Blind-Rough-v0` | SoleFoot | SFIdentifiedBlindRoughEnvCfg | Rough | Yes | No | No |
| `Isaac-Limx-SF-Identified-Blind-Rough-Play-v0` | SoleFoot | SFIdentifiedBlindRoughEnvCfg_PLAY | Rough | Yes | No | Yes |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-v0` | SoleFoot | SFHIMIdentifiedBlindRoughEnvCfg | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Play-v0` | SoleFoot | SFHIMIdentifiedBlindRoughEnvCfg_PLAY | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0` | SoleFoot | SFIdentifiedBlindRoughEnvUrdfCfg | Rough | Yes | No | No |
| `Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0` | SoleFoot | SFIdentifiedBlindRoughEnvUrdfCfg_PLAY | Rough | Yes | No | Yes |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0` | SoleFoot | SFHIMIdentifiedBlindRoughEnvUrdfCfg | Rough | Yes | Yes | No |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Play-Urdf-v0` | SoleFoot | SFHIMIdentifiedBlindRoughEnvUrdfCfg_PLAY | Rough | Yes | Yes | Yes |
| `Isaac-Limx-SF-Identified-Berkeley-v0` | SoleFoot | SFIdentifiedBerkeleyRoughEnvCfg | Flat | Yes | No | No |
| `Isaac-Limx-SF-Identified-Berkeley-Play-v0` | SoleFoot | SFIdentifiedBerkeleyRoughEnvCfg_PLAY | Flat | Yes | No | Yes |
| `Isaac-Limx-SF-Copt-Flat-v0` | SoleFoot | SFCoptBlindFlatEnvCfg | Flat | Yes | No | No |
| `Isaac-Limx-SF-Copt-Rough-v0` | SoleFoot | SFCoptBlindRoughEnvCfg | Rough | Yes | No | No |
| `Isaac-Limx-SF-Copt-Rough-Play-v0` | SoleFoot | SFCoptBlindRoughEnvCfg_PLAY | Rough | Yes | No | Yes |
| `Isaac-Limx-SF-Copt-Learned-Flat-v0` | SoleFoot | SFCoptBlindFlatEnvCfg | Flat | Yes | No | No |
| `Isaac-Limx-SF-Copt-Learned-Flat-Play-v0` | SoleFoot | SFCoptBlindFlatEnvCfg_PLAY | Flat | Yes | No | Yes |
| `Isaac-Limx-SF-Copt-Learned-Rough-v0` | SoleFoot | SFCoptBlindRoughEnvCfg | Rough | Yes | No | No |
| `Isaac-Limx-SF-Copt-Learned-Rough-Play-v0` | SoleFoot | SFCoptBlindRoughEnvCfg_PLAY | Rough | Yes | No | Yes |
| `Isaac-Limx-SDBRS1-Blind-Flat-v0` | SD_BRS1 | SDBRS1BlindFlatEnvCfg | Flat | No | No | No |
| `Isaac-Limx-SDBRS1-Blind-Flat-Play-v0` | SD_BRS1 | SDBRS1BlindFlatEnvCfg_PLAY | Flat | No | No | Yes |
| `Isaac-Limx-SDBRS1-Blind-Flat2-v0` | SD_BRS1 | SDBRS1BlindFlatEnv2Cfg | Flat | No | No | No |
| `Isaac-Limx-SDBRS1-Blind-Flat2-Play-v0` | SD_BRS1 | SDBRS1BlindFlatEnv2Cfg_PLAY | Flat | No | No | Yes |
| `Isaac-Limx-SDBRS1-Blind-Rough-v0` | SD_BRS1 | SDBRS1BlindRoughEnvCfg | Rough | No | No | No |
| `Isaac-Limx-SDBRS1-Blind-Rough-Play-v0` | SD_BRS1 | SDBRS1BlindRoughEnvCfg_PLAY | Rough | No | No | Yes |
| `Isaac-Limx-SDBRS1-HIM-Blind-Flat-v0` | SD_BRS1 | SDBRS1HIMBlindFlatEnvCfg | Flat | No | Yes | No |
| `Isaac-Limx-SDBRS1-HIM-Blind-Flat-Play-v0` | SD_BRS1 | SDBRS1HIMBlindFlatEnvCfg_PLAY | Flat | No | Yes | Yes |
| `Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0` | SD_BRS1 | SDBRS1HIMBlindRoughEnvCfg | Rough | No | Yes | No |
| `Isaac-Limx-SDBRS1-HIM-Blind-Rough-Play-v0` | SD_BRS1 | SDBRS1HIMBlindRoughEnvCfg_PLAY | Rough | No | Yes | Yes |

47 registrations, extracted from `robots/__init__.py` on 2026-07-30.

---

## 9. Developer Reference: Key Classes & Files

| Class / Object | File | Description |
|---|---|---|
| `SFEnvCfg` | `cfg/SF/limx_base_env_cfg.py` | Root common MDP config for all SF tasks |
| `SFHIMEnvCfg` | `cfg/SF/limx_base_env_cfg.py` | Root common MDP config for HIM tasks |
| `SFBerkeleyEnvCfg` | `cfg/SF/limx_berkeley_env_cfg.py` | Berkeley-style reward override config |
| `SFSceneCfg` | `cfg/SF/limx_base_env_cfg.py` | Scene: terrain, robot placeholder, sensors |
| `CommandsCfg` | `cfg/SF/limx_base_env_cfg.py` | Velocity command space definition |
| `ObservationsCfg` | `cfg/SF/limx_base_env_cfg.py` | Policy + critic observation groups |
| `HIMObservationsCfg` | `cfg/SF/limx_base_env_cfg.py` | Policy + obsHistory + critic + estimatorGT groups |
| `RewardsCfg` | `cfg/SF/limx_base_env_cfg.py` | All reward terms and their weights |
| `EventsCfg` | `cfg/SF/limx_base_env_cfg.py` | Domain randomization configuration |
| `CurriculumCfg` | `cfg/SF/limx_base_env_cfg.py` | Terrain + push + velocity curriculum |
| `TerminationsCfg` | `cfg/SF/limx_base_env_cfg.py` | Time-out and base contact termination |
| `SFBaseEnvCfg` | `robots/limx_solefoot_env_cfg.py` | Asset binding + default joint positions |
| `SFIdentifiedBlindRoughEnvCfg` | `robots/limx_solefoot_env_cfg.py` | Identified actuator, rough, blind training task |
| `SFIdentifiedBerkeleyRoughEnvCfg` | `robots/limx_solefoot_env_cfg.py` | Identified actuator, Berkeley-style training task |
| `SOLEFOOT_CFG` | `assets/config/solefoot_cfg.py` | SF articulation config (implicit actuators) |
| `SOLEFOOT_IDENTIFIED_CFG` | `assets/config/solefoot_identified_cfg.py` | SF articulation config (identified actuators) |
| `IdentifiedActuator` | `actuators/actuator_pd.py` | Custom motor model for sim-to-real |
| `SF_TRON1AFlatPPORunnerCfg` | `agents/limx_rsl_rl_ppo_cfg.py` | PPO runner config for standard SF tasks |
| `SF_Berkeley_PPORunnerCfg` | `agents/limx_rsl_rl_ppo_cfg.py` | PPO runner config for Berkeley tasks |
| `EncoderCfg` | `utils/wrappers/rsl_rl/rl_mlp_cfg.py` | Proprioceptive encoder configuration |
| `RslRlPpoAlgorithmMlpCfg` | `utils/wrappers/rsl_rl/rl_mlp_cfg.py` | Extended PPO algorithm config |
| `RslRlVecEnvWrapper` | `isaaclab_rl.rsl_rl` (external) | Isaac Lab → RSL-RL environment adapter |
| `OnPolicyRunner` | `rsl_rl.runners` (external) | Standard PPO training runner |
| `HIMOnPolicyRunner` | `himloco/runners/` | HIM PPO training runner |
| `CoptOnPolicyRunner` | `co_optimisation/runners/copt_on_policy_runner.py` | Co-optimisation runner, interleaves the evolutionary design update with the PPO update |
| `CoptActorCritic` | `co_optimisation/modules/copt_actor_critic.py` | Actor-critic whose estimator conditions the actor on the morphology |
| `GrowingDesignDistCMAESDesignGenerator` | `co_optimisation/runners/usd_generator.py` | CMA-ES design optimiser over the thigh and shank length scales |
| `DataLogger` | `scripts/rsl_rl/play.py` | Evaluation data collection and export |

Classes added since this reference was first written:

| Class / Object | File | Description |
|---|---|---|
| `CoptObservationsCfg` | `cfg/SF/limx_base_env_cfg.py` | Observation groups for both co-optimisation policy types, `policy`, `critic`, `commands`, `morphologyObs`, `predictedMorphologyObs`, `predictedPrivilegedObs`, `obsHistory` |
| `SFCoptEnvCfg` | `cfg/SF/limx_base_env_cfg.py` | Root MDP config for every `Copt` and `Copt-Learned` task |
| `CoptPPO` | `co_optimisation/algorithms/copt_ppo.py` | PPO with per-individual advantage normalisation and an explained variance metric |
| `CoptLearnedModelPPO` | `co_optimisation/algorithms/copt_ppo.py` | Adds the decoder model-estimation loss to the PPO objective |
| `CoptLearnedModelV2PPO` | `co_optimisation/algorithms/copt_ppo.py` | As above, regressing the morphology rather than the dynamic state, presently unreachable |
| `CoptEstimator` | `co_optimisation/modules/copt_estimator.py` | Encoder-decoder estimator trained jointly with PPO |
| `CoptLearnedModelActorCritic` | `co_optimisation/modules/copt_actor_critic.py` | Actor-critic pairing the estimator latent with the policy observations |
| `CoptLearnedModelV2ActorCritic` | `co_optimisation/modules/copt_actor_critic.py` | As above, with the estimator conditioned on observation history alone |
| `capture_env_state` / `restore_env_state` | `co_optimisation/utils/env_state.py` | Environment and manager state capture and restore, the basis of a complete resume |
| `HIMManagerBasedRLEnv` | `tasks/locomotion/envs/him_env.py` | Environment subclass supplying the members the HIM runner expects, registered as the entry point of every HIM task |
| `SDBRS1EnvCfg`, `SDBRS1HIMEnvCfg` | `cfg/SF/brs_base_env_cfg.py` | The two root MDP templates for the SD_BRS1 biped, independent of the SF template |
| `SDBRS1BlindFlatEnvCfg` and siblings | `robots/brs_solefoot_env_cfg.py` | The ten SD_BRS1 scenario classes, over three base classes and their play variants |
| `SD_BRS1FlatPPORunnerCfg` | `agents/limx_rsl_rl_ppo_cfg.py` | PPO runner config for SD_BRS1, the only one enabling symmetry augmentation |
| `SFCoptLearnedModelPPORunnerCfg` | `agents/limx_rsl_rl_ppo_cfg.py` | Agent config distinguishing `Copt-Learned` tasks from `Copt` tasks |
| `SD_BRS1_IDENTIFIED_CFG2` | `assets/config/sd_brs1_identified_cfg.py` | SD_BRS1 articulation on the simplified primitive URDF, bound by the `Flat2` tasks |
| SD_BRS1 observation and action mirror | `tasks/locomotion/mdp/symmetry/brs.py` | Sagittal reflection used by the rsl_rl symmetry hook |
| `GaitReward` | `tasks/locomotion/mdp/rewards.py` | Contact clock reward that broke the cadence collapse on SD_BRS1 |
| `foot_clearance_reward_v2` | `tasks/locomotion/mdp/rewards.py` | True sole clearance, replacing the ankle-frame proxy that tilting could farm |
| `knee_flexion_in_swing_v2` | `tasks/locomotion/mdp/rewards.py` | Second attempt at a swing-phase knee flexion reward |
| `terrain_levels_vel_delayed` | `tasks/locomotion/mdp/curriculums.py` | Terrain curriculum that suppresses its update during a morphology reset |
| `SD_BRS1_IDENTIFIED_CFG` | `assets/config/sd_brs1_identified_cfg.py` | SD_BRS1 articulation config |
