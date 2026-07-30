# mjlab.md — Architecture, Optimisations, and Design-Update Survey for mjlab

## 1. Purpose and Provenance

This document records a detailed investigation of the mjlab framework cloned at `/ws/mjlab/`, undertaken to determine whether mjlab can replace Isaac Lab as the simulation backend for the TRON1A design and policy co-optimisation pipeline described in [../ARCHITECTURE.md](../ARCHITECTURE.md) and [../CO_OPTIMISATION.md](../CO_OPTIMISATION.md). It answers four questions the investigation was charged with, namely the abstractions by which mjlab generates a scene and spawns robots, the GPU memory and compute optimisations mjlab and MuJoCo Warp perform for parallelisation, whether mjlab possesses an equivalent of Isaac Lab physics cloning, and what the equivalent of the extensive in-place USD updates would be for a mjlab morphology-update framework. The findings are grounded in a direct reading of the mjlab source tree, the mjlab documentation under `/ws/mjlab/docs/`, the MuJoCo Warp reference, and the mjlab paper. All prior context files in this workspace concern the Isaac Lab implementation only, so this file is the first mjlab-specific record and is intended to seed a subsequent plan and design document for co-optimisation on mjlab.

The evidence base combines codebase analysis and external literature. Codebase citations use the form `path:line`. External sources are collected in Section 14.

## 2. Executive Summary of Findings

mjlab reproduces the Isaac Lab manager-based environment API on top of MuJoCo Warp, a GPU-accelerated reimplementation of MuJoCo, and it is a deliberately lightweight framework whose entire simulation state is exposed as native MuJoCo `MjModel` and `MjData` structures rather than as a USD scene graph. The consequences for the co-optimisation port are substantial and largely favourable, and they can be stated up front.

First, mjlab does not clone physics prims. Isaac Lab replicates a robot subtree once per environment through `GridCloner`, producing `/World/envs/env_{i}/Robot` paths inside a single physics scene. mjlab instead compiles one shared `MjModel` and hands it to MuJoCo Warp, which adds a leading world dimension so that a single `MjData` holds the state of all parallel environments at once. Parallel worlds are non-interacting because they are independent slices of the batch, not because they are spatially separated, so the notion of cloning does not arise.

Second, the memory and compute optimisations follow directly from that batching model. The model is shared across worlds by default and only the specific fields that must vary per world are expanded into per-world arrays, the PyTorch and Warp buffers alias the same GPU memory through a zero-copy bridge, and the whole `step`, `forward`, `reset`, and `sense` pipeline is captured once as a CUDA graph and replayed with a single launch.

Third, and most important for co-optimisation, MuJoCo has no collision-cooking stage. In PhysX the collision geometry is baked irrevocably when the timeline starts, which is the root cause of the elaborate stop, delete, spawn, and play USD respawn sequence that the Isaac Lab pipeline performs to change a link length. In MuJoCo the collider is analytic, so link geometry, joint attachment points, and inertial properties remain ordinary model arrays that can be rewritten at runtime. mjlab already exploits this in its domain-randomisation engine, which writes `geom_size`, `body_pos`, and the full inertial set per world in place and recomputes the derived constants, with no restart and no respawn.

The practical conclusion is that the design-update framework on mjlab is far smaller than its Isaac Lab counterpart. The in-place USD update machinery, the multi-USD prototype spawning, and the stop-delete-spawn-play respawn all collapse into a single operation, namely a per-world write of the geometry and inertia fields followed by an environment reset. Section 10 sets out the full implementation blueprint.

## 3. Framework Identity and Stack

mjlab is authored by Kevin Zakka, Qiayuan Liao, Brent Yi, Louis Le Lay, Koushil Sreenath, and Pieter Abbeel, and is published as arXiv 2601.22074, titled mjlab, A Lightweight Framework for GPU-Accelerated Robot Learning. The version cloned here is 1.5.0 per `pyproject.toml`. The framework combines Isaac Lab's manager-based API with MuJoCo Warp, and it ships reference tasks for velocity tracking, motion imitation, and manipulation.

The dependency footprint is small and is the mechanism by which mjlab achieves its fast startup, which is one of its three stated commitments alongside transparent physics and tight MuJoCo ecosystem integration (`docs/source/motivation.rst`). The salient runtime dependencies from `pyproject.toml` are the following.

| Dependency | Version pin | Role |
|---|---|---|
| `mujoco` | `~=3.10.0` | CPU MuJoCo, compilation of `MjSpec` to `MjModel` |
| `mujoco-warp` | `>=3.10.0.1,~=3.10.0` | GPU batched physics backend |
| `warp-lang` | `>=1.14.0` | NVIDIA Warp kernel runtime and CUDA graph capture |
| `torch` | `>=2.7.0` | policy training, zero-copy tensor views of Warp arrays |
| `rsl-rl-lib` | `==5.4.0` | PPO training runner and actor-critic |
| `torchrunx` | `>=0.3.4` | multi-GPU distributed training |
| `viser`, `mjviser` | current | browser-based and native visualisation |

The console entry points are `train`, `play`, `demo`, `list-envs`, `viz-nan`, and `export-scene`, defined under `[project.scripts]`. An NVIDIA GPU is required for training, and macOS is supported for evaluation only.

mjlab positions itself against three neighbours in `docs/source/motivation.rst`, namely Isaac Lab, which it credits with the manager API and the USD pipeline but faults for the Omniverse runtime weight, MuJoCo Playground, which it credits with hackability but faults for code duplication across tasks, and Newton, which it credits with multi-physics and differentiable solvers. mjlab explicitly declares cross-simulator portability a non-goal, since it targets a single physics stack in order to keep the state inspectable.

## 4. Scene Generation and Robot Spawning Abstractions

mjlab is organised into a simulation layer that models the robot and the world, and a manager layer that defines the reinforcement-learning problem over it (`docs/source/architecture_overview.rst`). The simulation layer contains four components, the Entity, the actuators, the sensors, and the Scene, and the manager layer contains eight managers that mirror the Isaac Lab set. This section covers the abstractions that build the scene and place robots into it.

### 4.1 The Entity abstraction

The `Entity` class at `entity/entity.py:117` is the atomic physical object of the framework, and it corresponds to the Isaac Lab `Articulation` and `AssetBase` classes but is defined entirely through MuJoCo constructs. An entity models a single rigid or articulated body system with at most one free joint serving as its floating base (`entity.py:174-187`), and the class documents a two-by-two matrix over fixed versus floating base and non-articulated versus articulated, with the humanoid and quadruped falling into the floating articulated cell (`entity.py:137-145`).

An entity is configured by an `EntityCfg` dataclass (`entity/entity.py:69`). The most consequential field is `spec_fn`, a callable returning a `mujoco.MjSpec`, which is the editable in-memory representation of an MJCF model (`entity.py:85-87`). This is the substitution for the Isaac Lab `UsdFileCfg.usd_path`, and it means that a robot is described by an MJCF file or by procedural spec construction rather than by a USD asset. The `InitialStateCfg` inner class carries the spawn position, orientation, root velocities, and per-joint position and velocity dictionaries keyed by regex (`entity.py:71-82`), which parallels the Isaac Lab `ArticulationCfg.InitialStateCfg`. An optional `EntityArticulationInfoCfg` carries the actuator configurations and the soft joint-limit factor (`entity.py:111-115`).

Construction of an entity runs a fixed pipeline in `Entity.__init__` (`entity.py:147-155`), comprising `_build_spec`, `_identify_joints`, `_apply_spec_editors`, `_add_actuators`, and `_add_initial_state_keyframe`. The spec build either wraps the user `spec_fn` in `auto_wrap_fixed_base_mocap` for an ordinary entity, or dispatches to `build_merged_variant_spec` for a mesh-variant entity (`entity.py:157-163`), which is the branch discussed in Section 8. Joint identification isolates the single free joint from the actuated and passive joints (`entity.py:169-187`). The spec editors apply light, camera, texture, material, mesh, and collision modifier dataclasses onto the spec (`entity.py:189-199`), which is how appearance and collision filtering are layered onto a base MJCF without editing the file, the hybrid approach the architecture document highlights.

Naming and indexing are handled by stripping attachment prefixes. The entity exposes `body_names`, `joint_names`, `geom_names`, and the remaining element name tuples by taking the trailing path component after a `/` split (`entity.py:420-462`), which is necessary because scene attachment prefixes every element name. A rich family of `find_*` methods resolves regex name keys to local indices through `resolve_matching_names` (`entity.py:512-629`), which is the mjlab equivalent of the Isaac Lab `find_joints` and `find_bodies` helpers and is used pervasively by actuators, observations, and randomisation terms.

### 4.2 Actuators

The actuator configuration is where the Unitree G1 constants file (`asset_zoo/robots/unitree_g1/g1_constants.py`) is instructive, because it shows a production robot definition. Actuator groups are declared as `BuiltinPositionActuatorCfg` instances, each targeting a regex over joint names and carrying stiffness, damping, effort limit, and armature (`g1_constants.py:125-178`). The stiffness and damping are derived from motor first principles, the reflected inertia of a two-stage planetary gearbox computed by `reflected_inertia_from_two_stage_planetary`, a chosen natural frequency of ten hertz, and a damping ratio of two (`g1_constants.py:35-123`), which is a more physically grounded actuator parameterisation than a hand-tuned gain. Actuators are attached to the spec through `edit_spec`, and mjlab further vectorises the built-in actuators into a `BuiltinActuatorGroup` and fuses ideal PD actuators into a `FusedActuatorGroup` at initialisation, looping only over whatever custom actuators remain (`entity.py:666-679`). This grouping is a compute optimisation, since it applies one batched control write for the common case rather than a Python loop per joint.

The final robot configuration is assembled by `get_g1_robot_cfg`, which returns a fresh `EntityCfg` binding the keyframe, the collision configuration, the MJCF spec function, and the articulation actuator set (`g1_constants.py:264-275`). The action scale for each actuator is derived as a quarter of the effort limit over the stiffness (`g1_constants.py:278-286`), which is a deployment-convention detail worth carrying into any ported task.

### 4.3 The Scene and its composition

The `Scene` class at `scene/scene.py:52` composes entities into a single simulation model. It begins from a base `scene.xml`, optionally overrides the model extent, then adds the terrain, the entities, and the sensors, and finally applies an optional user `spec_fn` callback before compilation (`scene.py:62-69`). The `SceneCfg` dataclass carries `num_envs`, `env_spacing`, an optional terrain, a dictionary of named entity configs, a tuple of sensor configs, and the optional spec callback (`scene.py:24-49`).

The composition mechanism is the pivotal point of contrast with Isaac Lab. In `_add_entities` (`scene.py:208-245`), each entity config is built into an `Entity`, its first keyframe is extracted and deleted before attachment to avoid corruption, and the entity spec is attached into the scene worldbody through `MjSpec.attach` with a per-entity prefix such as `robot/` (`scene.py:235-236`). The per-entity keyframes are then concatenated in iteration order into one merged scene keyframe named `init_state` (`scene.py:238-245`). Terrain is attached the same way with an empty prefix (`scene.py:247-264`). The entire scene is therefore a single `MjSpec` that is compiled once by `Scene.compile` into one `MjModel` (`scene.py:71-72`).

The decisive observation is that this composition inserts one copy of each entity, not `num_envs` copies. There is no grid cloning at the scene-graph level. The multiplication into parallel environments happens later and elsewhere, at the transfer to MuJoCo Warp, which is the subject of Section 5. This is the single most important structural difference from the Isaac Lab pipeline described in ../CO_OPTIMISATION.md Section 4.3, where `GridCloner` produces one prim subtree per environment inside the USD stage before physics starts.

### 4.4 Environment origins and the absence of spatial cloning

Because parallel worlds are independent batch slices rather than co-located prims, mjlab does not need to spatially separate environments to keep them from colliding. The `env_origins` tensor exists for a narrower purpose, namely to place each robot onto its own tile of a shared terrain and to offset observations and commands. For a flat plane, `_compute_env_origins_grid` lays the origins out on a square grid at `env_spacing` metres (`terrains/terrain_entity.py:367-381`), and for procedural terrain, origins are drawn from the sub-terrain grid by difficulty row and terrain-type column (`terrain_entity.py:325-365`). The origin is applied to the robot only at reset, where `reset_scene_to_default` adds `env.scene.env_origins[env_ids]` to the default root position before writing the root state to the simulation (`envs/mdp/events.py:97-100`). The same additive offset appears in `reset_root_state_uniform` and the flat-patch reset (`events.py:152-175`).

The consequence is that the terrain is a single shared model attached once, and the many robots occupy different coordinates on that one terrain, yet a robot in world seven cannot collide with a robot in world nine because those are separate `MjData` slices. This is the resolution of the physics-cloning question. mjlab has no equivalent of `GridCloner` and needs none, and its equivalent of the Isaac Lab `replicate_physics` flag is simply the default sharing of one compiled model across the world dimension, discussed next.

## 5. GPU Parallelisation, Memory, and Compute Optimisations

The parallelisation model is inherited from MuJoCo Warp and orchestrated by the `Simulation` class at `sim/sim.py:199`. This section documents each optimisation, tying the mjlab code to the MuJoCo Warp reference behaviour.

### 5.1 The world dimension

MuJoCo Warp preserves the `MjModel` and `MjData` paradigm but adds a leading world dimension, so a single `MjData` holds the state of N independent simulation instances (`docs/source/architecture_overview.rst`, MuJoCo Warp reference). mjlab realises this in `_finish_init`, which calls `mjwarp.put_data(self._mj_model, self._mj_data, nworld=self.num_envs, nconmax=..., njmax=...)` to allocate the batched data once (`sim.py:327-333`). The model is transferred once by `mjwarp.put_model` (`sim.py:268`). Thousands of environments are therefore stepped by advancing one batched data object.

### 5.2 Shared model, selective per-world expansion

By default every `mjwarp.Model` field has a leading batch dimension of size one and applies identically to all worlds, and MuJoCo Warp indexes such a field by `worldid % field.shape[0]` so that a shared field costs the memory of a single world (MuJoCo Warp reference). A field is promoted to genuine per-world storage only when a term requires it to vary. mjlab performs this promotion in `expand_model_fields` (`sim/randomization.py:20-56`), which tiles a shared array into an `nworld` array through a Warp kernel and replaces the model attribute. The set of fields to expand is not guessed, it is declared. Each domain-randomisation function is decorated with `@requires_model_fields(...)`, which records the field names and a recompute level onto the function (`managers/event_manager.py:69-96`), and the `EventManager` collects the union of these declarations into `domain_randomization_fields` (`event_manager.py:381-385`). The environment then expands exactly that set once at load time through `self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)` (`envs/manager_based_rl_env.py:309`). The memory optimisation is therefore precise, only the randomised fields pay the per-world memory cost, and every other field stays shared.

### 5.3 Zero-copy PyTorch and Warp interoperability

The bridge between Warp arrays and PyTorch tensors is zero-copy. `TorchArray` wraps a Warp array and exposes it as a torch tensor through `wp.to_torch`, which shares the underlying GPU memory rather than copying it (`sim/sim_data.py:15-27`), and it routes indexed writes through the Warp CUDA stream to preserve ordering (`sim_data.py:67-73`). `WarpBridge` wraps an entire mjwarp struct and lazily wraps each Warp-array attribute as a `TorchArray` on access, caching the wrappers (`sim_data.py:183-221`). Crucially, `WarpBridge` forbids attribute assignment and permits only in-place mutation such as `obj.field[:] = value`, because replacing an array would change its memory address and silently break a captured CUDA graph (`sim_data.py:223-229`). The observation and reward managers thus read simulation state as torch tensors with no host round-trip and no copy, which is the central compute and memory optimisation for the reinforcement-learning loop. This contrasts with the Isaac Lab constraint noted in ../CO_OPTIMISATION.md Section 4.6, where physics writes must pass through CPU tensors of the PhysX Tensor API.

### 5.4 CUDA graph capture

The simulation captures the kernel sequences for `step`, `forward`, `reset`, and `sense` as CUDA graphs once at startup and replays each with a single launch, eliminating the CPU dispatch cost of thousands of kernels per step (`sim/sim.py:200-219`, `create_graph` at `sim.py:347-379`). Graph capture is gated on a CUDA device with memory pools enabled and a driver at least 12.4 (`sim.py:560-581`), and it is disabled gracefully otherwise. The class documentation states the invariant that a captured graph holds pointers to the arrays present at capture time, so any code that replaces a model or data array must recapture the graph (`sim.py:210-219`). `expand_model_fields` honours this by calling `create_graph` after it replaces arrays (`sim.py:451-452`). The consequence for co-optimisation, developed in Section 10, is that per-iteration in-place writes to already-expanded per-world arrays are graph-safe, and only the one-time expansion needs a recapture. The documentation reinforces this, noting that per-episode resets and domain-randomisation events run as ordinary Python between graph replays and do not break the capture (`docs/source/architecture_overview.rst`).

### 5.5 Contact and constraint memory, the solver, and broadphase

MuJoCo Warp bounds contact and constraint memory by the `nconmax` and `njmax` parameters, where memory and computation scale with these values and they should be set as small as the simulation allows (MuJoCo Warp reference). mjlab surfaces both on `SimulationCfg` with heuristic defaults when left unset (`sim/sim.py:155-166`), and the Unitree Go1 configuration tunes them per terrain, setting `njmax=300` and a smaller contact-sensor match bound on flat ground while raising continuous-collision iterations and the contact-sensor match bound on rough terrain (`tasks/velocity/config/go1/env_cfgs.py:38-41,265-268`). The solver is the MuJoCo Warp Newton solver, which early-exits once all worlds have converged and thereby avoids wasted iterations, and dense inertia factorisation uses Warp tile Cholesky (MuJoCo Warp reference). `MujocoCfg` exposes the integrator, cone, solver, iteration counts, and tolerances (`sim.py:97-153`), and `SimulationCfg` additionally exposes the broadphase algorithm and bounding-volume filters applied after `put_model` (`sim.py:168-196`). These are the compute knobs a co-optimisation task would tune, since a distribution of morphologies changes the contact and constraint counts.

### 5.6 NaN guarding

Given that the co-optimisation investigation in ../plans/COPT_INVESTIGATION_PLAN.md concerns training instability and NaN crashes in the Isaac Lab Berkeley task, it is worth noting that mjlab wraps every `step` in a `NanGuard` context (`sim/sim.py:345,492-498`) and ships a `viz-nan` script, which provides a native facility for the class of failure that has troubled the Isaac Lab runs.

## 6. The Environment Lifecycle and Step Loop

The manager-based environment is `ManagerBasedRlEnv` at `envs/manager_based_rl_env.py:166`, configured by `ManagerBasedRlEnvCfg` (`manager_based_rl_env.py:51`). The lifecycle has four phases, build, initialise, reset, and step, which map closely onto the Isaac Lab lifecycle in ../CO_OPTIMISATION.md Section 4.4 but without any USD or Fabric stage.

The build and initialise phases run in `__init__` (`manager_based_rl_env.py:177-251`). The scene is constructed, the simulation is constructed from the scene spec and any variant metadata, the scene is initialised against the compiled model and the warp model and data, and the sensor context is wired for the sense graph. Managers are then loaded by `load_managers` (`manager_based_rl_env.py:299-354`) in a deliberate order, the event manager first so that its domain-randomisation field set can drive `expand_model_fields`, then the command, action, observation, termination, reward, curriculum, metrics, and recorder managers, and finally the startup events fire once. The ordering is the reason the field expansion and the associated CUDA-graph recapture occur exactly once, before any training step.

The step loop is `step` (`manager_based_rl_env.py:378-479`), and its ordered body is the clearest statement of the MDP cycle.

```text
action_manager.process_action(action)
for _ in range(decimation):
    action_manager.apply_action()
    scene.write_data_to_sim()      # actuator targets -> ctrl
    sim.step()                     # one CUDA-graph replay of mjwarp.step
    scene.update(dt)               # advance actuator internal state
    metrics_manager.compute_substep()
termination_manager.compute()
reward_manager.compute(dt=step_dt)
metrics_manager.compute()
[reset terminated envs via _reset_idx, then scene.write_data_to_sim]
sim.forward()                      # single refresh of derived quantities
command_manager.compute(dt=step_dt)
event_manager.apply(mode="step") and apply(mode="interval")
sim.sense()                        # BVH refit, cameras, raycasts
observation_manager.compute(update_history=True)
```

Two design points matter for co-optimisation. First, a single `sim.forward()` refreshes derived quantities for all worlds, including both the decimation-advanced worlds and the freshly reset worlds, which the code justifies as a consistent one-substep staleness that keeps the MDP well defined (`manager_based_rl_env.py:386-410,450-454`). Second, `_reset_idx` (`manager_based_rl_env.py:553-591`) drives the curriculum, then `sim.reset(env_ids)`, then `scene.reset`, then the reset-mode events, then the per-manager resets, which is the hook set into which a per-morphology reset would slot.

The reset itself is a masked operation. `Simulation.reset` fills a boolean per-world mask and replays the reset graph over `mjwarp.reset_data`, which returns the masked worlds to the `init_state` keyframe (`sim/sim.py:500-511`). There is no timeline stop, no prim deletion, and no re-parse, which is why a mjlab reset is cheap relative to the Isaac Lab stop-and-play cycle.

## 7. Physics Cloning, the Isaac Lab Equivalence

The user's question of whether mjlab implements cloning of physics as Isaac Lab does can now be answered precisely. Isaac Lab clones a robot prim subtree once per environment through `GridCloner`, and it offers a `replicate_physics` flag that reuses one physics representation across the clones for speed, and the Isaac Lab co-optimisation task deliberately sets `replicate_physics=False` so that a distinct articulation can be spawned per environment (../CO_OPTIMISATION.md Section 4.7, ../ARCHITECTURE.md Section 5.1). mjlab has no such feature and requires none. The equivalent of replicated physics is the default behaviour of MuJoCo Warp, where one compiled model is shared across the world dimension and only selected fields are expanded per world. The equivalent of disabling replication, that is, of giving each environment a genuinely different physical model, is achieved by two distinct mechanisms in mjlab, the per-world mesh variant system for a fixed discrete set of bodies, and per-world model-field writes for continuous parameter variation. These are the subjects of Sections 8 and 9, and together they are the substrate onto which the co-optimisation design updates map.

## 8. Per-World Mesh Variants, the Static MultiUSD Analogue

mjlab supports running a single batched simulation in which different worlds use different mesh assets for the same logical entity, exposed through `VariantEntityCfg` and documented at `docs/source/entity/per_world_mesh.rst`. This is the closest structural analogue to the Isaac Lab `MultiUsdFileCfg` used by the co-optimisation task, and understanding both its power and its limits is essential.

### 8.1 What the mechanism does

A `VariantEntityCfg` supplies a dictionary of named variants, each a callable returning an `MjSpec`, and an optional assignment describing how worlds map to variants (`entity/variants.py:95-127`). The build proceeds in two phases (`variants.py:1-35`). At authoring time, `build_merged_variant_spec` validates that all variants share kinematic topology, computes a slot identity for each variable mesh geom, and merges every variant's mesh assets into one padded template spec (`variants.py:860-1066`). At simulation-init time, `build_variant_model` compiles the merged scene once into the canonical model, then compiles each variant's original source spec in isolation and scatters the geometry-dependent fields into per-world arrays keyed by a world-to-variant assignment (`variants.py:1317-1442`).

The per-world variation is carried by three categories of override (`docs/source/entity/per_world_mesh.rst`). The first is `geom_dataid`, an `(nworld, ngeom)` table whose row for a world selects which compiled mesh each slot points at, with minus one as the skip sentinel that MuJoCo Warp already understands. The second is the set of mesh-derived fields listed in `VARIANT_DEPENDENT_FIELDS`, namely `geom_size`, `geom_rbound`, `geom_aabb`, `geom_pos`, `geom_quat`, `body_mass`, `body_subtreemass`, `body_inertia`, `body_invweight0`, `body_ipos`, and `body_iquat`, stored as per-world arrays (`variants.py:59-71`). The third is the set of per-mesh-geom contact attributes captured verbatim per variant. The inertial correctness is guaranteed by compiling each variant's source spec alone and writing the resulting inertia into the per-world array, which the documentation verifies against independent compiles by regression test.

### 8.2 Cost and assignment

Construction cost is linear in the total variant count, since the merged scene compiles once and each variant's isolated source spec compiles once, at roughly one to two milliseconds per compile for typical procedural meshes, so a hundred variants cost a few hundred milliseconds at startup (`docs/source/entity/per_world_mesh.rst`). Per-step cost is unaffected by variant count, because the variant-dependent fields are accessed by world index in the existing kernels with no branching. World assignment uses the largest-remainder method over per-variant weights, or an explicit callable mapping world index to variant index (`variants.py:1288-1314`, `docs/source/entity/per_world_mesh.rst`), and the resolved assignment is readable at runtime through `env.sim.world_to_variant`.

### 8.3 The decisive limitation for co-optimisation

The variant assignment is fixed at simulation initialisation and there is no API to swap a world to a different variant on episode reset, so per-episode or per-iteration mesh reassignment is not supported (`docs/source/entity/per_world_mesh.rst`, limitations section, and `variants.py:106-109`). Furthermore, variants must share the body tree, the joint topology, the actuator and sensor counts, and the inertial representation, so only mesh geoms may differ. These constraints make `VariantEntityCfg` well suited to training a policy across a fixed heterogeneous distribution of bodies decided before training, but ill suited to a CMA-ES loop that emits a fresh continuous design every fixed number of iterations, since that loop requires the per-world design to change during training. The variant system is therefore not the vehicle for the co-optimisation morphology update, though it could serve a related purpose such as training robustness across a discrete catalogue of pre-authored morphologies. The vehicle is the model-field write path of Section 9.

## 9. Domain Randomisation as Per-World Model-Field Writes

mjlab's domain-randomisation subsystem, under `envs/mdp/dr/`, is the general facility for making parallel worlds physically differ by writing model fields per world at runtime. It is the true functional analogue of the Isaac Lab combination of the PhysX Tensor API writes and the in-place USD attribute updates, and it is the substrate for a continuous morphology update.

### 9.1 The write engine

The core engine is `_randomize_model_field` (`envs/mdp/dr/_core.py:48-148`). It resolves an entity and a set of element indices from a `SceneEntityCfg`, indexes the target model field as an `(env, entity, axis)` view, samples values under a distribution and an operation, combines them with either the current values or the stored defaults, and writes only the targeted axes back into the field. The operations are the familiar absolute, additive, and scaling combinators (`envs/mdp/dr/_types.py`, exported at `envs/mdp/dr/__init__.py`), and defaults are drawn from `sim.get_default_field`, with per-world defaults respected for variant scenes (`_core.py:412-436`). The write targets `env.sim.model`, which is the `WarpBridge`, so the assignment is an in-place mutation of the per-world Warp array and is therefore CUDA-graph safe once the field has been expanded.

### 9.2 The catalogue of writable fields

The exported randomisation functions, enumerated from `envs/mdp/dr/__init__.py`, define the full set of model fields mjlab can write per world at runtime. The subset relevant to morphology co-optimisation is the following.

| Function | Model field(s) written | Recompute level | Relevance to link-length design |
|---|---|---|---|
| `geom_size` | `geom_size`, and recomputes `geom_rbound`, `geom_aabb` | none | link geometry, the box or capsule half-extents |
| `geom_pos` | `geom_pos` | none | offset of a link geom within its body |
| `body_pos` | `body_pos` | `set_const_0` | joint attachment point, moves child links when a parent lengthens |
| `body_com_offset` / `body_ipos` | `body_ipos` | `set_const` | centre-of-mass shift with length |
| `body_mass` | `body_mass` | `set_const` | mass, warns to prefer `pseudo_inertia` |
| `pseudo_inertia` | `body_mass`, `body_ipos`, `body_inertia`, `body_iquat` jointly | `set_const` | physically consistent mass and inertia scaling |
| `dof_armature` / `joint_armature` | `dof_armature` | `set_const_0` | reflected actuator inertia if the actuator scales |
| `joint_stiffness`, `joint_damping`, `joint_friction` | corresponding dof fields | none | actuator gain reapplication after a design change |

The `geom_size` function is the direct evidence that runtime geometry mutation is a first-class operation, since after writing new sizes it recomputes the bounding sphere `geom_rbound` and the local box `geom_aabb` for the supported primitive types, sphere, capsule, ellipsoid, cylinder, and box, so that the broadphase stays consistent (`envs/mdp/dr/geom.py:34-111,250-286`). This is the operation that PhysX cannot perform at runtime, and it exists here as a supported, tested code path.

### 9.3 Recompute levels and derived constants

Every field write that changes a mass, an inertia, or a kinematic offset must refresh the derived model constants, and mjlab encodes this dependency as a `RecomputeLevel` enumeration (`managers/event_manager.py:25-66`). The level `set_const_fixed` recomputes `body_subtreemass`, the level `set_const_0` additionally recomputes the inverse-weight and tendon-length constants after changes to `dof_armature`, `body_inertia`, `body_pos`, `body_quat`, or `qpos0`, and the level `set_const` performs the full recomputation after changes to `body_mass` or `body_ipos`. The `EventManager.apply` method tracks the strongest level fired across all events in a mode and calls `self._env.sim.recompute_constants(strongest_fired)` exactly once at the end (`event_manager.py:260-332`), which dispatches to the corresponding MuJoCo Warp routine (`sim/sim.py:472-483`). A morphology update that changes geometry and inertia therefore has a precise, already-implemented recompute contract to satisfy.

### 9.4 The pseudo-inertia path

For inertial changes mjlab recommends `pseudo_inertia` (`envs/mdp/dr/body.py:416-541`), which jointly randomises mass, centre of mass, principal moments, and principal-frame orientation through the pseudo-inertia matrix factorisation of Rucker and Wensing, 2022, and is exact for any perturbation magnitude. This matters for a scale-based morphology change, because lengthening a link should scale both its mass and its inertia consistently, and `pseudo_inertia` provides that through its global density-scale parameter alpha while `body_mass` alone leaves the inertia untouched and emits a warning (`body.py:297-314`). The implementation avoids the cuSOLVER library that would otherwise allocate several gigabytes on first use, using an analytic four-by-four Cholesky and a Jacobi three-by-three eigensolver (`body.py:39-134`), which is itself a memory optimisation worth noting.

### 9.5 A worked configuration

The Unitree Go1 velocity configuration demonstrates the DR terms in practice, applying per-axis foot-friction randomisation at startup, a base centre-of-mass offset, an encoder bias, joint offsets and root-state resets on episode reset, and a periodic push (`tasks/velocity/config/go1/env_cfgs.py:147-184`, `tasks/velocity/velocity_env_cfg.py:201-258`). Each term is an `EventTermCfg` with a mode of startup, reset, interval, or step, a function, and a parameter dictionary, which is the same term-and-manager pattern the co-optimisation morphology update would follow if implemented as an event, or would emulate if driven from the training runner.

## 10. Implementation Blueprint for Co-optimisation on mjlab

This section maps each element of the Isaac Lab co-optimisation pipeline, documented in ../ARCHITECTURE.md Section 5.1 and ../CO_OPTIMISATION.md, onto its mjlab realisation, and sets out what must be built.

### 10.1 The mapping

| Isaac Lab COPT element | mjlab equivalent | Effort |
|---|---|---|
| `MultiUsdFileCfg` with `replicate_physics=False` per-env articulation | shared compiled model plus per-world field writes, no multi-asset spawn needed | eliminated |
| `apply_link_length_params` authoring absolute box extents on USD prototypes | a function writing `geom_size`, `geom_pos`, `body_pos`, and inertia per world | reimplement, smaller |
| `update_articulation_links` editing USD scale and collider size | direct in-place writes to `env.sim.model.*` per world | eliminated |
| Stop, delete prim, spawn USD, play respawn sequence | none, in-place writes plus `env.reset()` | eliminated |
| `CoptOnPolicyRunner` interleaving the CMA-ES update every 120 iterations | subclass of `MjlabOnPolicyRunner` overriding `learn` | reimplement |
| `GrowingDesignDistCMAESDesignGenerator` on the unit square | reusable unchanged, it is pure Python over numpy | reuse |
| `CoptActorCritic` conditioning the actor on a morphology latent | rsl-rl policy augmentation plus a morphology observation term | reuse or reimplement |
| Actuator reapplication after a design change | `dof_armature` and gain writes, or an actuator-group refresh | small |

### 10.2 The morphology-apply function

The Isaac Lab `apply_link_length_params` authors absolute box extents onto each design's USD prototype and runs one `sim.reset()` (copt.md, `co_optimisation/utils/update.py`). The mjlab counterpart is a function that, for each scalable link, writes the new half-extent into `geom_size`, shifts the dependent quantities, and updates inertia, all per world. Concretely, lengthening a thigh box requires four coordinated writes, the thigh `geom_size` axial half-extent, the thigh `geom_pos` if the geom is anchored at one end rather than centred, the child body `body_pos` so the knee stays attached at the new link end, and the thigh inertia through `pseudo_inertia` or an analytic box-inertia formula. These are exactly the fields the DR catalogue already writes, so the function is assembled from existing primitives rather than invented. After the writes, one call to `sim.recompute_constants(RecomputeLevel.set_const)` refreshes the derived constants, and one `env.reset()` returns the robots to a valid configuration on the terrain.

### 10.3 Field expansion and graph safety

For the per-world writes to take effect, the target fields must be expanded to per-world storage before training, which requires them to appear in the environment's `domain_randomization_fields`. Two routes achieve this. The first registers the morphology fields by declaring one or more event terms decorated with `requires_model_fields` for `geom_size`, `geom_pos`, `body_pos`, and the inertial set, so the standard `load_managers` expansion covers them. The second calls `env.unwrapped.sim.expand_model_fields((...))` once at runner construction for the same field tuple. Either route triggers the one-time CUDA-graph recapture inside `expand_model_fields` (`sim/sim.py:434-452`). Thereafter the per-iteration morphology writes are in-place mutations of already-expanded arrays and are graph-safe, so the outer co-optimisation loop imposes no further recapture cost. This is the central correctness invariant of the port, and it is the reason the mjlab morphology update is cheap where the Isaac Lab respawn is expensive.

### 10.4 The runner and the outer loop

The mjlab training runner is `MjlabOnPolicyRunner` (`rl/runner.py:11`), a subclass of the rsl-rl `OnPolicyRunner` that already extends `save` and `load` to persist the environment step counter across checkpoints (`rl/runner.py:67-141`), and the velocity task subclasses it further as `VelocityOnPolicyRunner`. The co-optimisation runner would subclass `MjlabOnPolicyRunner` and override `learn` to interleave the CMA-ES design update with the PPO update at a fixed cadence, which is the same override point the Isaac Lab `CoptOnPolicyRunner` uses. Within that override, at the update boundary, the runner advances the design generator, distributes the population across the worlds round-robin exactly as the Isaac Lab pipeline does over its 4096 environments, calls the morphology-apply function of Section 10.2, and resets. Because save and load already persist environment state, the checkpoint extension needed to persist the CMA-ES generator state and the design assignment is a small addition following the existing `save` and `load` pattern, and it directly addresses the redundant-retraining objective recorded in knowledge_base.md.

### 10.5 Morphology conditioning of the policy

The Isaac Lab `CoptActorCritic` concatenates a morphology latent, produced by an estimator from privileged observations, to the actor input, and its critic observes the true per-environment link lengths (../ARCHITECTURE.md Section 5.1). This half of the pipeline is backend-agnostic, since it lives in the rsl-rl policy and in the observation configuration, not in the simulator. On mjlab the true per-world design is directly available, either as the tensor the runner wrote or by reading `env.sim.model.geom_size` for the scalable geoms, so an observation term that exposes the link-length parameters to the critic, and optionally to an estimator feeding the actor, is straightforward to author as a standard observation function. The mjlab observation manager supports asymmetric actor and critic groups, so the privileged-critic pattern transfers without friction.

### 10.6 What does not transfer, and residual risks

The topology of the robot cannot vary per world under either mjlab mechanism, since both the variant system and the field-write path require a shared body and joint structure, so a co-optimisation over the number of links or joints is out of scope, exactly as it is under Isaac Lab. The scalable-length co-optimisation of the current TRON1A pipeline, which varies only `thigh_length_scale` and `shank_length_scale` within fixed topology, is fully supported. The residual risks to validate empirically are three. First, the box-inertia and centre-of-mass bookkeeping must be authored correctly, since an inconsistent inertia would corrupt the dynamics, and the `pseudo_inertia` path or a verified analytic formula should be used rather than `body_mass` alone. Second, the contact and constraint memory bounds `nconmax` and `njmax` must accommodate the largest morphology in the distribution, since the Go1 configuration shows these are tuned per scene. Third, the reset cadence at the morphology update must be validated against the training-variance concern already open for the Isaac Lab pipeline in ../plans/COPT_INVESTIGATION_PLAN.md and knowledge_base.md, because a reset of all worlds at the update boundary perturbs the reward and metric buffers in mjlab just as it does in Isaac Lab.

## 11. Point-by-Point Answers to the Investigation Questions

The scene-generation and robot-spawning abstractions are the `Entity` and `EntityCfg` classes built from a `mujoco.MjSpec`, composed by the `Scene` through `MjSpec.attach` with per-entity prefixes into one compiled `MjModel`, with actuators, sensors, and initial-state keyframes layered on by modifier configs (Section 4).

The GPU memory and compute optimisations are the shared model with selective per-world field expansion, the zero-copy `WarpBridge` and `TorchArray` tensor views, the CUDA-graph capture of the step, forward, reset, and sense pipelines, the bounded contact and constraint allocations, the early-exiting Newton solver, and the tile Cholesky factorisation, all inherited from MuJoCo Warp and orchestrated by the mjlab `Simulation` class (Section 5).

mjlab does not implement physics cloning and does not need it, because MuJoCo Warp represents parallel environments as independent slices of a batched `MjData` over a shared model rather than as replicated prims, so the Isaac Lab `GridCloner` and `replicate_physics` have no counterpart and `env_origins` serves only to place robots on shared terrain (Sections 5 and 7).

The equivalent of the Isaac Lab in-place USD updates is the domain-randomisation model-field write path, which mutates `geom_size`, `body_pos`, and the inertial fields per world in place at runtime and recomputes the derived constants, with no collision cooking and therefore no respawn, making the mjlab design-update framework a small per-world write followed by a reset rather than a stop-delete-spawn-play sequence (Sections 9 and 10).

## 12. Open Questions and Verification TODOs

The following items were not fully resolved and should be verified before or during the plan and design document.

The exact geom parameterisation of the intended mjlab TRON1A robot is unknown, since mjlab ships G1, Go1, and a YAM arm but not TRON1A, so whether the thigh and shank are boxes or capsules in the MJCF determines which `geom_size` axis encodes length and whether `geom_pos` needs adjustment. This should be settled by inspecting the MJCF once the robot is ported.

Whether `sim.recompute_constants` and the per-world inertia writes compose correctly under an active CUDA graph in the exact sequence the runner would call them should be confirmed by a small standalone test, following the pattern of the variant regression test that checks inertia against independent compiles.

The interaction between a full-batch reset at the morphology-update boundary and the reward and episode buffers should be examined in mjlab specifically, since the analogous Isaac Lab behaviour is a documented source of training-variance in ../plans/COPT_INVESTIGATION_PLAN.md.

The multi-GPU path through `torchrunx` and its effect on the morphology-update broadcast across ranks was not investigated and is relevant if training uses the `--gpu-ids` distributed mode.

## 13. File and Symbol Reference Map

| Symbol or file | Location | Role |
|---|---|---|
| `Entity`, `EntityCfg` | `entity/entity.py:117,69` | robot and object abstraction, built from `MjSpec` |
| `EntityData` | `entity/data.py` | per-world state and target tensors |
| `VariantEntityCfg`, `build_merged_variant_spec`, `build_variant_model` | `entity/variants.py:95,860,1317` | static per-world mesh variants, MultiUSD analogue |
| `VARIANT_DEPENDENT_FIELDS` | `entity/variants.py:59` | mesh-derived fields stored per world |
| `Scene`, `SceneCfg` | `scene/scene.py:52,24` | composition of entities into one `MjModel` |
| `Simulation`, `SimulationCfg`, `MujocoCfg` | `sim/sim.py:199,155,97` | MuJoCo Warp orchestration, CUDA graphs, recompute |
| `expand_model_fields` | `sim/randomization.py:20` | shared-to-per-world field promotion |
| `WarpBridge`, `TorchArray` | `sim/sim_data.py:183,15` | zero-copy torch views of Warp arrays |
| `ManagerBasedRlEnv`, `ManagerBasedRlEnvCfg` | `envs/manager_based_rl_env.py:166,51` | environment, step loop, lifecycle |
| `EventManager`, `RecomputeLevel`, `requires_model_fields` | `managers/event_manager.py:143,25,69` | event dispatch, derived-constant recompute contract |
| `_randomize_model_field` | `envs/mdp/dr/_core.py:48` | the per-world field write engine |
| `geom_size`, `_recompute_geom_bounds` | `envs/mdp/dr/geom.py:250,34` | runtime geometry mutation and bound recompute |
| `body_pos`, `body_mass`, `pseudo_inertia` | `envs/mdp/dr/body.py:360,284,416` | attachment-point and inertial writes |
| `reset_scene_to_default` | `envs/mdp/events.py:71` | applies `env_origins` and resets root state |
| `TerrainEntity`, `TerrainEntityCfg` | `terrains/terrain_entity.py:109,66` | terrain as a shared entity, `env_origins` |
| `RslRlVecEnvWrapper` | `rl/vecenv_wrapper.py:9` | mjlab to rsl-rl adapter |
| `MjlabOnPolicyRunner` | `rl/runner.py:11` | rsl-rl runner with checkpoint state persistence |
| `get_g1_robot_cfg`, `BuiltinPositionActuatorCfg` | `asset_zoo/robots/unitree_g1/g1_constants.py:264,125` | reference robot and actuator definition |
| `unitree_go1_rough_env_cfg` | `tasks/velocity/config/go1/env_cfgs.py:32` | reference quadruped task with DR and contact sensors |

## 14. Sources

Codebase, `/ws/mjlab/` at version 1.5.0, files enumerated in Section 13.

mjlab documentation, `/ws/mjlab/docs/source/`, in particular `architecture_overview.rst`, `motivation.rst`, `migration_isaac_lab.rst`, and `entity/per_world_mesh.rst`.

mjlab paper, Kevin Zakka, Qiayuan Liao, Brent Yi, Louis Le Lay, Koushil Sreenath, and Pieter Abbeel, mjlab, A Lightweight Framework for GPU-Accelerated Robot Learning, arXiv:2601.22074, 2026, https://arxiv.org/abs/2601.22074.

MuJoCo Warp reference documentation, https://mujoco.readthedocs.io/en/stable/mjwarp/index.html, on the world batch dimension, shared-versus-per-world model fields, `nconmax` and `njmax` allocation, CUDA graph capture, and the early-exiting Newton solver with tile Cholesky factorisation.

Isaac Lab comparison points, drawn from the workspace documents ../ARCHITECTURE.md, ../CO_OPTIMISATION.md, knowledge_base.md, and copt.md, for the `GridCloner`, `MultiUsdFileCfg`, `replicate_physics`, `apply_link_length_params`, and the stop-delete-spawn-play respawn sequence.

## 15. Citation re-verification against the restored clone (2026-07-30)

The clone that this survey cites had been removed from the workspace and has since been restored at `/ws/mjlab`, so the line citations of section 13 could be checked directly rather than taken on trust. The restored tree reports version 1.5.3 with head commit `15ebce88` dated 2026-07-28, which places it roughly a fortnight ahead of the sources this survey read.

Every file and every symbol named in the reference map of section 13 is present. The line numbers have drifted, in most cases by one or two lines, and in one case substantially, the `Entity` class that the map places at `entity/entity.py:117` now begins at line 30. Citations in this document should therefore be resolved by symbol name rather than by position, which is the general rule this workspace already follows.

The survey's substantive findings stand unaltered. Only two commits since the sources were read touch the domain randomisation or variant machinery, `f643d245` and `15ebce88`, and both merely add light configuration fields and their randomisation functions, so the write engine of section 9 and the co-optimisation blueprint of section 10 are unaffected. The randomisation package now carries a `light.py` beside the existing `actuator.py`, `body.py`, `camera.py`, `geom.py`, `joint.py`, `material.py`, `pair.py`, `site.py`, and `tendon.py`, which extends the catalogue of section 9.2 without changing its structure.

One symbol that did not exist when the survey was written deserves a note, since its name invites confusion. A `model_sync` module now appears under `viewer/`, consumed by the native viewer, the viser scene, and the offscreen renderer. It synchronises the model to the visualiser and has no part in the per-world field write path, so it must not be mistaken for a mechanism bearing on the morphology update.

The open questions of section 12 are re-examined as follows. The first stands, the asset zoo still ships only `unitree_g1`, `unitree_go1`, and `i2rt_yam`, so the geom parameterisation of a ported TRON1A remains undetermined and must be settled from its MJCF once written. The second and the third stand, both requiring a run on a GPU rather than a reading of the source. The fourth is narrowed rather than closed, the distributed path is confirmed to run through `torchrunx`, pinned at version 0.3.4 or later in `pyproject.toml`, with device selection handled by `utils/gpu.py`, and the behaviour of a morphology-update broadcast across ranks remains unverified.
