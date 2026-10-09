# design_generator_integration.md

Status, APPLIED 2026-09-25. Revision 2, 2026-09-25, incorporating the fifteen points of review raised against revision 1 of 2026-09-22. Phases 1 to 8 of section 14 are applied to the tree, phase 9 being the two ablations rather than a code change. Section 19 records the outcome, the four gates closed by measurement and the seven divergences from what this document specified. Do not apply this plan a second time.

## 0. What Changed in This Revision

Revision 1 kept the vendored generator untouched and bridged it to the simulator with four new modules. The review reversed that division of labour, and this revision moves every bridging concern into the generator itself, where all of its consumers see it. Four proposed modules therefore disappear, one of them absorbed into an existing file, and the section numbering is no longer comparable with revision 1's.

| Revision 1 | Revision 2 | Why |
| --- | --- | --- |
| §9 `runners/design_family.py` | gone | The submodule is now an importable package, so `from design_generator import biped, quadruped` needs no loader. See §4.9. |
| §10 `runners/mjcf_sanitiser.py` | gone, absorbed into `build_xml(spec, for_isaaclab=True)` | Isaac Lab consumes a robot description, MuJoCo a scene. The distinction belongs in the emitter. See §9.1. |
| §11 `runners/mjcf_defaults.py` | gone, absorbed into the two `constant_params.json` and `variable_params.json` | Every consumer of the generator should see the donor-matched parameter set, not only the co-optimisation package. See §9.3 and §9.4. |
| §12 `runners/mjcf_actuators.py` | gone, absorbed into `class Actuator` and `class CatalogActuator` | The simulator-side numbers are properties of an actuator, not of a side table. See §9.5. |
| §13 `runners/mjcf_generator.py` | gone, `runners/usd_generator.py` rewritten in place | See §10. |
| §14 `utils/update_mjcf.py` | gone, `utils/update.py` rewritten in place | See §11. |
| §15.3 `patch_dc_motor_envelope` | gone, folded into `apply_actuator_params` | See §12. |
| §4.6, six paragraphs | §4.6, nine subsections with both pipelines drawn, walked, and compared | See §4.6. |
| `class RandomPopulation` | gone, `class Population` made concrete | See §10. |
| `SOLEFOOT_IDENTIFIED_CFG_URDF` retained as an alias | gone, every call site moves to `SOLEFOOT_IDENTIFIED_CFG` | See §6.1. |
| CMA-ES with rounding over a relaxed integer | `cmaes.CatCMAwM` | See §10. |

One further change was forced by the review rather than requested by it. Reading the Isaac Sim MuJoCo importer line by line, as §4.6 demanded, showed that it parses every geom's `density` attribute and never uses it, which silently discards the entire linear mass model the generator is built on. §4.4 states the finding and §9.2 states the repair.

## 1. Introduction

The co-optimisation pipeline searches over robot morphologies while a policy trains on them. Its present design generator at `co_optimisation/co_optimisation/runners/usd_generator.py` does that by mutating a copy of `environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf` per individual, rewriting a box size, a mass, an inertia and a joint origin, and converting the result to Universal Scene Description. That approach carries the donor robot's topology as an immovable given, cannot reach the actuator, and has no counterpart for the quadruped.

The vendored submodule at `co_optimisation/co_optimisation/runners/design_generator` replaces the mutation with an emission. It holds a parametric model of each robot family, a catalogue of real actuators, and a builder that writes a complete MuJoCo document from a `DesignSpec`. This plan retires the mutating generator, moves both families onto the emitted document, and widens the search from two link lengths to link lengths and actuator choices together.

The document is written to be executed. Every module it changes appears in full or as an exact edit, every number it quotes was computed rather than recalled, and §15 lists the gates that separate what has been verified here from what can only be verified against a running simulator.

## 2. The Incumbent Pipeline, Read End to End

`CoptOnPolicyRunner` drives the search. At construction it reads a `copt` block from the agent configuration for the update interval, the late-start iteration and the population size (`copt_on_policy_runner.py:106-121`). Every `ea_update_interval` iterations it computes a per-individual fitness, tells the generator, asks for a new population, and applies it.

### 2.1 The three interfaces the runner depends upon

The runner never touches a URDF, a document or a converter. It consumes a `Population` through three methods, `get_usd_files`, `get_actuator_params` and `get_link_length_params` (`usd_generator.py:261-283`), and a generator through four, `generate_population`, `update_with_fitness`, `sample_batch` and the `get_state` and `load_state` pair used by checkpointing (`copt_on_policy_runner.py:175, :221, :574-581`). Preserving those seven names is what lets this plan replace everything behind them without touching the runner's control flow.

### 2.2 The two morphology pathways

`_reload_morphology` deletes every robot prim and respawns the population through `spawn_multi_usd_file` (`copt_on_policy_runner.py:569-599`, `utils/respawn.py:33-227`). `_update_morphology` instead edits the existing assets in place through `apply_link_length_params` and resets (`copt_on_policy_runner.py:643-706`, `utils/update.py:129-175`). The first is correct always and costs a full respawn, the second is cheap and correct only while the number of distinct assets is unchanged.

### 2.3 What the generators search over

`RandomDesignGenerator` samples eighteen scale factors and applies them to the URDF (`usd_generator.py:569-693`). `CMAESDesignGenerator` narrows the search to two, `thigh_length_scale` and `shank_length_scale` (`usd_generator.py:238-243`), and `GrowingDesignDistCMAESDesignGenerator` widens its late-start prior from a point to the full box over the pre-training schedule (`usd_generator.py:854-908`).

## 3. The Submodule's Three Interfaces

The generator presents three things to a consumer. `DesignSpec` is a frozen dataclass of the free design variables and a tier of fixed-constant clearance fields (`biped/generate.py:69-158`, `quadruped/generate.py:68-137`). `build_xml(spec)` emits a complete MuJoCo document (`biped/builder.py:412`, `quadruped/builder.py:395`). `ACTUATOR_CATALOG` is a flat list of `CatalogActuator` records loaded from `catalogs/actuators.json` (`common/actuator_catalog.py:478-508`), which is the shelf mode, the alternative being an enumeration over the published mass and diameter tables of COMPAct [3] that the submodule vendors under `catalogs/compact/`. This plan searches the shelf, every entry of which is a real unit with a measured envelope, and leaves the enumeration untouched.

Two further layers sit behind those. `constant_params.json` holds every hardware constant the builder reads at import time, from box cross-sections to joint ranges to the linear mass densities. `variable_params.json` holds one complete `DesignSpec` as a flat object, which `load_design_spec` turns into the default design.

## 4. The Findings That Amend the Requirements

### 4.1 The format bridge exists, and the Isaac Lab MuJoCo converter's options do not all bind

`isaaclab.sim.converters.MjcfConverter` wraps the `isaacsim.asset.importer.mjcf` extension and produces an instanceable Universal Scene Description asset in the same shape the URDF converter does (`mjcf_converter.py:20-105`). `MjcfFileCfg` spawns from it exactly as `UrdfFileCfg` does. The bridge the requirements assumed to be missing is therefore present, and the co-optimisation pathway needs no new spawner.

Four of the configuration's options do not reach the importer, and a reader who sets them will be misled.

`make_instanceable` is never read by the plugin. The field is forwarded to `set_make_instanceable` and stored in `ImportConfig::makeInstanceable`, which no line of the importer consults, and `instanceableMeshUsdPath` is read once into a local that is passed down the recursion and never used (`MjcfImporter.cpp:278`). Instancing nonetheless happens, unconditionally, because the importer marks every body's `visuals` and `collisions` scope instanceable as it creates them (`MjcfImporter.cpp:533-535`, `MjcfImporter.cpp:1279`). The effect the option promises is delivered, the option itself being inert.

`import_inertia_tensor` is likewise inert. `MjcfConverterCfg.import_inertia_tensor` defaults to True and is forwarded, but the condition that would have consulted it is commented out at the point of use, `if (body->inertial) // && config.importInertiaTensor)` (`MjcfUsd.cpp:398`). An inertial block is always imported when present and never when absent.

`merge_fixed_joints` has no MuJoCo counterpart at all. `ImportConfig::mergeFixedJoints` exists and is bound, and nothing reads it. Its place is taken by `createBodyForFixedJoint`, which the Isaac Lab converter does not expose and which defaults to True (`IMjcf.h:75`), so a jointless MuJoCo body becomes its own link joined by a `UsdPhysicsFixedJoint`. That default is the one this plan depends on, and §17 states what happens if it ever changes.

`import_sites` is declared on the configuration and then ignored, the converter calling `import_config.set_import_sites(True)` with a literal (`mjcf_converter.py:86`). Sites are always imported. §4.5.1 says what that costs and how this plan answers it.

One option that the URDF converter sets is missing from the MuJoCo one, and it is not inert. The URDF path calls `set_create_physics_scene(False)` (`urdf_converter.py:131`), where the MuJoCo path leaves the call commented out (`mjcf_converter.py:84`) and the underlying default is True (`IMjcf.h:69`), so the importer defines a `/physicsScene` prim with gravity, a solver type and a broadphase into the asset's physics layer (`PluginInterface.cpp:209-222`). A referenced asset that carries its own physics scene is a hazard in a scene that already has one. §17 records it as a risk and the gate in §15 checks for it, since it cannot be suppressed from the Isaac Lab configuration as it stands.

### 4.2 Naming, and why no rename layer is needed

Three conventions exist for the quadruped and two for the biped. The donor URDFs use `abad_{LIMB}_actuator_Link`, `hip_{LIMB}_actuator_Link`, `hip_{LIMB}_thigh_Link`, `knee_{LIMB}_Link` and `foot_{LIMB}_Link`, with joints `abad_{LIMB}_Joint`, `hip_{LIMB}_Joint` and `knee_{LIMB}_Joint`. The biped emitter uses `abad|hip|knee|ankle_{SIDE}_act`, `thigh|shank_{SIDE}_Link` and `foot_{SIDE}_Link`, with joints `abad|hip|knee|ankle_{SIDE}_Joint`. The quadruped emitter uses the same body convention but names its root body `trunk` and its joints `{LEG}_hip_joint`, `{LEG}_thigh_joint` and `{LEG}_calf_joint`, following the model predictive control package that also consumes this submodule (`quadruped/builder.py:455-457`).

Revision 1 proposed a rename layer that mapped the quadruped emitter's output onto the biped's convention. The review established that no such layer is needed, and it is right. Isaac Lab reads body and joint names off the prim tree, so any valid identifier serves, and the configuration is the cheaper place to change a name than the document. This revision therefore adopts the emitted names verbatim in both families and moves the environment configurations onto them, which is what §5 and §6.3 specify.

The residual is an asymmetry between the two families, the quadruped's root body being `trunk` where the biped's is `base_Link`, and the quadruped's joints carrying the limb first where the biped's carry it second. The asymmetry could be removed by renaming inside `quadruped/builder.py`, at the cost of breaking the model predictive control package that pins those joint suffixes, so this plan leaves it standing and records it in §16.

### 4.3 The quadruped's topology matches the donor body for body and the biped's does not

The quadruped emitter's chain is `trunk` to `abad_{LEG}_act` to `hip_{LEG}_act` to `knee_{LEG}_act` to `thigh_{LEG}_Link` to `shank_{LEG}_Link` to `foot_{LEG}_Link`, seven bodies per leg counting the trunk once. `quadruped.urdf` has exactly the same seven, under the donor names. The correspondence is one to one.

The biped's is not. `base_robot.urdf` carries an eighth body per leg, `hip_{SIDE}_Link`, a bracket of 0.5 kg between the hip joint and the thigh (`base_robot.urdf` link `hip_R_Link`), which the emitter's chain has no slot for. It also carries `limx_imu`, a 0.01 kg body fixed to the base. This plan folds the bracket's mass into the knee housing, which occupies the same position in the chain, and the inertial measurement unit's into the trunk, which is where it sits. §8.1 shows that the two foldings preserve the total mass exactly.

### 4.4 The Isaac Sim importer discards every geom density, which is a blocker

This is the finding that most changes the shape of the work, and it was invisible until the importer was read.

The generator's mass model is a linear density. A thigh of length L weighs `LAMBDA_THIGH * L`, and the builder realises that by giving the collision box a volumetric `density` chosen so that MuJoCo's own mass computation returns exactly that (`biped/builder.py:155-181`, `quadruped/builder.py:213-236`). No `<inertial>` element is emitted for the thigh, the shank or the foot, so the density is the whole model.

The Isaac Sim MuJoCo importer parses that attribute into `MJCFGeom::density` at `MjcfParser.cpp:175` and never reads it again. A search of the plugin for the field finds the parse and nothing else. Body mass is taken solely from an `<inertial>` element, and where none exists the importer authors a body-level `physics:density` equal to the import configuration's own default (`MjcfUsd.cpp:395-426`). Isaac Lab passes `link_density`, which defaults to 0.0 (`mjcf_converter_cfg.py:16`), and a `UsdPhysicsMassAPI` density of zero means the PhysX default of 1000 kg per cubic metre.

The consequence is that every density-driven link of both families would arrive in the simulator at a mass PhysX invented from the collision box's volume. For the biped's thigh the emitted box is 0.05 by 0.032 by 0.25 metres, a volume of 4.0e-4 cubic metres, which at 1000 kg per cubic metre is 0.4 kg against the model's 1.5 kg. The quadruped's foot, whose collision sphere carries `density="0"`, is worse still, since a massless body is legal in MuJoCo and is not legal in a PhysX articulation.

The repair is to emit the inertial the density would have produced, which §9.2 specifies. It is exact rather than approximate, MuJoCo's own automatic inertia for a single box geom being the same closed form, so the emitted document means the same thing in both engines and the MuJoCo consumers see no change. It also subsumes the review's request that the quadruped's foot carry the donor's mass, since the foot's inertial is emitted on the same footing as every other link's.

### 4.5 The emitted document is a scene, not a description

MuJoCo takes a document to be a complete world. The emitters accordingly write a skybox, a ground material, a directional light, a ten metre ground plane, a home keyframe, a sensor suite and, for the quadruped, a tracking camera (`biped/builder.py:495-539`, `quadruped/builder.py:487-531`). Isaac Lab takes a document to be a robot description and supplies the world itself.

Every one of those elements is a liability under Isaac Lab. The floor becomes a kinematic rigid body with a collision plane inside the referenced asset (`MjcfImporter.cpp:686-712`), so four thousand environments would carry four thousand ground planes through the reference. The root body's `pos` carries the spawn height, which would compose with `ArticulationCfg.InitialStateCfg.pos` rather than replace it. The keyframe and the sensors have no reader. The review's prescription, an optional `for_isaaclab` argument on `build_xml`, is the right shape, and §9.1 gives it.

#### 4.5.1 How the converter treats sites, and what becomes of the geom that shares a name with one

A site becomes a prim. `addVisualSites` creates `<body>/sites/<name>` for each one, an `Xform` when the site has no geometry and, when it has, a primitive mesh under `/meshes/` that the site prim references and is then marked instanceable (`MjcfImporter.cpp:539-578`). Both emitters give every site an explicit `size`, so every site has geometry, and each therefore costs a mesh prim, an instanced reference and a material binding per robot. Isaac Lab reads contacts from the PhysX body prims through `ContactSensor`, never from a MuJoCo touch sensor, so not one of those prims has a reader.

The name clash revision 1 flagged is real but survivable. The quadruped's foot names both its collision geom and its site `{LEG}` (`quadruped/builder.py:379-383` before this plan's edit). The two do not collide in the prim namespace, sites living under `<body>/sites` and collision geometry under `/collisions/<body>`, and the importer uniquifies the site's mesh path by appending a counter when `/meshes/<name>` is already taken (`MjcfImporter.cpp:551-559`). What it does not uniquify is its `convertedMeshes` cache, which the site's entry overwrites, so a later mesh-typed geom of the same name would reference the site's sphere instead of its own mesh. Neither family emits such a geom today, which is why the clash has never bitten.

This plan closes it from both ends. The foot's collision geom is renamed `foot_{LEG}_collision`, matching the biped's existing `foot_{SIDE}_collision` and removing the shared name outright, and the sites are omitted entirely when `for_isaaclab` is set, since their only consumers are MuJoCo sensors that are omitted with them. The MuJoCo path keeps both, so the model predictive control package sees no change beyond the geom's new name, which it does not reference.

#### 4.5.2 The second articulation root, which is a hard blocker

`addWorldGeomsAndSites` runs unconditionally and begins by defining `<root>/worldBody` and applying `UsdPhysicsArticulationRootAPI` and `PhysxSchemaPhysxArticulationAPI` to it (`MjcfImporter.cpp:580-597`). The prim exists whether or not the world body holds anything.

Isaac Lab searches the subtree under `ArticulationCfg.prim_path` for prims carrying that schema and raises when it finds more than one, naming them (`articulation.py:1523-1538`). Every asset the MuJoCo importer produces therefore fails to initialise as an articulation unless the ambiguity is resolved. The resolution is one line of configuration, `articulation_root_prim_path`, which names the root explicitly and skips the search entirely (`articulation.py:1510-1512`, `articulation_cfg.py:41-51`). §6.1 sets it for both families.


### 4.6 The two conversion pipelines, read side by side

The review asked for this section because revision 1 asserted that the in-place link-length edit could not be assumed to survive the format change without saying what the two pipelines actually do. What follows is that reading, taken from the plugin sources at `/ws/IsaacSim/source/extensions/isaacsim.asset.importer.{urdf,mjcf}` rather than from the Isaac Lab wrappers, since the wrappers only choose options. Isaac Sim is at 5.1.0-rc.19, and the Isaac Lab URDF converter pins the importer extension to 2.4.31 on that version to keep the older fixed-joint merge behaviour (`urdf_converter.py:42-68`), which changes that pipeline's merge default and nothing about the prim layout described here.

#### 4.6.1 The URDF pipeline, drawn

```
              base_robot.urdf
                    |
   [UrdfConverter.__init__]  pins extension 2.4.31 on Isaac Sim >= 5.1   urdf_converter.py:65-68
                    |
   [URDFParseFile] ----------------------> UrdfRobot model in memory     urdf_converter.py:89
                    |
   [_update_joint_parameters]  only when cfg.joint_drive is set          urdf_converter.py:94-96
                    |                      (the co-optimisation cfg passes None, so skipped)
   [URDFImportRobot] --------------------> five USD stages
                    |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
   <name>.usd   _base.usd   _physics.usd _robot.usd  _sensor.usd
     (root)     geometry     rigid body    robot      sensors
                             joints,       schema
                             mass
                    |
   layer graph:  _base  <-sublayer-  _physics  <-sublayer-  <name>.usd
                 _robot <-sublayer-  _base
                    |
   at save: sublayers cleared, re-added as payloads inside variant sets   PluginInterface.cpp:395-467
                 Physics = {None, PhysX}, Sensor = {None, Sensors},
                 Robot   = {None, Robot}
                    |
                    v
        /<robot>                         default prim, Xform
          /<link>                        one Xform per link, WORLD transform, scale (1,1,1)
            /visuals      -> instanceable internal ref to  /visuals/<link>
            /collisions   -> instanceable internal ref to  /colliders/<link>
          /joints/<joint>                one prim per joint
        /visuals/<link>/mesh_<i>         Xform, scale = FULL edge lengths
                        /box             UsdGeomCube, size = 1.0
        /colliders/<link>/mesh_<i>/box   the same, plus UsdPhysicsCollisionAPI
        /physicsScene                    suppressed, cfg.create_physics_scene is False
```

#### 4.6.2 The URDF pipeline, walked

The converter parses the file into an in-memory `UrdfRobot`, optionally rewrites every joint's drive type, target type and gains, optionally overrides the root link, and calls the importer (`urdf_converter.py:87-109`). The co-optimisation configuration passes `joint_drive=None`, so the whole drive-rewriting branch is skipped.

`addRigidBody` then does the work per link (`UrdfImporter.cpp:333-610`). It defines `<robot>/<link>` as an `Xform` and authors a translate, an orient and a unit scale, the translate being the link's accumulated world pose rather than its pose relative to its parent (`UrdfImporter.cpp:352-360`). It applies `UsdPhysicsRigidBodyAPI` and `UsdPhysicsMassAPI`, writes `physics:mass` from `<inertial><mass>` when present and otherwise a `physics:density` equal to the configuration default, diagonalises the full inertia matrix into `physics:diagonalInertia` and `physics:principalAxes`, and writes `physics:centerOfMass` from the inertial origin (`UrdfImporter.cpp:377-432`).

Geometry follows. For each visual it builds `/visuals/<link>/<name>`, where `<name>` is the `<visual name>` when the URDF gives one, the mesh file's stem for a mesh, and otherwise the positional `mesh_<i>` (`UrdfImporter.cpp:447-466`). `addMesh` authors that prim's translate, orient and scale, the scale being the geometry's size, and then defines the shape beneath it (`UrdfImporter.cpp:242-287`). For a box the shape is a `UsdGeomCube` named `box` with its `size` attribute set to 1.0 and its `extent` set to the half-edges, so the box's actual dimensions live entirely in the parent `Xform`'s scale, which holds the full edge lengths (`UrdfImporter.cpp:200-214`, `UrdfImporter.cpp:140-151`). Finally `<link>/visuals` takes an internal reference to `/visuals/<link>` and is marked instanceable (`UrdfImporter.cpp:530-531`). Collisions repeat the pattern under `/colliders/<link>`, adding `UsdPhysicsCollisionAPI` to each shape (`UrdfImporter.cpp:534-578`).

#### 4.6.3 What the URDF pipeline assumes

It assumes a link is a `<link>` element with at most one `<visual>` and one `<collision>` that the plan cares about, that an unnamed visual or collision may be addressed by its ordinal, that mass and inertia are stated on the link rather than derived from geometry, and that a joint's `<origin>` is expressed in the parent link's frame. It assumes fixed joints may be merged away, and Isaac Lab's pin to 2.4.31 exists precisely because that assumption changed in the newer extension (`urdf_converter.py:42-50`). It assumes the caller supplies the physics scene, `createPhysicsScene` defaulting to False (`IUrdf.h` line 67 in the extension's configuration struct) and Isaac Lab setting it to False explicitly anyway.

#### 4.6.4 What the URDF pipeline produces

A link prim directly beneath the robot prim. A box whose dimensions are the referencing `Xform`'s scale, in full edge lengths, over a unit cube. A geometry child addressable as `mesh_0` when the source names nothing. Mass, inertia and centre of mass on the link prim, in the physics layer. One joint prim per URDF joint, under `<robot>/joints`, holding `physics:localPos0` equal to the child link's origin in the parent link's frame.

#### 4.6.5 The MuJoCo pipeline, drawn

```
              individual_0001.xml
                    |
   [MjcfConverter._convert_asset]  basename.split(".") unpacked into two   mjcf_converter.py:61
                    |               names, so exactly one dot is allowed
   [MJCFCreateAsset  prim_path=/<basename>] ------> five USD stages
                    |
   config: set_import_sites(True) LITERAL, set_density(0.0),
           set_import_inertia_tensor(...) inert, set_make_instanceable(...) inert,
           set_fix_base(False), set_self_collision(False)                  mjcf_converter.py:77-105
   defaults that Isaac Lab never sets:
           createBodyForFixedJoint = True   <- jointless bodies survive    IMjcf.h:75
           createPhysicsScene      = True   <- a /physicsScene is authored IMjcf.h:69
           makeDefaultPrim         = True, forced again at PluginInterface.cpp:121
                    |
   layer graph and variant sets: identical in shape to the URDF pipeline's
                    |
                    v
        /<basename>                      default prim, Xform
          /<rootbody>                    Xform, RobotAPI                   MjcfImporter.cpp:280-286
            /<rootbody>                  the root RIGID BODY, ArticulationRootAPI
            /<body>                      EVERY other body, FLAT, WORLD transform
              /visuals    -> instanceable internal ref to /visuals/<body>
              /collisions -> instanceable internal ref to /collisions/<body>
              /sites/<site>              one prim per site, always imported
          /joints/<joint>                hinge and slide joints, by JOINT name
          /joints/<body>                 fixed joints, by BODY name        MjcfImporter.cpp:1353
          /worldBody                     ALWAYS created, ArticulationRootAPI
        /visuals/<body>/<geom>           Xform, scale = MuJoCo HALF-extents
        /collisions/<body>/<geom>        the same, plus UsdPhysicsCollisionAPI
        /meshes/<geom>/<geom>            UsdGeomCube, size left at its default 2.0
        /physicsScene                    authored, gravity, TGS, MBP
```

#### 4.6.6 The MuJoCo pipeline, walked

`AddPhysicsEntities` sets stage metadata, creates the root prim, and for each top-level body defines `<basename>/<rootbody>` as an `Xform` carrying the robot schema before descending (`MjcfImporter.cpp:262-291`). The descent is `CreatePhysicsBodyAndJoint`, and its first act decides whether the body exists at all. A body with no joint that is not the root is dropped, together with its entire subtree, unless `createBodyForFixedJoint` is set (`MjcfImporter.cpp:1170-1176`). The flag defaults to True and Isaac Lab does not clear it, so the welded bodies both emitters rely upon survive.

Every surviving body becomes `<basename>/<rootbody>/<bodyname>`, a flat sibling of every other body regardless of its depth in the MuJoCo tree, because the recursion passes the same `rootPrimPath` down (`MjcfImporter.cpp:1183-1184`, `MjcfImporter.cpp:1298-1301`). The prim carries the accumulated world transform and a uniform scale equal to the distance scale (`MjcfUsd.cpp:369-385`). `applyRigidBody` then writes mass, centre of mass and diagonal inertia from the `<inertial>` element, or, in its absence, a body-level density from the import configuration (`MjcfUsd.cpp:387-426`). §4.4 is the consequence of that branch.

Collision geometry is built first so that the visual pass can tell whether the body has visuals at all. Each non-visual geom, meaning each geom whose `contype` and `conaffinity` are not both zero, is turned into a primitive under `/meshes/<geomname>` and referenced by an `Xform` under `/collisions/<bodyname>` whose translate, orient and scale `moveVisualGeom` authors (`MjcfImporter.cpp:1228-1266`, `MjcfImporter.cpp:367-487`). For a box that scale is `geom->size`, which in MuJoCo is the half-extent, and the cube itself is defined with only its `extent` set, leaving `size` at the schema default of 2.0 (`MjcfUsd.cpp:820-826`). The half-extent convention therefore reaches USD unchanged. `<body>/collisions` then references `/collisions/<bodyname>` and is marked instanceable (`MjcfImporter.cpp:1276-1279`). Visuals repeat the pattern under `/visuals/<bodyname>` (`MjcfImporter.cpp:487-537`), and sites follow under `<body>/sites` (`MjcfImporter.cpp:539-578`).

Joints are authored last per body. A body with one hinge becomes a `UsdPhysicsRevoluteJoint` at `<basename>/joints/<jointname>`, with the axis rotated into local x and the range converted to degrees (`MjcfImporter.cpp:1362-1397`). A body with none becomes a `UsdPhysicsFixedJoint` at `<basename>/joints/<bodyname>`, whose parent-side frame is the body's own `pos` and `quat` (`MjcfImporter.cpp:1353-1361`). In both cases the parent-side local position reduces to the child body's origin expressed in the parent's frame, the MuJoCo joint position being zero in the child frame for every joint both emitters write.

The world body is then materialised whether or not it holds anything, and given an articulation root (`MjcfImporter.cpp:580-597`). §4.5.2 is the consequence.

#### 4.6.7 What the MuJoCo pipeline assumes

It assumes a body's mass is stated in an `<inertial>` element and that a geom's `density` is decoration, which §4.4 shows to be the reverse of what MuJoCo itself assumes. It assumes an unnamed geom may be addressed by the document-wide running name `_geom_<N>`, assigned in parse order (`MjcfParser.cpp:180`), so the identity of a geometry prim depends on how many geoms precede it in the file. It assumes sites are wanted. It assumes the caller has no physics scene. It assumes the asset's file name contains exactly one dot, the converter unpacking `basename.split(".")` into two names (`mjcf_converter.py:61`). It assumes a world body is worth a prim and an articulation root.

#### 4.6.8 What the MuJoCo pipeline produces

A link prim two levels beneath the referenced asset prim, under the root body's own `Xform`. A box whose dimensions are the referencing `Xform`'s scale, in half-extents, over a cube of nominal size 2. A geometry child named after the MuJoCo geom, or `_geom_<N>` when the geom is unnamed. Mass, inertia and centre of mass on the link prim, in the physics layer, present only when the document stated an inertial. One joint prim per joint under `<asset>/joints`, named after the joint for a moving joint and after the child body for a welded one, holding `physics:localPos0` equal to the child body's origin in the parent body's frame. A spurious articulation root at `<asset>/worldBody` and a physics scene at `/physicsScene`.

#### 4.6.9 The differences, and what each one costs

| # | URDF pipeline | MuJoCo pipeline | Cost to `update.py` |
| --- | --- | --- | --- |
| 1 | link at `<robot>/<link>` | body at `<robot>/<root>/<body>` | the link path gains a segment, and the joints scope is two levels up rather than one |
| 2 | box scale is the full edge length | box scale is the MuJoCo half-extent | every extent written must be halved, or written as a half-extent throughout |
| 3 | cube `size` set to 1.0 | cube `size` left at the default 2.0 | none directly, the two conventions being consistent with their own scales |
| 4 | geometry child named `mesh_<i>` | geometry child named after the geom, or `_geom_<N>` | the constant `GEOM_MESH = "mesh_0"` has no counterpart, and the child must be discovered |
| 5 | mass always authored when the URDF states one | mass authored only when an `<inertial>` exists | the density recovery `mass0 / (x0*y0*z0)` reads `None` and raises |
| 6 | fixed joints merged by default | fixed joints kept, named after the child body | the child joint of a scalable link may be named after a body |
| 7 | one articulation root | two, the second at `/worldBody` | none directly, but the asset will not initialise without `articulation_root_prim_path` |
| 8 | no physics scene | a physics scene | none directly, recorded as a risk in §17 |
| 9 | a link grows along its own z | a link grows along a fixed direction that the specification knows | the `-(z + offset)` form cannot express the biped's thigh, which grows along a bend of 145 degrees |

Difference 9 is the one that decides the shape of the new `update.py`. The incumbent derives the child joint's new anchor from the box's z extent plus a module constant (`update.py:124-126`), which is only meaningful when a link grows along its own z. The biped's thigh does not, its child sitting along `THIGH_DIR_XZ` at 145 degrees from vertical (`biped/builder.py:141-142`). Rather than teach `update.py` a direction, this plan has the generator hand it the child's position outright, which is §11's central move and is what the review asked for in asking that the generator provide the data bundled together.

Differences 5 and 2 together are why the incumbent would not have failed loudly. `update.py` reads the prototype's `mesh_0` child and returns early when it is absent (`update.py:83-85`), so under the MuJoCo layout it would have found no `mesh_0`, returned, and left every design at generation zero while the logs reported new populations. The new module raises instead, and §11's `assert_link_lengths_applied` reads the realised separations back out of the simulation after the reset, which is the only check that distinguishes an edit that was authored from an edit that took effect.

### 4.7 The actuator contract has three gaps, two of them standing defects

`apply_actuator_params` walks each actuator group, and for each environment writes the design's overrides into the actuator's tensor attributes, skipping anything that is not a tensor (`respawn.py:249-259`). Three things pass through that sieve.

`saturation_effort` is not a tensor. `DCMotor.__init__` reads it from the configuration into a Python float under a leading underscore, `self._saturation_effort` (`actuator_pd.py:268`), so `getattr(actuator, "saturation_effort")` returns nothing and the override is silently discarded. Every `RandomDesignGenerator` run to date has sampled a `saturation_effort_scale` that the simulation never saw.

`_vel_at_effort_lim` is a tensor, but it is a cached product computed once at construction from the velocity limit, the effort limit and the stall torque (`actuator_pd.py:270`). Writing any of the three afterwards leaves the corner velocity at its construction value, and `_clip_effort` clamps the joint velocity against it every step (`actuator_pd.py:297`). The envelope therefore bends around the old numbers.

`armature` is a tensor and is written, and the simulation never reads it. No line of `IdealPDActuator.compute` or `DCMotor._clip_effort` consults it, it being a PhysX joint property that `Articulation._process_actuators_cfg` pushes once at construction through `write_joint_armature_to_sim` (`articulation.py:1774`) and nothing pushes again. The `armature_scale` of the incumbent generator has been inert for the same reason.

All three are repaired inside `apply_actuator_params` in §12, which is where the review asked for them.

### 4.8 The specification vector round trip is not the identity

`DesignSpec.vector` emits only the free fields and `DesignSpec.from_vector` reconstructs the rest from their Python defaults (`biped/generate.py:167-213`). For the biped's donor parameter set that loses fourteen clearances, among them `clearance_ankleAct2foot_x` at 0.01 and `clearance_ankleAct2foot_z`, so the round trip moves the foot. The requirement that the generators exchange designs as vectors through that interface cannot be met, and §10 uses `dataclasses.replace` on a pinned baseline instead. The side effect is welcome, the length-only generators staying two dimensional and their existing checkpoints therefore loadable.

### 4.9 The two sibling packages now import into one interpreter

Revision 1 proposed an import-isolating loader because the two families' modules were flat siblings that bound the same global names. The submodule has since been restructured around `design_generator.common.actuator_catalog`, with `ActuatorCatalog` owning one robot's catalogues as instance state and each family's `actuator.py` re-exposing its own (`common/actuator_catalog.py:513-545`). Both families import cleanly into one process, which was confirmed by importing them together, and the loader is therefore deleted.

What the restructuring needs is that `design_generator` be importable as a top-level package, its modules using absolute imports of that name (`biped/generate.py:10-20`). The submodule ships a `pyproject.toml` that declares exactly that, so the install step is `pip install -e co_optimisation/co_optimisation/runners/design_generator`, added to the repository's setup instructions in §13. No code in this repository imports it any other way.

### 4.10 The biped's observation dimension changes and no checkpoint transfers

`robot_link_lengths` returns one number per parent-child pair, and the SF configuration passes four pairs (`cfg/SF/limx_base_env_cfg.py:152-157`). The pairs are stated by name, and the names change, so the term's configuration changes while its dimension does not. The policy's input dimension is therefore unchanged and its meaning is not, the pairs now measuring thigh and shank on the emitted chain rather than on the donor's. Together with the joint-axis change of §16 this means no URDF-trained biped checkpoint transfers, which the phased programme in §14 treats as a given rather than a risk to be mitigated.

### 4.11 The vendor Universal Scene Description asset is retired

`SOLEFOOT_IDENTIFIED_CFG` presently spawns `assets/usd/SF_TRON1A/SF_TRON1A.usd` (`solefoot_identified_cfg.py:10, :103-113`). The asset's prim names are the vendor's and cannot be made to match the emitted document's, so a task that spawned it would need a second set of name-bearing configuration terms. The decision taken on 2026-09-22 retires it, and every TRON1 task moves onto the emitted asset.

The scope of the retirement is `usd/SF_TRON1A/` alone. `usd/PF_TRON1A/`, `usd/WF_TRON1A/` and `usd/SD_BRS1/` are kept, those robots having no emitted asset to move onto. That scoping was this document's judgement rather than an instruction, and it is restated here so that it can be overruled.

### 4.12 Both default parameter sets reproduce their donor, one of them exactly

The requirements assumed the emitted robot would approximate the donor. It does better than that. §8 gives the measurements. The quadruped reproduces `quadruped.urdf` to zero residual at every body station, exactly at every link mass, exactly in total mass, exactly in centre of mass, and to within 0.03 percent in whole-robot inertia about the centre of mass. The biped reproduces `base_robot.urdf` to 4.0e-7 metres at every body station, exactly at every link mass under the two foldings of §4.3, exactly in total mass, to 7.3 millimetres in centre of mass height, and to within 4.9 percent in whole-robot inertia.

The biped's two residuals have a single cause between them, which §8.1 quantifies term by term. The emitter centres an actuator housing on the joint it serves and centres a shortened link box on the link's midpoint, where the donor offsets the housing outboard by its own half-length and makes the box flush at the proximal end. Both are modelling conventions of the generator, neither is wrong, and changing either would be a change to the builder's geometry that the review did not ask for.

## 5. The Naming Convention

The convention is whatever the emitter writes. There is no rename layer, in either direction, in either family.

### 5.1 What the convention changes in the assets

| Concern | Donor URDF | Emitted document, both modes |
| --- | --- | --- |
| biped root body | `base_Link` | `base_Link` |
| biped hip abduction housing | `abad_{S}_actuator_Link` | `abad_{S}_act` |
| biped hip pitch housing | `hip_{S}_actuator_Link` | `hip_{S}_act` |
| biped bracket | `hip_{S}_Link` | folded into `knee_{S}_act` |
| biped knee housing | `knee_{S}_actuator_Link` | `knee_{S}_act` |
| biped thigh | `hip_{S}_thigh_Link` | `thigh_{S}_Link` |
| biped shank | `knee_{S}_Link` | `shank_{S}_Link` |
| biped ankle housing | `ankle_{S}_actuator_Link` | `ankle_{S}_act` |
| biped foot | `ankle_{S}_Link` | `foot_{S}_Link` |
| biped inertial unit | `limx_imu` | folded into `base_Link` |
| biped joints | `abad|hip|knee|ankle_{S}_Joint` | unchanged |
| quadruped root body | `base_Link` | `trunk` |
| quadruped housings | `abad|hip|knee_{LEG}_actuator_Link` | `abad|hip|knee_{LEG}_act` |
| quadruped thigh | `hip_{LEG}_thigh_Link` | `thigh_{LEG}_Link` |
| quadruped shank | `knee_{LEG}_Link` | `shank_{LEG}_Link` |
| quadruped foot | `foot_{LEG}_Link` | unchanged |
| quadruped joints | `abad|hip|knee_{LEG}_Joint` | `{LEG}_hip|thigh|calf_joint` |

The biped's joints already conform, which is why only the body names move on that side. The quadruped's joints do not, and §16 records the asymmetry and its cause.

### 5.2 What the convention changes in the configurations

For the biped the changes are confined to `cfg/SF/limx_base_env_cfg.py`, the two analysis scripts and the play script.

| Line | Present | Becomes | Note |
| --- | --- | --- | --- |
| `limx_base_env_cfg.py:76` | `Robot/base_Link` | `Robot/base_Link/base_Link` | the importer interposes the root body's own `Xform` |
| `:86` | `Robot/.*` | `Robot/base_Link/.*` | the contact sensor must reach the bodies, which are one level deeper |
| `:153` | `hip_{S}_thigh_Link`, `knee_{S}_Link` | `thigh_{S}_Link`, `shank_{S}_Link` | parents of the link-length observation |
| `:156` | `knee_{S}_Link`, `ankle_{S}_actuator_Link` | `shank_{S}_Link`, `ankle_{S}_act` | children of the same |
| `:287, :300, :344, :349, :653, :666, :1010, :1021, :1022` | `ankle_.*` | `foot_.*_Link` | the four body resolution, repaired. See §6.4 |
| `:920, :929, :969, :975` | `ankle_[RL]_Link` | `foot_[RL]_Link` | the foot, by exact name |
| `:957` | `["abad_.*", "hip_.*", "knee_.*", "base_Link"]` | `["abad_.*", "hip_.*", "knee_.*", "thigh_.*", "shank_.*", "base_Link"]` | see below |
| `:720, :1035` | `base_Link` | unchanged | the biped's root body keeps its name |
| `:750, :762, :775, :788, :918` | `abad|hip|knee|ankle_[LR]_Joint` | unchanged | the biped's joints already conform |

The change at line 957 needs stating carefully, revision 1 having described it wrongly. The term is `pen_undesired_contacts`, a `RewTerm` carrying a weight of minus 0.5 (`cfg/SF/limx_base_env_cfg.py:951-960`). It is a penalty and not a termination, the only contact-based termination the configuration carries being `base_contact`, which watches `base_Link` alone (`cfg/SF/limx_base_env_cfg.py:1032-1038`) and is unaffected by any of this. Under the donor names the patterns `hip_.*` and `knee_.*` reached `hip_{S}_thigh_Link` and `knee_{S}_Link` as well as the housings, so the penalised set was thirteen bodies. Under the emitted names they reach only the housings, and the set would fall to seven without the addition. Adding `thigh_.*` and `shank_.*` brings it to eleven and restores the intent, the two bodies the donor set held that the new one would not being the bracket and the donor's separate hip link, both of which are folded away.

For the quadruped the changes are larger, because the joint names move as well.

| Line | Present | Becomes |
| --- | --- | --- |
| `cfg/quadruped/base_env_cfg.py:76` | `Robot/base_Link` | `Robot/trunk/trunk` |
| `:86` | `Robot/.*` | `Robot/trunk/.*` |
| `:425-432, :555-562` | `hip_{LEG}_thigh_Link`, `knee_{LEG}_Link` | `thigh_{LEG}_Link`, `shank_{LEG}_Link` |
| `:435-438, :565-568` | `knee_{LEG}_Link` | `shank_{LEG}_Link` |
| `:934, :1206` | `base_Link` | `trunk` |
| `:967` | `knee_.._Joint` | `.._calf_joint` |
| `:978` | `hip_.._Joint` | `.._thigh_joint` |
| `:989, :1123` | `abad_.._Joint` | `.._hip_joint` |
| `:1111` | `["base_Link", "abad_.*", "hip_.*", "knee_.*"]` | `["trunk", "abad_.*", "hip_.*", "knee_.*", "thigh_.*", "shank_.*"]` |
| `:319, :332, :439-442, :463, :468, :529, :542, :569-572, :872` | `foot_.*_Link`, `foot_{LEG}_Link` | unchanged |

`mdp/observations.py` takes every body name from its term configuration and hardcodes none (`observations.py:253-269`), so the shared package is untouched and the warning of `CLAUDE.md` rule 3 does not apply. The same holds for `mdp/rewards.py`, whose terms take `SceneEntityCfg` objects.

## 6. The Environment Side

### 6.1 The asset configurations

Both configurations lose their URDF spawn and gain an emitted one. The document is produced once by a tool script at `scripts/tools/generate_default_mjcf.py`, which §13 gives, and committed beside the assets, so that a task that merely trains a policy needs neither the generator nor a conversion at import time.

The three settings that §4 showed to be necessary are set here. `articulation_root_prim_path` resolves the two articulation roots of §4.5.2. `fix_base` must be stated because `MjcfConverterCfg` leaves it `MISSING` (`mjcf_converter_cfg.py:28`). `self_collision` keeps the value the URDF spawn used.

### 6.2 The retirement of the URDF and of the vendor asset

Every code reference to a URDF is removed in this plan, the assets themselves being kept for later deletion once the emitted pipeline has been exercised. That means `urdf_path` and `UrdfFileCfg` leave both asset configurations, `SOLEFOOT_IDENTIFIED_CFG_URDF` is deleted and its twelve call sites in `robots/limx_solefoot_env_cfg.py` collapse onto `SOLEFOOT_IDENTIFIED_CFG`, `base_urdf_path` leaves the generator and both scripts, and `co_optimisation/tests/cmaes_design_generator_test.py`, which parses the donor URDF, is rewritten against the emitted document or deleted.

`assets/urdf/quadruped/gen_quadruped_urdf.py` is left where it is. It is the provenance of a retained asset rather than a consumer of one, and it is scheduled for deletion together with the asset it emits.

### 6.3 The environment configuration classes

The tables of §5.2 are the whole of the change. No class is added, no term is added or removed, and no weight moves.

### 6.4 The price of repairing the four body resolution

`body_names="ankle_.*"` presently resolves to four bodies on the donor robot, `ankle_{R,L}_actuator_Link` and `ankle_{R,L}_Link`, where every term that uses it means the foot. Under the emitted names the same pattern resolves to the two housings and no foot at all, so leaving it alone is not an option and the repair is forced rather than chosen.

Nine terms change meaning as a result. `rew_feet_air_time`, `feet_slide`, `rew_feet_clearance` and the gait terms that take `body_names="ankle_.*"` at lines 287, 300, 344, 349, 653, 666, 1010, 1021 and 1022 go from averaging over a housing and a foot to reading the foot alone. Every one of them becomes correct, and every one of them becomes incomparable with its value in any earlier TRON1 run. §14 places the repair in the same phase as the asset change so that a single boundary separates the two regimes in the logs.


## 7. The Catalogue

### 7.1 The schema extension

`CatalogActuator` carried a joint-side envelope, a mass, a cost and an optional electrical triple, and nothing a simulator needs beyond torque and speed. Revision 1 answered that with a side table keyed by actuator name, which the review rightly rejected, a reflected rotor inertia being a property of an actuator rather than of a lookup. The class therefore gains ten optional fields and one method, and `Actuator` gains the same method over the COMPAct decomposition, so that a consumer never has to know which of the two modes produced the pick. §9.5 gives the code.

The fields are `rotor_inertia`, `armature_value`, `effort_limit`, `velocity_limit`, `saturation_effort`, `friction_static`, `friction_dynamic`, `activation_vel`, `stiffness` and `damping`, every one of them optional and every one of them read verbatim from the JSON entry when stated. `sim_params` resolves them, falling back to `peak_torque` for the effort limit, to `peak_speed` for the velocity limit, and to the effort limit times `DEFAULT_STALL_RATIO` for the stall torque. That ratio is 6.7, the median of the seven measured `saturation_effort` over `effort_limit` pairs in the two identified configurations, which span 6.22 to 10.05.

Friction and the two gains have no fallback and are simply omitted when unstated, so that a consumer keeps whatever its own configuration holds rather than being handed a fabricated number. The armature has no fallback either and resolves to zero, which is a real hazard under a categorical search and is why §10's `CatCMAESDesignGenerator` names the bare entries at construction.

### 7.2 The four TRON1 entries

Four entries are added to `biped/catalogs/actuators.json`, one per joint group, each carrying the donor's housing envelope and mass and the identified configuration's dynamics. Together they make the biped's default design reproduce both the donor's geometry and its actuator model exactly.

| Field | TRON1-ABAD | TRON1-HIP | TRON1-KNEE | TRON1-ANKLE | Source |
| --- | --- | --- | --- | --- | --- |
| `peak_torque` | 60.0 | 60.0 | 90.0 | 40.0 | `solefoot_identified_cfg.py` `effort_limit` |
| `peak_speed` | 23.0 | 20.0 | 14.0 | 20.0 | the same `velocity_limit` |
| `mass` | 0.950 | 1.469 | 1.450 | 0.950 | `base_robot.urdf`, the knee folding in `hip_{S}_Link` |
| `diameter` | 0.0966 | 0.0966 | 0.0966 | 0.0600 | twice the URDF cylinder radius |
| `length` | 0.0545 | 0.0545 | 0.0545 | 0.0545 | the URDF cylinder length |
| `rotor_inertia` | 6.9e-05 | 9.4e-05 | 1.5e-04 | 6.9e-05 | the identified armature divided by 81 |
| `ratio` | 9.0 | 9.0 | 9.0 | 9.0 | the square root of the same 81 |
| `saturation_effort` | 402.0 | 443.0 | 560.0 | 402.0 | the identified configuration |
| `stiffness` | 55.0 | 80.0 | 60.0 | 10.0 | the same |
| `damping` | 13.5 | 13.0 | 4.0 | 0.5 | the same |
| `friction_static` | 0.3 | 0.3 | 0.8 | 0.1 | the same |
| `friction_dynamic` | 0.02 | 0.02 | 0.02 | 0.02 | the same |
| `activation_vel` | 0.1 | 0.1 | 0.1 | 0.1 | the same |

`peak_torque` is taken from the identified continuous ceiling rather than from the URDF's `<limit effort>` of 80, 80, 80 and 20, because it is the number the simulated actuator can actually deliver and because the URDF limits never reached the simulator in the first place, the co-optimisation spawn passing `joint_drive=None` and Isaac Lab writing `effort_limit_sim` over whatever the importer left (`articulation.py:1770-1771`).

The derivation of `rotor_inertia` is worth stating. The identified configuration writes the armature as a product, for instance `6.9e-5 * 81` (`solefoot_identified_cfg.py:24`), which is a rotor inertia reflected through a gear ratio of nine. Splitting it back into its factors lets `CatalogActuator.armature` derive the joint-side value as `rotor_inertia * ratio ** 2` and reproduces the original to the last digit, which §8.1's gate checks.

### 7.3 The three quadruped entries

Three entries are added to `quadruped/catalogs/actuators.json` on the same footing. The quadruped's builder takes only mass and peak torque from a catalogue entry, its housing envelope being a fixed constant, so the diameter and length are recorded for completeness rather than read.

| Field | A1-ABAD | A1-HIP | A1-KNEE | Source |
| --- | --- | --- | --- | --- |
| `peak_torque` | 23.622511 | 23.622511 | 35.238 | `quadruped_identified_cfg.py`, matching `quadruped.urdf`'s `<limit effort>` exactly |
| `peak_speed` | 30.0 | 30.0 | 20.0 | the same, matching `<limit velocity>` |
| `mass` | 1.028375 | 1.028375 | 1.434746 | `quadruped.urdf` |
| `diameter` | 0.092 | 0.092 | 0.092 | twice `ABD_RADIUS` |
| `length` | 0.040 | 0.040 | 0.040 | twice `ABD_HALFLEN` |
| `armature_value` | 0.01 | 0.01 | 0.01 | the identified configuration, stated directly rather than factored |
| `saturation_effort` | 158.0 | 158.0 | 236.0 | the same |
| `stiffness` | 40.0 | 40.0 | 50.0 | the same |
| `damping` | 1.2 | 1.1 | 0.9 | the same |
| `friction_static` | 0.2 | 0.2 | 0.2 | the same |
| `friction_dynamic` | 0.02 | 0.02 | 0.02 | the same |
| `activation_vel` | 0.1 | 0.1 | 0.1 | the same |

The existing eleven biped and eleven quadruped entries are untouched, and the `_schema` block of each file gains the ten new field names.

## 8. The Parameter Sets and Their Verification

The parameter sets below were derived by inverting the builders' own placement formulas against the donor URDFs, and then checked by emitting the document and walking it by forward kinematics. The derivation script, the two kinematics helpers and the comparison harness were run for this plan, and §15 marks which gates they close and which remain open.

### 8.1 The biped

`biped/variable_params.json` becomes the parameter set below. Every clearance is solved rather than chosen, and several come out negative, which is legitimate, the clearance being an additive correction to a rule-derived base offset rather than a physical gap.

```json
{
  "trunk_length": 0.27,
  "trunk_width": 0.26,
  "trunk_thickness": 0.19,

  "act_abduction": "TRON1-ABAD",
  "clearance_trunk2abadAct_x": 0.06,
  "clearance_trunk2abadAct_y": 0.0,
  "clearance_trunk2abadAct_z": 0.0442,

  "act_hip": "TRON1-HIP",
  "clearance_abadAct2hipAct_x": 0.02,
  "clearance_abadAct2hipAct_y": -0.008877339,
  "clearance_abadAct2hipAct_z": 0.0,

  "act_knee": "TRON1-KNEE",
  "clearance_hipAct2kneeAct_x": 0.0,
  "clearance_hipAct2kneeAct_y": -0.09425,
  "clearance_hipAct2kneeAct_z": 0.0,

  "thigh_length": 0.3,
  "clearance_kneeAct2thigh_x": -0.027703732,
  "clearance_kneeAct2thigh_y": -0.05925,
  "clearance_kneeAct2thigh_z": -0.03956505,

  "shank_length": 0.35,

  "act_ankle": "TRON1-ANKLE",
  "clearance_shank2ankleAct_x": -0.017221012,
  "clearance_shank2ankleAct_y": 0.0,
  "clearance_shank2ankleAct_z": -0.003079015,

  "foot_length": 0.2,
  "clearance_ankleAct2foot_x": 0.01,
  "clearance_ankleAct2foot_y": 0.0,
  "clearance_ankleAct2foot_z": -0.01,

  "body_height": 0.80435432
}
```

`biped/constant_params.json` changes in nine values and keeps the rest.

| Key | Was | Becomes | Why |
| --- | --- | --- | --- |
| `LINK_BOX_LENGTH_FRAC` | 0.8 | 0.833333333 | makes the thigh box exactly the donor's 0.25 m over a 0.30 m link |
| `THIGH_BEND_DEG` | -145.0 | -145.000014 | the donor's 0.610865 rad hip rotation, to the digit |
| `SHANK_BEND_DEG` | -40.0 | -40.000189 | the donor's 0.610865 minus 1.3090 rad |
| `ANKLE_BEND_DEG` | 40.0 | 40.003913 | the donor's 0.6982 rad |
| `LAMBDA_THIGH` | 3.24 | 5.0 | 1.5 kg over 0.30 m |
| `LAMBDA_SHANK` | 1.48 | 3.485714286 | 1.22 kg over 0.35 m |
| `LAMBDA_FOOT` | 1.48 | 3.1 | 0.62 kg over 0.20 m |
| `TRUNK_SHELL_THICKNESS` | 0.003 | 1e-06 | see below |
| `TRUNK_EXTRA_MASS` | 7.0 | 5.00907714 | closes the trunk to 5.01 kg, the donor's base plus its inertial unit |

`THIGH_BOX_HALF`, `SHANK_BOX_HALF`, `FOOT_BOX_HALF`, the four joint ranges, `TRUNK_SHELL_DENSITY`, `JOINT_DYNAMICS` and `LINK_COLORS` already hold the donor's values and are unchanged.

The shell thickness deserves its own paragraph, the review having asked that the trunk's two knobs be set according to the generator rather than to the donor. They are. The generator models a trunk as a thick-walled shell of surface area times thickness times density, plus a solid payload filling the cavity (`biped/builder.py:433-451`). The donor's trunk is a solid box, its stated inertia of 0.043208, 0.045417 and 0.058542 matching `(1/12) m (b^2 + c^2)` for a solid 0.27 by 0.26 by 0.19 box of 5.01 kg to three decimal places. A solid box is what the generator's own model becomes in the limit of a vanishing wall, the cavity growing to fill the shell and the payload with it, so the thickness is taken to that limit rather than to an arbitrary value. At 0.003 m the model returned half the donor's inertia, at 0.001 m 81 percent of it, and at 1e-06 m it returns the solid-box answer, which is what the measurement below reports.

The verification, run against `base_robot.urdf`, compares the emitted document body by body at the zero pose.

| Quantity | Result |
| --- | --- |
| worst body station error over both legs, seven stations each | 3.952e-07 m |
| worst per-link mass error over all fifteen bodies | 0.0 kg |
| total mass | 21.328000 kg against 21.328000 |
| centre of mass | (-0.021183, 0.0, -0.299771) against (-0.021204, 0.0, -0.292472) |
| whole-robot inertia about the centre of mass, worst diagonal | 4.907 percent |
| bodies | 15 |
| joints | `abad|hip|knee|ankle_{R,L}_Joint`, eight |

The 7.299 millimetre centre-of-mass drop has three contributions and no fourth, which the harness accounts for term by term.

| Body | Mass, per side | Height shift | Contribution |
| --- | --- | --- | --- |
| `thigh_{S}_Link` | 1.500 kg | -20.479 mm | -2.881 mm |
| `shank_{S}_Link` | 1.220 kg | -19.151 mm | -2.191 mm |
| `ankle_{S}_act` | 0.950 kg | -25.000 mm | -2.227 mm |
| every other body | | 0.000 mm | 0.000 mm |
| total | | | -7.299 mm |

The first two are the box-centring convention, the emitter placing a shortened box on the link's midpoint where the donor makes it flush at the proximal end. The third is the housing convention, the emitter centring the ankle housing on the ankle joint where the donor offsets it outboard by its own half-length. Both are properties of the builder's geometry rather than of this parameter set, and changing either would be a change to the builder that the review did not ask for. The shift is downward, which is mildly stabilising, and it is recorded in §16.

### 8.2 The quadruped

`quadruped/variable_params.json` becomes the parameter set below. Every clearance but two is zero, the rule-derived placement of `GEOMETRY_RULES.md` reproducing the donor outright once the trunk dimensions are the donor's.

```json
{
  "trunk_length": 0.17,
  "trunk_width": 0.196,
  "trunk_thickness": 0.092,

  "act_abduction": "A1-ABAD",
  "clearance_trunk2abadAct_x": 0.0,
  "clearance_trunk2abadAct_y": 0.0,
  "clearance_trunk2abadAct_z": 0.0,

  "act_hip": "A1-HIP",
  "clearance_abadAct2hipAct_x": 0.005,
  "clearance_abadAct2hipAct_y": 0.0,
  "clearance_abadAct2hipAct_z": 0.0,

  "act_knee": "A1-KNEE",
  "clearance_hipAct2kneeAct_x": 0.0,
  "clearance_hipAct2kneeAct_y": 0.01,
  "clearance_hipAct2kneeAct_z": 0.0,

  "thigh_length": 0.213,
  "clearance_kneeAct2thigh_x": 0.0,
  "clearance_kneeAct2thigh_y": 0.0,
  "clearance_kneeAct2thigh_z": 0.0,

  "shank_length": 0.213,
  "clearance_shank2foot_x": 0.0,
  "clearance_shank2foot_y": 0.0,
  "clearance_shank2foot_z": 0.0,

  "body_height": 0.32
}
```

Rule 4 pins the trunk thickness to twice the abduction housing's radius, which is 0.092 m, and the donor's trunk is 0.092 m thick. Rule 2 requires the trunk to be at least 0.184 m wide and the donor's is 0.196 m. Rules 1 and 3 place the abduction housing at 0.105 m and 0.052 m from the trunk centre and the donor places it at 0.105 m and 0.052 m. The agreement is not a coincidence, the donor URDF having been generated from the same geometric rules by `assets/urdf/quadruped/gen_quadruped_urdf.py`, and it is why the quadruped reproduces exactly where the biped does not.

`quadruped/constant_params.json` changes in five values and gains one.

| Key | Was | Becomes | Why |
| --- | --- | --- | --- |
| `VISUAL_LINK_SCALE` | 0.8 | 1.0 | the donor's thigh and shank cylinders span the full link, so no shortening is wanted and the link offsets carry the clearance instead |
| `LAMBDA_THIGH` | 3.24 | 0.879647887 | 0.187365 kg over 0.213 m |
| `LAMBDA_SHANK` | 1.48 | 0.431028169 | 0.091809 kg over 0.213 m |
| `TRUNK_SHELL_THICKNESS` | 0.003 | 1e-06 | the same degenerate limit as the biped's, for the same reason |
| `TRUNK_EXTRA_MASS` | 4.0 | 0.674035243 | closes the trunk to 0.674397 kg, the donor's base link |
| `FOOT_MASS` | absent | 0.02 | new, the donor's `foot_{LEG}_Link` mass. See §9.2 |

`ABD_RADIUS`, `ABD_HALFLEN` and their hip and knee twins already hold the donor's 0.046 m and 0.020 m, `FOOT_RADIUS` already holds its 0.022 m, `THIGH_BOX_HALF` and `SHANK_BOX_HALF` already inscribe the donor's cylinders, and the three joint ranges already hold the donor's limits, so none of them moves.

The verification, run against `quadruped.urdf`, is exact.

| Quantity | Result |
| --- | --- |
| worst body station error over four legs, six stations each | 0.0 m |
| worst per-link mass error over all twenty five bodies | 0.0 kg |
| total mass | 15.837077 kg against 15.837077 |
| centre of mass | (0.0, 0.0, -0.0146005) against (0.0, 0.0, -0.0146005) |
| whole-robot inertia about the centre of mass, worst diagonal | 0.026 percent |
| bodies | 25 |
| joints | `{FR,FL,RR,RL}_{hip,thigh,calf}_joint`, twelve |

The remaining 0.026 percent is the box cross-section standing in for the donor's cylinder, the two having the same mass and length and slightly different second moments.

## 9. The Design Generator Changes

The review moved four concerns into the submodule, so this section is where most of revision 1's new code went. Every change below is additive at the call site, `build_xml(spec)` keeping its present meaning and every MuJoCo consumer therefore seeing the same document it saw before, with the two exceptions §9.2 and §4.5.1 name and justify.

### 9.1 `build_xml` gains `for_isaaclab`

The argument turns a MuJoCo scene into a robot description. When it is set the emitter omits the `<visual>` and `<asset>` blocks that exist only to dress the world, the world light and the ten metre floor plane, every `<site>` and the `<sensor>` block that anchors to them, the home `<keyframe>`, and, for the quadruped, the tracking camera. It also writes the root body at `0 0 0` rather than at `spec.body_height`, so that Isaac Lab's `init_state.pos` is the only thing setting the spawn height.

The omitted material is factored into module-level string constants rather than deleted, so the MuJoCo path emits character for character what it emitted before. That was checked by emitting both families with the argument unset and diffing against the present output.

### 9.2 The explicit inertials

§4.4 is the reason and this is the repair. Each of the density-driven links gains an `<inertial>` element carrying the mass and the diagonal inertia that its geom density would have produced, computed by a new `box_inertial_block` helper, and the quadruped's foot gains one from a new `sphere_inertial_block`.

The repair is exact rather than approximate and it is unconditional rather than gated on `for_isaaclab`. MuJoCo's own automatic inertia for a body with one non-zero-density geom is the solid box's closed form about that box's own centre, in that box's own frame, which is precisely what the helper writes, so a MuJoCo consumer computes the same numbers it computed before and an Isaac Lab consumer finally sees them at all. Emitting it unconditionally also keeps one code path where two would have to be kept in step.

The quadruped's foot is the one place where a number changes in the MuJoCo path as well. Its collision sphere carried `density="0"` and the body carried no inertial, so it weighed nothing. It now weighs `FOOT_MASS`, which is the donor's 0.02 kg, through a `FOOT_DENSITY` derived from that mass and `FOOT_RADIUS` so that the geom density and the explicit inertial agree. That is what the review asked for in asking that the foot carry the mass its URDF variant carries, and it is also what makes the total mass of §8.2 come out exactly.

The foot's collision geom is renamed from `{LEG}` to `foot_{LEG}_collision` in the same pass, for the reason §4.5.1 gives.

```diff
--- a/co_optimisation/co_optimisation/runners/design_generator/quadruped/builder.py
+++ b/co_optimisation/co_optimisation/runners/design_generator/quadruped/builder.py
@@ -123,8 +123,11 @@
 LAMBDA_SHANK = _C["LAMBDA_SHANK"]   # [kg/m] shank mass-per-length, hardware constant -- already
                                      # includes the real foot hardware's mass (see FOOT_RADIUS below)
 
-FOOT_RADIUS = _C["FOOT_RADIUS"]     # [m] foot contact sphere radius -- massless, see FOOT_RADIUS's
-                                     # use in _leg_xml
+FOOT_RADIUS = _C["FOOT_RADIUS"]     # [m] foot contact sphere radius
+FOOT_MASS = _C["FOOT_MASS"]         # [kg] foot contact sphere mass -- the donor URDF's own
+                                     # foot_{LEG}_Link mass. A massless body is legal in
+                                     # MuJoCo but not in a PhysX articulation, so this is
+                                     # read here rather than left to the geom density.
 
 KNEE_RANGE = tuple(_C["KNEE_RANGE"])    # same all legs
 ABAD_RANGE = tuple(_C["ABAD_RANGE"])    # same all legs
@@ -233,6 +236,33 @@
 LAMBDA_SHANK_CORRECTED = LAMBDA_SHANK / VISUAL_LINK_SCALE
 THIGH_DENSITY = _linear_density(LAMBDA_THIGH_CORRECTED, THIGH_BOX_HALF)
 SHANK_DENSITY = _linear_density(LAMBDA_SHANK_CORRECTED, SHANK_BOX_HALF)
+# Uniform sphere density giving the foot geom exactly FOOT_MASS, so the
+# MuJoCo auto-mass and the explicit inertial below agree.
+FOOT_DENSITY = FOOT_MASS / ((4.0 / 3.0) * math.pi * FOOT_RADIUS ** 3)
+
+
+def box_inertial_block(mass: float, half: Tuple[float, float, float], pos: str) -> str:
+    """``<inertial>`` for a solid box of the given mass and half-extents.
+
+    MuJoCo derives a jointless box link's mass from the geom ``density``
+    attribute, but the Isaac Sim MJCF importer parses ``density`` and discards
+    it (MjcfParser.cpp:175 is its only use), falling back to the body-level
+    default density of the import config. Emitting the inertial the density
+    would have produced makes the two engines agree exactly and is a no-op for
+    MuJoCo, whose own auto-inertia is this same expression.
+    """
+    hx, hy, hz = half
+    ixx = _solid_box_axis_inertia(mass, 2.0 * hy, 2.0 * hz)
+    iyy = _solid_box_axis_inertia(mass, 2.0 * hx, 2.0 * hz)
+    izz = _solid_box_axis_inertia(mass, 2.0 * hx, 2.0 * hy)
+    return (f'<inertial pos="{pos}" mass="{_fmt(mass)}" '
+            f'diaginertia="{_vec(ixx, iyy, izz)}"/>')
+
+
+def sphere_inertial_block(mass: float, radius: float) -> str:
+    """``<inertial>`` for a solid sphere centred on the body origin."""
+    i = (2.0 / 5.0) * mass * radius * radius
+    return f'<inertial pos="0 0 0" mass="{_fmt(mass)}" diaginertia="{_vec(i, i, i)}"/>'
 
 
 def _solid_box_axis_inertia(mass: float, d1: float, d2: float) -> float:
@@ -298,7 +328,8 @@
             f'diaginertia="{_vec(i_perp, i_perp, i_axial)}"/>')
 
 
-def _leg_xml(leg: str, spec, act_abad, act_hip, act_knee) -> str:
+def _leg_xml(leg: str, spec, act_abad, act_hip, act_knee,
+             for_isaaclab: bool = False) -> str:
     sx, sy = SIDE_SIGN_X[leg], SIDE_SIGN_Y[leg]
 
     # abad_{LEG}_act, relative to base_Link -- Rules 1/3 (base offset) plus
@@ -350,6 +381,21 @@
         -L2 + spec.clearance_shank2foot_z,
     )
 
+    thigh_inertial = box_inertial_block(
+        LAMBDA_THIGH * L1,
+        (THIGH_BOX_HALF[0], THIGH_BOX_HALF[1], (L1 * VISUAL_LINK_SCALE) / 2.0),
+        f"0 0 {_fmt(-L1 / 2.0)}")
+    shank_inertial = box_inertial_block(
+        LAMBDA_SHANK * L2,
+        (SHANK_BOX_HALF[0], SHANK_BOX_HALF[1], (L2 * VISUAL_LINK_SCALE) / 2.0),
+        f"0 0 {_fmt(-L2 / 2.0)}")
+    foot_inertial = sphere_inertial_block(FOOT_MASS, FOOT_RADIUS)
+    # The site is a MuJoCo sensor anchor, and it shares its name with the foot's
+    # collision geom. Isaac Lab reads contacts from the PhysX body prims and its
+    # converter hard-codes site import on (mjcf_converter.py:86), so the site
+    # would become a dead instanced prim under a clashing name.
+    foot_site = "" if for_isaaclab else f'\n                  <site name="{leg}" pos="0 0 0" size="0.01"/>'
+
     return f"""
       <!-- ==================== {leg} LEG (actuator cluster + thigh + shank) ==================== -->
       <body name="abad_{leg}_act" pos="{_vec(*abad_pos)}">
@@ -368,19 +414,21 @@
             {_act_geom_block(KNEE_RADIUS, KNEE_HALFLEN, QUAT_ALIGN_Y, LINK_COLORS['knee'])}
 
             <body name="thigh_{leg}_Link" pos="{_vec(*thigh_pos)}">
+              {thigh_inertial}
               <geom class="visual" type="box" pos="0 0 {_fmt(-L1 / 2.0)}" size="{_vec(THIGH_BOX_HALF[0] * VISUAL_LINK_SCALE, THIGH_BOX_HALF[1] * VISUAL_LINK_SCALE, (L1 * VISUAL_LINK_SCALE) / 2.0)}" rgba="{LINK_COLORS['thigh']}"/>
               <geom class="collision" type="box" pos="0 0 {_fmt(-L1 / 2.0)}" size="{_vec(THIGH_BOX_HALF[0], THIGH_BOX_HALF[1], (L1 * VISUAL_LINK_SCALE) / 2.0)}" density="{_fmt(THIGH_DENSITY)}"/>
 
               <body name="shank_{leg}_Link" pos="0 0 {_fmt(-L1)}">
+                {shank_inertial}
                 <joint class="knee" name="{leg}_calf_joint" axis="0 1 0" range="{_vec(*KNEE_RANGE)}"/>
                 <geom class="visual" type="box" pos="0 0 {_fmt(-L2 / 2.0)}" size="{_vec(SHANK_BOX_HALF[0] * VISUAL_LINK_SCALE, SHANK_BOX_HALF[1] * VISUAL_LINK_SCALE, (L2 * VISUAL_LINK_SCALE) / 2.0)}" rgba="{LINK_COLORS['shank']}"/>
                 <geom class="collision" type="box" pos="0 0 {_fmt(-L2 / 2.0)}" size="{_vec(SHANK_BOX_HALF[0], SHANK_BOX_HALF[1], (L2 * VISUAL_LINK_SCALE) / 2.0)}" density="{_fmt(SHANK_DENSITY)}"/>
 
                 <body name="foot_{leg}_Link" pos="{_vec(*foot_pos)}">
+                  {foot_inertial}
                   <geom class="visual" type="sphere" size="{_fmt(FOOT_RADIUS)}" rgba="0.05 0.05 0.05 1"/>
-                  <geom class="collision" name="{leg}" type="sphere" size="{_fmt(FOOT_RADIUS)}" density="0"
-                        friction="0.8 0.02 0.01" priority="1" solimp="0.015 1 0.023" condim="6"/>
-                  <site name="{leg}" pos="0 0 0" size="0.01"/>
+                  <geom class="collision" name="foot_{leg}_collision" type="sphere" size="{_fmt(FOOT_RADIUS)}" density="{_fmt(FOOT_DENSITY)}"
+                        friction="0.8 0.02 0.01" priority="1" solimp="0.015 1 0.023" condim="6"/>{foot_site}
                 </body>
               </body>
             </body>
@@ -392,7 +440,17 @@
 # ---------------------------------------------------------------------------
 # Top-level MJCF assembly
 # ---------------------------------------------------------------------------
-def build_xml(spec) -> str:
+def build_xml(spec, for_isaaclab: bool = False) -> str:
+    """Emit the MuJoCo document for ``spec``.
+
+    ``for_isaaclab`` narrows the document from a complete MuJoCo *scene* to a
+    bare robot *description*, which is all the Isaac Sim MJCF importer wants.
+    It drops the skybox and ground material, the world light and floor plane,
+    the sites and the sensors that anchor to them, the tracking camera and the
+    home keyframe, and it places the root body at the world origin so that
+    Isaac Lab's own ``init_state.pos`` is the only thing setting the spawn
+    height. See plans/design_generator_integration.md section 4.5.
+    """
     check_geometry(spec)   # Rule 2
 
     # Rule 4: trunk height is pinned to the ab-ad actuator's diameter,
@@ -425,7 +483,7 @@
     base_z = spec.body_height   # direct spawn height
 
     legs_xml = "\n".join(
-        _leg_xml(leg, spec, act_abad, act_hip, act_knee) for leg in LEG_NAMES
+        _leg_xml(leg, spec, act_abad, act_hip, act_knee, for_isaaclab) for leg in LEG_NAMES
     )
 
     limits = {"abad": act_abad.peak_torque, "hip": act_hip.peak_torque, "knee": act_knee.peak_torque}
@@ -464,6 +522,27 @@
         for leg in LEG_NAMES
     )
 
+    scene_xml = "" if for_isaaclab else _SCENE_BLOCK
+    world_xml = "" if for_isaaclab else _WORLD_BLOCK
+    base_pos = "0 0 0" if for_isaaclab else f"0 0 {_fmt(base_z)}"
+    trunk_extras = "" if for_isaaclab else (
+        '<site name="imu" pos="0 0 0" size="0.01"/>\n'
+        '      <camera name="tracking" mode="trackcom" pos="0 -1.2 0.6" xyaxes="1 0 0 0 1 1"/>')
+    sensors_xml = "" if for_isaaclab else f'''
+  <sensor>
+    <framequat name="base_quat" objtype="site" objname="imu"/>
+    <framepos name="base_pos" objtype="site" objname="imu"/>
+    <gyro name="base_gyro" site="imu"/>
+    <accelerometer name="base_accel" site="imu"/>
+{foot_sensors_xml}
+    <subtreecom name="base_subtreecom" body="trunk"/>
+    <subtreelinvel name="base_subtreelinvel" body="trunk"/>
+  </sensor>'''
+    keyframe_xml = "" if for_isaaclab else f'''
+  <keyframe>
+    <key name="home" qpos="{qpos_str}"/>
+  </keyframe>'''
+
     return f"""<mujoco model="design_quadruped">
   <compiler angle="radian" autolimits="true"/>
 
@@ -471,17 +550,7 @@
     <flag eulerdamp="disable"/>
   </option>
 
-  <visual>
-    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
-    <global azimuth="120" elevation="-15"/>
-  </visual>
-
-  <asset>
-    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
-    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
-              rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
-    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
-  </asset>
+{scene_xml}
 
   <default>
     <geom contype="1" conaffinity="0" friction="0.9 0.02 0.001" solimp="0.9 0.95 0.001" solref="0.01 1"/>
@@ -494,16 +563,11 @@
 {defaults_xml}
   </default>
 
-  <worldbody>
-    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
-    <geom name="floor" type="plane" size="10 10 0.05" material="groundplane" friction="0.9 0.02 0.001"
-          contype="1" conaffinity="1"/>
-
-    <body name="trunk" pos="0 0 {_fmt(base_z)}">
+  <worldbody>{world_xml}
+    <body name="trunk" pos="{base_pos}">
       <freejoint name="root"/>
       <inertial pos="0 0 0" mass="{_fmt(m_trunk)}" diaginertia="{_vec(trunk_Ixx, trunk_Iyy, trunk_Izz)}"/>
-      <site name="imu" pos="0 0 0" size="0.01"/>
-      <camera name="tracking" mode="trackcom" pos="0 -1.2 0.6" xyaxes="1 0 0 0 1 1"/>
+      {trunk_extras}
       <geom class="visual" type="box" size="{_vec(spec.trunk_length / 2.0, spec.trunk_width / 2.0, spec.trunk_thickness / 2.0)}" rgba="0.79216 0.81961 0.93333 1"/>
       <geom class="collision" type="box" size="{_vec(spec.trunk_length / 2.0, spec.trunk_width / 2.0, spec.trunk_thickness / 2.0)}"/>
 {legs_xml}
@@ -513,19 +577,29 @@
   <actuator>
 {actuators_xml}
   </actuator>
+{sensors_xml}{keyframe_xml}
+</mujoco>
+"""
 
-  <sensor>
-    <framequat name="base_quat" objtype="site" objname="imu"/>
-    <framepos name="base_pos" objtype="site" objname="imu"/>
-    <gyro name="base_gyro" site="imu"/>
-    <accelerometer name="base_accel" site="imu"/>
-{foot_sensors_xml}
-    <subtreecom name="base_subtreecom" body="trunk"/>
-    <subtreelinvel name="base_subtreelinvel" body="trunk"/>
-  </sensor>
 
-  <keyframe>
-    <key name="home" qpos="{qpos_str}"/>
-  </keyframe>
-</mujoco>
+# ---------------------------------------------------------------------------
+# MuJoCo-only scene furniture, omitted when for_isaaclab is set.
+# ---------------------------------------------------------------------------
+_SCENE_BLOCK = """  <visual>
+    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
+    <global azimuth="120" elevation="-15"/>
+  </visual>
+
+  <asset>
+    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
+    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
+              rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
+    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
+  </asset>
+"""
+
+_WORLD_BLOCK = """
+    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
+    <geom name="floor" type="plane" size="10 10 0.05" material="groundplane" friction="0.9 0.02 0.001"
+          contype="1" conaffinity="1"/>
 """
```

The biped's is the same shape without the foot sphere, its foot already being a box with a non-zero density.

```diff
--- a/co_optimisation/co_optimisation/runners/design_generator/biped/builder.py
+++ b/co_optimisation/co_optimisation/runners/design_generator/biped/builder.py
@@ -203,6 +203,26 @@
     return (1.0 / 12.0) * mass * (d1 * d1 + d2 * d2)
 
 
+def box_inertial_block(mass: float, half: Tuple[float, float, float],
+                       pos: str, quat: str = None) -> str:
+    """``<inertial>`` for a solid box of the given mass and half-extents.
+
+    MuJoCo derives a jointless box link's mass from the geom ``density``
+    attribute, but the Isaac Sim MJCF importer parses ``density`` and
+    discards it (MjcfParser.cpp:175 is its only use), falling back to the
+    body-level default density of the import config. Emitting the inertial
+    the density would have produced makes the two engines agree exactly and
+    is a no-op for MuJoCo, whose own auto-inertia is this same expression.
+    """
+    hx, hy, hz = half
+    ixx = _solid_box_axis_inertia(mass, 2.0 * hy, 2.0 * hz)
+    iyy = _solid_box_axis_inertia(mass, 2.0 * hx, 2.0 * hz)
+    izz = _solid_box_axis_inertia(mass, 2.0 * hx, 2.0 * hy)
+    q = f' quat="{quat}"' if quat else ""
+    return (f'<inertial pos="{pos}"{q} mass="{_fmt(mass)}" '
+            f'diaginertia="{_vec(ixx, iyy, izz)}"/>')
+
+
 def hollow_box_inertia(mass: float, a: float, b: float, c: float, t: float) -> Tuple[float, float, float]:
     """(Ix, Iy, Iz) for a thick-walled hollow cuboid shell of uniform wall
     thickness `t`, outer dims a(x) * b(y) * c(z), and total shell mass
@@ -311,7 +331,8 @@
             f'diaginertia="{_vec(i_perp, i_perp, i_axial)}"/>')
 
 
-def _leg_xml(side: str, spec, act_abad, act_hip, act_knee, act_ankle) -> str:
+def _leg_xml(side: str, spec, act_abad, act_hip, act_knee, act_ankle,
+             for_isaaclab: bool = False) -> str:
     L = _cluster_layout(side, spec, act_abad, act_hip, act_knee)
     abad_axis = "-1 0 0" if side == "R" else "1 0 0"
 
@@ -362,6 +383,28 @@
 
     knee_lo, knee_hi = KNEE_RANGE
 
+    # Explicit inertials for the three density-driven links. See
+    # box_inertial_block: the values are exactly MuJoCo's own auto-inertia, so
+    # this is inert there and is the only thing the Isaac Sim importer reads.
+    thigh_inertial = box_inertial_block(
+        LAMBDA_THIGH * spec.thigh_length,
+        (THIGH_BOX_HALF[0], THIGH_BOX_HALF[1], box_half_len),
+        _vec3(box_center_local), box_quat)
+    shank_inertial = box_inertial_block(
+        LAMBDA_SHANK * spec.shank_length,
+        (SHANK_BOX_HALF[0], SHANK_BOX_HALF[1], shank_half_len),
+        shank_box_pos)
+    foot_inertial = box_inertial_block(
+        LAMBDA_FOOT * spec.foot_length,
+        (spec.foot_length / 2.0, FOOT_BOX_HALF[0], FOOT_BOX_HALF[1]),
+        f"{_fmt(x_offset_ankleAct2foot)} 0.0 {_fmt(z_offset_ankleAct2foot)}")
+    # The site is a MuJoCo sensor anchor. Isaac Lab reads contacts from the
+    # PhysX body prims instead, and its converter hard-codes site import on
+    # (mjcf_converter.py:86), so the site would become a dead instanced prim.
+    foot_site = "" if for_isaaclab else (
+        f'\n                    <site name="foot_{side}" '
+        f'pos="{_fmt(x_offset_ankleAct2foot)} 0.0 {_fmt(z_bottom_ankleAct2foot)}" size="0.01"/>')
+
     return f"""
       <!-- ==================== {'RIGHT' if side == 'R' else 'LEFT'} LEG (actuator cluster + thigh) ==================== -->
       <body name="abad_{side}_act" pos="{_vec3(L['abad_pos'])}">
@@ -380,10 +423,12 @@
             {_act_geom_block(L['r_knee'], L['hl_knee'], QUAT_ALIGN_Y, LINK_COLORS['knee'])}
 
             <body name="thigh_{side}_Link" pos="{_vec3(L['thigh_pos_rel'])}">
+              {thigh_inertial}
               <geom class="visual" type="box" size="{box_size}" pos="{_vec3(box_center_local)}" quat="{box_quat}" rgba="{LINK_COLORS['thigh']}"/>
               <geom class="collision" type="box" size="{box_size}" pos="{_vec3(box_center_local)}" quat="{box_quat}" density="{_fmt(THIGH_DENSITY)}"/>
 
               <body name="shank_{side}_Link" pos="{shank_pos_rel}" quat="{SHANK_QUAT}">
+                {shank_inertial}
                 <joint class="knee" name="knee_{side}_Joint" axis="0 -1 0" range="{_fmt(knee_lo)} {_fmt(knee_hi)}"/>
                 <geom class="visual" type="box" size="{shank_box_size}" pos="{shank_box_pos}" rgba="{LINK_COLORS['shank']}"/>
                 <geom class="collision" type="box" size="{shank_box_size}" pos="{shank_box_pos}" density="{_fmt(SHANK_DENSITY)}"/>
@@ -393,10 +438,11 @@
                   {_act_geom_block(r_ankle, hl_ankle, QUAT_ALIGN_Y, LINK_COLORS['ankle'])}
 
                   <body name="foot_{side}_Link" pos="{foot_pos_rel}">
+                    {foot_inertial}
                     <joint class="ankle" name="ankle_{side}_Joint" axis="0 -1 0" range="{_vec(*ANKLE_RANGE)}"/>
                     <geom class="visual" type="box" size="{foot_size}" pos="{_fmt(x_offset_ankleAct2foot)} 0.0 {_fmt(z_offset_ankleAct2foot)}" rgba="{LINK_COLORS['ankle']}"/>
                     <geom class="collision" name="foot_{side}_collision" type="box" size="{foot_size}" pos="{_fmt(x_offset_ankleAct2foot)} 0.0 {_fmt(z_offset_ankleAct2foot)}" density="{_fmt(FOOT_DENSITY)}"/>
-                    <site name="foot_{side}" pos="{_fmt(x_offset_ankleAct2foot)} 0.0 {_fmt(z_bottom_ankleAct2foot)}" size="0.01"/>
+{foot_site}
                   </body>
                 </body>
               </body>
@@ -409,7 +455,17 @@
 # ---------------------------------------------------------------------------
 # Top-level MJCF assembly
 # ---------------------------------------------------------------------------
-def build_xml(spec) -> str:
+def build_xml(spec, for_isaaclab: bool = False) -> str:
+    """Emit the MuJoCo document for ``spec``.
+
+    ``for_isaaclab`` narrows the document from a complete MuJoCo *scene* to a
+    bare robot *description*, which is all the Isaac Sim MJCF importer wants.
+    It drops the skybox and ground material, the world light and floor plane,
+    the sites and the sensors that anchor to them, and the home keyframe, and
+    it places the root body at the world origin so that Isaac Lab's own
+    ``init_state.pos`` is the only thing setting the spawn height. See
+    plans/design_generator_integration.md section 4.5.
+    """
     act_abad, act_hip, act_knee, act_ankle = spec.actuators()
     check_geometry(spec, act_abad, act_hip)
 
@@ -442,8 +498,8 @@
 
     base_z = spec.body_height   # direct spawn height -- no hip_z_drop composition in this topology
 
-    right_xml = _leg_xml("R", spec, act_abad, act_hip, act_knee, act_ankle)
-    left_xml = _leg_xml("L", spec, act_abad, act_hip, act_knee, act_ankle)
+    right_xml = _leg_xml("R", spec, act_abad, act_hip, act_knee, act_ankle, for_isaaclab)
+    left_xml = _leg_xml("L", spec, act_abad, act_hip, act_knee, act_ankle, for_isaaclab)
 
     limits = {"abad": act_abad.peak_torque, "hip": act_hip.peak_torque,
               "knee": act_knee.peak_torque, "ankle": act_ankle.peak_torque}
@@ -470,6 +526,16 @@
         for side in LEG_SIDES for joint in ("abad", "hip", "knee", "ankle")
     )
 
+    scene_xml = "" if for_isaaclab else _SCENE_BLOCK
+    world_xml = "" if for_isaaclab else _WORLD_BLOCK
+    base_pos = "0 0 0" if for_isaaclab else f"0 0 {_fmt(base_z)}"
+    imu_site = "" if for_isaaclab else '<site name="imu" pos="0 0 0" size="0.01"/>'
+    sensors_xml = "" if for_isaaclab else _SENSOR_BLOCK
+    keyframe_xml = "" if for_isaaclab else f'''
+  <keyframe>
+    <key name="home" qpos="{qpos_str}"/>
+  </keyframe>'''
+
     return f"""<mujoco model="design_biped">
   <compiler angle="radian" autolimits="true"/>
 
@@ -477,17 +543,7 @@
     <flag eulerdamp="disable"/>
   </option>
 
-  <visual>
-    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
-    <global azimuth="120" elevation="-15"/>
-  </visual>
-
-  <asset>
-    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
-    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
-              rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
-    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
-  </asset>
+{scene_xml}
 
   <default>
     <geom contype="1" conaffinity="0" friction="0.9 0.02 0.001" solimp="0.9 0.95 0.001" solref="0.01 1"/>
@@ -500,15 +556,11 @@
 {defaults_xml}
   </default>
 
-  <worldbody>
-    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
-    <geom name="floor" type="plane" size="10 10 0.05" material="groundplane" friction="0.9 0.02 0.001"
-          contype="1" conaffinity="1"/>
-
-    <body name="base_Link" pos="0 0 {_fmt(base_z)}">
+  <worldbody>{world_xml}
+    <body name="base_Link" pos="{base_pos}">
       <freejoint name="root"/>
       <inertial pos="0 0 0" mass="{_fmt(m_trunk)}" diaginertia="{_vec(trunk_Ixx, trunk_Iyy, trunk_Izz)}"/>
-      <site name="imu" pos="0 0 0" size="0.01"/>
+      {imu_site}
       <geom class="visual" type="box" size="{_vec(spec.trunk_length / 2.0, spec.trunk_width / 2.0, spec.trunk_thickness / 2.0)}" rgba="0.79216 0.81961 0.93333 1"/>
       <geom class="collision" type="box" size="{_vec(spec.trunk_length / 2.0, spec.trunk_width / 2.0, spec.trunk_thickness / 2.0)}"/>
 {right_xml}
@@ -519,7 +571,33 @@
   <actuator>
 {actuators_xml}
   </actuator>
+{sensors_xml}{keyframe_xml}
+</mujoco>
+"""
 
+# ---------------------------------------------------------------------------
+# MuJoCo-only scene furniture, omitted when for_isaaclab is set.
+# ---------------------------------------------------------------------------
+_SCENE_BLOCK = """  <visual>
+    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
+    <global azimuth="120" elevation="-15"/>
+  </visual>
+
+  <asset>
+    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
+    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
+              rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
+    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
+  </asset>
+"""
+
+_WORLD_BLOCK = """
+    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
+    <geom name="floor" type="plane" size="10 10 0.05" material="groundplane" friction="0.9 0.02 0.001"
+          contype="1" conaffinity="1"/>
+"""
+
+_SENSOR_BLOCK = """
   <sensor>
     <framequat name="base_quat" objtype="site" objname="imu"/>
     <framepos name="base_pos" objtype="site" objname="imu"/>
@@ -532,9 +610,4 @@
     <subtreecom name="base_subtreecom" body="base_Link"/>
     <subtreelinvel name="base_subtreelinvel" body="base_Link"/>
   </sensor>
-
-  <keyframe>
-    <key name="home" qpos="{qpos_str}"/>
-  </keyframe>
-</mujoco>
-"""
\ No newline at end of file
+"""
```

### 9.3 The two `constant_params.json`

The values are in §8.1 and §8.2. They are edits to two JSON files and carry no code.

### 9.4 The two `variable_params.json`

The same, and likewise.

### 9.5 The actuator classes

`common/actuator_catalog.py` gains a constants block, ten optional fields and an `armature` property on `CatalogActuator`, an `armature` and a `saturation_effort` property on `Actuator`, a `sim_params` method on both, and ten keys in the JSON loader.

```diff
--- a/co_optimisation/co_optimisation/runners/design_generator/common/actuator_catalog.py
+++ b/co_optimisation/co_optimisation/runners/design_generator/common/actuator_catalog.py
@@ -258,6 +258,35 @@
 
 
 # ---------------------------------------------------------------------------
+# Simulator-side identification
+# ---------------------------------------------------------------------------
+# Every downstream simulator needs more of an actuator than the joint-side
+# torque, speed, mass and cost this catalog was built around: a stall torque
+# for the four-quadrant DC-motor envelope, a reflected rotor inertia, a
+# friction pair, and the position-loop gains the controller runs at. Both
+# actuator classes below therefore carry those numbers and expose them under
+# one name, `sim_params`, so a consumer never has to know which of the two
+# modes produced the pick.
+#
+# The stall ratio is the median of the seven measured
+# `saturation_effort / effort_limit` pairs in
+# environments/assets/config/{solefoot,quadruped}_identified_cfg.py, which
+# span 6.22 to 10.05 and cluster at 6.7. It is used only when an entry does
+# not state its own saturation_effort.
+DEFAULT_STALL_RATIO = 6.7
+
+# Keys of the dict `sim_params` returns. They are exactly the attribute names
+# of isaaclab.actuators.DCMotor and environments.actuators.IdentifiedActuator,
+# so the dict can be handed to co_optimisation.utils.respawn.apply_actuator_params
+# unchanged.
+SIM_PARAM_KEYS = (
+    "effort_limit", "velocity_limit", "saturation_effort", "armature",
+    "friction_static", "friction_dynamic", "activation_vel",
+    "stiffness", "damping",
+)
+
+
+# ---------------------------------------------------------------------------
 # Composed actuator (joint-side view)
 # ---------------------------------------------------------------------------
 @dataclass(frozen=True)
@@ -334,6 +363,34 @@
             return self.motor.length
         return 0.6 * self.diameter
 
+    # ---- simulator-side identification -------------------------------------
+    @property
+    def armature(self) -> float:
+        """Joint-side reflected rotor inertia [kg·m²], ``I_rotor · N²``."""
+        return float(self.motor.rotor_inertia * self.ratio ** 2)
+
+    @property
+    def saturation_effort(self) -> float:
+        """Stall torque [N·m] of the four-quadrant DC-motor envelope."""
+        return float(self.peak_torque * DEFAULT_STALL_RATIO)
+
+    def sim_params(self, **overrides) -> Dict[str, float]:
+        """Simulator-side actuator parameters for one joint group.
+
+        Keys are :data:`SIM_PARAM_KEYS`. A COMPAct-composed actuator fixes the
+        torque envelope, the speed ceiling and the reflected inertia; friction
+        and the position-loop gains have no counterpart in the published data,
+        so they are omitted unless the caller supplies them.
+        """
+        out = {
+            "effort_limit": float(self.peak_torque),
+            "velocity_limit": float(self.peak_speed),
+            "saturation_effort": self.saturation_effort,
+            "armature": self.armature,
+        }
+        out.update({k: float(v) for k, v in overrides.items() if v is not None})
+        return out
+
     # ---- electrical-power model -------------------------------------------
     def electrical_power(self, tau_joint: float, omega_joint: float) -> float:
         """Instantaneous electrical input power [W] given joint-side τ and ω.
@@ -442,6 +499,62 @@
     ratio: Optional[float] = None     # gearbox ratio
     notes: str = ""
 
+    # ---- simulator-side identification, all optional ----------------------
+    # Stated per entry in catalogs/actuators.json where the unit has been
+    # identified on hardware, and derived below where it has not. See
+    # SIM_PARAM_KEYS for what each one drives.
+    rotor_inertia:     Optional[float] = None   # [kg·m²] motor side
+    armature_value:    Optional[float] = None   # [kg·m²] joint side, overrides the derivation
+    effort_limit:      Optional[float] = None   # [N·m] continuous ceiling
+    velocity_limit:    Optional[float] = None   # [rad/s]
+    saturation_effort: Optional[float] = None   # [N·m] stall torque
+    friction_static:   Optional[float] = None   # [N·m]
+    friction_dynamic:  Optional[float] = None   # [N·m·s/rad]
+    activation_vel:    Optional[float] = None   # [rad/s]
+    stiffness:         Optional[float] = None   # [N·m/rad]
+    damping:           Optional[float] = None   # [N·m·s/rad]
+
+    @property
+    def armature(self) -> float:
+        """Joint-side reflected rotor inertia [kg·m²].
+
+        Taken from ``armature_value`` when the entry states one, else derived
+        as ``I_rotor · N²`` from ``rotor_inertia`` and ``ratio``, else zero.
+        """
+        if self.armature_value is not None:
+            return float(self.armature_value)
+        if self.rotor_inertia is not None and self.ratio is not None:
+            return float(self.rotor_inertia * self.ratio ** 2)
+        return 0.0
+
+    def sim_params(self, **overrides) -> Dict[str, float]:
+        """Simulator-side actuator parameters for one joint group.
+
+        Keys are :data:`SIM_PARAM_KEYS`. Entries the catalog does not state
+        fall back to the joint-side envelope this class was built around:
+        ``effort_limit`` to ``peak_torque``, ``velocity_limit`` to
+        ``peak_speed``, ``saturation_effort`` to
+        ``effort_limit * DEFAULT_STALL_RATIO``. Friction and the two gains are
+        omitted when unstated, so a consumer keeps whatever its own
+        configuration holds rather than being handed a fabricated number.
+        """
+        effort = self.effort_limit if self.effort_limit is not None else self.peak_torque
+        out: Dict[str, float] = {
+            "effort_limit": float(effort),
+            "velocity_limit": float(self.velocity_limit if self.velocity_limit is not None
+                                    else self.peak_speed),
+            "saturation_effort": float(self.saturation_effort if self.saturation_effort is not None
+                                       else effort * DEFAULT_STALL_RATIO),
+            "armature": self.armature,
+        }
+        for key in ("friction_static", "friction_dynamic", "activation_vel",
+                    "stiffness", "damping"):
+            value = getattr(self, key)
+            if value is not None:
+                out[key] = float(value)
+        out.update({k: float(v) for k, v in overrides.items() if v is not None})
+        return out
+
     # ---- electrical-power model ------------------------------------------
     def electrical_power(self, tau_joint: float, omega_joint: float) -> float:
         """Direction-aware electrical input power [W].
@@ -475,6 +588,15 @@
 ActuatorLike = Union[Actuator, CatalogActuator]
 
 
+# The simulator-side fields a JSON entry may state, read verbatim when present
+# and left at None otherwise so CatalogActuator.sim_params can fall back.
+_SIM_JSON_KEYS = (
+    "rotor_inertia", "armature_value", "effort_limit", "velocity_limit",
+    "saturation_effort", "friction_static", "friction_dynamic",
+    "activation_vel", "stiffness", "damping",
+)
+
+
 def load_actuator_catalog_json(path: Union[str, Path]) -> List["ActuatorLike"]:
     """Load a flat actuator catalog from JSON (see catalogs/actuators.json).
 
@@ -503,6 +625,7 @@
             R=(float(d["R"])  if "R"  in d else None),
             ratio=(float(d["ratio"]) if "ratio" in d else None),
             notes=str(d.get("notes", "")),
+            **{k: (float(d[k]) if k in d else None) for k in _SIM_JSON_KEYS},
         ))
     return out
 
```

`sim_params` returns a dict keyed exactly as `IdentifiedActuator`'s attributes are named, so it can be handed to `apply_actuator_params` with no translation. That was checked against all seven new entries, each of which reproduces its identified configuration exactly, the armature of `TRON1-ABAD` for instance resolving to 0.005589 from `6.9e-05 * 9.0 ** 2`.

## 10. `co_optimisation/co_optimisation/runners/usd_generator.py`

The file is rewritten in place and keeps its name, the review having asked that the URDF pathway be deprecated inside it rather than beside it. Five of its six public names survive unchanged in meaning, `DesignGeneratorBase`, `RandomDesignGenerator`, `CMAESDesignGenerator`, `GrowingDesignDistCMAESDesignGenerator` and `Population`. `RandomPopulation` is deleted and `Population` is made concrete, absorbing its three methods and gaining two more. `CatCMAESDesignGenerator` is added.

Four things carry over verbatim, deliberately. The `ask` and `tell` discipline, the late-start machinery and its widening prior, the cost sanitisation and the `get_state` and `load_state` pair are unchanged, so an existing checkpoint of a length-only run still loads, the search dimension still being two.

Four things change. The base class takes a family name where it took a URDF path. A specification is assembled with `dataclasses.replace` on a pinned baseline rather than through `from_vector`, for the reason §4.8 gives, which also preserves the two-decimal rounding the URDF-era generator applied and with it the conversion cache's hit rate. The actuator parameters come from the picked catalogue entry's `sim_params` rather than from a hardcoded baseline table. And the in-place payload is a complete geometric description rather than three extents, which is what lets §11 do no inference of its own.

The categorical search is `cmaes.CatCMAwM`, which the review named. It carries a multivariate Gaussian over the continuous block and a categorical distribution per discrete block, and applies the margin correction of Hamano et al. [1], which lower-bounds each discrete marginal so that the search cannot collapse onto one catalogue entry before the continuous block has converged. Revision 1 had proposed a continuous relaxation with rounding instead, which is the failure mode that correction exists to answer.

The feasibility guard is new and is not decoration. `check_geometry` raises when a sampled trunk width cannot accommodate the chosen abduction housing (`biped/builder.py:404-424`) or when Rule 2 is violated (`quadruped/builder.py:283-289`), and under a categorical search that is reachable rather than hypothetical. A raised generation would abort a training run, so an infeasible individual silently repeats the last feasible design, which costs one individual's diversity instead. Both branches were exercised.

```python
"""MJCF-based robot design generator for co-optimisation.

Every design in a population is a document emitted by the vendored parametric
generator at ``co_optimisation/runners/design_generator``, converted to
Universal Scene Description by :class:`isaaclab.sim.converters.MjcfConverter`
and spawned through the multi-asset pathway.  The generator is the single
source of truth: it owns the geometry, the mass model and the actuator
catalog, and it hands this module the three things the runner consumes, a
Universal Scene Description path per individual, the actuator parameters that
cannot be expressed in the asset, and the absolute link geometry the in-place
editor in ``co_optimisation/utils/update.py`` writes.

``param_ranges`` dict keys accepted by :class:`RandomDesignGenerator`:

- ``"thigh_length_scale"``     – scale on ``DesignSpec.thigh_length``. Range: (0.75, 1.25)
- ``"shank_length_scale"``     – scale on ``DesignSpec.shank_length``. Range: (0.75, 1.25)
- ``"joint_effort_scale"``     – scale on the picked actuator's ``effort_limit``. Range: (0.70, 1.30)
- ``"velocity_limit_scale"``   – scale on its ``velocity_limit``. Range: (0.80, 1.20)
- ``"saturation_effort_scale"``– scale on its ``saturation_effort``. Range: (0.70, 1.30)
- ``"armature_scale"``         – scale on its ``armature``. Range: (0.70, 1.30)
- ``"friction_static_scale"``  – scale on its ``friction_static``. Range: (0.70, 1.40)
- ``"friction_dynamic_scale"`` – scale on its ``friction_dynamic``. Range: (0.70, 1.40)

Per-group stiffness/damping scales, applied post-spawn via
``apply_actuator_params``.  Ranges are the symmetric equivalents of the runtime
domain-randomisation absolute ranges, an asymmetric absolute range being
resolved by taking the smaller half:

- ``"abad_stiffness_scale"``  – Range: (0.909, 1.091)
- ``"abad_damping_scale"``    – Range: (0.889, 1.111)
- ``"hip_stiffness_scale"``   – Range: (0.875, 1.125)
- ``"hip_damping_scale"``     – Range: (0.846, 1.154)
- ``"knee_stiffness_scale"``  – Range: (0.833, 1.167)
- ``"knee_damping_scale"``    – Range: (0.750, 1.250)
- ``"ankle_stiffness_scale"`` – Range: (0.800, 1.200), biped only
- ``"ankle_damping_scale"``   – Range: (0.800, 1.200), biped only

Three keys of the URDF-era generator are gone.  ``link_mass_scale`` has no
counterpart because a link's mass is ``lambda * length`` under the generator's
own model, so it moves with the length scales rather than independently;
``actuator_radius_scale`` and ``actuator_length_scale`` have none because the
housing envelope is a property of the catalog entry the design picks, so it
moves with :class:`CatCMAESDesignGenerator`'s categorical variables instead.
"""

from __future__ import annotations

import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

import cma

from design_generator import biped, quadruped

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def _box_inertia(mass: float, half) -> tuple[float, float, float]:
    """Diagonal moment of inertia of a solid box about its own centre."""
    hx, hy, hz = (float(v) for v in half)
    return (
        mass * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12.0,
        mass * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12.0,
        mass * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12.0,
    )


def _entry(half, pos, mass, child_joint, child_body, child_pos) -> dict:
    """One scalable link's absolute geometry, as ``update.py`` consumes it.

    ``half`` are MuJoCo half-extents, which is also the scale the Isaac Sim
    MuJoCo importer authors on the box prototype, so they are written through
    unchanged.  ``pos`` is the box centre in the link frame, which for these
    links is also the centre of mass.  ``child_pos`` is the child body's origin
    in the link frame, which is what the child joint's ``physics:localPos0``
    holds.
    """
    mass = float(mass)
    return {
        "half_extents": [float(v) for v in half],
        "box_pos": [float(v) for v in pos],
        "mass": mass,
        "inertia": list(_box_inertia(mass, half)),
        "child_joint": child_joint,
        "child_body": child_body,
        "child_pos": [float(v) for v in child_pos],
    }


def _biped_links(spec) -> dict[str, dict]:
    b = biped.builder
    l1, l2 = float(spec.thigh_length), float(spec.shank_length)
    out: dict[str, dict] = {}
    for side in b.LEG_SIDES:
        out[f"thigh_{side}_Link"] = _entry(
            half=(b.THIGH_BOX_HALF[0], b.THIGH_BOX_HALF[1],
                  b.LINK_BOX_LENGTH_FRAC * l1 / 2.0),
            pos=(l1 / 2.0) * b.THIGH_BOX_DIR,
            mass=b.LAMBDA_THIGH * l1,
            child_joint=f"knee_{side}_Joint",
            child_body=f"shank_{side}_Link",
            child_pos=(l1 * b.THIGH_DIR_XZ[0], 0.0, l1 * b.THIGH_DIR_XZ[1]),
        )
        out[f"shank_{side}_Link"] = _entry(
            half=(b.SHANK_BOX_HALF[0], b.SHANK_BOX_HALF[1],
                  b.LINK_BOX_LENGTH_FRAC * l2 / 2.0),
            pos=(0.0, 0.0, -l2 / 2.0),
            mass=b.LAMBDA_SHANK * l2,
            # ankle_{S}_act carries no joint, so the importer names its fixed
            # joint after the body (MjcfImporter.cpp:1353).
            child_joint=f"ankle_{side}_act",
            child_body=f"ankle_{side}_act",
            child_pos=(spec.clearance_shank2ankleAct_x,
                       spec.clearance_shank2ankleAct_y,
                       -l2 + spec.clearance_shank2ankleAct_z),
        )
    return out


def _quadruped_links(spec) -> dict[str, dict]:
    q = quadruped.builder
    l1, l2 = float(spec.thigh_length), float(spec.shank_length)
    out: dict[str, dict] = {}
    for leg in q.LEG_NAMES:
        out[f"thigh_{leg}_Link"] = _entry(
            half=(q.THIGH_BOX_HALF[0], q.THIGH_BOX_HALF[1],
                  l1 * q.VISUAL_LINK_SCALE / 2.0),
            pos=(0.0, 0.0, -l1 / 2.0),
            mass=q.LAMBDA_THIGH * l1,
            child_joint=f"{leg}_calf_joint",
            child_body=f"shank_{leg}_Link",
            child_pos=(0.0, 0.0, -l1),
        )
        out[f"shank_{leg}_Link"] = _entry(
            half=(q.SHANK_BOX_HALF[0], q.SHANK_BOX_HALF[1],
                  l2 * q.VISUAL_LINK_SCALE / 2.0),
            pos=(0.0, 0.0, -l2 / 2.0),
            mass=q.LAMBDA_SHANK * l2,
            child_joint=f"foot_{leg}_Link",
            child_body=f"foot_{leg}_Link",
            child_pos=(spec.clearance_shank2foot_x,
                       spec.clearance_shank2foot_y,
                       -l2 + spec.clearance_shank2foot_z),
        )
    return out


class Family:
    """Everything this module needs to know about one robot family."""

    def __init__(self, name, module, groups, spec_fields, root_body, links):
        self.name = name
        self.module = module
        self.groups = groups
        self.spec_fields = spec_fields      # actuator group -> DesignSpec field
        self.root_body = root_body
        self.links = links

    def baseline(self, path=None):
        return (self.module.generate.load_design_spec(path) if path is not None
                else self.module.generate.load_design_spec())

    def build_xml(self, spec) -> str:
        return self.module.builder.build_xml(spec, for_isaaclab=True)

    def catalog(self) -> list:
        return list(self.module.actuator.ACTUATOR_CATALOG)


FAMILIES: dict[str, Family] = {
    "biped": Family(
        "biped", biped,
        groups=("abad", "hip", "knee", "ankle"),
        spec_fields={"abad": "act_abduction", "hip": "act_hip",
                     "knee": "act_knee", "ankle": "act_ankle"},
        root_body="base_Link", links=_biped_links,
    ),
    "quadruped": Family(
        "quadruped", quadruped,
        groups=("abad", "hip", "knee"),
        spec_fields={"abad": "act_abduction", "hip": "act_hip", "knee": "act_knee"},
        root_body="trunk", links=_quadruped_links,
    ),
}

# ---------------------------------------------------------------------------
# Search ranges
# ---------------------------------------------------------------------------

DEFAULT_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "thigh_length_scale": (0.75, 1.25),
    "shank_length_scale": (0.75, 1.25),
    "joint_effort_scale": (0.70, 1.30),
    "velocity_limit_scale": (0.80, 1.20),
    "saturation_effort_scale": (0.70, 1.30),
    "armature_scale": (0.70, 1.30),
    "friction_static_scale": (0.70, 1.40),
    "friction_dynamic_scale": (0.70, 1.40),
    "abad_stiffness_scale": (0.909, 1.091),
    "abad_damping_scale": (0.889, 1.111),
    "hip_stiffness_scale": (0.875, 1.125),
    "hip_damping_scale": (0.846, 1.154),
    "knee_stiffness_scale": (0.833, 1.167),
    "knee_damping_scale": (0.750, 1.250),
    "ankle_stiffness_scale": (0.800, 1.200),
    "ankle_damping_scale": (0.800, 1.200),
}

# CMA-ES restricts its search to the two length scales it actually drives.
CMAES_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "thigh_length_scale": (0.75, 1.25),
    "shank_length_scale": (0.75, 1.25),
}

# The DesignSpec field each length scale multiplies.
LENGTH_SCALE_FIELDS: dict[str, str] = {
    "thigh_length_scale": "thigh_length",
    "shank_length_scale": "shank_length",
}

# The scale key each simulator-side actuator parameter answers to. A parameter
# with no entry here is passed through from the catalog unscaled.
ACTUATOR_SCALE_KEYS: dict[str, str] = {
    "effort_limit": "joint_effort_scale",
    "velocity_limit": "velocity_limit_scale",
    "saturation_effort": "saturation_effort_scale",
    "armature": "armature_scale",
    "friction_static": "friction_static_scale",
    "friction_dynamic": "friction_dynamic_scale",
}


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


class Population:
    """A fixed set of robot designs, one per individual.

    Each individual is a Universal Scene Description path, an actuator
    parameter dict overriding the Python-side ``IdentifiedActuator``
    attributes, and an absolute link-geometry dict for in-place morphology
    updates.  ``root_body`` is the name of the emitted document's root body,
    which the Isaac Sim MuJoCo importer interposes between the referenced
    asset prim and every link prim, so ``utils/update.py`` needs it to address
    a link.
    """

    def __init__(
        self,
        usd_files: list[str],
        actuator_params: list[dict[str, dict]],
        link_length_params: list[dict[str, dict]],
        root_body: str,
        specs: list | None = None,
    ) -> None:
        self._usd_files = usd_files
        self._actuator_params = actuator_params
        self._link_length_params = link_length_params
        self._root_body = root_body
        self._specs = specs if specs is not None else []

    def get_usd_files(self) -> list[str]:
        """Return a list of Universal Scene Description paths, one per individual."""
        return self._usd_files

    def get_actuator_params(self) -> list[dict[str, dict]]:
        """Return actuator override dicts, one per individual.

        Each element is keyed by actuator group name (``"abad"``, ``"hip"``,
        ``"knee"``, ``"ankle"``), whose value is a dict of scalar overrides for
        the ``IdentifiedActuator`` attributes, as
        ``design_generator.common.actuator_catalog.SIM_PARAM_KEYS`` names them.
        """
        return self._actuator_params

    def get_link_length_params(self) -> list[dict[str, dict]]:
        """Return absolute link-geometry dicts, one per individual."""
        return self._link_length_params

    def get_root_body(self) -> str:
        """Return the emitted document's root body name."""
        return self._root_body

    def get_specs(self) -> list:
        """Return the ``DesignSpec`` behind each individual, for logging."""
        return self._specs


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DesignGeneratorBase(ABC):
    """Abstract base class for design generators.

    Provides all shared specification, emission and conversion machinery.
    Subclasses implement :meth:`_generate_individual` to choose one
    individual's :class:`DesignSpec`.

    Args:
        family: ``"biped"`` or ``"quadruped"``, selecting which of the vendored
            generator's two packages supplies the geometry and the catalog.
        num_individuals: Number of designs per generation.
        param_ranges: Optional dict overriding entries in
            :data:`DEFAULT_PARAM_RANGES`.  Only the keys given are overridden.
        output_dir: Directory for the emitted documents and their conversions.
        variable_params_path: Optional path to a ``variable_params.json`` that
            supplies the baseline specification.  Defaults to the one the
            family's own package ships, which reproduces the donor robot.
    """

    def __init__(
        self,
        family: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
        variable_params_path: str | None = None,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(
                f"Unknown robot family {family!r}. Known families: {sorted(FAMILIES)}."
            )
        self.family = FAMILIES[family]
        self.num_individuals = num_individuals
        self.output_dir = output_dir
        self.param_ranges: dict[str, tuple[float, float]] = {**DEFAULT_PARAM_RANGES}
        if param_ranges is not None:
            self.param_ranges.update(param_ranges)
        self.baseline = self.family.baseline(variable_params_path)
        # The last specification that emitted without raising, substituted for
        # any later specification the builder rejects. See _build_xml.
        self._last_feasible = self.baseline
        self._current_scales: dict[str, float] | None = None

    # ---- abstract design hooks ----------------------------------------------

    @abstractmethod
    def _generate_individual(self, generation: int, idx: int):
        """Return ``(spec, actuator_params)`` for one individual."""
        ...

    @abstractmethod
    def generate_population(self, generation: int) -> Population | None:
        """Generate a new population for *generation*."""
        ...

    def update_with_fitness(self, fitness: list[float]) -> None:
        """Optionally update internal state using per-individual fitness scores.

        The default implementation is a no-op (random search).
        """
        pass

    def sample_batch(self) -> None:
        pass

    # ---- shared population pipeline -----------------------------------------

    def _build_population(self, generation: int, indices) -> Population:
        """Build generation *generation* for the individual indices *indices*."""
        usd_files: list[str] = []
        actuator_params: list[dict[str, dict]] = []
        link_length_params: list[dict[str, dict]] = []
        specs: list = []
        for idx in indices:
            spec, act = self._generate_individual(generation, idx)
            xml_path = self._generate_individual_mjcf(generation, idx, spec)
            usd_files.append(self._generate_individual_usd(xml_path, idx))
            actuator_params.append(act)
            link_length_params.append(self.family.links(spec))
            specs.append(spec)
        return Population(usd_files, actuator_params, link_length_params,
                          self.family.root_body, specs)

    # ---- scale sampling ------------------------------------------------------

    def _sample_scales(self, rng: np.random.Generator) -> dict[str, float]:
        return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.param_ranges.items()}

    def _sample_scales_v2(self, rng: np.random.Generator, scale: float) -> dict[str, float]:
        scales = {}
        for k, (lo, hi) in self.param_ranges.items():
            mean = (lo + hi) / 2
            lo_n = (lo - mean) * scale + mean
            hi_n = (hi - mean) * scale + mean
            scales[k] = float(rng.uniform(lo_n, hi_n))
        return scales

    # ---- specification assembly ---------------------------------------------

    def _spec_from_scales(self, scales: dict[str, float], picks: dict[str, str] | None = None):
        """Baseline specification with the length scales and actuator picks applied.

        Built with :func:`dataclasses.replace` on the baseline rather than
        through ``DesignSpec.from_vector``, which is not the identity on
        ``DesignSpec.vector`` because the vector form carries only the free
        fields and drops every fixed-constant clearance back to its Python
        default.  Length scales are rounded to two decimals, preserving the
        quantisation the URDF-era generator applied to the box extents, so the
        effective search lattice and therefore the conversion cache hit rate
        are unchanged.
        """
        changes: dict[str, float | str] = {}
        for key, field in LENGTH_SCALE_FIELDS.items():
            if key in scales:
                changes[field] = round(getattr(self.baseline, field) * scales[key], 2)
        if picks:
            for group, name in picks.items():
                changes[self.family.spec_fields[group]] = name
        return replace(self.baseline, **changes)

    def _actuator_params(self, spec, scales: dict[str, float] | None = None) -> dict[str, dict]:
        """Simulator-side actuator parameters for every group of one design."""
        scales = scales or {}
        out: dict[str, dict] = {}
        for group, actuator in zip(self.family.groups, spec.actuators()):
            params = actuator.sim_params()
            for key, scale_key in ACTUATOR_SCALE_KEYS.items():
                if key in params and scale_key in scales:
                    params[key] *= scales[scale_key]
            for key in ("stiffness", "damping"):
                scale_key = f"{group}_{key}_scale"
                if key in params and scale_key in scales:
                    params[key] *= scales[scale_key]
            out[group] = {k: round(float(v), 6) for k, v in params.items()}
        return out

    # ---- emission and conversion --------------------------------------------

    def _build_xml(self, spec) -> str:
        """Emit *spec*, falling back to the last feasible specification.

        ``check_geometry`` raises when a sampled trunk width or actuator radius
        makes the placement chain degenerate.  A raised generation would abort
        training, so an infeasible individual silently repeats the last design
        that did emit, which costs one individual's diversity rather than the
        run.
        """
        try:
            xml = self.family.build_xml(spec)
        except ValueError as exc:
            print(f"[design] infeasible specification, reusing the last feasible one: {exc}")
            return self.family.build_xml(self._last_feasible)
        self._last_feasible = spec
        return xml

    def _generate_individual_mjcf(self, generation: int, idx: int, spec) -> str:
        gen_dir = os.path.join(self.output_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        # Exactly one dot in the basename: MjcfConverter._convert_asset splits
        # the basename on "." and unpacks the result into two names, so a second
        # dot raises ValueError there (mjcf_converter.py:61).
        xml_path = os.path.join(gen_dir, f"individual_{idx:04d}.xml")
        with open(xml_path, "w") as fh:
            fh.write(self._build_xml(spec))
        return xml_path

    def _generate_individual_usd(self, xml_path: str, idx: int) -> str:
        # Imported here rather than at module scope so that the generators can
        # be exercised without a running simulator.
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        usd_out_dir = os.path.join(os.path.dirname(xml_path), f"individual_{idx:04d}_usd")
        cfg = MjcfConverterCfg(
            asset_path=xml_path,
            usd_dir=usd_out_dir,
            usd_file_name=f"{self.family.name}_{idx}.usd",
            link_density=0.0,
            import_inertia_tensor=True,
            fix_base=False,
            self_collision=False,
            make_instanceable=True,
            force_usd_conversion=True,
        )
        return MjcfConverter(cfg).usd_path


# ---------------------------------------------------------------------------
# Random design generator
# ---------------------------------------------------------------------------


class RandomDesignGenerator(DesignGeneratorBase):
    """Generates robot design populations by randomly perturbing the baseline.

    Each individual is produced by sampling scalar scale factors from
    ``param_ranges``, applying the length scales to the baseline
    specification, emitting the document, converting it, and recording the
    actuator parameters that the asset cannot carry.
    """

    def generate_population(self, generation: int) -> Population | None:
        return self._build_population(generation, range(self.num_individuals))

    def _generate_individual(self, generation: int, idx: int):
        rng = np.random.default_rng(seed=generation * 10000 + idx)
        scales = self._sample_scales(rng)
        self._current_scales = scales
        spec = self._spec_from_scales(scales)
        return spec, self._actuator_params(spec, scales)


# ---------------------------------------------------------------------------
# CMA-ES design generator
# ---------------------------------------------------------------------------


class CMAESDesignGenerator(DesignGeneratorBase):
    """Generates robot designs by sampling from a CMA-ES search distribution.

    The design vector is the unit-hypercube encoding of the scale-factor
    dictionary used by :class:`RandomDesignGenerator`; CMA-ES adapts the
    multivariate Gaussian search distribution from the per-individual mean
    episode return reported by :class:`CoptOnPolicyRunner`.
    """

    def __init__(
        self,
        family: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
        sigma0: float = 0.2,
        seed: int = 0,
        es_state_path: str | None = None,
        late_start: bool = False,
        late_start_it: int = 8000,
        late_start_prior_pop_size: int = 4096,
        max_cma_iter: int = 10000,
        variable_params_path: str | None = None,
    ) -> None:
        super().__init__(
            family=family,
            num_individuals=num_individuals,
            param_ranges=param_ranges,
            output_dir=output_dir,
            variable_params_path=variable_params_path,
        )
        self.late_start_prior_num_individual = late_start_prior_pop_size
        # Restrict CMA-ES to only the length scales this generator drives.
        # Caller-supplied param_ranges may tune the bounds of these two keys
        # but cannot add new dimensions (any extras would be inert).
        self.param_ranges = {**CMAES_PARAM_RANGES}
        if param_ranges is not None:
            for key, rng in param_ranges.items():
                if key in self.param_ranges:
                    self.param_ranges[key] = rng

        self.param_keys: list[str] = list(self.param_ranges.keys())
        self.num_params: int = len(self.param_keys)
        self._sigma0: float = float(sigma0)
        self._seed: int = int(seed)
        self.late_start = late_start
        self.late_start_it = late_start_it
        self._max_cma_iter = max_cma_iter

        if es_state_path is not None:
            with open(es_state_path, "rb") as fh:
                self._es = pickle.loads(fh.read())
            self._check_dimension()
        else:
            self._es = self._make_strategy()

        self._pending_solutions: list | None = None
        self._last_solutions: list | None = None
        self._terminated = False

    # --- strategy ------------------------------------------------------

    def _make_strategy(self):
        opts = cma.CMAOptions()
        opts.set({
            "popsize": int(self.num_individuals),
            "bounds": [np.zeros(self.num_params), np.ones(self.num_params)],
            "seed": self._seed,
            "verb_disp": 1,
            "verb_filenameprefix": str(Path(self.output_dir) / "cma_log") + "/",
            "maxiter": self._max_cma_iter,
            "BoundaryHandler": "BoundTransform",
        })
        return cma.CMAEvolutionStrategy(
            np.full(self.num_params, 0.5), self._sigma0, opts
        )

    def _check_dimension(self) -> None:
        assert self._es.N == self.num_params, (
            f"Checkpoint dimension mismatch: "
            f"pickled {self._es.N}, expected {self.num_params}"
        )

    def _dumps(self) -> bytes:
        return self._es.pickle_dumps()

    # --- Public API ---------------------------------------------------

    def toggle_late_start(self) -> None:
        self.late_start = not self.late_start

    def sample_batch(self) -> None:
        assert self._pending_solutions is None, "sample_batch called twice consecutively"
        self._pending_solutions = self._es.ask()

    def update_with_fitness(self, fitness: list[float]) -> None:
        assert self._last_solutions is not None, (
            "update_with_fitness called before generate_population"
        )
        if not self._terminated:
            print("Updating Designs with PPO Training Roll")
            costs = [self._sanitise_cost(-float(f)) for f in fitness]
            self._es.tell(self._last_solutions, costs)
            if self._es.stop():
                print(f"[CMA-ES] terminated: {self._es.stop()}")
                self._terminated = True
            self.sample_batch()

    def generate_population(self, generation: int) -> Population | None:  # type: ignore[override]
        if self._terminated:
            print("CMAES Terminated. Fine-tuning parent policy on selected design only")
            self._pending_solutions = self._last_solutions
            return None
        print("Generating New Designs using CMAES output")
        if self.late_start:
            print("late start enabled sampling random designs")
            pop = self._build_population(generation, range(self.late_start_prior_num_individual))
        else:
            pop = self._build_population(generation, range(self.num_individuals))
        self._last_solutions = self._pending_solutions
        self._pending_solutions = None
        return pop

    # --- Internal helpers --------------------------------------------

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=generation * 10000 + idx)
            scales = self._sample_scales(rng)
        else:
            assert self._pending_solutions is not None, (
                "_generate_individual called before sample_batch"
            )
            scales = self._denormalise(self._pending_solutions[idx])
        self._current_scales = scales
        spec = self._spec_from_scales(scales)
        return spec, self._actuator_params(spec)

    def _denormalise(self, x) -> dict[str, float]:
        scales: dict[str, float] = {}
        for i, key in enumerate(self.param_keys):
            lo, hi = self.param_ranges[key]
            xi = float(np.clip(np.asarray(x)[i], 0.0, 1.0))
            scales[key] = float(lo + xi * (hi - lo))
        return scales

    @staticmethod
    def _sanitise_cost(c: float, fallback: float = 1e6) -> float:
        if not np.isfinite(c):
            return fallback
        return c

    def save_state(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self._dumps())

    def get_state(self) -> dict:
        """Return a picklable snapshot of the full search state.

        Widens the strategy-object-only :meth:`save_state` into the complete
        bookkeeping required for an identical resume, namely the pending and
        last ``ask`` solutions that pair genotypes with their fitness, the
        termination flag, and the late-start flag.  The two subclasses inherit
        this unchanged.
        """
        return {
            "es": self._dumps(),
            "pending_solutions": self._pending_solutions,
            "last_solutions": self._last_solutions,
            "terminated": self._terminated,
            "late_start": self.late_start,
        }

    def load_state(self, state: dict) -> None:
        """Restore the search state produced by :meth:`get_state`."""
        self._es = pickle.loads(state["es"])
        self._check_dimension()
        self._pending_solutions = state["pending_solutions"]
        self._last_solutions = state["last_solutions"]
        self._terminated = state["terminated"]
        self.late_start = state["late_start"]


class GrowingDesignDistCMAESDesignGenerator(CMAESDesignGenerator):
    """CMA-ES whose late-start prior widens from a point to the full box."""

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=generation * 10000 + idx)
            # Saturation point of the design distribution's growth
            s = 0.1
            # Total number of generations of random population sampling
            n = self.late_start_it
            scale = 0.95 * (generation / (n - s)) + 0.05
            scales = self._sample_scales_v2(rng, scale)
        else:
            assert self._pending_solutions is not None, (
                "_generate_individual called before sample_batch"
            )
            scales = self._denormalise(self._pending_solutions[idx])
        self._current_scales = scales
        spec = self._spec_from_scales(scales)
        return spec, self._actuator_params(spec)


# ---------------------------------------------------------------------------
# Mixed continuous and categorical CMA-ES design generator
# ---------------------------------------------------------------------------


class CatCMAESDesignGenerator(CMAESDesignGenerator):
    """CMA-ES with Margin over the link lengths *and* the actuator picks.

    The continuous part is the same unit-hypercube encoding of the length
    scales that :class:`CMAESDesignGenerator` searches.  Alongside it sits one
    categorical variable per actuator group, whose categories are the entries
    of that family's catalog, so a genotype names both how long the leg is and
    which unit drives each joint.  The two are optimised jointly by
    ``cmaes.CatCMAwM``, which carries a multivariate Gaussian over the
    continuous block and a categorical distribution per discrete block and
    applies the margin correction of Hamano et al. [1] so the discrete
    marginals cannot collapse before the continuous block has converged.

    Args:
        actuator_groups: Which groups the search is allowed to vary.  Defaults
            to every group of the family.  A group left out keeps whatever the
            baseline specification picks for it.
        catalog_names: Optional explicit shortlist of catalog entry names, in
            preference to the family's whole catalog.  Restricting the shelf is
            the cheapest way to keep the categorical block small.
    """

    def __init__(
        self,
        family: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
        sigma0: float = 0.2,
        seed: int = 0,
        es_state_path: str | None = None,
        late_start: bool = False,
        late_start_it: int = 8000,
        late_start_prior_pop_size: int = 4096,
        max_cma_iter: int = 10000,
        variable_params_path: str | None = None,
        actuator_groups: tuple[str, ...] | None = None,
        catalog_names: list[str] | None = None,
    ) -> None:
        # Resolved before super().__init__ so that _make_strategy, which it
        # calls, already sees the categorical block.
        self._pending_groups = actuator_groups
        self._pending_names = catalog_names
        super().__init__(
            family=family,
            num_individuals=num_individuals,
            param_ranges=param_ranges,
            output_dir=output_dir,
            sigma0=sigma0,
            seed=seed,
            es_state_path=es_state_path,
            late_start=late_start,
            late_start_it=late_start_it,
            late_start_prior_pop_size=late_start_prior_pop_size,
            max_cma_iter=max_cma_iter,
            variable_params_path=variable_params_path,
        )

    # --- strategy ------------------------------------------------------

    def _resolve_categories(self) -> None:
        if getattr(self, "cat_names", None) is not None:
            return
        groups = self._pending_groups or self.family.groups
        unknown = set(groups) - set(self.family.groups)
        if unknown:
            raise ValueError(
                f"Unknown actuator group(s) {sorted(unknown)} for family "
                f"{self.family.name!r}. Known groups: {list(self.family.groups)}."
            )
        names = (self._pending_names if self._pending_names is not None
                 else [a.name for a in self.family.catalog()])
        by_name = {a.name: a for a in self.family.catalog()}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(
                f"Actuator name(s) {missing} are not in the {self.family.name} catalog. "
                f"Available: {sorted(by_name)}."
            )
        if len(names) < 2:
            raise ValueError(
                "A categorical search needs at least two catalog entries per group; "
                f"got {names}."
            )
        self.cat_groups: tuple[str, ...] = tuple(groups)
        self.cat_names: list[str] = list(names)
        # A catalog entry that states no rotor inertia and no armature yields an
        # armature of zero, which apply_actuator_params would then write over
        # the identified reflected inertia. Name them once at construction
        # rather than let the run discover it in the gait.
        bare = [n for n in self.cat_names
                if getattr(by_name[n], "armature", 0.0) == 0.0]
        if bare:
            print(
                "[design] warning: the following catalog entries state no rotor "
                "inertia or armature, so a design that picks one runs with zero "
                f"reflected inertia at that joint: {bare}. Pass catalog_names to "
                "restrict the shelf to identified entries."
            )

    def _make_strategy(self):
        from cmaes import CatCMAwM

        self._resolve_categories()
        n_cat = len(self.cat_names)
        # The continuous block keeps the unit-hypercube convention of the
        # parent class, so _denormalise is reused unchanged.
        x_space = [[0.0, 1.0] for _ in range(self.num_params)]
        c_space = [n_cat for _ in self.cat_groups]
        # Start every categorical marginal uniform except for the baseline's
        # own pick, which is given the plurality so that generation zero is
        # centred on the donor robot rather than on an arbitrary shelf entry.
        cat_param = np.full((len(self.cat_groups), n_cat), 1.0 / n_cat)
        for row, group in enumerate(self.cat_groups):
            pick = getattr(self.baseline, self.family.spec_fields[group])
            if pick in self.cat_names:
                cat_param[row, :] = 0.5 / (n_cat - 1)
                cat_param[row, self.cat_names.index(pick)] = 0.5
        return CatCMAwM(
            x_space=x_space,
            c_space=c_space,
            cat_param=cat_param,
            population_size=int(self.num_individuals),
            mean=np.full(self.num_params, 0.5),
            sigma=self._sigma0,
            seed=self._seed,
        )

    def _check_dimension(self) -> None:
        self._resolve_categories()
        got = int(self._es.population_size)
        assert got == int(self.num_individuals), (
            f"Checkpoint population size mismatch: pickled {got}, "
            f"expected {self.num_individuals}"
        )

    def _dumps(self) -> bytes:
        # CatCMAwM carries no pickle_dumps of its own, unlike cma's strategy.
        return pickle.dumps(self._es)

    # --- ask and tell ---------------------------------------------------

    def sample_batch(self) -> None:
        assert self._pending_solutions is None, "sample_batch called twice consecutively"
        self._pending_solutions = [self._es.ask() for _ in range(self._es.population_size)]

    def update_with_fitness(self, fitness: list[float]) -> None:
        assert self._last_solutions is not None, (
            "update_with_fitness called before generate_population"
        )
        if self._terminated:
            return
        print("Updating Designs with PPO Training Roll")
        costs = [self._sanitise_cost(-float(f)) for f in fitness]
        self._es.tell(list(zip(self._last_solutions, costs)))
        self.sample_batch()

    # --- decoding -------------------------------------------------------

    @staticmethod
    def _category_indices(c) -> list[int]:
        """Category index per variable, from either encoding CatCMAwM emits.

        ``Solution.c`` is documented as a one-hot matrix of shape
        ``(n_variables, n_categories)``; a one-dimensional array of indices is
        accepted too so that a future release which drops the encoding does not
        break the decode.
        """
        arr = np.asarray(c)
        if arr.ndim == 2:
            return [int(i) for i in arr.argmax(axis=1)]
        return [int(round(float(v))) for v in arr.reshape(-1)]

    def _picks_from_solution(self, solution) -> dict[str, str]:
        indices = self._category_indices(solution.c)
        n = len(self.cat_names)
        return {
            group: self.cat_names[int(np.clip(indices[row], 0, n - 1))]
            for row, group in enumerate(self.cat_groups)
        }

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=generation * 10000 + idx)
            scales = self._sample_scales_v2(
                rng, 0.95 * (generation / (self.late_start_it - 0.1)) + 0.05)
            picks = {group: self.cat_names[int(rng.integers(len(self.cat_names)))]
                     for group in self.cat_groups}
        else:
            assert self._pending_solutions is not None, (
                "_generate_individual called before sample_batch"
            )
            solution = self._pending_solutions[idx]
            scales = self._denormalise(solution.x)
            picks = self._picks_from_solution(solution)
        self._current_scales = scales
        spec = self._spec_from_scales(scales, picks)
        return spec, self._actuator_params(spec)
```

## 11. `co_optimisation/co_optimisation/utils/update.py`

The file is rewritten in place, revision 1's separate `update_mjcf.py` being withdrawn. Its structure is the incumbent's, and every one of its three authoring steps is the incumbent's, resolved per attribute through the property stack exactly as before. What changes is what §4.6.9 tabulated, plus the failure mode.

The link path gains a segment and the joints scope loses one, so both are built from a `root_body` the population carries rather than from string surgery on the link path. The box scale is written as a half-extent. The geometry child is discovered by structure rather than named, because an unnamed MuJoCo geom's prim name depends on how many geoms precede it in the document. The mass, the inertia and the centre of mass are written from numbers the generator computed rather than from a density recovered by dividing. The child joint's anchor is written as a three-vector rather than as a z-displacement, which is what lets the biped's thigh grow along a bend of 145 degrees.

The failure mode inverts. The incumbent returns early when it does not recognise the layout (`update.py:84-88`), so a mismatch costs a whole schedule of training on generation zero while the logs report new populations. The new module collects every prim it could not reach and raises, naming them. `assert_link_lengths_applied` goes further and reads the realised body separations back out of the simulation after the reset, comparing each against the separation the population asked for, which is the only check that can distinguish an edit that was authored from an edit that took effect.

```python
"""In-place primitive-geometry updates for *instanced* articulation links.

The Isaac Sim MuJoCo importer authors each body's geometry as an instanceable
internal reference: ``<robot>/<root>/<body>/{visuals,collisions}`` are instance
prims whose box lives under ``/visuals/<body>/<geom>`` and
``/collisions/<body>/<geom>`` inside the per-individual Universal Scene
Description layer, each an ``Xform`` carrying the box's ``xformOp:scale`` and
``xformOp:translate`` and referencing a unit ``UsdGeomCube`` in ``/meshes``
(MjcfImporter.cpp:1236-1266, MjcfUsd.cpp:820-826).  Instanced geometry cannot
be overridden through an environment prim, so the box is set on the
prototype's own source ``Xform`` via the ``Sdf`` layer API.

Three conventions of that layout differ from the URDF importer's and are the
reason this module cannot be shared with it.  The scale is the MuJoCo
half-extent rather than the full edge length; the geometry ``Xform`` under the
prototype is named after the MuJoCo geom, which is ``_geom_<N>`` when the geom
is unnamed, rather than the URDF importer's positional ``mesh_<i>``; and every
body prim sits one level deeper, under the root body's own ``Xform``, while
the joints scope stays directly beneath the referenced asset prim.  The first
is handled by writing half-extents, the second by discovering the single box
child rather than naming it, and the third by taking the root body's name from
the population.  See plans/design_generator_integration.md section 4.6.

Mass, inertia and centre of mass (the body prim) and the child-joint anchor
(the joints scope) are NOT instanced; they reach each environment by the
*reference* to the design's Universal Scene Description, and they are
generally authored in a DIFFERENT layer than the geometry, the importer
emitting base, physics, robot and sensor into separate sub-stages that the
root layer then carries as variant-selected payloads
(PluginInterface.cpp:395-460).  Each quantity is therefore resolved from its
own authoring layer via the corresponding prim's property stack, never assumed
to share the geometry layer.  Editing one design's prototype and source
updates all of its instances on the next ``sim.reset()``.

Every number written here is computed by the design generator from the same
specification that emitted the document, so this module performs no geometry
inference of its own: it neither recovers a density nor assumes an axis along
which a link grows.  See ``co_optimisation/runners/usd_generator.py``'s
``_entry``.

The simulation must be stopped before calling these helpers; a single timed
``sim.reset()`` reactivates physics after all per-design edits are authored.
"""

from __future__ import annotations

import time

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage


def _source_spec(attr):
    """(layer, Sdf.Path) of the strongest authored opinion for ``attr`` (or None)."""
    if not attr:
        return None, None
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    return (stack[0].layer, stack[0].path) if stack else (None, None)


def _box_xform(prototype) -> Usd.Prim | None:
    """The one ``Xform`` under *prototype* that positions and scales a box.

    A scalable link carries exactly one collision box and one visual box, and
    the two land in different prototypes, so the search is unambiguous.  The
    prim is identified by structure rather than by name because an unnamed
    MuJoCo geom is named ``_geom_<N>`` by the importer, with ``N`` a
    document-wide running counter (MjcfParser.cpp:180) that shifts whenever the
    emitted document gains or loses a geom.
    """
    for child in prototype.GetChildren():
        if not child.GetAttribute("xformOp:scale"):
            continue
        for descendant in Usd.PrimRange(child):
            if descendant.IsA(UsdGeom.Cube):
                return child
    return None


def update_articulation_links(
    prototype_path: str,
    link_prim_path: str,
    joints_prim_path: str,
    params: dict,
    stage=None,
) -> bool:
    """Author ONE geometry prototype and its link's physics on their source layers.

    Sets the prototype box ``Xform``'s ``xformOp:scale`` and
    ``xformOp:translate`` and -- idempotently -- the owning link's mass,
    inertia and centre of mass and the child joint's ``physics:localPos0``.
    Each quantity is authored on whichever layer actually defines it, resolved
    per attribute, since geometry and physics live in separate layers.  Editing
    one prototype and source updates all of its instances on the next
    ``sim.reset()``.  Must run while the simulation is stopped.

    Args:
        prototype_path: Path of the implicit geometry prototype
            (``/__Prototype_<N>``) backing one ``<body>/{visuals,collisions}``
            instance.
        link_prim_path: Composed (non-instanced) body prim of a representative
            environment, e.g.
            ``"/World/envs/env_0/Robot/base_Link/thigh_R_Link"``.
        joints_prim_path: The design's joints scope in the same environment,
            e.g. ``"/World/envs/env_0/Robot/joints"``.
        params: One entry of a population's link-length dict, carrying
            ``half_extents``, ``box_pos``, ``mass``, ``inertia``,
            ``child_joint`` and ``child_pos``.
        stage: Optional explicit USD stage; defaults to the current stage.

    Returns:
        True when the box was found and written, False when the prototype
        carries no box, which :func:`apply_link_length_params` turns into a
        raised error rather than a silent skip.
    """
    stage = stage or get_current_stage()

    hx, hy, hz = (float(v) for v in params["half_extents"])
    px, py, pz = (float(v) for v in params["box_pos"])
    mass = float(params["mass"])
    ixx, iyy, izz = (float(v) for v in params["inertia"])
    cx, cy, cz = (float(v) for v in params["child_pos"])

    # --- geometry source (instanced): via the prototype's own box Xform ---
    box = _box_xform(stage.GetPrimAtPath(prototype_path))
    if not box:
        return False
    g_layer, g_scale_path = _source_spec(box.GetAttribute("xformOp:scale"))
    if g_layer is None or g_scale_path is None:
        return False
    g_trans_path = g_scale_path.GetPrimPath().AppendProperty("xformOp:translate")

    # --- physics sources (reference-shared, NOT instanced): each on its OWN layer ---
    link_prim = stage.GetPrimAtPath(link_prim_path)
    joint_prim = stage.GetPrimAtPath(f"{joints_prim_path}/{params['child_joint']}")
    m_layer, m_path = _source_spec(link_prim.GetAttribute("physics:mass"))
    i_layer, i_path = _source_spec(link_prim.GetAttribute("physics:diagonalInertia"))
    c_layer, c_path = _source_spec(link_prim.GetAttribute("physics:centerOfMass"))
    j_layer, j_path = _source_spec(joint_prim.GetAttribute("physics:localPos0"))

    with Sdf.ChangeBlock():
        # (1) geometry -- prototype source; the only legal resize of instanced
        #     geometry. The scale is the MuJoCo half-extent, the importer having
        #     left the cube at its default size of 2 and put the half-extent on
        #     the referencing Xform (MjcfUsd.cpp:820-826, MjcfImporter.cpp:399).
        g_layer.GetAttributeAtPath(g_scale_path).default = Gf.Vec3d(hx, hy, hz)
        if g_layer.GetAttributeAtPath(g_trans_path) is not None:
            g_layer.GetAttributeAtPath(g_trans_path).default = Gf.Vec3d(px, py, pz)
        # (2) mass, inertia and centre of mass -- link prim, reference-shared
        if m_layer is not None and m_path is not None:
            m_layer.GetAttributeAtPath(m_path).default = mass
        if i_layer is not None and i_path is not None:
            i_layer.GetAttributeAtPath(i_path).default = Gf.Vec3f(ixx, iyy, izz)
        if c_layer is not None and c_path is not None:
            c_layer.GetAttributeAtPath(c_path).default = Gf.Vec3f(px, py, pz)
        # (3) child-joint anchor -- joints scope, reference-shared. localPos0 is
        #     the child body's own origin expressed in this link's frame, the
        #     importer having taken the joint's own position to be the child
        #     body origin (MjcfImporter.cpp:1367-1376), so relocating it is what
        #     lengthens the limb.
        if j_layer is not None and j_path is not None:
            j_layer.GetAttributeAtPath(j_path).default = Gf.Vec3f(cx, cy, cz)
    return True


def apply_link_length_params(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict]],
    root_body: str,
    strict: bool = True,
) -> None:
    """Author absolute link geometry on every design's source layer, then a single
    timed ``sim.reset()``.

    Iterating the first ``num_individuals`` environments visits each design
    exactly once (round-robin assignment, identical to the multi-asset
    spawner), and editing a design's geometry prototype updates all of its
    instances.  The simulation must be stopped before this function is called;
    a safety-net stop is included.

    Args:
        env: The unwrapped ``ManagerBasedRLEnv`` instance.
        link_length_params_list: Per-individual link-geometry dicts, one per
            individual, each mapping a body name to the entry documented in
            :func:`update_articulation_links`.
        root_body: The emitted document's root body name, which the Isaac Sim
            MuJoCo importer interposes between the referenced asset prim and
            every body prim.  Supplied by ``Population.get_root_body``.
        strict: Raise when a design's link could not be reached, rather than
            leaving it silently at the previous generation's geometry.
            Defaults to True.  The URDF-era editor returned early instead, so a
            layout it did not recognise cost a whole schedule of training on
            generation zero while the logs reported new populations.
    """
    sim, scene = env.sim, env.scene
    num_individuals = len(link_length_params_list)
    env_paths = scene.env_prim_paths

    if sim.is_playing():  # safety net
        sim._disable_app_control_on_stop_handle = True
        sim.stop()
        sim._disable_app_control_on_stop_handle = False

    stage = get_current_stage()
    missed: list[str] = []
    for individual in range(num_individuals):  # first N envs = each design once
        env_path = env_paths[individual]
        joints_prim_path = f"{env_path}/Robot/joints"
        for link_name, params in link_length_params_list[individual].items():
            link_prim_path = f"{env_path}/Robot/{root_body}/{link_name}"
            for purpose in ("visuals", "collisions"):
                inst = stage.GetPrimAtPath(f"{link_prim_path}/{purpose}")
                proto = inst.GetPrototype() if inst else None
                if not proto:
                    missed.append(f"{link_prim_path}/{purpose} (no prototype)")
                    continue
                if not update_articulation_links(
                    proto.GetPath().pathString, link_prim_path,
                    joints_prim_path, params, stage
                ):
                    missed.append(f"{link_prim_path}/{purpose} (no box under prototype)")

    if missed and strict:
        raise RuntimeError(
            "In-place link update reached none of the following "
            f"{len(missed)} prim(s), so their designs would have stayed at the "
            "previous generation's geometry:\n  " + "\n  ".join(missed[:20])
            + ("\n  ..." if len(missed) > 20 else "")
        )

    print("Reactivating Physics (in-place prototype update)")
    start = time.perf_counter()
    sim.reset()
    print(
        f"Total time taken for sim.reset(): {time.perf_counter() - start:.4f} seconds"
    )


def assert_link_lengths_applied(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict]],
    tol: float = 1e-3,
) -> None:
    """Read the realised body separations back out of the simulation.

    Called after the reset that follows :func:`apply_link_length_params`, this
    compares each scalable link's realised distance to its child body against
    the distance the population asked for.  It is the only check that can tell
    an edit that was authored from an edit that took effect, the authoring API
    reporting success whether or not physics ever reads the attribute.

    Raises:
        RuntimeError: naming every link whose realised separation differs from
            the requested one by more than *tol* metres.
    """
    import torch

    robot = env.scene.articulations["robot"]
    names = robot.body_names
    positions = robot.data.body_link_pos_w
    num_individuals = len(link_length_params_list)
    bad: list[str] = []
    for individual in range(num_individuals):
        for link_name, params in link_length_params_list[individual].items():
            child = params.get("child_body")
            if child is None or link_name not in names or child not in names:
                continue
            want = float(torch.tensor(params["child_pos"]).norm())
            got = float(
                (positions[individual, names.index(child)]
                 - positions[individual, names.index(link_name)]).norm()
            )
            if abs(got - want) > tol:
                bad.append(
                    f"individual {individual} {link_name} -> {child}: "
                    f"realised {got:.6f} m, requested {want:.6f} m"
                )
    if bad:
        raise RuntimeError(
            "The in-place link update was authored but did not reach the "
            "simulation for:\n  " + "\n  ".join(bad)
        )
```

## 12. `co_optimisation/co_optimisation/utils/respawn.py`

`class respawn_robots` is untouched. `apply_actuator_params` is replaced in full, absorbing what revision 1 had proposed as a separate `patch_dc_motor_envelope` and repairing the three gaps of §4.7 in one place.

```python
# ---------------------------------------------------------------------------
# Replaces respawn.apply_actuator_params in full. The rest of respawn.py,
# including class respawn_robots, is untouched.
# ---------------------------------------------------------------------------

# Attributes of the DC-motor envelope that IdentifiedActuator keeps as Python
# scalars rather than per-environment tensors, so a plain tensor write skips
# them. DCMotor.__init__ reads cfg.saturation_effort into a float
# (actuator_pd.py:268) and derives the corner velocity from it once
# (actuator_pd.py:270), after which _clip_effort uses both every step
# (actuator_pd.py:297-303).
_ENVELOPE_SCALARS = ("saturation_effort",)

# Attributes the actuator model never reads, which reach the simulation only
# through an explicit write. compute() uses stiffness, damping, effort_limit,
# velocity_limit, saturation_effort and the friction triple, but armature is a
# PhysX joint property that Articulation._process_actuators_cfg pushes once at
# construction (articulation.py:1774) and nothing pushes again.
_SIM_WRITE_THROUGH = {"armature": "write_joint_armature_to_sim"}


def apply_actuator_params(
    env: ManagerBasedRLEnv,
    actuator_params_list: list[dict[str, dict]],
) -> None:
    """Patch ``IdentifiedActuator`` attributes per environment.

    Each environment is assigned an individual round-robin; its actuator
    tensor rows are overwritten with the corresponding design's parameter
    values.

    Three kinds of parameter are handled, because three reach the simulation by
    different routes.  A per-environment tensor attribute is written row by
    row, as before.  A scalar of the DC-motor envelope is first promoted to a
    tensor of the same shape, so that it can carry one value per environment at
    all, and the corner velocity that ``DCMotor.__init__`` derived from it once
    is then recomputed, without which the envelope would clip at the previous
    generation's numbers.  An attribute the actuator model never reads is
    pushed into the simulation explicitly.

    Args:
        env: The unwrapped ``ManagerBasedRLEnv``.
        actuator_params_list: List of actuator-param dicts, one per individual.
            Each dict maps actuator group name to scalar overrides, keyed as
            ``design_generator.common.actuator_catalog.SIM_PARAM_KEYS`` names
            them.
    """
    num_envs: int = env.num_envs
    num_individuals = len(actuator_params_list)
    articulation = env.scene.articulations["robot"]

    for group_name, actuator in articulation.actuators.items():
        touched: set[str] = set()
        for env_idx in range(num_envs):
            individual_idx = env_idx % num_individuals
            overrides = actuator_params_list[individual_idx].get(group_name, {})

            for attr_name, value in overrides.items():
                tensor_attr = getattr(actuator, attr_name, None)
                if isinstance(tensor_attr, torch.Tensor):
                    # tensor_attr shape: (num_envs, num_joints_in_group)
                    tensor_attr[env_idx, :] = value
                    touched.add(attr_name)
                elif attr_name in _ENVELOPE_SCALARS:
                    _promote_envelope_scalar(actuator, attr_name)[env_idx, :] = value
                    touched.add(attr_name)

        if not touched:
            continue
        # The corner velocity is a cached product of three quantities any of
        # which may have just moved, so it is recomputed whenever the group was
        # touched at all rather than only when saturation_effort was.
        if hasattr(actuator, "_vel_at_effort_lim") and hasattr(actuator, "_saturation_effort"):
            actuator._vel_at_effort_lim = actuator.velocity_limit * (
                1.0 + actuator.effort_limit / actuator._saturation_effort
            )
        for attr_name, writer in _SIM_WRITE_THROUGH.items():
            if attr_name in touched:
                getattr(articulation, writer)(
                    getattr(actuator, attr_name), joint_ids=actuator.joint_indices
                )


def _promote_envelope_scalar(actuator, attr_name: str) -> torch.Tensor:
    """Return the envelope attribute as a per-environment tensor, promoting once.

    ``DCMotor`` stores the stall torque under a leading underscore and as a
    Python float, which makes it both invisible to a ``getattr`` on the public
    name and incapable of holding one value per design.  Promoting it to a
    tensor shaped like ``effort_limit`` leaves ``_clip_effort`` correct, its
    arithmetic being elementwise either way, and lets each environment carry
    its own individual's stall torque.
    """
    private = f"_{attr_name}"
    value = getattr(actuator, private)
    if not isinstance(value, torch.Tensor):
        value = torch.full_like(actuator.effort_limit, float(value))
        setattr(actuator, private, value)
    return value
```

Three kinds of parameter are now handled because three reach the simulation by different routes. A per-environment tensor attribute is written row by row, as before. A scalar of the DC-motor envelope is promoted once to a tensor of the same shape, so that it can carry one value per environment at all, after which the corner velocity that `DCMotor.__init__` derived from it is recomputed. An attribute the actuator model never reads is pushed into the simulation explicitly, which at present means the armature alone.

The promotion is safe. `_clip_effort` uses the stall torque only in elementwise arithmetic against tensors of the same shape (`actuator_pd.py:298-303`), so a tensor is as valid there as a float and carries the per-design values a float cannot. It is also idempotent, a second application finding a tensor and leaving it alone, which matters because `_update_morphology` calls this function on every generation.

This was exercised against a mock that mirrors the attribute kinds of `ActuatorBase`, `DCMotor` and `IdentifiedActuator` exactly. Two designs of differing envelope were applied over four environments, and the stall torque, the effort limit, the corner velocity, the gains and the pushed armature all came out per environment and correct, the corner velocity moving from a uniform 26.4329 to 16.2500 and 21.9900 by individual.

## 13. The Edits to the Remaining Files

### 13.1 `co_optimisation/co_optimisation/runners/__init__.py`

```python
from co_optimisation.runners.copt_on_policy_runner import CoptOnPolicyRunner
from co_optimisation.runners.usd_generator import (
    CatCMAESDesignGenerator,
    CMAESDesignGenerator,
    DesignGeneratorBase,
    GrowingDesignDistCMAESDesignGenerator,
    Population,
    RandomDesignGenerator,
)
```

`RandomPopulation` leaves the export list, `CMAESDesignGenerator` and `CatCMAESDesignGenerator` join it. No module outside `usd_generator.py` imports `RandomPopulation`, which was checked.

### 13.2 `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`

Two call sites gain the root body, which `apply_link_length_params` needs to address a link under the interposed root prim of §4.6.8.

```python
# in _apply_restored_state, at the call site :621-624
        apply_link_length_params(
            unwrapped_env,
            self.current_population.get_link_length_params(),
            self.current_population.get_root_body(),
        )

# in _update_morphology, at the call site :694-697
                apply_link_length_params(
                    unwrapped_env,
                    self.current_population.get_link_length_params(),
                    self.current_population.get_root_body(),
                )
```

Two further lines are added after each of the two `unwrapped_env.reset()` calls that follow an in-place edit, and after the reset inside `_apply_restored_state`, closing gate 9.

```python
                from co_optimisation.utils.update import assert_link_lengths_applied
                assert_link_lengths_applied(
                    unwrapped_env, self.current_population.get_link_length_params()
                )
```

Nothing else in the runner moves. `_reload_morphology`, `_compute_individual_fitness`, the checkpoint pair and the fitness accumulators are untouched.

### 13.3 `environments/environments/assets/config/solefoot_identified_cfg.py`

```python
import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from environments.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
# The MuJoCo document generated by scripts/tools/generate_default_mjcf.py from the
# parameter set of plans/design_generator_integration.md section 8.1, which
# reproduces the retired urdf/solefoot/tron1/base_robot.urdf to within four tenths
# of a micrometre at every body station and exactly in total mass.
mjcf_path = os.path.join(current_dir, "../mjcf/solefoot/tron1/base_robot.xml")
usd_path_pf = os.path.join(current_dir, "../usd/PF_TRON1A/PF_TRON1A.usd")
usd_path_wf = os.path.join(current_dir, "../usd/WF_TRON1A/WF_TRON1A.usd")

# ... TRON1_ABAD_ACTUATOR_CFG through TRON1_ANKLE_ACTUATOR_CFG unchanged ...
# ... rigid_props, articulation_props, activate_contact_sensors unchanged ...
# ... init_state, soft_joint_pos_limit_factor, actuators unchanged ...

SOLEFOOT_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=mjcf_path,
        fix_base=False,
        self_collision=True,
        import_inertia_tensor=True,
        link_density=0.0,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=activate_contact_sensors,
    ),
    # The Isaac Sim MuJoCo importer interposes the root body's own Xform between the
    # referenced asset prim and every link, and unconditionally applies a second
    # ArticulationRootAPI to a /worldBody prim it always creates
    # (MjcfImporter.cpp:580-597). Isaac Lab raises on the resulting ambiguity unless
    # the root is named outright (articulation.py:1533-1538).
    articulation_root_prim_path="/base_Link/base_Link",
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)
```

`usd_path`, `usd_path_sf`, `urdf_path`, the `UrdfConverter` import, the `spawn` object and `SOLEFOOT_IDENTIFIED_CFG_URDF` are all deleted, together with the commented-out multi-asset block at lines 132 to 165. `usd_path_pf` and `usd_path_wf` are retained, the pointfoot and wheelfoot tasks having no emitted asset.

### 13.4 `environments/environments/assets/config/quadruped_identified_cfg.py`

The same shape, with the quadruped's root body.

```python
mjcf_path = os.path.join(current_dir, "../mjcf/quadruped/quadruped.xml")

# ... the three actuator configurations and every property object unchanged ...

QUADRUPED_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=mjcf_path,
        fix_base=False,
        self_collision=True,
        import_inertia_tensor=True,
        link_density=0.0,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=activate_contact_sensors,
    ),
    articulation_root_prim_path="/trunk/trunk",
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)
```

The three actuator configurations' `joint_names_expr` move with §5.2's table, from `abad_.._Joint` to `.._hip_joint`, from `hip_.._Joint` to `.._thigh_joint` and from `knee_.._Joint` to `.._calf_joint`, and `init_state.joint_pos` moves with them.

### 13.5 `environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py`

The import loses one name and the twelve `SOLEFOOT_IDENTIFIED_CFG_URDF` call sites at lines 145, 893, 910, 927, 944, 961, 1029, 1047, 1064, 1081, 1098 and the remainder become `SOLEFOOT_IDENTIFIED_CFG`. The ten existing `SOLEFOOT_IDENTIFIED_CFG` call sites are already correct. Every `.replace(prim_path=...)` argument is unchanged, the prim path being the robot's own and not a link's.

### 13.6 `scripts/tools/generate_default_mjcf.py`

A new tool, thirty lines, that writes each family's default document beside the assets so that training needs neither the generator nor a conversion at import time.

```python
"""Emit each family's default MuJoCo document from its own parameter set.

Run after editing a family's variable_params.json or constant_params.json, or
after changing a catalog entry that its variable_params.json names. The output
is committed, so that environments/assets/config/*_identified_cfg.py can spawn
without importing the generator.

    python scripts/tools/generate_default_mjcf.py
"""

from __future__ import annotations

import pathlib

from design_generator import biped, quadruped

ASSETS = pathlib.Path(__file__).resolve().parents[2] / "environments/environments/assets/mjcf"

TARGETS = {
    "solefoot/tron1/base_robot.xml": biped,
    "quadruped/quadruped.xml": quadruped,
}


def main() -> None:
    for relative, family in TARGETS.items():
        spec = family.generate.load_design_spec()
        out = ASSETS / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(family.builder.build_xml(spec, for_isaaclab=True))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

### 13.7 `scripts/rsl_rl/train.py`

The COPT branch at lines 204 to 229 loses its URDF path and gains a family.

```python
        _num_individuals = 256
        ea_update_interval = 480
        ea_late_start = 12000
        param_ranges = {
            "thigh_length_scale": (0.75, 1.25),
            "shank_length_scale": (0.75, 1.25),
        }
        design_generator = GrowingDesignDistCMAESDesignGenerator(
            family="biped",
            num_individuals=_num_individuals,
            param_ranges=param_ranges,
            output_dir=os.path.join(log_dir, "copt_usds"),
            sigma0=0.25,
            seed=42,
            late_start=True,
            late_start_it=int(ea_late_start / ea_update_interval),
            late_start_prior_pop_size=int(env.num_envs / 4),
            max_cma_iter=(agent_cfg.max_iterations - ea_late_start) / ea_update_interval,
        )
```

The third experiment of the brief, which varies actuators and link lengths together, selects `CatCMAESDesignGenerator` with the same arguments plus a shortlist.

```python
        design_generator = CatCMAESDesignGenerator(
            family="biped",
            num_individuals=_num_individuals,
            param_ranges=param_ranges,
            output_dir=os.path.join(log_dir, "copt_usds"),
            sigma0=0.25,
            seed=42,
            late_start=False,
            max_cma_iter=agent_cfg.max_iterations / ea_update_interval,
            # Every entry the search may pick states its own rotor inertia and
            # gains, so no design runs with a fabricated actuator. See section 16.
            catalog_names=["TRON1-ABAD", "TRON1-HIP", "TRON1-KNEE", "TRON1-ANKLE"],
        )
```

### 13.8 `scripts/rsl_rl/play.py`

The COPT branch at lines 799 to 816 takes the same treatment, `base_urdf_path=_base_urdf` becoming `family="biped"` and `_base_urdf` being deleted. The robot profile table at line 190 loses its `("ankle_.*", None)` row, TRON1's feet now resolving under the row below it, `("foot_.._Link", None)`, which already covers the quadruped and now covers the biped as well, `foot_R_Link` and `foot_L_Link` matching `foot_.._Link`.

### 13.9 `scripts/analysis/dashboard.py`

`_TRON_FEET_NAMES` at line 97 becomes `("foot_R_Link", "foot_L_Link")` and the comment above it, which records that `play.py` resolved four feet against these two, is replaced by a dated note that the four body resolution was repaired. `_TRON_JOINT_NAMES` at lines 88 to 91 is unchanged, the biped's joints keeping their names.

### 13.10 The install step

`design_generator` must be importable as a top-level package, for the reason §4.9 gives. One line is added to the repository's setup instructions, beside the existing `pip install -e co_optimisation`.

```bash
pip install -e co_optimisation/co_optimisation/runners/design_generator
```

`cmaes` is also added to the environment, `CatCMAESDesignGenerator` importing `CatCMAwM` from it. It is a separate distribution from `cma`, which the existing generators use and which stays.

## 14. The Phased Programme

Each phase leaves the tree working and is separately revertible.

| Phase | Content | Gate |
| --- | --- | --- |
| 1 | The design generator changes of §9, with the parameter sets of §8 and the catalogue entries of §7. Nothing outside the submodule moves. | 1, 2 |
| 2 | `scripts/tools/generate_default_mjcf.py` and the two committed documents. | 2 |
| 3 | The two asset configurations of §13.3 and §13.4, the vendor asset retired, the URDF spawns deleted. | 3, 4 |
| 4 | The environment configuration renames of §5.2, both families, including the repair of §6.4. | 5, 6 |
| 5 | `usd_generator.py` rewritten, §10. `runners/__init__.py`, §13.1. | 7 |
| 6 | `respawn.apply_actuator_params`, §12. | 8 |
| 7 | `update.py` rewritten, §11, and the runner call sites of §13.2. | 9 |
| 8 | The scripts of §13.7 to §13.9, and a pre-training run on link-length-varied designs. | 10 |
| 9 | The two ablations of the brief, `GrowingDesignDist` against `CMAES`, then `CatCMAES`. | 10 |

Phases 3 and 4 must land together in one commit for each family, an asset whose names have changed being unusable by a configuration whose names have not.

## 15. The Verification Gates

| # | Gate | Status |
| --- | --- | --- |
| 1 | Both families emit for their own default parameter set, and the emitted document reproduces its donor to the tolerances of §8. | Closed here. Worst station error 3.952e-07 m and 0.0 m, total mass exact for both. |
| 2 | Every emitted body carries a mass and a diagonal inertia, so that no link depends on the discarded geom density of §4.4. | Closed here. Fifteen and twenty five bodies respectively, none massless. |
| 3 | `MjcfConverter` converts both documents without error, and the asset carries exactly one `ArticulationRootAPI` under the configured root. | Open, needs Isaac Sim. |
| 4 | The articulation initialises, and `body_names` and `joint_names` are the emitted names in the emitted order. | Open, needs Isaac Sim. |
| 5 | Every welded body survives the import as its own link, `createBodyForFixedJoint` having held at its default. The biped must show `abad_{S}_act`, `thigh_{S}_Link` and `ankle_{S}_act`. | Open, needs Isaac Sim. This is the gravest of the open gates. See §17. |
| 6 | Every `SceneEntityCfg` in both configurations resolves to the number of bodies or joints its term intends, checked by a resolution dump at startup. | Open, needs Isaac Sim. |
| 7 | All five generators produce a population for both families, and the in-place payload matches the emitted document term for term. | Closed here for the payload and the population. The conversion step ran against a stub. |
| 8 | `apply_actuator_params` writes per-environment stall torques, recomputes the corner velocity and pushes the armature. | Closed here against a mock of the Isaac Lab classes. |
| 9 | After an in-place update and its reset, the realised body separations equal the requested ones for every individual. | Open, needs Isaac Sim. `assert_link_lengths_applied` is the check. |
| 10 | A pre-training run reaches a reward trajectory comparable with the URDF-era run's over the first thousand iterations. | Open, needs a training run. |

Gates 1, 2, 7 and 8 were closed by running the code in this document. The derivation and comparison harnesses, the four generators under a stubbed converter and a stubbed `cmaes`, and the actuator mock all executed, and every number quoted in §7, §8 and §12 came out of them.

## 16. The Defects Left Standing

1. The quadruped's names stay asymmetric with the biped's, its root body being `trunk` and its joints carrying the limb first. Renaming inside `quadruped/builder.py` would fix it and would break the model predictive control package that pins those suffixes (`quadruped/builder.py:449-452`), so the asymmetry is kept deliberately. A later session that wants the symmetry should change the builder and the MPC package together.

2. The biped's left hip joint reverses sign. `base_robot.urdf` gives `hip_L_Joint` the axis `0 1 0` and `hip_R_Joint` the axis `0 -1 0`, with mirrored ranges, where the emitter gives both `0 -1 0` and one range (`biped/builder.py:373-396`). The emitted convention is the symmetric one, a positive command pitching both thighs the same way, and it is almost certainly what was wanted, but it is a behavioural change and no URDF-trained biped policy transfers across it.

3. The donor's knee ranges are asymmetric and the emitted ones are not. `knee_R_Joint` spans -1.21357 to 1.172665 and `knee_L_Joint` spans -0.872665 to 1.361357 under the same axis, which has the look of a transcription error, and the emitter uses the right leg's range for both. This is a repair, recorded so that a later reader does not mistake it for a regression.

4. The biped's centre of mass sits 7.3 mm lower than the donor's, for the three reasons §8.1 tabulates. Both causes are builder conventions rather than parameter choices. A later session that wants the donor's distribution exactly should make the link box flush at its proximal end and offset the actuator housing outboard by its own half-length, which are two changes to `_leg_xml` and would move every clearance in §8.1 with them.

5. The biped's thigh box matches the donor's length exactly and its shank box is 8.3 mm short, `LINK_BOX_LENGTH_FRAC` being one constant serving two links of different ratio. The shorter box is inside the donor's envelope, so it can only reduce contact, never add it.

6. `MjcfConverterCfg.make_instanceable`, `import_inertia_tensor` and `merge_fixed_joints` do not reach the importer, and `import_sites` is overridden by a literal, for the reasons §4.1 gives. This plan works around all four rather than patching the vendored converter, since a patch there would diverge from upstream Isaac Lab.

7. A catalogue entry that states no rotor inertia and no armature yields an armature of zero, which `apply_actuator_params` would write over the identified reflected inertia. Eleven of the biped's fifteen entries and eleven of the quadruped's fourteen are in that state. `CatCMAESDesignGenerator` prints the list at construction and `catalog_names` restricts the shelf, but nothing prevents a caller from searching over a bare entry.

8. The Isaac Sim MuJoCo importer authors a `/physicsScene` into every asset it produces, the Isaac Lab converter not disabling it. The consequence of referencing such an asset into a scene that already has a physics scene is untested here and is gate 3's business.

9. `spawn_multi_usd_file` is unchanged and still assumes each asset has a default prim. The MuJoCo importer sets one, forced at `PluginInterface.cpp:121`, so the assumption holds, and it is recorded here because it is an assumption rather than a check.

## 17. Risks and Contingencies

1. `createBodyForFixedJoint` defaults to True and the Isaac Lab converter does not expose it (`IMjcf.h:75`). If a future Isaac Sim were to flip that default, `CreatePhysicsBodyAndJoint` would drop every jointless body together with its entire subtree (`MjcfImporter.cpp:1170-1176`), and both families would lose most of each leg. There is no contingency inside MuJoCo, a body cannot be given a joint it does not have. The contingency is to pin the Isaac Sim version, as the URDF converter already pins its own extension, and gate 5 is what would catch it.

2. The `/worldBody` articulation root is resolved by configuration rather than by suppression, since §4.5.2 shows the prim is created unconditionally. If a future importer changed the prim's path, `articulation_root_prim_path` would still name the right root and nothing would break, the field being an assertion rather than a workaround.

3. The `cmaes` API is taken from its published documentation and source rather than from a local installation, the library not being present in this container. The constructor keywords, the `Solution` fields, the one-hot encoding of `c`, the `population_size` property and the minimising `tell` were all read from the upstream sources [2]. `_category_indices` accepts both a one-hot matrix and an index vector so that a change of encoding does not break the decode, and gate 7 exercises the path under a stub that matches the documented surface.

4. The two-decimal rounding of the length scales is retained from the URDF-era generator. It bounds the number of distinct documents and therefore of conversions, which matters at 256 individuals, and it keeps the search lattice as coarse as it was. A run that wants a finer lattice should change the rounding in `_spec_from_scales` and accept the conversion cost.

5. Emitting a document and converting it per individual is the same cost the URDF path paid, one conversion per distinct design per generation. The MuJoCo importer writes five layer files rather than the URDF importer's five, so the disk cost is comparable. This was not measured and gate 10 is where it would show.

6. No URDF-trained biped checkpoint transfers, for the three independent reasons §4.10, §16.2 and §6.4 give. The phased programme treats a fresh pre-training run as the starting point rather than a resume.

## 18. Bibliography

1. M. Hamano, S. Saito, M. Nomura, S. Shirakawa. CMA-ES with Margin, Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization. Genetic and Evolutionary Computation Conference (GECCO), 2022. arXiv:2205.13482.
2. CyberAgent AI Lab. cmaes, a Python library for CMA Evolution Strategy. Repository and documentation, read 2026-09-25, https://github.com/CyberAgentAILab/cmaes.
3. A. Singh, D. Kapa, S. Joshi, S. Kolathaya. COMPAct, Computational Optimization and Automated Modular Design of Planetary Actuators. IEEE International Conference on Robotics and Automation (ICRA), 2026. arXiv:2510.07197.

## 19. The Outcome, Recorded on Applying

Phases 1 to 8 of section 14 were applied in order on 2026-09-25. Every file section 13 names was changed, the three diffs of section 9 applied cleanly to the live tree, and each of the nineteen touched Python files compiles. A static check confirmed that all thirty names imported from the four rewritten modules resolve in them.

Four of the ten gates of section 15 were closed by measurement against the applied tree, and their numbers are exactly the ones sections 7, 8 and 12 predicted.

| Gate | Result |
| --- | --- |
| 1, both families reproduce their donor | Biped, worst station error 3.952e-07 m, worst link mass error 0.0 kg, total mass 21.328000 against 21.328000, centre of mass 7.299 mm low, inertia 4.907 per cent. Quadruped, worst station error 0.0 m, worst link mass error 0.0 kg, total mass 15.837077 against 15.837077, centre of mass exact, inertia 0.026 per cent. |
| 2, no link depends on the discarded geom density | Both families, both emission modes, every body carries an inertial and a positive mass, and the total is identical across modes at 21.328000 kg and 15.837077 kg. |
| 7, the generators and the in-place payload | All four generators produce a population for both families. The payload matches the emitted document in half extents, box centre, mass, diagonal inertia and child body position on every individual. The length-only searches stay two dimensional, the categorical search picks by name, a shortlist confines the shelf, an unknown family, group or actuator name is refused, the infeasible-specification fallback fires for both families, and the search state round trips through pickle for all three CMA-ES classes. |
| 8, the actuator envelope | Two designs of differing envelope over four environments. The stall torque is promoted to a tensor and carries per-environment values, the corner velocity moves from a uniform 26.4329 to 16.2500 and 21.9900 by individual, the armature is pushed into the simulation, and a second application is idempotent. |

Those checks now live in the repository at `co_optimisation/tests/design_generator_test.py`, forty three tests that pass without Isaac Sim, stubbing the converter and, when absent, `cma` and `cmaes`. They replace `co_optimisation/tests/cmaes_design_generator_test.py`, every test of which exercised the retired URDF mutation machinery.

Seven things diverge from what this document specified, all of them discovered while applying it.

1. `copt_on_policy_runner.py` imported `RandomPopulation` directly, which section 13.1 had not enumerated, and `_population_to_dict` and `_population_from_dict` serialised a population without its root body. Both are corrected, and `_population_from_dict` defaults a checkpoint that predates the field to `base_Link`, which is the only family the URDF-era pipeline ever ran.

2. A second TRON1 asset configuration existed that section 13 had not enumerated. `environments/assets/config/solefoot_cfg.py` carried a non-identified `SOLEFOOT_CFG` on the retired vendor asset and a `SOLEFOOT_CFG_URDF` on `urdf/solefoot/base_robot.urdf`, a path that has not existed in the tree for some time, so the two tasks that used it could not have spawned. `SOLEFOOT_CFG` is moved onto the emitted document and `SOLEFOOT_CFG_URDF` is deleted, its four call sites collapsing onto `SOLEFOOT_CFG`. Two task identifiers, `SFBaseEnvUrdfCfg` and `SFHIMBaseEnvUrdfCfg`, are therefore now duplicates of their parents rather than URDF variants of them, and are left registered rather than removed, a task registry change being wider than this plan.

3. `assets/config/__init__.py` exported both retired names and is rewritten.

4. The third experiment of the brief is selectable rather than a commented alternative. `scripts/rsl_rl/train.py` gains `--design_search`, taking `length` for the incumbent CMA-ES over the two link length scales and `cat` for the mixed continuous and categorical search, defaulting to `length` so the existing invocation is unchanged.

5. The emitted document formats every float to six decimal places where the in-place editor writes the generator's full-precision numbers, so a design reached by an in-place edit carries an inertia up to 5e-07 kg m squared different from the same design reached by a respawn. The difference is three orders of magnitude below the smallest link inertia either family carries and no correction is applied.

6. `FOOT_MASS` is placed beside `FOOT_RADIUS` in the quadruped's `constant_params.json` rather than appended, the two being one hardware fact about the same body.

7. The three comments in `quadruped_identified_cfg.py` that read "MuJoCo names this joint FR_hip_joint" were rewritten, the joint now carrying that name itself rather than a translation of it.

One item of section 6.2 is deliberately not done. `environments/assets/urdf/quadruped/gen_quadruped_urdf.py` is left where it is, being the provenance of a retained asset rather than a consumer of one, and is scheduled for deletion together with the asset it emits.

Six gates remain open and every one of them needs Isaac Sim. Gates 3, 4, 5, 6 and 9 are the conversion, the articulation, the survival of the welded bodies, the resolution of every `SceneEntityCfg`, and the realised body separations after an in-place edit. Gate 10 is a pre-training run. Gate 5 remains the gravest, for the reason section 17 gives.
