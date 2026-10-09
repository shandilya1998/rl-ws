# The parametric MJCF design generator

## 1. Purpose and scope of this document

This document records the software architecture, the application programming interface, the modelling choices and the mathematical content of the parametric MJCF design generator vendored as a git submodule at `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/design_generator`. It is written so that a later session may write a design search loop against that code, or extend it to a morphology it does not presently serve, such as a wheeled legged machine, without rereading four and a half thousand lines of source and eight hundred lines of markdown whose claims the source has in several places outgrown.

Three questions govern the reading. What does the generator take as input and emit as output, which is answered by sections 3 to 6. What physical model stands behind each number it writes into the MJCF, which is answered by sections 7 to 10. And what must a caller supply to drive it, which is answered by sections 11 and 12, the first recording the outer loop already written against an earlier revision of this code and the second recording the contract the co-optimisation runner in this workspace would require it to satisfy. Section 13 is a register of the defects and the documentation divergences that reading established, and section 14 states the rules a future extension should observe.

The document sits at the workspace level rather than in the simulation repository because its subject spans three trees, the submodule itself, the `co_optimisation` package that hosts it inside `tron1-rl-isaaclab-cozum`, and the separate repository at `/ws/Co-Design-Optimization-using-CMA-ES` from which the reference usage comes. The repository level index at `../tron1-rl-isaaclab-cozum/context/README.md` states that the co-optimisation pipeline record belongs one level up, and this is part of that record. The companion account of the pipeline itself is [copt.md](copt.md), of the evolutionary strategy that drives it [cmaes.md](cmaes.md), and of whether a periodic design swap is compatible with proximal policy optimisation [copt_ppo_nonstationarity.md](copt_ppo_nonstationarity.md).

Every claim below about the behaviour of this code carries a `path:line` citation resolved against the submodule root unless the path says otherwise, and every claim drawn from a published work carries a bracketed number resolved against the bibliography of section 15. Numerical results attributed to this session were computed by executing the module under `python3` with the shipped parameter files, and are marked as such at the point of use. The MuJoCo Python package is absent from this container, so no claim below rests on compiling the emitted XML, and the two places where the repository's own documentation asserts a compiler behaviour are attributed to that documentation rather than restated as verified fact.

## 2. Provenance, and the two places this tree is mounted

The submodule is registered at `tron1-rl-isaaclab-cozum/.gitmodules` against the remote `git@github.com:shandilya1998/design_generator.git` on branch `main`, and the working tree stands at commit `f0ba756`, whose single message reads "Merge quadruped design/MPC code into the combined repo layout". The same remote is mounted a second time in this workspace, at `/ws/Co-Design-Optimization-using-CMA-ES/design_builder`, registered in that repository's own `.gitmodules`. A recursive comparison of the two checkouts returns no difference other than a `__pycache__` directory, so the two mounts are one tree and a change to either reaches the other only through the remote.

That second mount is the more informative of the pair, because it sits beside the convex model predictive control stack and the two design search harnesses that the submodule's own documentation repeatedly names without shipping. Those harnesses are read in section 11. The lineage the source comments record runs further back still, to a biped generator under a project called `convex-mpc-biped`, from which the file layout, the two tier parameter file split and the `clearance_{source}2{target}_{axis}` naming were taken, and to a package called `design` carrying `spec.py`, `builder_v2.py` and `evaluator.py`, from which the quadruped placement rules were taken. Neither of those predecessors is present anywhere in this workspace, which matters because a large fraction of the shipped markdown is written as a diff against them and cannot be checked.

## 3. File structure

The submodule holds two sibling directories, `biped` and `quadruped`, which mirror one another module for module. Neither is a Python package, there being no `__init__.py` anywhere in the tree, and each module imports its siblings by bare name, so a caller must place the chosen directory on `sys.path` before importing anything from it. The two directories cannot be imported into one interpreter under their shipped names, since both define a module called `builder` and a module called `actuator`.

| Path | Lines | Role |
|---|---|---|
| `quadruped/generate.py` | 271 | `DesignSpec`, `DesignBounds`, actuator reference resolution, the command line entry point |
| `quadruped/builder.py` | 531 | `build_xml(spec)`, the MJCF emitter and every placement, mass and inertia formula |
| `quadruped/actuator.py` | 623 | `Motor`, `GearboxType`, `MADRow`, `Actuator`, `CatalogActuator`, the catalog loaders and the mode switch |
| `quadruped/constant_params.json` | – | Fixed hardware constants read by `builder.py` at import time |
| `quadruped/variable_params.json` | – | One flat object carrying every `DesignSpec` field, the baseline design |
| `quadruped/catalogs/actuators.json` | – | Eleven pre-built joint side actuators |
| `quadruped/catalogs/motors.json` | – | Six bare brushless motors, a fallback never reached in practice |
| `quadruped/catalogs/compact/` | – | The vendored COMPAct artefacts, one configuration file, four topology parameter files and four mass and diameter tables |
| `quadruped/graphs/` | – | Three throwaway verification scripts and their rendered figures |
| `quadruped/GEOMETRY_RULES.md` | 263 | The placement rule reference |
| `quadruped/DESIGN_PARAMETERS.md` | 97 | The design variable reference |
| `quadruped/README.md` | 62 | Orientation and quickstart |
| `biped/*` | – | The same set, with an ankle actuator and a foot link added to the chain, no `README.md`, and a `graphs/README.md` the quadruped lacks |

Two files that the markdown instructs the reader to run, `quadruped/view_joints.py` and its biped counterpart, are absent from both directories.

## 4. The three layer architecture

The generator is a straight pipeline of three layers with no cycles and no shared mutable state beyond one module level catalog, and its whole behaviour follows from reading them in order.

The lowest layer is the actuator model in `actuator.py`. It answers one question, which is what joint side torque, speed, mass, cost, efficiency and housing envelope follow from a named choice of hardware. It offers two entirely separate ways of answering it, a flat shelf of pre-built units read from JSON and a combinatorial enumeration of motor against gearbox topology against gear ratio drawn from the published artefacts of COMPAct [1]. Which of the two is live is a property of the module, held in the rebindable global `ACTUATOR_CATALOG` at `quadruped/actuator.py:569` and swapped by `set_active_catalog` at `quadruped/actuator.py:580`.

The middle layer is the design specification in `generate.py`. It is a frozen dataclass of scalar fields, a bounds object describing the box those fields may range over, and the machinery to move between the dataclass and a flat float vector that an external optimiser can perturb. It holds no geometry and no physics, only the design variables and the rules for sampling them.

The upper layer is the emitter in `builder.py`. It takes a specification, resolves its actuator references against whichever catalog is live, applies the placement rules to obtain every body position, applies the mass and inertia models to obtain every inertial property, and returns one complete MJCF document as a Python string. It writes no file of its own, the entry point at `quadruped/generate.py:251` being the only thing in the tree that touches disk on the output side.

The separation of the two parameter files is the architectural decision that most affects how the code is used. Values that a design search may vary live in `variable_params.json` and reach the emitter only through the `DesignSpec` it is passed. Values that describe the manufacturing process rather than the design, the housing shell dimensions, the link cross sections, the joint limits, the material densities and the joint dynamics, live in `constant_params.json` and are read once at import time into module level constants at `quadruped/builder.py:95`. The consequence is that changing a constant requires reimporting the module, and that no two designs within a single process may disagree about a constant. A caller that wishes to search over a quantity presently held constant must move it from the second file to the first and thread it through `DesignSpec`, which is the substance of section 14.

The data flow is therefore as follows. A JSON object is loaded by `load_design_spec` at `quadruped/generate.py:243` and splatted into the `DesignSpec` constructor, so the JSON key set must match the field set exactly. Each `act_*` field is resolved by `_resolve` at `quadruped/generate.py:43`, which accepts either an integer index into the live catalog or a catalog name, and raises `IndexError` or `KeyError` respectively with the available names enumerated. `build_xml` at `quadruped/builder.py:395` then validates, computes and formats, and the caller writes the returned string wherever it wishes.

## 5. The actuator model

### 5.1 The two catalog modes

The module exposes two disjoint representations of an actuator behind a duck typed union declared at `quadruped/actuator.py:527`. Nothing downstream distinguishes them, both carrying `name`, `peak_torque`, `peak_speed`, `mass`, `cost`, `eta_gb`, `diameter`, `length` and `electrical_power`.

The default mode is the flat catalog, loaded from `catalogs/actuators.json` by `load_actuator_catalog_json` at `quadruped/actuator.py:530` and bound at import time at `quadruped/actuator.py:569`. The module docstring records why this is the default rather than the COMPAct enumeration, namely that the `compact/` artefacts were once absent, which left the enumeration silently empty. Those artefacts are present now, so the reasoning no longer applies, but the default was not revisited. A name keyed dictionary is built alongside the list at `quadruped/actuator.py:574` so that a specification may reference an actuator by name, which the documentation correctly presses as the robust choice, since an index is a position in a file and shifts whenever that file is edited.

The second mode is the COMPAct enumeration, built by `build_compact_catalog` at `quadruped/actuator.py:434` as the cross product of every motor, every gearbox topology and a subsample of that topology's published ratios. Executed in this session against the shipped artefacts it yields eighty four entries, being seven motors against four topologies against the default three ratios each.

### 5.2 The motor

`Motor` at `quadruped/actuator.py:61` is a frozen record of a flat brushless machine, carrying the torque constant, the phase resistance, the peak current, the no load speed, the rotor inertia, the mass, the cost and, optionally, the body diameter and axial length. Its one derived property is the motor side peak torque, the product of the torque constant and the peak current.

The loader at `quadruped/actuator.py:100` accepts three schemas, this project's own list form, this project's dict form, and the official COMPAct configuration, which it detects by the presence of a top level `Motors` object. Under the third of these two quantities are reconstructed rather than read. The torque constant follows from the published speed constant by the standard conversion at `quadruped/actuator.py:80`,

    Kt = 60 / (2 pi Kv)

with `Kv` in revolutions per minute per volt, and the phase resistance is estimated at `quadruped/actuator.py:85` by assuming a fixed electrical to mechanical efficiency at the rated point, so that the resistive drop carries the complement of that efficiency,

    R = V_rated (1 - eta_motor) / I_max

with `eta_motor` defaulting to 0.85. This is a heuristic and is documented as one. It is the only place in the tree where a physical parameter is invented rather than read, and it propagates into the copper loss term of the power model of section 5.6, so a power figure computed under the COMPAct mode carries that assumption and a figure computed under the flat catalog mode does not, the flat entries carrying a measured resistance.

The loading order at `quadruped/actuator.py:176` prefers the COMPAct configuration over the project's own `motors.json`, falling back to three hardcoded entries only if both fail. Since the COMPAct configuration is present, the shipped `motors.json` is never reached, and the live motor list in this session was the seven COMPAct motors, `MotorU12_framed`, `Motor8020_framed`, `MotorU8_framed`, `MotorU10_framed`, `MotorMN8014_framed`, `MotorMAD_M6C12_framed` and `MotorU15_framed`. This is worth stating plainly because `catalogs/README.md` builds its whole ranking table on the assumption that `motors.json` is the live motor catalog, and it is not.

### 5.3 The gearbox and the COMPAct tables

COMPAct is a framework for identifying optimal gearbox parameters for a given motor across four planetary topologies, the single stage planetary gearbox, the compound planetary gearbox, the Wolfrom planetary gearbox and the double stage planetary gearbox, minimising mass and actuator width while maximising efficiency, and automating the generation of printable actuator computer aided design geometry [1]. The artefacts bundled under `catalogs/compact/` are the published outputs of that framework rather than a reimplementation of it, and this is the design decision that gives the enumeration its value. Each row of a mass and diameter table is a gearbox that satisfies an integer tooth count at a specific ratio, so every actuator the enumeration produces corresponds to a design the published optimiser actually returned, and no ratio is interpolated. The module docstring records that an earlier parametric model did interpolate, and produced ratios that no tooth count could realise.

`MADRow` at `quadruped/actuator.py:193` captures the seven fields the downstream model needs from a row, the ratio, the gearbox only mass, the theoretical efficiency, the joint side peak torque, the actuator outer width, the dimensionless cost objective and the planet count. The parser at `quadruped/actuator.py:211` reads the four tables through `csv.DictReader`, converts the published width from millimetres to metres, tolerates the two different planet count column spellings the tables use, silently discards any row whose fields do not parse, and sorts by ratio. Read in this session the tables carry seven, fourteen, fifty one and twenty eight usable rows for the single stage, compound, Wolfrom and double stage topologies respectively.

`GearboxType` at `quadruped/actuator.py:235` binds one topology's optimisation bounds, read from its own parameter file, to its table of realisable designs. It exposes both the declared bounds and the bounds the published rows actually span, and these disagree materially. The single stage topology declares a range of four to fifteen and publishes rows spanning 4.45 to 7.00, and the Wolfrom topology declares four to sixty and publishes 10.5 to 60.8. A caller reasoning about the reachable ratio space must read `ratio_min_published` and `ratio_max_published`, not the declared pair.

The reference motor against which the published rows were computed is fixed by name at `quadruped/actuator.py:265` as `MotorU8_framed`, and its motor side peak torque is resolved at `quadruped/actuator.py:268`, evaluating in this session to 2.29183 newton metres. Every scaling law of the next section is expressed as a ratio against that number.

### 5.4 The composed actuator and its scaling laws

`Actuator` at `quadruped/actuator.py:313` bundles a motor, a topology and a resolved table row, and presents the joint side view through four derived properties, each of which is a modelling choice worth stating separately.

The efficiency is taken directly from the row, being the theoretical figure the gear pair model produced.

The peak torque at `quadruped/actuator.py:334` takes the row's published joint side torque and scales it linearly by the ratio of the fitted motor's peak torque to the reference motor's,

    tau_peak = tau_peak_row * (Kt I_max)_motor / (Kt I_max)_ref

which preserves the published dependence on ratio and efficiency while honouring the chosen motor. This is exact under the assumption that the gearbox is torque transparent at a fixed ratio, which is the assumption a fixed efficiency already encodes.

The peak speed at `quadruped/actuator.py:345` is the motor's no load speed divided by the ratio, with no allowance for the voltage headroom consumed by the load, so it is a no load figure and is documented as one.

The mass at `quadruped/actuator.py:350` adds the motor's own mass to the row's gearbox mass scaled by the square root of the same torque ratio,

    m = m_motor + m_gearbox_row * sqrt(tau_peak_motor / tau_peak_ref)

The square root is a surrogate rather than a derivation. Its qualitative justification is that gear face width scales with the torque a tooth must carry while the pitch diameters do not, so mass grows sublinearly in torque, and a square root is the simplest exponent with that property. No source in the tree derives it and none is cited for it, so a caller who cares about the mass of an off reference motor should treat this term as an estimate.

The cost at `quadruped/actuator.py:359` discards the dimensionless COMPAct objective entirely and substitutes a dollars per kilogram surrogate at three hundred and eighty dollars per kilogram, applied to the gearbox mass and added to the motor cost. This is stated in the source and should be carried into any objective that weights cost, since the resulting figure is a mass proxy and not a price.

The envelope properties close the set. The diameter at `quadruped/actuator.py:366` is the greater of the motor body diameter and the gearbox width, the assembly being no narrower than either part. The length at `quadruped/actuator.py:374` is the motor's own body length, with a fallback of six tenths of the diameter, and its docstring is careful to record that this is a lower bound rather than a measurement, the published rows carrying no axial gearbox dimension. That omission matters only for the biped, whose housings are sized from these two fields, and not for the quadruped, whose housings are fixed constants.

### 5.5 The pre-built actuator

`CatalogActuator` at `quadruped/actuator.py:467` is the flat alternative, a record of joint side figures with no decomposition. Five fields are required of a JSON entry, the name, the peak torque, the peak speed, the mass and the cost, and the remainder carry defaults, the efficiency at 0.85, the diameter at 0.080 metres and the length at six tenths of the diameter. The loader at `quadruped/actuator.py:530` is the only consumer of the JSON schema, and it reads the keys `diameter` and `length`.

The shipped quadruped catalog does not carry those keys. It carries `width`, which nothing reads. Executing the loader in this session against `quadruped/catalogs/actuators.json` returns eleven entries every one of which resolves to a diameter of exactly 0.080 metres and a length of exactly 0.048 metres, the two defaults, regardless of the `width` its own entry declares. The biped's copy of the catalog is a later revision that carries `diameter` and `length` properly, together with a combined ranking index, and resolves to the real envelopes. The two catalogs have diverged in their numbers as well as their schema, the quadruped's Unitree B1 declaring a peak torque of 360 newton metres against the biped's 210, and its AK10-9 declaring 48 against 53. This is recorded as a defect in section 13 rather than repaired here, since repairing it would change the mass of every design the quadruped generator has ever produced.

### 5.6 The electrical power model

Both actuator classes expose the same direction aware power model, at `quadruped/actuator.py:387` and `quadruped/actuator.py:498`. It is the one piece of genuine physics in the module that does not appear in the emitted XML at all, existing solely for an outer loop objective.

Given a joint side torque and angular velocity, the sign of their product decides the regime. Under motoring the gearbox loss is charged to the input, so the motor side torque is the joint torque divided by both the ratio and the efficiency. Under braking the loss is charged against the returning power, so the motor torque is the joint torque multiplied by the efficiency and divided by the ratio. The motor speed is the joint speed multiplied by the ratio in both regimes. The current follows from the torque constant, the copper loss from the square of the current against the phase resistance, and the electrical input is the mechanical power plus the copper loss under motoring and the copper loss alone under braking, floored at zero in both cases, so no regenerative credit is granted,

    tau_m = tau_j / (N eta)   motoring,          tau_m = tau_j eta / N   braking
    omega_m = omega_j N
    I = tau_m / Kt
    P = max(0, omega_m tau_m + I^2 R)   motoring,  P = max(0, I^2 R)   braking

`CatalogActuator` degrades gracefully when the electrical fields are absent, falling back to an efficiency only estimate that charges the mechanical power divided by the efficiency under motoring and nothing under braking. The model neglects iron loss, friction and the inverter, and charges no quiescent current, so it is a lower bound on the true electrical draw.

### 5.7 The catalog hot swap

`set_active_catalog` at `quadruped/actuator.py:580` rebinds four module level globals, the actuator list, the motor list, the gearbox list and the name index, and returns the new actuator list. The contract that makes this safe is that no consumer captures the catalog at import time. `generate.py` observes it, reading through the indirections `_active_catalog` and `_active_catalog_by_name` at `quadruped/generate.py:34` and `quadruped/generate.py:39`, which resolve the attribute on the module object at every call. The `DesignBounds` actuator fields observe it too, through `default_factory` lambdas at `quadruped/generate.py:204`, so a bounds object constructed after a swap carries the new catalog's index range. A bounds object constructed before a swap does not, which is why both shipped harnesses call the swap before anything else in `main`.

## 6. The design specification and the search space

### 6.1 Two tiers of field

`DesignSpec` at `quadruped/generate.py:70` is a frozen dataclass whose twenty four fields fall into two tiers by whether they carry a Python default.

The nine free fields are the trunk length, width and thickness, the thigh and shank lengths, the body height, and the three actuator references. These are the search space, being exactly the fields that `vector`, `from_vector`, `labels`, `DesignBounds.as_pairs` and `DesignBounds.sample` enumerate.

The fifteen fixed constant fields are the clearances, each defaulting to zero, named `clearance_{source}2{target}_{axis}` and chained down the leg from the trunk through the three housings and the thigh to the foot. Every one is read by the emitter, each added on top of its link's rule derived base offset, so leaving them at zero reproduces the pure rule geometry and setting one nudges a single body along a single axis. None of them appears in the vector form, so an optimiser cannot reach them and `from_vector` reconstructs them at their defaults.

The distinction is worth holding onto, because it is the mechanism by which this generator separates a design decision from a manufacturing correction. A clearance exists so that a specific actuator housing shape can be accommodated without perturbing a formula that is otherwise an exact geometric identity.

### 6.2 The vector interface

`vector` at `quadruped/generate.py:148` emits the nine free fields as a float array in a fixed order and `from_vector` at `quadruped/generate.py:165` inverts it, rounding each actuator reference and clipping it into the live catalog's index range. The docstring states the restriction that follows, which is that a specification whose actuator references are names cannot be vectorised, a string not being castable to a float, so a caller who needs the vector form must build the specification with integer references.

`DesignBounds` at `quadruped/generate.py:194` gives an inclusive interval per free field, `integrality_mask` at `quadruped/generate.py:219` marks the three actuator dimensions as integer valued in a form that scipy's differential evolution accepts directly, and `sample` at `quadruped/generate.py:223` draws the six continuous fields uniformly and the three discrete ones by integer draw, then routes the concatenation through `from_vector` so that the clipping applies. The trunk thickness bound is degenerate at (0.09, 0.09), which is honest, the field being overridden in the emitter and the bound existing only to preserve the vector's shape.

`per_joint_tau_max` at `quadruped/generate.py:140` returns the twelve vector of torque clips in leg major order, for an external controller to consume. Nothing inside the submodule calls it. The quadruped joint ordering it assumes, being `FR`, `FL`, `RR`, `RL` against abduction, hip and knee, matches the leg ordering the convex control stack uses at `/ws/Co-Design-Optimization-using-CMA-ES/mpc/quadruped/mpc/robot.py:23`.

### 6.3 Where the biped specification differs

The biped specification at `biped/generate.py:71` is not a superset of the quadruped's, and the differences are not only the ankle. It carries fifteen free fields rather than nine, having promoted four clearances into the search space, the abduction housing's vertical drop and three cluster gaps, and having added a foot length and a fourth actuator reference. It carries no trunk thickness override, that field remaining exactly as the caller set it. And its one non zero default is the vertical gap from the ankle housing to the foot at `biped/generate.py:142`, which must stay positive because the foot hangs below the ankle.

The deeper divergence is philosophical. On the biped a clearance is frequently the whole of an offset, the base term being zero, whereas on the quadruped every offset has a geometrically determined base and a clearance is a correction upon it. The quadruped builder's own docstring records this as the intended difference at `quadruped/builder.py:83`.

## 7. The quadruped geometry

### 7.1 Topology

Each of the four legs is a chain of six bodies emitted by `_leg_xml` at `quadruped/builder.py:301`, three of them actuator housings and three of them structure,

    trunk -> abad_{LEG}_act -> hip_{LEG}_act -> knee_{LEG}_act -> thigh_{LEG}_Link -> shank_{LEG}_Link -> foot_{LEG}_Link

The placement of the joints within that chain is the point on which the architecture turns, and it is not the naive one. The abduction housing is welded to the trunk and carries no joint at all. The abduction degree of freedom lives on the hip housing, the hip pitch degree of freedom on the knee housing, and the knee degree of freedom on the shank. Each housing therefore carries the joint of the stage above it, which is what makes the cluster a cluster, the three motors sitting together near the trunk rather than distributed down the leg, and the mass of the knee motor consequently riding proximal of the knee rather than upon it. This is the mass distribution a real quadruped leg is built to achieve and it is the reason the topology is preferred to a naive one link per joint chain.

A second naming layer sits on top. The internal token for a joint type, `abad`, `hip` or `knee`, names the default class, the joint dynamics entry and the link colour, while the emitted joint and actuator names carry the convention the model predictive control stack expects, mapped at `quadruped/builder.py:453`, so that the abduction axis is named `hip`, the thigh flexion axis is named `thigh` and the knee is named `calf`. The emitted joint names are therefore `{LEG}_hip_joint`, `{LEG}_thigh_joint` and `{LEG}_calf_joint`, and these match the suffixes the controller looks up at `/ws/Co-Design-Optimization-using-CMA-ES/mpc/quadruped/mpc/robot.py:24`. A reader tracing a torque from the specification to the simulator must pass through this map twice and should expect the confusion it causes.

### 7.2 The four placement rules

Four rules fix the relationship between the trunk and the abduction housing, and the emitter treats them as exact equalities rather than as targets. All four are stated in `quadruped/GEOMETRY_RULES.md` and implemented in `quadruped/builder.py`.

The first rule places the abduction housing flush against the trunk's fore or aft end face, the offset being half the trunk length plus the housing's own half length, implemented at `quadruped/builder.py:186` and signed per leg by `SIDE_SIGN_X`,

    x_abad = +- (trunk_length / 2 + ABD_HALFLEN)

The second rule constrains the trunk to be at least twice the abduction actuator's diameter wide, so that the two housings on one axle do not overlap, and this is the only hard constraint in the generator. It is enforced at `quadruped/builder.py:281`, which raises `ValueError` rather than clipping, and the derived floor at `quadruped/builder.py:178` evaluates to 0.184 metres against the shipped baseline width of 0.20.

    trunk_width >= 2 * (2 * ABD_RADIUS)

The third rule nests the abduction housing inside the trunk's width envelope rather than pushing it outside, the offset being half the trunk width minus the housing radius, so the housing's outer surface is flush with the trunk's side face. It is implemented at `quadruped/builder.py:194` and, because the second rule guarantees half the width exceeds the radius, the two rules are always mutually consistent.

    y_abad = +- (trunk_width / 2 - ABD_RADIUS)

The fourth rule pins the trunk's height to the abduction actuator's diameter, an equality rather than a floor, and the emitter enforces it by overwriting the incoming field at `quadruped/builder.py:400` before any downstream computation reads it. This is the only field the generator silently discards, and it is silent by design, the documentation preferring an override to a rejection so that a specification drawn from a wider box remains buildable. Against the shipped constants it fixes the trunk height at 0.092 metres.

    trunk_thickness = 2 * ABD_RADIUS

The fourth rule also explains why the abduction housing has no vertical base offset, sitting at the trunk's own mid height and flush with its top and bottom faces, so that `clearance_trunk2abadAct_z` is the only source of vertical displacement available to it.

### 7.3 The cluster offsets

Three further offsets carry the chain past the abduction housing, and all three are flush contact identities in which each neighbour contributes its own half extent toward the shared boundary, so that the bodies touch with neither gap nor overlap. They are derived at `quadruped/builder.py:173` and applied at `quadruped/builder.py:326`, `quadruped/builder.py:332` and `quadruped/builder.py:343`,

    x_hip   = +- (ABD_HALFLEN + HIP_RADIUS)
    y_knee  = +- (HIP_HALFLEN + KNEE_HALFLEN)
    y_thigh = +- (KNEE_HALFLEN + THIGH_BOX_HALF[1])

The thigh shares the sign of the knee offset rather than opposing it, so the stack continues outboard past the knee housing's far face instead of doubling back. Evaluated in this session at the shipped constants the two derived quantities are 0.066 and 0.040 metres, and the thigh's lateral station measured from the trunk centre line, including the one non zero clearance the baseline sets, is 0.129 metres against a trunk half width of 0.10, so the legs stand 0.029 metres outboard of the trunk's side face.

The critical modelling simplification of the whole geometry is stated at `quadruped/builder.py:14`. Every housing cylinder is a fixed size, `ABD_RADIUS` through `KNEE_HALFLEN`, independent of which catalog actuator is chosen for that joint. Only the mass varies with the pick. This makes the geometry invariant under an actuator swap, which is what allows the placement rules to be constants rather than functions of the design, and it is the single largest difference between the quadruped and the biped, whose housings are sized from the catalog envelope and whose offsets are therefore design dependent.

### 7.4 The distal links and the foot

The thigh and the shank are straight cuboids hanging along the parent frame's negative vertical, emitted at `quadruped/builder.py:371` and `quadruped/builder.py:376` with an explicit position and size rather than through the `fromto` convenience. `quadruped/GEOMETRY_RULES.md` records that this choice is forced, asserting that the MuJoCo [4] compiler silently squares a box's cross section when `fromto` is given, reading only the first size component, which would double the mass whenever the two half extents differ. That claim is the repository's own, established by its author through recompilation, and it is repeated here as attribution rather than as a result verified in this session, the MuJoCo package being absent from this container. It is noted that the module docstring at `quadruped/builder.py:38` still describes the boxes as placed via `fromto`, which the code no longer does.

The foot is a fixed radius sphere carrying no mass of its own, its collision geometry declaring a density of zero at `quadruped/builder.py:381`. The reason is a double counting argument rather than an approximation, the shank's mass per length constant already accounting for the real hardware's foot, so giving the sphere its own mass would charge it twice. The body is legal in MuJoCo despite having no mass because it carries no joint and is therefore welded to its parent. The sphere carries its own friction, solver impedance and contact dimension, overriding the class defaults, and a site is placed at its centre for the touch and position sensors.

### 7.5 The placement verified at the baseline

Evaluated in this session against the shipped `variable_params.json` and `constant_params.json`, the derived geometry is as follows, and these numbers are the ones a reader should check a modified build against.

| Quantity | Value | Source |
|---|---|---|
| Minimum trunk width, rule 2 | 0.184 m | `quadruped/builder.py:178` |
| Trunk thickness after the rule 4 override | 0.092 m | `quadruped/builder.py:183` |
| Abduction housing station in x | 0.120 m | `quadruped/builder.py:186` |
| Abduction housing station in y | 0.054 m | `quadruped/builder.py:194` |
| Hip housing offset in x | 0.066 m | `quadruped/builder.py:173` |
| Knee housing offset in y | 0.040 m | `quadruped/builder.py:174` |
| Thigh lateral station from the trunk centre | 0.129 m | computed, includes the baseline clearance of 0.005 m |
| Thigh volumetric density | 5062.5 kg/m³ | `quadruped/builder.py:234` |
| Shank volumetric density | 2359.694 kg/m³ | `quadruped/builder.py:235` |

## 8. The mass and inertia model

### 8.1 The linear mass model and the density inversion

The generator declines to write an explicit inertial element for the distal links, and instead hands MuJoCo [4] a geometry density and lets the compiler integrate mass, centre of mass and inertia from the volume. That decision creates a problem, because the design variable is a length and the physical model the project wants is a mass proportional to that length, whereas a density multiplied by a volume already varies with length. The resolution at `quadruped/builder.py:212` is to invert the relation, choosing the density that makes the product reduce to the intended linear law.

For a box of cross section `2 h_x` by `2 h_y` extended along its length `L`, the compiler computes `rho * 4 h_x h_y L`, and setting this equal to `lambda L` gives a density independent of the length,

    rho = lambda / (4 h_x h_y)

so the emitted mass is exactly `lambda L` for every length the design search may propose. The docstring records the reason a box is preferred to a capsule here, which is that a capsule's hemispherical end caps break the exactness.

Verified in this session at the baseline, the thigh geometry integrates to 0.648000 kilogrammes against the intended `3.24 * 0.20`, and the shank to 0.296000 against `1.48 * 0.20`, agreeing to every digit the formatting carries.

### 8.2 The visual scale correction

A complication sits on top. Both links are drawn at eight tenths of their true length so that adjacent links do not visually interpenetrate at the joints, and the shortening is applied to the collision geometry as well as the visual geometry, at `quadruped/builder.py:371` and `quadruped/builder.py:372`. Since it is the collision geometry that carries the density, the mass would fall short by that same fraction. The compensation at `quadruped/builder.py:232` raises the linear mass constant by the reciprocal of the scale before the density inversion is applied,

    lambda_corrected = lambda / VISUAL_LINK_SCALE

so that the shortened box still integrates to the true hardware mass. Evaluated in this session the corrected constants are 4.05 and 1.85 kilogrammes per metre against the declared 3.24 and 1.48.

The consequence is a modelling compromise that should be stated rather than hidden. The link's collision extent is four fifths of its kinematic length while its mass is that of the full length, so its density is a quarter higher than the material's and its own moment of inertia about a transverse axis is smaller than the true link's by roughly the square of the scale. For a link whose mass is dominated by the actuator masses at its ends this is a small error, and for a design search that varies length it is a consistent one, but a caller computing a reflected inertia from the emitted model should know that the number is not the hardware's.

### 8.3 The trunk as a shell and a payload

The trunk is the one body whose inertial element is written explicitly, and it is modelled as the superposition of two rigid bodies sharing a centre, computed at `quadruped/builder.py:410`. The first is a thick walled hollow cuboid shell, standing for the structure. The second is a solid cuboid filling that shell's cavity, standing for the battery and the electronics.

The shell's mass is the outer surface area multiplied by the wall thickness and the material density, which is the thin wall approximation and is exact only to first order in the thickness, overcounting the eight corners where three faces meet. Its inertia is computed at `quadruped/builder.py:245` by subtracting a cavity from a solid block, where the cavity is assigned the mass it would have at the same uniform density, being the shell mass scaled by the volume ratio, so that the subtraction is dimensionally consistent,

    m_cavity = m_shell * (a - 2t)(b - 2t)(c - 2t) / (a b c)
    I_x = (1/12) m_shell (b² + c²) - (1/12) m_cavity ((b - 2t)² + (c - 2t)²)

and cyclically for the two remaining axes. The function raises rather than returning a negative inertia when the wall thickness exceeds half of any outer dimension, which is the correct failure mode for a design search that may propose a small trunk. The payload's inertia is the solid cuboid formula evaluated on the cavity dimensions at the declared payload mass, and the two are added.

Verified in this session at the baseline, the trunk's outer surface is 0.153600 square metres, the shell mass is 1.244160 kilogrammes and the total trunk mass is 5.244160 kilogrammes against a declared payload of 4.0.

### 8.4 The housing inertias

Each housing writes an explicit inertial element at `quadruped/builder.py:295`, taking the catalog actuator's mass as given and computing a solid cylinder's principal moments about its own centroid at `quadruped/builder.py:204`, the axial moment being half the mass times the squared radius and the transverse moment the standard `(1/12) m (3 r² + L²)`. The same alignment quaternion that orients the geometry orients the inertia tensor, so the axial term lands on the joint axis. Since the housing dimensions are fixed constants, the inertia of a housing varies with the actuator pick only through the mass, which is a solid approximation for a machine whose rotor and stator fill the shell but which understates the polar moment of a pancake motor whose mass concentrates at the rim.

### 8.5 The mass budget verified at the baseline

Computed in this session by evaluating the emitter's own formulas against the shipped baseline, with all three joints carrying the `Unitree-A1` entry at 0.605 kilogrammes.

| Component | Count | Each | Total |
|---|---|---|---|
| Trunk shell | 1 | 1.244160 kg | 1.244160 kg |
| Declared payload | 1 | 4.000000 kg | 4.000000 kg |
| Actuator housings | 12 | 0.605000 kg | 7.260000 kg |
| Thigh links | 4 | 0.648000 kg | 2.592000 kg |
| Shank links | 4 | 0.296000 kg | 1.184000 kg |
| Feet | 4 | 0 kg | 0 kg |
| Whole robot | | | 16.280160 kg |

Actuation accounts for 44.6 per cent of that total, which is the figure a co-design objective weighting mass against torque is ultimately trading against.

## 9. The home pose, the joint limits and the emitted model

### 9.1 The pose is set, not solved

The home keyframe's hip pitch and knee angles are hand set constants read from the parameter file at `quadruped/builder.py:145` and written into the keyframe at `quadruped/builder.py:443`, replicated across all four legs. An earlier revision solved a two link inverse kinematic problem for a foot landing at the ground plane, and the docstring at `quadruped/builder.py:48` records why that was withdrawn, namely that it made the pose an indirect function of the body height and the leg lengths rather than something a user could set, and that it failed silently when the body height exceeded the leg's reach.

The consequence is that the body height and the pose are now independent knobs and the feet are not guaranteed to touch the ground. Computed in this session at the baseline, rotating the thigh by seventy degrees and the knee by minus one hundred and fifty four, the foot centre sits 0.089310 metres below the hip pitch axis and 0.010966 metres forward of it, so with a body height of 0.4 metres the foot sphere's lower surface stands 0.288690 metres above the floor. This is not a bug, but it does mean that a design search which drops the emitted model onto a controller must either set the two bend angles to a pose that closes the chain or accept a drop at the start of every trial.

The two angles are not validated against the joint ranges, and the documentation states plainly that an out of range value means the joint starts pressed against its own limit stop rather than that the build fails. Checked in this session, the baseline values do clear their ranges, the knee angle of minus 2.687807 radians sitting 0.008723 radians inside a floor of minus 2.696530, a margin of half a degree.

### 9.2 The joint limits

The three ranges in the shipped parameter file are, converted in this session, minus forty six to forty six degrees for abduction, minus sixty to two hundred and forty degrees for hip pitch, and minus one hundred and fifty four and a half to minus fifty two and a half degrees for the knee. These are exactly the values the MuJoCo Menagerie Unitree A1 model declares for its three joint classes, verified in this session by retrieval of that model [2], and the 33.5 newton metre force range that model declares is likewise exactly the peak torque the shipped catalog assigns to its `Unitree-A1` entry. The quadruped's kinematic envelope is therefore an A1 envelope regardless of what the design search does to its link lengths, which is a constraint a caller should be aware of when searching outside that machine's scale.

### 9.3 What reaches the emitted model, and what does not

The emitted document declares a two millisecond timestep, fifty solver and line search iterations, the implicit fast integrator and an elliptic friction cone, at `quadruped/builder.py:470`. It carries a visual and a collision default class, the visual class declaring a density of zero so that visual geometry never contributes mass, and one default class per joint type carrying that type's damping, armature and friction loss together with the control and force range. It carries a free joint at the trunk, a tracking camera, an inertial measurement site, a ground plane, twelve direct torque motors, a sensor block of base orientation, position, rate and acceleration together with a position and a touch sensor at each foot and the subtree centre of mass and linear velocity, and one keyframe.

Of everything the actuator model computes, exactly two quantities reach the document. Verified in this session by enumerating every attribute access on a resolved actuator within the emitter, the quadruped builder reads `mass` at `quadruped/builder.py:318` and `peak_torque` at `quadruped/builder.py:431`, and nothing else. The peak speed, the efficiency, the cost, the rotor inertia and the power model are all inert with respect to the emitted asset. Three of those are inert by necessity, MJCF carrying no joint velocity limit and no notion of price, and they exist for the outer loop objective of section 11. But the rotor inertia is a real omission, because MJCF does carry an armature, and the armature the generator writes is a flat constant per joint class rather than the reflected inertia `I_rotor N²` the chosen actuator would actually present. That omission is registered as a known task in the harness repository's own `TODO.md` and it is restated here in section 13 because it bears directly on any gain derivation performed against a generated asset, the armature being the term that dominates the effective inertia at a lightly loaded joint, as this workspace's own record establishes at `../tron1-rl-isaaclab-cozum/context/KScale.md`.

## 10. The biped variant

The biped directory implements the same architecture over a different topology and a different placement philosophy, and it is the reference a wheeled legged extension should read most closely, being the one case in the tree where the geometry is a function of the actuator chosen.

Its chain adds two bodies, an ankle housing welded to the shank and a foot link carrying the ankle joint, so a leg is eight bodies and four degrees of freedom, and a build emits eight motors rather than twelve. Its housings are sized from the catalog at `biped/builder.py:185`, halving the actuator's diameter and length to obtain the cylinder's radius and half length, which is why the biped's catalog had to be revised to carry those fields and the quadruped's did not.

Its placement formulas are chained clearances rather than rule derived identities, computed at `biped/builder.py:267`. The abduction housing hangs below the trunk rather than sitting at its mid height, the vertical offset being the trunk's half thickness plus the housing radius plus a clearance. The hip housing's lateral offset is the one genuinely non trivial formula in either directory,

    y_hip = -+ ( sqrt( (y_abad + clearance)² - r_hip² ) - 2 hl_hip )

which is the tangency condition for placing a cylinder of radius `r_hip` against a lateral station at distance `y_abad + clearance`, and it can be imaginary, which is why `check_geometry` at `biped/builder.py:242` guards the discriminant explicitly and raises with a remedy named. The emitter guards it a second time by flooring the discriminant at zero at `biped/builder.py:279`, so a specification that slipped past the check degrades rather than raising. The source comment at `biped/builder.py:280` records that this offset alone carries the opposite sign convention to its neighbours, an asymmetry established when the formula was first given and preserved deliberately.

Its thigh points along a fixed computer aided design bend direction rather than straight down, the direction being derived once at import from a constant bend angle at `biped/builder.py:150`, and the thigh length scales along that direction without changing it. The documentation is candid that this costs about two millimetres of fidelity against the reference model and calls the trade deliberate. Its shank and ankle carry their own fixed bend quaternions. Its links are shortened by a fraction in the same spirit as the quadruped's visual scale, with the same density compensation, and its foot is a box with a real mass rather than a massless sphere.

Its home pose is a hardcoded four vector at `biped/builder.py:464`, taken from the reference model's own keyframe, with no relation to the specification at all.

Computed in this session at the shipped biped baseline, with all four joints carrying `MIT-Cheetah-3`, the housings resolve to a radius of 0.048 and a half length of 0.020 metres, the trunk shell weighs 2.946780 kilogrammes against a declared payload of 7.0, the thigh weighs 0.874800 and the shank 0.444000 and the foot 0.296000 kilogrammes, each agreeing exactly with its linear model, and the whole robot weighs 17.176380 kilogrammes.

## 11. The reference usage, and what it establishes

The two harnesses the user identified, at `/ws/Co-Design-Optimization-using-CMA-ES/mpc/quadruped/cmaes_design_search.py` and `optimize_design.py`, are the intended consumers of this generator, and they establish the shape of an outer loop around it. They must be read with one caveat stated first, which is that neither runs as shipped. Both import a package named `design`, and a directed search of that repository in this session found no such package anywhere, the import failing under `python3` with `ModuleNotFoundError`. Both also use an older specification surface that the submodule no longer offers, naming `calf_length` where the submodule names `shank_length`, naming `hip_x` and `hip_y` which the submodule has removed entirely, calling `build_xml_v2` and passing a `payload_mass` keyword which `build_xml` does not accept. They are therefore a design document rather than executable code, and the value in them is the contract they imply.

That contract has four parts. First, an evaluator, `evaluate(spec, constraints, weights, builder)`, which builds the MJCF from a specification, drops it onto a convex model predictive controller of the MIT Cheetah 3 family [3], runs a battery of velocity and yaw rate tracking trials against a required payload, and returns a record carrying the mass, the cost, the tracking error, the mean electrical power, a crash flag, a constraint violation magnitude and a scalar score. Second, a weight set, defaulting to one per kilogramme of mass, one hundredth per dollar, fifty per squared metre per second of tracking error and one tenth per watt. Third, a constraint set, carrying a minimum payload, a minimum ground clearance and a maximum trunk envelope, whose violations attract a heavy penalty so that the optimiser discards them without a feasibility projection. Fourth, a search, either an unstructured random draw from `DesignBounds.sample`, a differential evolution over the full mixed integer vector using `integrality_mask` directly, or a covariance matrix adaptation evolution strategy over a four dimensional continuous subspace of thigh length, shank length, trunk length and trunk width with every other field pinned at a baseline.

Three details of that last variant are worth carrying forward because they are lessons rather than choices. The lower bound on trunk width is raised to the rule 2 floor before the strategy is constructed, because a design below that floor is rejected outright by the emitter and a rejected candidate carries no gradient. The catalog mode is switched before any bounds object is built, for the reason section 5.7 gives. And the population's MJCF files are written to a fixed set of sample directories that are overwritten each generation, so that disk usage is bounded by the population size rather than by the generation count, with the single best design written once at the end.

The parameter reference beside those harnesses records that the controller is rebound to the design at three points, the commanded body height taking the specification's own, the torque clip taking the twelve vector from `per_joint_tau_max`, and the model predictive control force ceiling scaling with the built robot's measured total mass. That is the minimum coupling a design conditioned controller requires, and it is worth noting that the first and third of those are quantities a reinforcement learning environment would also have to rebind per design.

## 12. What this workspace's co-optimisation runner would require

The submodule sits inside `co_optimisation/co_optimisation/runners/`, beside `usd_generator.py`, and the placement is a statement of intent rather than a wiring, since no module in the package imports it. A directed search in this session found no reference to `design_generator`, `build_xml` or `DesignSpec` anywhere in `co_optimisation` outside the submodule itself.

One naming hazard should be cleared before anything else. The phrase "design generator" already means something else in this workspace, namely the `RandomDesignGenerator` and `CMAESDesignGenerator` classes of `usd_generator.py`, and `../CO_OPTIMISATION.md` uses it in that sense throughout. The submodule shares the name and is not the same object, so a sentence about the design generator must say which of the two it means.

The contract it would have to satisfy is fully specified by the abstract base classes already in place. `DesignGeneratorBase` at `../tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py:287` requires `_generate_individual`, returning a per individual pair of absolute link extents and actuator overrides, and `generate_population`, returning a population or `None`, together with the two optional hooks `update_with_fitness` and `sample_batch` that the runner calls at `../tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:574` and `:577`, and the state serialisation the checkpoint path reads at `:175` and `:221`. `Population` at `usd_generator.py:251` requires three accessors, one returning a USD path per individual, one returning an actuator override dictionary keyed by actuator group, and one returning absolute link extents for the in place morphology edit.

Three gaps stand between the submodule and that contract, and they should be recorded before any integration is attempted.

The first is the format. The incumbent generator mutates a URDF with `xml.etree` and converts it to USD through Isaac Lab's converter at `usd_generator.py:403`, whereas this submodule emits MJCF and nothing else. A bridging step is required, and this workspace has already performed that conversion once by hand and recorded the general procedure for it at `../tron1-rl-isaaclab-cozum/context/quadruped_xml_to_urdf_conversion.md`, whose half extent rule, capsule caveat and frame placement conventions are exactly what an automated bridge would have to implement.

The second is the parameterisation. The incumbent searches over scale factors applied to a donor asset, a thigh length scale and a shank length scale bounded at 0.75 and 1.25 at `usd_generator.py:240`, and every design in the population is therefore a perturbation of one machine. This submodule searches over absolute dimensions and discrete hardware picks, and constructs each design from primitives with no donor at all. The second space strictly contains the first and is far larger, and the consequences of enlarging it for the stationarity of the learning problem are the subject of [copt_ppo_nonstationarity.md](copt_ppo_nonstationarity.md), whose central finding, that the present design space is a finite lattice, does not survive the change.

The third is the actuator. The incumbent emits overrides against Isaac Lab actuator configurations, stiffness, damping, effort limit, saturation effort, armature and friction, whereas this submodule emits a peak torque and nothing else, and does not compute a reflected inertia at all. Translating a catalog pick into an Isaac Lab actuator configuration requires choosing gains, and this workspace's own record establishes that choosing a gain from a swing inertia alone produces a robot that cannot stand, the derivation and the training run it cost being recorded at `../tron1-rl-isaaclab-cozum/context/KScale.md`. A design search that varies both link length and actuator must recompute both regimes per design, and nothing in the submodule does so.

## 13. Defects and divergences established by reading

The tree carries a substantial quantity of documentation that the source has outgrown. What follows is a register rather than a repair, since several of the divergences could only be closed by changing numbers that historical builds depend upon. Each entry names the two places that disagree.

| # | Matter | Established at |
|---|---|---|
| 1 | The builder docstring states that `DesignSpec` no longer carries any clearance field, which is false | `quadruped/builder.py:28` against `quadruped/generate.py:108` and `quadruped/builder.py:311` |
| 2 | The constants table gives the thigh cross section as (0.020, 0.020), the mass per length constants as 0.879646 and 0.431027, and the knee bend as minus seventy degrees | `quadruped/GEOMETRY_RULES.md` against `quadruped/constant_params.json`, which declares [0.020, 0.010], 3.24, 1.48 and minus one hundred and fifty four |
| 3 | The same table states the collision geometry is drawn at full unscaled size, so that mass and physical length are unaffected by the visual scale | `quadruped/GEOMETRY_RULES.md` against `quadruped/builder.py:372`, which shrinks it, and `quadruped/builder.py:232`, which exists only to compensate for that shrinking |
| 4 | The builder docstring describes the distal boxes as placed via `fromto`, a form the rules file separately explains must not be used | `quadruped/builder.py:38` against `quadruped/builder.py:371` |
| 5 | The rules file states the baseline sets two clearances to 0.005 and 0.010 metres | `quadruped/GEOMETRY_RULES.md` against `quadruped/variable_params.json`, which sets 0.0 and 0.005 |
| 6 | The parameter reference gives baselines of 0.36 metres trunk length and 1.0 metres body height, and states every clearance is zero | `quadruped/DESIGN_PARAMETERS.md` against `quadruped/variable_params.json`, which declares 0.2, 0.4 and one non zero clearance |
| 7 | The README states the home pose is solved by two link inverse kinematics and that any body height lands the feet on the floor automatically | `quadruped/README.md` against `quadruped/builder.py:443`, and against the 0.288690 metre foot clearance computed in this session at the baseline |
| 8 | Both quickstarts instruct the reader to run `view_joints.py`, which is absent from the submodule | `quadruped/README.md` against the file listing |
| 9 | Every reference to `design/spec.py`, `design/builder_v2.py`, `design/GEOMETRY_RULES.md`, `optimize_design.py` and `cmaes_design_search.py` resolves nowhere within the submodule, and the last two, which do exist one repository away, import a `design` package absent from that repository too | throughout both markdown sets, against `/ws/Co-Design-Optimization-using-CMA-ES/mpc/quadruped/optimize_design.py:36` |
| 10 | The quadruped actuator catalog declares a `width` field that no loader reads, so all eleven entries resolve to the default envelope of 0.080 by 0.048 metres | `quadruped/catalogs/actuators.json` against `quadruped/actuator.py:551`. Inert for the quadruped, whose housings are fixed, but fatal were this catalog used with the biped emitter |
| 11 | The catalogs README tabulates torques, masses and envelopes that its own quadruped JSON does not carry, matching the biped's later revision instead, and disagrees on the Unitree B1 at 210 against 360 newton metres and the AK10-9 at 53 against 48 | `quadruped/catalogs/README.md` against `quadruped/catalogs/actuators.json` |
| 12 | `clearance_ankleAct2foot_y` is declared and documented as positioning the foot body, and is never read | `biped/generate.py:139` and both biped markdown files, against a directed search of `biped/builder.py`, where the foot body's position is the literal origin at `biped/builder.py:355` |
| 13 | The biped constants table gives mass per length constants of 4.32, 2.16 and 2.1, and names three constants that no longer exist | `biped/GEOMETRY_RULES.md` against `biped/constant_params.json`, which declares 3.24, 1.48 and 1.48 |
| 14 | A guard raises with a message about clearing the knee actuator's radius, a subtraction the formula no longer performs | `biped/builder.py:324` |
| 15 | The quadruped's root body is named `trunk` while every docstring and both markdown files call it `base_Link` | `quadruped/builder.py:502` against `quadruped/builder.py:9` |
| 16 | The armature is a flat per class constant rather than the reflected rotor inertia the chosen actuator presents, and `CatalogActuator` carries no rotor inertia field at all so the quantity cannot be computed under the default catalog mode | `quadruped/builder.py:437` and `quadruped/actuator.py:467`, the omission being separately registered at `/ws/Co-Design-Optimization-using-CMA-ES/mpc/quadruped/TODO.md` |
| 17 | The peak speed, the efficiency and the cost never reach the emitted document, so half of the actuator model is visible only to an outer loop objective, and a generated asset cannot enforce a velocity ceiling | verified in this session by enumerating every actuator attribute read within `quadruped/builder.py`, which finds `mass` and `peak_torque` alone |
| 18 | `Actuator.length` returns the motor body length excluding the gearbox stage, and is documented as a lower bound rather than a measurement | `quadruped/actuator.py:374` |
| 19 | The motor catalog file is never read under the shipped configuration, the loader preferring the COMPAct configuration, so the combined ranking the catalogs README maintains describes a list that is not live | `quadruped/actuator.py:176` against `quadruped/catalogs/README.md` |
| 20 | The declared gearbox ratio bounds and the ratios the published tables actually span disagree materially, the single stage topology declaring four to fifteen and publishing 4.45 to 7.00 | `quadruped/actuator.py:277` against the tables, measured in this session |

Two of these deserve emphasis over the rest. Entry 16 is the one that bears on this workspace's own gain derivations, since a design search that silently holds the armature fixed while varying the actuator produces a family of assets whose effective joint inertias are wrong in a way no reward will reveal. Entry 17 is the one that bounds what the generator can be asked to do, since a design whose merit rests on a fast actuator is indistinguishable in simulation from the same design with a slow one unless the outer loop applies the speed limit itself.

## 14. Rules for extending the generator

The following are offered as rules for a future change rather than as observations, and they follow from the architecture read above together with the workspace's own change discipline, which requires that no edit alter the behaviour of a caller that did not ask for the change.

A new design variable belongs in `variable_params.json` and in `DesignSpec`, and a new manufacturing constant belongs in `constant_params.json`. The test for which of the two a quantity is, is whether two designs within one search may legitimately disagree about it. Anything promoted from the second file to the first must be given a Python default equal to its former constant value, so that a specification which does not mention it reproduces the previous geometry exactly, and must be added to `vector`, `from_vector`, `labels`, `as_pairs`, `integrality_mask` and `sample` together or to none of them, those six being a single interface.

A new placement offset should be expressed as a rule derived base term plus a clearance, following the quadruped's pattern rather than the biped's, because a base term of zero cannot be checked against a geometric intent while a flush contact identity can. A new hard constraint belongs in `check_geometry` and should raise with the remedy named, as rule 2 does, rather than clip, because a clipped design is a design the optimiser believes it evaluated and did not.

A change that alters an emitted number for an unchanged specification is a change to every historical build and must be treated as such. Where the new behaviour can be carried by an optional argument whose default reproduces the old, it should be, and where it cannot, a second emitter should be added beside the first rather than the first edited, with the calling configuration repointed and the defect in the original recorded.

For a wheeled legged machine specifically, four consequences follow from the reading above. The chain is the easy part, a wheel being one further body below the shank carrying a continuous joint rather than a limited one, which means that the joint range machinery must learn to emit a joint with no range at all rather than a wide one, since a range of plus or minus a large number is not a continuous joint and will accumulate a limit stop. The mass model does not transfer, a wheel being a body whose mass is fixed rather than proportional to a length, so it belongs with the housings under an explicit inertial element and not with the distal links under the linear density inversion, and its inertia is genuinely that of a thick ring rather than a solid cylinder, so `cyl_inertia` would understate its polar moment. The contact model does transfer, the foot sphere already carrying its own friction, solver impedance and contact dimension, and a wheel would need the same treatment with a rolling friction term added. And the actuator model needs the half that is presently inert, because a wheel's merit is almost entirely a speed and power question rather than a torque question, which makes entry 17 of section 13 a prerequisite for this extension rather than a blemish upon it.

Finally, the home pose machinery should be reconsidered before it is extended rather than after. It was once an inverse kinematic solve, was withdrawn for good reasons recorded at `quadruped/builder.py:48`, and now leaves the baseline design's feet nearly three tenths of a metre above the floor. A wheeled machine's resting pose is determined by its wheel radius and cannot be hand set per design at all, so an extension will have to reintroduce a solve for the vertical offset even though the joint angles stay fixed, and the right shape for that is a derived spawn height computed from the chain rather than a free `body_height` field.

## 15. Bibliography

1. Singh, A., Kapa, D., Joshi, S., Kolathaya, S. (2026). COMPAct, Computational Optimization and Automated Modular design of Planetary Actuators. IEEE International Conference on Robotics and Automation (ICRA) 2026. arXiv:2510.07197. Official code at https://github.com/singhaman1750/COMPAct, MIT licensed, from which the artefacts under `catalogs/compact/` are vendored verbatim.
2. Google DeepMind. MuJoCo Menagerie, `unitree_a1` model. https://github.com/google-deepmind/mujoco_menagerie. Read in this session at `unitree_a1/a1.xml` on the `main` branch, which declares the three joint ranges and the 33.5 newton metre force range that `quadruped/constant_params.json` reproduces exactly.
3. Di Carlo, J., Wensing, P. M., Katz, B., Bledt, G., Kim, S. (2018). Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2018. DOI 10.1109/IROS.2018.8594448. The controller family the reference evaluator of section 11 drops each candidate design onto.
4. Todorov, E., Erez, T., Tassa, Y. (2012). MuJoCo, A physics engine for model-based control. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2012, pages 5026 to 5033. DOI 10.1109/IROS.2012.6386109. The engine whose model format this generator emits and whose compiler performs the density to mass integration that section 8.1 inverts.

## 16. Corrections established on 2026-09-22 during the integration drafting

This section is appended rather than folded into the text above, per the rule that a discovery contradicting an existing entry is recorded as a dated correction so that the reasoning which produced the earlier claim survives. Every figure below was computed by executing the submodule against the parameter sets recorded in [../plans/design_generator_integration.md](../plans/design_generator_integration.md) section 8.

Defect 2 of section 13 is withdrawn as a documentation defect and restated as a version skew. `quadruped/GEOMETRY_RULES.md` gives the mass per length constants as 0.879646 and 0.431027 against the shipped JSON's 3.24 and 1.48, and those two figures are exactly the donor asset's, being 0.187365 kilogrammes over 0.213 metres and 0.091809 over 0.213. The markdown therefore records the parameter set that produced `environments/assets/urdf/quadruped/quadruped.urdf`, and the JSON was changed afterwards. The table is stale with respect to the JSON and correct with respect to the asset, which is a different fault from the one entry 2 alleged.

Section 12's first gap is closed. Isaac Lab does carry a MuJoCo importer, `MjcfConverter` at `IsaacLab/source/isaaclab/isaaclab/sim/converters/mjcf_converter.py:20`, a configuration class at `mjcf_converter_cfg.py:13` and a spawner `MjcfFileCfg` at `from_files_cfg.py:136`, so the bridging step that section states as required is a conversion rather than a reimplementation. Two settings its URDF counterpart makes are left commented out in it, `set_make_default_prim` at `mjcf_converter.py:82` against `urdf_converter.py:129`, and `set_create_physics_scene` at `:84`.

Section 12's second gap is narrower than stated. The submodule's search space does not strictly contain the incumbent's only in the sense of being larger, it reaches the incumbent's machine exactly. The quadruped emitter reproduces `quadruped.urdf` to zero residual at six decimal places at every one of the twenty four leg body stations and to a total mass of 15.837077 kilogrammes against that asset's 15.837077, requiring only two non zero clearances and an injected foot inertial. The biped emitter reproduces `urdf/solefoot/tron1/base_robot.urdf` to within five hundredths of a millimetre at every leg body station, a foot collision box centre agreeing to 2.7 micrometres and a sole drop of 0.804356 metres against 0.804354, at a total mass of 21.318001 against 21.328000, the deficit of 0.01 being exactly the `limx_imu` link the emitted document replaces with a site.

Three emitter properties bear on any Isaac Lab use and none is recorded above. The quadruped's `foot_{LEG}_Link` at `quadruped/builder.py:380-382` carries a collision sphere of declared density zero and no `<inertial>`, which MuJoCo admits and a PhysX articulation does not. The emitted `<worldbody>` at `biped/builder.py:503-506` and `quadruped/builder.py:497-500` carries a directional light and a ten by ten metre ground plane, which a reference into a scene composes once per environment. And the quadruped names a foot collision geom `{LEG}` at `quadruped/builder.py:381` and a foot site `{LEG}` at `:383`, which MuJoCo admits because geoms and sites occupy separate name spaces and USD does not.

Two `DesignSpec` properties bear on any optimiser use and neither is recorded above. `vector` raises on a specification whose actuator references are catalogue names, which is the shipped `variable_params.json` of both families. And `from_vector` is not the identity on a round trip, returning every fixed constant field to its Python default, which on the shipped biped baseline moves `clearance_ankleAct2foot_x` from 0.01 to 0.0 and `_z` from 0.01 to 0.03.


## 17. Corrections and additions established on 2026-09-25, on applying the integration plan

This section is appended rather than folded into the text above, per the rule that a discovery contradicting an existing entry is recorded as a dated correction so that the reasoning which produced the earlier claim survives. Everything below was established by reading the Isaac Sim importer plugins line by line and by executing the generator against both donor URDFs, and the work it records is now applied to the tree. The full reading of the two conversion pipelines, drawn, walked and compared, is section 4.6 of [../plans/design_generator_integration.md](../plans/design_generator_integration.md), which remains the single home for it.

The Isaac Sim MuJoCo importer parses every geom's `density` attribute and never reads it again. `MjcfParser.cpp:175` is the only line in the plugin that touches the field. Body mass is taken solely from an `<inertial>` element, and where none exists the importer authors a body-level `physics:density` equal to the import configuration's own default (`MjcfUsd.cpp:395-426`), which Isaac Lab leaves at 0.0 (`mjcf_converter_cfg.py:16`) and which PhysX then reads as its own default of 1000 kg per cubic metre. Since the generator's entire mass model is a linear density realised through that attribute, every density-driven link of both families would have arrived in the simulator at a mass PhysX invented. The biped's thigh, a box of 0.05 by 0.032 by 0.25 metres, would have weighed 0.4 kg against the model's 1.5 kg. Defect 3 of section 13 recorded the massless quadruped foot as the only mass hazard, and it is the narrower case of this one.

The repair applied is an explicit `<inertial>` on every density-driven link, emitted unconditionally rather than only for Isaac Lab, because MuJoCo's own automatic inertia for a body carrying one non-zero-density geom is the same closed form about the same centre in the same frame. The emitted document therefore means the same thing in both engines, and the total mass of both families is unchanged in the MuJoCo path, which was confirmed by emitting each family in both modes and summing.

The importer creates a second articulation root. `addWorldGeomsAndSites` runs unconditionally and applies `UsdPhysicsArticulationRootAPI` and `PhysxSchemaPhysxArticulationAPI` to a `<root>/worldBody` prim whether or not the world body holds anything (`MjcfImporter.cpp:580-597`). Isaac Lab searches the subtree under `ArticulationCfg.prim_path` for that schema and raises when it finds more than one (`articulation.py:1533-1538`), so every asset this importer produces fails to initialise unless `articulation_root_prim_path` names the root outright. Both asset configurations now set it.

The importer places every body flat under the root body's own Xform. `CreatePhysicsBodyAndJoint` passes the same `rootPrimPath` down its recursion (`MjcfImporter.cpp:1183`, `MjcfImporter.cpp:1298-1301`), so a link prim is at `<asset>/<rootbody>/<bodyname>` regardless of its depth in the MuJoCo tree, while the joints scope stays at `<asset>/joints`. A box's dimensions are the referencing Xform's `xformOp:scale` in MuJoCo half-extents over a cube left at the schema default size of 2 (`MjcfUsd.cpp:820-826`), where the URDF importer writes full edge lengths over a cube of size 1. A jointless body survives as its own link joined by a `UsdPhysicsFixedJoint` named after the body rather than after a joint (`MjcfImporter.cpp:1353`), the `createBodyForFixedJoint` flag defaulting to True (`IMjcf.h:75`) and Isaac Lab not exposing it.

Four options of `MjcfConverterCfg` do not reach the importer. `make_instanceable` and `instanceableMeshUsdPath` are never read, instancing happening unconditionally instead (`MjcfImporter.cpp:533-535`, `MjcfImporter.cpp:1279`). `import_inertia_tensor` is commented out of its own condition (`MjcfUsd.cpp:398`). `merge_fixed_joints` has no MuJoCo counterpart at all. `import_sites` is overridden by a literal (`mjcf_converter.py:86`), so sites are always imported. The URDF converter's `set_create_physics_scene(False)` also has no counterpart, and the underlying default is True (`IMjcf.h:69`), so every asset carries its own `/physicsScene`.

Defect 2 of section 13, restated as a version skew on 2026-09-22, is now closed by measurement rather than by argument. Both families' parameter sets were derived by inverting the builders' own placement formulas against their donor URDFs and checked by walking the emitted document by forward kinematics. The quadruped reproduces `quadruped.urdf` at zero residual at all twenty four leg stations, exactly at every one of its twenty five link masses, exactly in total mass at 15.837077 kg, exactly in centre of mass, and to 0.026 per cent in whole-robot inertia about the centre of mass. The biped reproduces `base_robot.urdf` to 3.952e-07 m at every station, exactly at every link mass once the donor's `hip_{SIDE}_Link` bracket is folded into the knee housing and its `limx_imu` into the trunk, exactly in total mass at 21.328000 kg, and to 4.907 per cent in inertia, its centre of mass sitting 7.299 mm lower for three reasons that are each a builder convention rather than a parameter, the thigh and shank boxes being centred on their links' midpoints where the donor makes them flush at the proximal end, and the ankle housing being centred on its joint where the donor offsets it outboard by its own half-length.

The generator's trunk shell model reaches the donor's inertia only in the degenerate limit. Both donors model a trunk as a solid box, the biped's stated inertia of 0.043208, 0.045417 and 0.058542 matching the solid-box closed form for its dimensions and mass to three decimal places. The generator models it as a thick-walled shell plus a payload filling the cavity, which at a 3 mm wall returns half the donor's inertia and at 1 mm eighty one per cent of it, converging on the solid-box answer only as the wall vanishes. `TRUNK_SHELL_THICKNESS` is therefore set to 1e-06 for both families and `TRUNK_EXTRA_MASS` carries the whole trunk.

The `LINK_BOX_LENGTH_FRAC` of the biped is one constant serving two links of different ratio. The donor's thigh box is 0.25 m of a 0.30 m link and its shank box 0.30 m of a 0.35 m link, so the constant is set to 0.833333333, which matches the thigh exactly and leaves the shank box 8.3 mm short. The shorter box is inside the donor's envelope, so it can only reduce contact.

`CatalogActuator` and `Actuator` now carry the simulator-side numbers themselves rather than leaving them to a consumer. Ten optional fields were added to the former, a `rotor_inertia` and an `armature_value` among them, and both classes expose one `sim_params` method whose keys are exactly the attribute names of `IdentifiedActuator`. Section 13's defect that the armature is a flat per class constant rather than the reflected rotor inertia of the chosen actuator is thereby closed for any consumer that reads `sim_params`, the seven newly added identified entries each reproducing their configuration exactly, `TRON1-ABAD` for instance resolving 0.005589 from `6.9e-05 * 9.0 ** 2`. An entry that states neither a rotor inertia nor an armature resolves to zero, and the categorical generator names such entries at construction rather than letting a run discover it.

One divergence between the two morphology pathways was found while testing and is smaller than it sounds. The emitted document formats every float to six decimal places, where the in-place editor writes the generator's full-precision numbers, so a design reached by an in-place edit carries an inertia up to 5e-07 kg m squared different from the same design reached by a respawn. The tolerance is three orders of magnitude below the smallest link inertia either family carries.

## 18. The importer extension, established on 2026-09-29 on the first headless conversion

The first attempt to spawn the emitted document inside a training run failed before any geometry was read, with `AttributeError: 'NoneType' object has no attribute 'set_import_sites'` raised at `IsaacLab/source/isaaclab/isaaclab/sim/converters/mjcf_converter.py:86`, preceded in the Kit log by `Can't execute command: "MJCFCreateImportConfig", it wasn't registered or ambigious`. The two messages are one fault. `_get_mjcf_import_config` obtains its configuration object by dispatching a Kit command at `mjcf_converter.py:77`, and `omni.kit.commands.execute` answers an unregistered name by logging that line and returning a pair whose second member is `None`, which the next statement dereferences.

The command is registered by the `isaacsim.asset.importer.mjcf` extension, and whether that extension is loaded depends entirely on which experience file the application was launched with. The windowed experience lists it at `IsaacLab/apps/isaaclab.python.kit:40`, beside the pinned URDF importer at `:41`. The headless experience at `IsaacLab/apps/isaaclab.python.headless.kit` carries two dependency blocks, at lines 23 and 196, and neither names any importer, while `isaaclab.python.headless.rendering.kit` inherits from it and so inherits the omission. `AppLauncher` selects the headless file at `IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py:739` whenever the headless flag is set without cameras, and the headless rendering file at `:725` when cameras are enabled as well, so every training run reaches a converter whose command was never registered.

The reason this was never seen under the retired URDF pathway is an asymmetry between the two converters rather than a property of the descriptions. `UrdfConverter.__init__` enables its own extension before doing anything else, at `IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py:65-68`, and then acquires the importer interface at `:71-73`. `MjcfConverter.__init__` did neither, passing straight to `AssetConverterBase.__init__`, which calls `_convert_asset` at `IsaacLab/source/isaaclab/isaaclab/sim/converters/asset_converter_base.py:106` on the first conversion of a given hash. The URDF pathway therefore carried its own extension into a headless application and the MJCF pathway relied on an experience file that does not provide it.

The repair adopts the sibling class's own idiom, four guarded lines in `MjcfConverter.__init__` that fetch the extension manager and enable `isaacsim.asset.importer.mjcf` immediately where it is not already enabled. It is placed in the converter rather than in the experience files because the converter is the one place every caller passes through, where the experience files would need amending once for the headless variant, once for the headless rendering variant and again for each further experience a later run selects. It is guarded, so an application that already loaded the extension, the windowed one among them, is unaffected. The change is local to the vendored Isaac Lab tree and will not survive a fresh checkout of it, so it belongs in the container's patch set rather than in this repository.

One contingency remains open. Enabling an extension that the local extension cache does not hold sends Kit to its registry, which fails on a machine without network access, so a first run on a fresh image may report that the extension cannot be resolved rather than that its command is missing. The check is to ask the extension manager for the extension before converting anything, which distinguishes the unregistered command from an absent extension and from the second reading of the Kit message, an ambiguous registration arising where a legacy `omni.importer.mjcf` is enabled beside the current one.

The repair was subsequently found to be the upstream one, established on 2026-09-29 by reading the fetched refs of the vendored clone. Isaac Lab closed this defect on 2026-05-30 in pull request 5889, whose description states that it "Adds missing isaac sim extension enable call for MJCF importer", and whose diff to `mjcf_converter.py` adds an `omni.kit.app` import and the same three guarded lines, the extension manager fetched, `is_extension_enabled` asked and `set_extension_enabled_immediate` called where it answers false. The commit also carries a changelog fragment and a regression test, `test_converter_enables_importer_extension`, which constructs a converter with the extension disabled and asserts that the extension is enabled afterwards. On 2026-07-22 pull request 6499 refactored the three lines into a single call of a helper, `enable_extension` imported from `isaaclab.sim.utils`, which changes the shape and not the behaviour.

The reach of that fix is the reason this workspace met the defect at all. It sits on `develop` and on `release/3.0.0`, both at version 3.0.0, and on the `v3.0.0-beta2` tags. It is absent from `main`, at version 2.3.2 and last advanced on 2026-07-24, and from `release/2.3.0`, at version 2.3.1, where `MjcfConverter.__init__` still passes straight to its parent. This checkout stands at version 2.3.0 on a fork of the main line, so no advance within the two three series would have acquired the fix, and the three lines had to be carried back rather than pulled forward. Upstream also declined to amend the experience files, `develop` still listing no importer in `apps/isaaclab.python.headless.kit`, which settles the question of where the repair belongs.

One consequence bears on section 4.1 of the integration plan. The three series converter is not the two three one with a fix applied, having been reworked for an importer that runs without Kit, in pull requests 6460 and 6824, which also changes the converted file to `{basename}/{basename}.usda` and states in its own docstring that several `MjcfConverterCfg` options the plan enumerated are ignored by the new importer. Any later move to three zero therefore requires the option audit of that section to be redone rather than assumed.
