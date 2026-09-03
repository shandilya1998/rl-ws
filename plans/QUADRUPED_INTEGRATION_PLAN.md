# QUADRUPED_INTEGRATION_PLAN.md

Status, OUTSTANDING as of 2026-08-19, no phase implemented. Sub-task 1.6, the `setup.py` package rename, is the sole exception and is already applied, recorded here for completeness and marked complete in its own section.

## 1. Introduction

The simulation repository was built to train bipedal locomotion policies for one robot family, the LimX TRON1 in its solefoot, pointfoot and wheelfoot variants, and was later extended to a second biped, SD_BRS1. Its every naming decision records that origin, the extension directory being called `bipedal_locomotion`, the shared Markov decision process package sitting beneath `tasks/locomotion/mdp`, and the environment configuration tree being partitioned by foot type rather than by morphology. The project now requires a quadruped, supplied as the MuJoCo model `my_design.xml`, to be trained by the same three runners the bipeds use, the vanilla proximal policy optimisation runner, the hybrid internal model runner in `himloco/`, and the co-optimisation runner in `co_optimisation/`. Nothing about that requirement is peculiar to four legs at the level of the learning algorithm, so the work divides cleanly into two halves, a generalisation of the repository so that it no longer asserts bipedality in its names, and the addition of one robot with its asset, its actuator identification, its environment configuration and its task registrations.

This document plans both halves. It establishes first what the rename has and has not accomplished, since every later citation depends on the path being settled, then derives the physical model of the quadruped from the supplied MuJoCo file and validates it against the primitive geometry it declares, then computes the actuator gains and velocity ranges from that model by the method already applied to the bipeds, then enumerates what of the existing Markov decision process package transfers unaltered, and finally specifies the environment configuration, the configuration class hierarchy, the task registrations and the container tooling changes that make the quadruped trainable. The literature survey in section 3 governs the reward configuration alone, since that is the only part of the specification where the biped precedent is an unreliable guide.

The change discipline of `../CLAUDE.md` governs throughout. No function in `environments/environments/tasks/locomotion/mdp/` is edited in place, because SD_BRS1 and the three TRON1 tasks read those functions and an in place edit would invalidate comparison against runs already flown. Every quadruped specific behaviour is instead carried by an argument the calling configuration sets, or by a new file that no existing configuration imports.

## 2. Task 1, completing the package rename

### 2.1 What the rename has accomplished

The two directory moves are already staged in the index. Git reports `exts/bipedal_locomotion/config` moved to `environments/config`, `exts/bipedal_locomotion/docs` to `environments/docs`, and the whole of `exts/bipedal_locomotion/bipedal_locomotion/` to `environments/environments/`, with `exts/` now an empty directory. The installation script has also been updated, `environments/setup.py` declaring `name="environments"` and `packages=["environments"]`.

### 2.2 What the rename has not accomplished

The Python package is still named `bipedal_locomotion` at every point where the name is written rather than implied by a directory. Forty one import statements across the repository, the extension manifest, the isort configuration and the container tooling all still address a package that no longer has a directory to occupy, so the tree as it stands does not import. The following table enumerates every site, established by grepping the working tree with the build and egg-info directories excluded.

| File | Lines | Nature of the reference |
|---|---|---|
| `environments/config/extension.toml` | 10, 14, 24 | Extension title, repository URL, and the declared `[[python.module]] name` |
| `environments/setup.py` | 1 | Module docstring naming the package |
| `environments/environments/ui_extension_example.py` | 6, 17, 41 | Log prefixes |
| `environments/environments/actuators` importers, `assets/config/solefoot_identified_cfg.py` | 7 | `from bipedal_locomotion.actuators import IdentifiedActuatorCfg` |
| `environments/environments/assets/config/sd_brs1_identified_cfg.py` | 6 | Same import |
| `environments/environments/tasks/locomotion/cfg/PF/limx_base_env_cfg.py` | 23 | `from bipedal_locomotion.tasks.locomotion import mdp` |
| `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py` | 23, 24 | `mdp` and `curriculums` imports |
| `environments/environments/tasks/locomotion/cfg/SF/brs_base_env_cfg.py` | 21, 22 | `mdp` and `curriculums` imports |
| `environments/environments/tasks/locomotion/cfg/SF/limx_berkeley_env_cfg.py` | 26 | `mdp` import |
| `environments/environments/tasks/locomotion/cfg/WF/limx_base_env_cfg.py` | 21 | `mdp` import |
| `environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py` | 11, 14 | Symmetry and wrapper imports |
| `environments/environments/tasks/locomotion/robots/__init__.py` | 3 | Agent configuration import |
| `environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py` | 9, 10, 14, 15, 20, 23 | Six imports |
| `environments/environments/tasks/locomotion/robots/limx_pointfoot_env_cfg.py` | 5, 6, 7, 15 | Four imports |
| `environments/environments/tasks/locomotion/robots/limx_wheelfoot_env_cfg.py` | 5, 6, 7, 15 | Four imports |
| `environments/environments/tasks/locomotion/robots/brs_solefoot_env_cfg.py` | 8, 12, 16 | Three imports |
| `co_optimisation/co_optimisation/runners/usd_generator.py` | 73 | Imports the four TRON1 actuator configurations |
| `co_optimisation/tests/cmaes_design_generator_test.py` | 24, 25 | Absolute path to the TRON1 URDF through the old `exts/` tree |
| `scripts/rsl_rl/train.py` | 96, 200 | Wrapper import, and an absolute URDF path through the old `exts/` tree |
| `scripts/rsl_rl/play.py` | 84, 85, 658, 692 | Package import, wrapper import, two absolute URDF paths |
| `scripts/scratchpad/*.py` | various | Four scratch scripts holding stale paths |
| `pyproject.toml` | 58 | `known_firstparty = "bipedal_locomotion"` |
| `../djinn` | 145, 187, 203 | `pip install -e biped/exts/bipedal_locomotion` |

### 2.3 Prescribed edits

Sub-task 1.1, rewrite every Python import. The transformation is purely lexical, `bipedal_locomotion.` becoming `environments.` at the head of an import path, and it is safely applied by the following command run from the repository root. The `build/` and egg-info trees are excluded because they are regenerated artefacts, and `scripts/scratchpad/` is included because a stale scratch script is worse than a deleted one.

```bash
cd /ws/tron1-rl-isaaclab-cozum
grep -rl 'bipedal_locomotion' --include='*.py' environments/environments co_optimisation scripts \
  | grep -v '/build/' | grep -v 'egg-info' \
  | xargs sed -i 's/\bfrom bipedal_locomotion\./from environments./g; s/\bimport bipedal_locomotion\b/import environments/g'
```

Sub-task 1.2, rewrite the four absolute URDF paths. These are container paths of the form `/workspace/isaaclab/biped/exts/bipedal_locomotion/bipedal_locomotion/assets/...` and must become `/workspace/isaaclab/biped/environments/environments/assets/...`. Note in passing that two of them, `scripts/rsl_rl/play.py:658` and `:692`, additionally point at `urdf/solefoot/base_robot.urdf` whereas the file actually sits at `urdf/solefoot/tron1/base_robot.urdf`, so these paths were already dead before the rename and must be corrected in the same pass.

```bash
cd /ws/tron1-rl-isaaclab-cozum
sed -i 's#exts/bipedal_locomotion/bipedal_locomotion/#environments/environments/#g; s#urdf/solefoot/base_robot.urdf#urdf/solefoot/tron1/base_robot.urdf#g' \
  scripts/rsl_rl/train.py scripts/rsl_rl/play.py co_optimisation/tests/cmaes_design_generator_test.py
sed -i 's#/exts/bipedal_locomotion/#/environments/#; s#^\s*"bipedal_locomotion/#    "environments/#' co_optimisation/tests/cmaes_design_generator_test.py
```

Sub-task 1.3, update `environments/config/extension.toml`. Line 24 must read `name = "environments"`, since Isaac Sim's extension loader resolves the Python module by that field and would otherwise fail to import the extension. Line 10 becomes `title = "quadruped_and_bipedal_locomotion_isaaclab"` and the description on line 13 becomes `description="Locomotion tasks for legged robots in IsaacLab"`, both being cosmetic but both asserting bipedality where the repository no longer intends it. The `repository` field on line 14 is an upstream provenance record and is left untouched.

Sub-task 1.4, update `pyproject.toml:58` to `known_firstparty = "environments"`, so that isort continues to group the repository's own imports.

Sub-task 1.5, update `../djinn`. Three lines install the package by its old path, and all three become `biped/environments`.

```bash
sed -i 's#biped/exts/bipedal_locomotion#biped/environments#g' /ws/djinn
```

Sub-task 1.6, `environments/setup.py`. COMPLETE. The script already declares `name="environments"` and `packages=["environments"]`. Only the docstring on line 1 remains, and it is covered by sub-task 1.7 below. One latent defect is recorded rather than repaired, `packages=["environments"]` lists the top level package alone and not its sub-packages, so a non editable install would ship an empty package. Every install performed by `djinn` uses `pip install -e`, under which the path entry makes the whole tree importable regardless, so the defect is inert today and is left standing deliberately. Should a wheel ever be built, the field must become `packages=find_packages()`.

Sub-task 1.7, cosmetic strings. `environments/setup.py:1` and the three log prefixes in `ui_extension_example.py` name the old package in human readable text. Rewrite them to `environments`.

Sub-task 1.8, purge the stale editable install. This is the step most easily forgotten and the one whose omission produces the most confusing failure, since a container that already has `bipedal_locomotion` installed editable will keep resolving the old name from a `.pth` entry pointing at a directory that no longer exists. Inside the container, run `pip uninstall -y bipedal_locomotion` before the first reinstall (This must be run in the runtime container, whereas all tasks are performed in the development container), and delete `environments/bipedal_locomotion.egg-info/` and `environments/build/` from the working tree so that no regenerated metadata reintroduces the name.

Sub-task 1.9, documentation. The following files reference the old path in prose and must be updated, `README.md`, `GEMINI.md`, `ARCHITECTURE.md` and `context/README.md` within the repository, and `../ARCHITECTURE.md`, `../CO_OPTIMISATION.md`, `../CLAUDE.md`, `../context/knowledge_base.md`, `../context/isaaclab_env.md`, `../context/gait_metrics.md`, `../context/copt_ppo_nonstationarity.md`, `../context/sd_brs_urdf.md`, `../context/literature.md`, `../context/brs_gait.md` and the five workspace plans `COPT_LEARNED_MODEL.md`, `SYMMETRY_PLAN.md`, `HIMLOCO_INTEGRATION_PLAN.md`, `GAIT_STATISTICS_PLAN.md`, `GAIT_STRATEGY.md`, `NATURAL_GAIT_PLAN.md` and `LIPM_REWARD.md` at the workspace level. Two of these carry a further defect. `../CLAUDE.md` clause 3 names the `bipedal_locomotion` mdp package as the shared module whose callers must be enumerated before editing, and must now name `environments`. `context/README.md` closes by directing the reader to `../exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/tron1/base_robot.md`, a file the rename deleted, so that pointer must be removed and the observation it supported retained.

## 3. Literature survey, the reward configuration

### 3.1 Why the survey is confined to the rewards

The observation set, the action interface, the command sampler, the terminations and the domain randomisation of the existing biped configuration are all robot agnostic in their construction and require only that a body name or a sensor be renamed. The reward configuration is not, because the biped set contains eight terms whose entire purpose is to shape a two legged gait, and because the weights of the terms that do transfer were tuned against a biped whose failure modes differ. The survey therefore establishes what the canonical quadruped configurations weight each shared term at, what they add that this repository does not have, and where the biped's tuned budget has drifted from the field's consensus.

### 3.2 A note on scale

Every weight quoted from the literature is on the scale of `legged_gym`, in which the linear velocity tracking reward carries a scale of 1.0 [1]. This repository carries the same term at a weight of 25 in `cfg/SF/limx_base_env_cfg.py:1140`, and both frameworks multiply the term by the control period before accumulating it, so the two scales differ by a uniform factor of twenty five and a literature weight of `w` corresponds to a weight of `25w` here. That factor is stated once and applied where it is used, since quoting a canonical weight without it invites a hundredfold error.

One consequence of the scale is that the biped configuration is not uniformly twenty five times the canon. The angular velocity tracking reward stands at 7.5 against a canonical 12.5, so the linear to angular ratio is 3.33 to 1 where every reference retrieved uses 2 to 1 [1][2][5]. The vertical velocity penalty stands at -0.5 against a canonical -50, a hundredfold divergence. The angular velocity penalty about the horizontal axes stands at -0.05 against a canonical -1.25. These divergences are recorded rather than corrected wholesale, for the reason given in section 3.5.

### 3.3 Canonical weights for the terms already selected

The following table gives, for each term the specification retains, the value the canonical references use, and the value this plan proposes for the quadruped. The canonical column is on the `legged_gym` scale and the proposal column is on this repository's scale.

| Term | Biped weight here | Canonical | Proposed | Justification |
|---|---|---|---|---|
| `keep_balance` (`stay_alive`) | 0.05 | absent from every reference [1][2][3][6][7] | 0.05 | Retained unchanged. The references achieve the same protection through `only_positive_rewards`, which this framework does not implement, so the alive bonus is the mechanism available |
| `rew_lin_vel_xy` | 25, std 0.4 | 1.0, sigma 0.5 [1], 1.5 for Go2 [2] | 25, std 0.4 | Unchanged, it defines the scale |
| `rew_ang_vel_z` | 7.5, std 0.4 | 0.5 [1], 0.75 for Go2 [2] | 12.5, std 0.4 | Restores the 2 to 1 ratio universal across the references. Safe to change, see section 3.6 |
| `pen_lin_vel_z` | -0.5 | -2.0 [1][2][5], -1.0 [7] | -2.0 | A fourfold increase rather than the hundredfold the canon implies, see section 3.5 |
| `pen_ang_vel_xy` | -0.05 | -0.05 invariant across all references [1][2][3][5] | -0.5 | A tenfold increase, likewise partial |
| `pen_joint_torque` | -8e-5 | -1e-5 for ANYmal, -2e-4 for Go2 in two independent ports [2][5] | -2e-4 | The Go2 value, this robot sharing Go2's 0.213 m segments and near identical torque ceilings |
| `pen_joint_accel` | -5e-7 | -2.5e-7 invariant across all references [1][2][3][7] | -2.5e-7 | Adopted outright, this being the single most stable constant in the survey |
| `pen_action_rate` | -0.01 | -0.01 [1][2][5], -0.1 [7] | -0.05 | Between the two, the quadruped's higher joint count raising the aggregate |
| `pen_joint_pos_limits` | -2.0 | 0 in the ETH lineage, -10.0 in Go2 and Walk These Ways [5][6] | -2.0 | Retained, the soft limit factor of 0.9 doing most of the work |
| `pen_undesired_contacts` | -0.5 | -1.0 [1][2][5], -10.0 for parkour [7], removed for Go2 rough [2] | -2.5 | Raised, a quadruped's thighs and shanks meeting terrain far more often than a biped's |
| `pen_action_smoothness` | -0.075 | present only in Walk These Ways, weight not recovered [6] | -0.075 | Retained unchanged, no canonical value exists to prefer |
| `pen_flat_orientation` | -1.0 | 0 on rough in the ETH lineage, -0.2 [3], -1.0 [7], -5.0 on flat [6] | -1.0 | Retained. The rough terrain band is -0.2 to -1.0 and the flat value is far higher, so the single value is a compromise the flat variant may raise |
| `pen_joint_vel_l2` | -5e-5 | 0, folded into a power term at -2e-5 [3][4] | -5e-5 | Retained. The specification names this term `pen_joint_vel_v2`, which does not exist, the tree defining `pen_joint_vel_l2` over `mdp.joint_vel_l2` |
| `feet_air_time` | 2.0, min 0.2, max 0.5 | 1.0 threshold 0.5 [1], 0.125 rough [2], 0.01 for Go2 rough [2] | 2.0, min 0.25, max 0.40 | Threshold retuned, see section 3.4 |
| `feet_slide` | -0.25 | present but unwired in Isaac Lab, present in Walk These Ways [2][6] | -0.25 | Retained unchanged |

### 3.4 The air time threshold, and why zero is the wrong value

The specification proposes a minimum threshold of zero for the quadruped. The survey establishes that this is the one value the term must not take. With `threshold_min` at zero the reward reduces to the air time itself gated on first contact, which is non negative and monotonically increasing, so every additional millisecond aloft is paid for and nothing whatever penalises a foot that hangs. The positive threshold is precisely the mechanism that supplies the lower branch, a foot returning sooner than the threshold earning a negative contribution and thereby being pushed toward lift off, and removing it leaves a term that on a light robot is maximised by hopping [1].

The canonical 0.5 seconds is not arbitrary either. It is half the stride period of ANYmal's roughly one hertz trot, so it is the swing duration of the gait the term is meant to elicit at a duty factor of one half [1]. The same derivation at the two hertz trot this quadruped's scale suggests gives a stride period of 0.5 seconds and a swing duration of 0.25 seconds, which is the value this plan adopts for `threshold_min`. The existing `feet_air_time` in this repository additionally clamps the excess at `threshold_max - threshold_min`, a refinement the canonical form lacks, and 0.40 seconds is proposed for the ceiling so that the term saturates at 1.6 times the target swing rather than paying without limit.

### 3.5 Why the penalty rebalancing is partial

Adopting the canonical penalties outright, which at this repository's scale would mean -50 for the vertical velocity and -1.25 for the horizontal angular velocity, would be defensible in `legged_gym` and is not defensible here, because every reference that carries those weights also sets `only_positive_rewards`, which clips the summed reward at zero and so prevents a policy from discovering that termination is cheaper than locomotion [1][2][7]. This framework has no such clip, and the alive bonus of 0.05 is the whole of the protection. A hundredfold increase in the dominant penalty against an unclipped return is therefore a change whose failure mode is a policy that lies down, and it must not be made blind.

The proposal accordingly moves each divergent penalty part of the way, by a factor of four for the vertical velocity and ten for the horizontal angular velocity, and records the canonical target so that a later pass may close the gap deliberately once a first policy has trained. Should the trained gait bounce vertically, the vertical velocity penalty is the first weight to raise, and the canonical -50 is the ceiling to raise it toward.

### 3.6 Weight changes are safe with respect to the curriculum

One objection to rebalancing the tracking weights would be that the curriculum keys off them. It does not, in the sense that matters. Every curriculum term that reads a reward compares the episode mean against `update_threshold * term_cfg.weight * env.step_dt`, at `mdp/curriculums.py:302`, `:436` and the corresponding lines of the other three, so the comparison is against a fraction of the maximum the term can attain and is invariant under a change of weight. Raising `rew_ang_vel_z` from 7.5 to 12.5 therefore leaves `modify_command_velocity_ang_z` and `modify_angular_tracking_reward_std` behaving exactly as before.

### 3.7 Terms the survey recommends adding

Foot clearance during swing. Present and weighted in both HIMLoco and DreamWaQ at -0.01 against a commanded clearance, and implemented in Walk These Ways as a phase gated squared error against a swing height command offset by the foot radius [3][4][6]. It is the standard mechanism against toe stubbing on rough terrain, which air time does not police because air time rewards duration and says nothing about trajectory. This repository already implements it, as `mdp.foot_clearance_reward` at `mdp/rewards.py:234`, with an optional contact gate. It is nonetheless left out of the configuration, section 8.9 giving the reason, which is that the implementation measures a world frame height and is therefore correct on flat ground alone.

Abduction joint deviation from the default pose. Present in Extreme Parkour as `hip_pos` at -0.5, a squared deviation of the abduction joints alone from the default configuration [7], and structurally the same term appears in the Unitree humanoid configuration [5]. It is not universal, being absent from the ANYmal, Go2 and HIMLoco configurations retrieved, but it addresses a failure mode this robot is specifically exposed to, namely that the abduction joints are the first degree of freedom a policy widens to buy itself a larger support polygon, and this robot's stance is only 0.264 metres wide against a 0.426 metre leg. This repository has the function already, `mdp.joint_deviation_l1` reaching the configuration through the Isaac Lab star import chain at `mdp/__init__.py:1`. Its use is prescribed in section 8.9.

### 3.8 Terms the survey recommends against

A gait or phase symmetry term, of the kind Walk These Ways and Siekmann et al. implement, requires a phase clock, a desired contact schedule and per command gait parameters, and is machinery rather than a reward line [6][8]. None of the plain tracking baselines carries it and the air time term already supplies a soft periodicity signal, so it is deferred. A base acceleration penalty is a biped motivated term addressing double support hopping and appears in none of the quadruped configurations retrieved [9]. A stumble penalty is present in the base `legged_gym` function set at a scale of zero and revived only for parkour [1][7], and it largely detects the same event the slide and contact penalties already price. A power penalty is an alternative parameterisation of the torque and joint velocity terms already carried rather than an addition [3][4]. A termination penalty is switched off in every reference, the clipping mechanism serving in its place [1].

### 3.9 Command ranges, curricula and terrain

The command ranges of the biped configuration, namely a linear x range of -1.0 to 1.0, a linear y range of -0.5 to 0.5 and an angular z range of -0.75 to 0.75, sit inside the band the survey establishes for a robot of this class, which is -1.0 to 1.5 in x, -0.5 to 1.0 in y and -1.0 to 1.5 in z [1][3][6]. They are adopted unchanged, the existing curricula widening them.

Two curriculum precedents bear directly. Isaac Lab's own Go2 port scales the terrain generator's difficulty parameters down explicitly because the robot is small, taking the box height range to 0.025 to 0.1 metres and the random rough noise to 0.01 to 0.06 metres [2], which establishes that obstacle amplitude must be parameterised against leg length rather than inherited from an ANYmal tuned generator. The same port disables the push event outright on rough terrain [2], on the reasoning that a light robot already receives ample perturbation from the terrain. This plan adopts the first in full, prescribing a quadruped specific terrain configuration in section 8.2, and adopts the second in part, retaining the push event but at half the biped's magnitude, since the push curriculum is one of the mechanisms the co-optimisation experiments rely upon and removing it would make the quadruped runs non comparable with the biped runs.

### 3.10 The base height target

The survey brackets this robot's proposed standing ratio from both sides. The A1 configuration in HIMLoco targets 0.30 metres over a 0.40 metre leg, a ratio of 0.75, and the Go1 configuration in Walk These Ways targets 0.34 metres, a ratio near 0.80 [3][6], while Go2, which shares this robot's 0.213 metre segments exactly, targets 0.25 metres over a 0.426 metre leg, a ratio of 0.587 [2][5]. The MuJoCo keyframe places the trunk at a ratio of 0.634, between the Go2 precedent and the A1 one, so the target requires no adjustment on the literature's evidence. It requires an adjustment of 0.022 metres on the evidence of the model itself, for the reason given in section 4.3.

## 4. Task 2, the quadruped asset

### 4.1 Reading the MuJoCo model

`my_design.xml` declares a trunk carrying four identical legs, each of six bodies, for twenty five bodies and twelve actuated degrees of freedom in total. Every body's inertial properties are given explicitly rather than inferred from density, and every geometry is an analytic primitive, so the model is fully determined and the conversion has no fidelity to lose.

The joint naming of the MuJoCo model does not match this repository's convention and the mismatch is the single most consequential fact of the conversion. The MuJoCo joint `FR_hip_joint` has axis `1 0 0`, which is the roll axis, and is therefore the abduction and adduction degree of freedom that TRON1 calls `abad`. The MuJoCo joint `FR_thigh_joint` has axis `0 1 0` and is the pitch degree of freedom that TRON1 calls `hip`. The MuJoCo joint `FR_calf_joint` is TRON1's `knee`. A conversion that carries the MuJoCo names across would therefore place the abduction joint under the name `hip`, and every reward, event and observation that selects joints by regular expression would then act on the wrong axis. The mapping is recorded in the table below and must be applied.

| MuJoCo joint | Axis | Physical degree of freedom | URDF joint |
|---|---|---|---|
| `FR_hip_joint` | `1 0 0` | Abduction and adduction, roll | `abad_FR_Joint` |
| `FR_thigh_joint` | `0 1 0` | Hip flexion and extension, pitch | `hip_FR_Joint` |
| `FR_calf_joint` | `0 1 0` | Knee flexion and extension, pitch | `knee_FR_Joint` |

### 4.2 Link topology and naming

The requested convention is `[abad/hip/knee]_[FR/FL/RR/RL]_[thigh/actuator]_[Link/joint]`. Two departures from the literal form are proposed. The joint suffix is `_Joint` with a capital letter for the twelve actuated joints, because that is what TRON1 uses at `assets/urdf/solefoot/tron1/base_robot.urdf` and consistency with TRON1 was the stated reason for the convention, while the fixed joints take a lowercase `_joint` exactly as TRON1's `abad_R_fixed_joint` and `knee_R_actuator_joint` do. The foot takes the name `foot_FR_Link` rather than a form built from a joint name, because the convention has no slot for a terminal segment and because a feet selecting expression of `foot_.*_Link` is then disjoint from the `abad_.*`, `hip_.*` and `knee_.*` expressions that the undesired contact penalty needs.

The resulting chain, given for the front right leg and repeated for the other three, is as follows.

| URDF link | MuJoCo body | Parent joint | Type | Origin in parent |
|---|---|---|---|---|
| `base_Link` | `trunk` | root | free | root |
| `abad_FR_actuator_Link` | `FR_abd_act` | `abad_FR_fixed_joint` | fixed | `0.105 -0.052 0` |
| `hip_FR_actuator_Link` | `FR_hip_act` | `abad_FR_Joint` | revolute, axis `-1 0 0`, mirrored, see below | `0.071 0 0` |
| `knee_FR_actuator_Link` | `FR_knee_act` | `hip_FR_Joint` | revolute, axis `0 1 0` | `0 -0.050 0` |
| `hip_FR_thigh_Link` | `FR_thigh` | `hip_FR_thigh_joint` | fixed | `0 -0.030 0` |
| `knee_FR_Link` | `FR_calf` | `knee_FR_Joint` | revolute, axis `0 1 0` | `0 0 -0.213` |
| `foot_FR_Link` | `FR_foot` | `foot_FR_joint` | fixed | `0 0 -0.213` |

The abduction axes are mirrored between the two sides, and this is the third correction applied to `my_design.xml`. The MuJoCo model gives all four legs the axis `1 0 0`, whereas TRON1 gives `abad_R_Joint` the axis `-1 0 0` and `abad_L_Joint` the axis `1 0 0`, at `assets/urdf/solefoot/tron1/base_robot.urdf:121` and `:393`, over an identical limit range of -0.38397 to 1.39626. The reason for TRON1's convention is a sign semantic rather than a kinematic one. A downward pointing leg rotated by a positive angle about `1 0 0` swings its foot toward positive `y`, which is outward for a left leg and inward for a right one, so an unmirrored model makes a positive command mean abduction on one side and adduction on the other. Mirroring makes positive mean outward on both.

Three things follow from adopting it. A laterally symmetric stance takes equal abduction values on all four legs rather than opposed ones, which is what the initial joint position of section 6 already assumes. A left to right symmetry augmentation becomes a pure permutation of the twelve joint states rather than a permutation composed with a sign flip on four of them, which is what any future augmentation for this robot will want, `mdp/symmetry/brs.py` being written for SD_BRS1 alone today. And the two robots' abduction states carry the same meaning, so a reward or an event written against one transfers to the other without a hidden sign.

The mirroring costs nothing in reachable configuration, the MuJoCo range of -1.0 to 1.0 being symmetric about zero, so the limits transcribe unchanged and only the axis vector differs. The URDF gives `abad_FR_Joint` and `abad_RR_Joint` the axis `-1 0 0` and `abad_FL_Joint` and `abad_RL_Joint` the axis `1 0 0`, and `my_design.xml` receives the same change on its `FR_hip_joint` and `RR_hip_joint`. The pitch axes are already consistent, `0 1 0` on all four legs meaning forward flexion everywhere, and are carried across unaltered.

### 4.3 Inertia validation, and three corrections to the MuJoCo model

Every inertial entry of the MuJoCo model was recomputed from its declared primitive and its declared mass. The MuJoCo `size` attribute of a box is a half extent triple and of a cylinder is a radius with a half length, so the full dimensions are twice those figures, a conversion that is itself a common source of a factor of four in an inertia. The formulae are those of `../tron1-rl-isaaclab-cozum/context/tron1.md`, namely `Ixx = m(b^2+c^2)/12` and its cyclic permutations for a box, `I_axial = m r^2/2` with `I_transverse = m(3r^2+L^2)/12` for a cylinder, and `I = 2 m r^2 / 5` for a sphere.

| Link | Primitive | Mass | Declared diagonal | Recomputed | Verdict |
|---|---|---|---|---|---|
| `base_Link` | Box 0.170 by 0.196 by 0.092 | 0.674397 | 0.002635, 0.002100, 0.003783 | 0.002635, 0.002100, 0.003783 | Exact |
| `abad_*_actuator_Link` | Cylinder r 0.046, L 0.040, axis X | 1.028375 | 0.001088, 0.000681, 0.000681 | 0.001088, 0.000681, 0.000681 | Exact |
| `hip_*_actuator_Link` | Cylinder r 0.046, L 0.040, axis Y | 1.028375 | 0.000681, 0.001088, 0.000681 | 0.000681, 0.001088, 0.000681 | Exact |
| `knee_*_actuator_Link` | Cylinder r 0.046, L 0.040, axis Y | 1.434746 | 0.000950, 0.001518, 0.000950 | 0.000950, 0.001518, 0.000950 | Exact |
| `hip_*_thigh_Link` | Cylinder r 0.020, L 0.213, axis Z | 0.187365 | 0.000727, 0.000727, 0.000037 | 0.000727, 0.000727, 0.000037 | Exact |
| `knee_*_Link` | Cylinder r 0.014, L 0.213, axis Z | 0.187365 | 0.000727, 0.000727, 0.000037 | 0.000352, 0.000352, 0.000009 | Wrong, at a mass of 0.091809 |
| `foot_*_Link` | Sphere r 0.022 | 0.020000 | 0.000004 on all three | 0.000004 | Exact to the stated precision |

The audit establishes the model's own convention before it establishes its errors, and the convention is decisive. Dividing the thigh's declared mass of 0.187365 by the volume of a solid cylinder of radius 0.020 and length 0.213 gives 700.0015 kilogrammes per cubic metre, which is 700 to five significant figures, and its declared axial term of 0.000037 is exactly `m r^2 / 2` for that radius. The leg segments were therefore dimensioned as cylinders at a uniform density of 700, and the `capsule` geometry type the file declares for them is the inconsistency rather than the inertia. Three corrections follow, and all three are applied to `my_design.xml` and to the URDF together, since the specification requires the generated inertia to be correct and a correction applied to one file alone would open a silent divergence between the two models.

The first correction is the calf. It carries the thigh's mass and the thigh's inertia verbatim, although its declared radius is 0.014 and not 0.020. At the model's own density of 700 the calf's mass is 0.091809 kilogrammes and its diagonal is 0.000352, 0.000352, 0.000009. The declared mass is therefore too large by a factor of 2.04 and the declared axial term by a factor of 4.16, and the error is not dynamically inert as an axial error alone would be, since the mass and the transverse terms both enter the knee's effective inertia and the whole robot's weight. Correcting it reduces the total mass from 16.219301 to 15.837077 kilogrammes and the knee's effective inertia by 6.9 percent, which section 5.2 carries through.

The second correction is the geometry type. The thigh and the calf are declared `capsule` but inertially treated as cylinders. Rendering a capsule of the same radius and the same length at the same density would give the thigh a mass of 0.210822 and a transverse term of 0.001034, and the calf a mass of 0.099855 and a transverse term of 0.000452, since the two hemispherical caps add an eighth of the volume and extend the segment by a radius at each end. URDF has no capsule primitive in any case, so the URDF must declare a cylinder whatever the MJCF says. The correction is therefore to change the two `geom` types in `my_design.xml` from `capsule` to `cylinder`, keeping the `fromto` attribute, which MuJoCo accepts for both, so that the declared shape, the declared mass and the declared inertia agree in both files. The capsule figures are recorded above so that the opposite resolution, recomputing the inertias for capsules and keeping the smoother collision primitive, remains available without repeating the derivation.

The third correction is the abduction axis, and section 4.2 sets out its reasoning.

The corrected total mass is 15.837077 kilogrammes, being 0.674397 for the trunk and 3.790670 for each leg, and the weight is 155.36 newtons. This is a materially heavier robot than the trunk mass alone suggests, the four legs carrying 95.7 percent of it, and the two actuator housings on each leg carrying 91 percent of the leg. The distribution matters to section 5.2, where it makes the abduction joint the most loaded of the three rather than the least.

One property of the model is recorded as a defect rather than corrected, because it is a modelling choice and not an internal inconsistency. The armature of 0.01 kilogramme metres squared is declared identically on all twelve joints. A GO-M8010-6 with a reduction of 6.33 and a rotor inertia of the order of 4 times 10 to the minus 5 reflects roughly 1.6 times 10 to the minus 3, so the declared figure is high by something near a factor of six, and it dominates the knee, whose entire distal chain amounts to 0.0023. It is carried across unaltered so that the two models remain interchangeable, and the consequence, that the derived gains of section 5.3 are larger than a physically identified armature would justify, is recorded there.

### 4.4 The standing pose, and the foot sphere

The MuJoCo keyframe places the trunk at 0.270 metres with the hip pitch at 0.884337 radians and the knee at -1.768673 radians. Since the knee angle is the negation of twice the hip angle to within a microradian, the calf's absolute pitch is the negation of the thigh's and the foot lands exactly beneath the hip pitch axis, the fore and aft components cancelling. The vertical drop is `0.213 cos(0.884337) + 0.213 cos(-0.884336)`, which evaluates to 0.270000 metres.

The hip pitch axis lies at the trunk's own height, every intermediate offset in the chain being purely lateral or longitudinal, so a trunk at 0.270 metres places the foot frame's origin at exactly zero. That origin is the centre of the contact sphere and not its lowest point, so the keyframe buries 22 millimetres of the foot in the ground. The correction is the fourth applied to `my_design.xml`, raising the keyframe's third `qpos` entry from 0.270000 to 0.292000, and 0.292 is thereafter the standing height that the articulation's initial position, the base height reward target and the terrain scaling of section 8.2 all use.

The distinction is worth stating plainly because it recurs whenever a spherical or capsular foot is converted. A box or a mesh foot usually has its frame at the sole, so the trunk height and the kinematic drop coincide. A sphere has its frame at the centre, so the standing height is the kinematic drop plus the radius, and the discrepancy is exactly the radius on every such robot. The conversion document of sub-task 2.4 records the rule.

### 4.5 Clearance audit, and the self collision decision

Requirement 2.5 asks that no self collision occur between links. The audit below establishes that no unintended overlap exists at the nominal pose, and that self collision must nevertheless remain enabled.

At the nominal configuration every non adjacent pair clears. The abduction housing spans x from 0.085 to 0.125 and the hip housing from 0.130 to 0.222, a gap of 5 millimetres. The hip housing spans y from -0.072 to -0.032 and the knee housing from -0.122 to -0.082, a gap of 10 millimetres. The hip housing and the thigh clear by 40 millimetres in y. The trunk, whose half extent in x is 0.085, abuts the abduction housing exactly without penetrating it. The single overlapping pair is the thigh against the knee housing, which interpenetrate in y over the interval -0.122 to -0.112, and that pair is joined by the fixed `hip_FR_thigh_joint`, so PhysX excludes it from self collision automatically as it excludes every directly jointed pair.

Self collision cannot be disabled, however, because the abduction range permits the legs to reach one another. Rotating the front right foot about its abduction axis by the upper limit of 1.0 radian carries it from `y = -0.132` to `y = +0.132`, which is the front left foot's nominal position, so an unconstrained policy can drive one leg into another during exploration. The prescription is therefore `self_collision=True` on the spawn configuration and `enabled_self_collisions=True` on the articulation root properties, matching both the biped configuration and MuJoCo's own default of colliding all pairs except parents and children.

### 4.6 The conversion procedure

The tool named in the specification, `mjcf_urdf_simple_converter`, requires the `mujoco` Python package, which is not installed in this container. Its role in this plan is confined to producing a reference draft against which the topology, the joint origins and the inertias are cross checked, because the tool exports every geometry as a triangulated mesh, and a mesh URDF would forfeit three things this project needs, the analytic inertia validation of section 4.3, the link length scaling the co-optimisation generator performs by rewriting joint origins, and the small asset footprint that lets Isaac Lab convert the URDF at every design swap.

Sub-task 2.1, produce the reference draft. In the development container run `pip install mujoco 'git+https://github.com/Yasu31/mjcf_urdf_simple_converter'` and convert `my_design.xml`. Compare its link count, its joint origins and its inertial diagonals against section 4.2 and section 4.3, and record any disagreement in the conversion document. Should the tool be unavailable, the cross check may be performed against MuJoCo's own `mj_printModel` output, and the omission recorded.

Sub-task 2.2, correct `my_design.xml`. Four corrections were established above and all four are applied to the MuJoCo file before the URDF is generated, so that the two models describe one robot. The patch is small enough to state in full.

```diff
   on the front right and rear right legs, the abduction axis
-              <joint name="FR_hip_joint" axis="1 0 0" range="-1.0 1.0"
+              <joint name="FR_hip_joint" axis="-1 0 0" range="-1.0 1.0"
-              <joint name="RR_hip_joint" axis="1 0 0" range="-1.0 1.0"
+              <joint name="RR_hip_joint" axis="-1 0 0" range="-1.0 1.0"

   on all four legs, the thigh geometry
-              <geom type="capsule" fromto="0 0 0 0 0 -0.213000"
+              <geom type="cylinder" fromto="0 0 0 0 0 -0.213000"
                     size="0.020000" rgba="0.6 0.6 0.6 1"/>

   on all four legs, the calf mass, inertia and geometry
-              <inertial pos="0 0 -0.106500" mass="0.187365"
-                        diaginertia="0.000727 0.000727 0.000037"/>
+              <inertial pos="0 0 -0.106500" mass="0.091809"
+                        diaginertia="0.000352 0.000352 0.000009"/>
-              <geom type="capsule" fromto="0 0 0 0 0 -0.213000"
+              <geom type="cylinder" fromto="0 0 0 0 0 -0.213000"
                     size="0.014000" rgba="0.45 0.45 0.45 1"/>

   and the standing keyframe
-      <key name="home" qpos="0.000000 0.000000 0.270000 1.000000 ...
+      <key name="home" qpos="0.000000 0.000000 0.292000 1.000000 ...
```

The corrected file is placed beside the URDF at `environments/environments/assets/urdf/quadruped/my_design.xml`, so that the MuJoCo source of the asset travels with the asset and a later divergence between the two is visible within one directory rather than across the workspace.

Sub-task 2.3, author the final URDF from primitives. The generator below is complete and has been executed against the corrected figures, producing twenty five links, twenty four joints of which twelve are revolute, a single root at `base_Link`, an acyclic tree in which every link reaches the root, mirrored abduction axes of `-1 0 0` on the right and `1 0 0` on the left, and a total mass of 15.837077 kilogrammes matching the corrected MuJoCo model exactly. Place it at `environments/environments/assets/urdf/quadruped/gen_quadruped_urdf.py` and its output at `environments/environments/assets/urdf/quadruped/quadruped.urdf`, keeping the generator beside the asset so that a later change to a length or a mass is made once in the source of truth rather than twelve times in the XML.

```python
"""Emit quadruped.urdf from the primitive parameters of my_design.xml.

Every number below is transcribed from the MuJoCo model and none is recomputed, so the
URDF is equivalent to the MJCF by construction rather than by a converter's fidelity.
The MJCF cylinder ``size`` is a radius and a HALF length, so the URDF length is twice
the second entry, and the MJCF box ``size`` is a half extent triple, so the URDF box is
twice each entry.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

PI_2 = 1.5707963267948966

# leg -> (abad mount x, abad mount y, hip actuator x offset, knee actuator y offset,
#         thigh y offset, abduction axis sign). The first five are transcribed from the
# four <body> chains of the MJCF. The sixth mirrors the abduction axis so that a positive
# command means outward on both sides, as TRON1 does, see section 4.2.
LEGS = {
    "FR": (0.105, -0.052, 0.071, -0.050, -0.030, -1),
    "FL": (0.105, 0.052, 0.071, 0.050, 0.030, 1),
    "RR": (-0.105, -0.052, -0.071, -0.050, -0.030, -1),
    "RL": (-0.105, 0.052, -0.071, 0.050, 0.030, 1),
}

THIGH_L = 0.213
CALF_L = 0.213
FOOT_R = 0.022

LIMITS = {           # joint -> (lower, upper, effort, velocity)
    "abad": (-1.0, 1.0, 23.622511, 30.0),
    "hip": (-1.2, 3.5, 23.622511, 30.0),
    "knee": (-2.9, -0.5, 35.238000, 20.0),
}


def _sub(parent, tag, **attrib):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrib.items()})


def _inertial(link, mass, ixx, iyy, izz, xyz="0 0 0"):
    node = _sub(link, "inertial")
    _sub(node, "origin", xyz=xyz, rpy="0 0 0")
    _sub(node, "mass", value=f"{mass:.6f}")
    _sub(node, "inertia", ixx=f"{ixx:.9f}", ixy="0", ixz="0",
         iyy=f"{iyy:.9f}", iyz="0", izz=f"{izz:.9f}")


def _shape(link, kind, geom_attr, xyz, rpy, rgba):
    for tag in ("visual", "collision"):
        node = _sub(link, tag)
        _sub(node, "origin", xyz=xyz, rpy=rpy)
        geom = _sub(node, "geometry")
        _sub(geom, kind, **geom_attr)
        if tag == "visual":
            mat = _sub(node, "material", name=rgba[0])
            _sub(mat, "color", rgba=rgba[1])


def _joint(root, name, jtype, parent, child, xyz, axis=None, limit=None):
    j = _sub(root, "joint", name=name, type=jtype)
    _sub(j, "origin", xyz=xyz, rpy="0 0 0")
    _sub(j, "parent", link=parent)
    _sub(j, "child", link=child)
    if axis is not None:
        _sub(j, "axis", xyz=axis)
    if limit is not None:
        lo, hi, eff, vel = limit
        _sub(j, "limit", lower=f"{lo}", upper=f"{hi}", effort=f"{eff}", velocity=f"{vel}")
        _sub(j, "dynamics", damping="0.0", friction="0.0")


def build():
    root = ET.Element("robot", {"name": "quadruped"})

    base = _sub(root, "link", name="base_Link")
    _inertial(base, 0.674397, 0.002635, 0.002100, 0.003783)
    _shape(base, "box", {"size": "0.170000 0.196000 0.092000"},
           "0 0 0", "0 0 0", ("trunk", "0.20 0.20 0.25 1"))

    for leg, (ax, ay, hx, ky, ty, asign) in LEGS.items():
        # abduction actuator housing, rigid to the trunk, cylinder axis along X
        link = _sub(root, "link", name=f"abad_{leg}_actuator_Link")
        _inertial(link, 1.028375, 0.001088, 0.000681, 0.000681)
        _shape(link, "cylinder", {"radius": "0.046000", "length": "0.040000"},
               "0 0 0", f"0 {PI_2} 0", ("abad", "0.85 0.35 0.10 1"))
        _joint(root, f"abad_{leg}_fixed_joint", "fixed", "base_Link",
               f"abad_{leg}_actuator_Link", f"{ax:.6f} {ay:.6f} 0.000000")

        # hip pitch actuator housing, carried by the abduction joint, axis along Y
        link = _sub(root, "link", name=f"hip_{leg}_actuator_Link")
        _inertial(link, 1.028375, 0.000681, 0.001088, 0.000681)
        _shape(link, "cylinder", {"radius": "0.046000", "length": "0.040000"},
               "0 0 0", f"{PI_2} 0 0", ("hip", "0.15 0.45 0.85 1"))
        _joint(root, f"abad_{leg}_Joint", "revolute", f"abad_{leg}_actuator_Link",
               f"hip_{leg}_actuator_Link", f"{hx:.6f} 0.000000 0.000000",
               axis=f"{asign} 0 0", limit=LIMITS["abad"])

        # knee actuator housing, carried by the hip pitch joint, axis along Y
        link = _sub(root, "link", name=f"knee_{leg}_actuator_Link")
        _inertial(link, 1.434746, 0.000950, 0.001518, 0.000950)
        _shape(link, "cylinder", {"radius": "0.046000", "length": "0.040000"},
               "0 0 0", f"{PI_2} 0 0", ("knee", "0.15 0.75 0.30 1"))
        _joint(root, f"hip_{leg}_Joint", "revolute", f"hip_{leg}_actuator_Link",
               f"knee_{leg}_actuator_Link", f"0.000000 {ky:.6f} 0.000000",
               axis="0 1 0", limit=LIMITS["hip"])

        # thigh segment, rigid to the knee actuator. The MJCF declares a capsule but
        # dimensions and inertias it as a cylinder, and after the correction of section
        # 4.3 both files say cylinder.
        link = _sub(root, "link", name=f"hip_{leg}_thigh_Link")
        _inertial(link, 0.187365, 0.000727, 0.000727, 0.000037,
                  xyz=f"0 0 {-THIGH_L / 2:.6f}")
        _shape(link, "cylinder", {"radius": "0.020000", "length": f"{THIGH_L:.6f}"},
               f"0 0 {-THIGH_L / 2:.6f}", "0 0 0", ("thigh", "0.6 0.6 0.6 1"))
        _joint(root, f"hip_{leg}_thigh_joint", "fixed", f"knee_{leg}_actuator_Link",
               f"hip_{leg}_thigh_Link", f"0.000000 {ty:.6f} 0.000000")

        # calf segment, carried by the knee joint
        # The mass and the inertia below are the CORRECTED figures of section 4.3,
        # computed for a cylinder of radius 0.014 at the model's own leg density of
        # 700 kg/m^3, not the thigh's figures that the MJCF erroneously repeats here.
        link = _sub(root, "link", name=f"knee_{leg}_Link")
        _inertial(link, 0.091809, 0.000352, 0.000352, 0.000009,
                  xyz=f"0 0 {-CALF_L / 2:.6f}")
        _shape(link, "cylinder", {"radius": "0.014000", "length": f"{CALF_L:.6f}"},
               f"0 0 {-CALF_L / 2:.6f}", "0 0 0", ("calf", "0.45 0.45 0.45 1"))
        _joint(root, f"knee_{leg}_Joint", "revolute", f"hip_{leg}_thigh_Link",
               f"knee_{leg}_Link", f"0.000000 0.000000 {-THIGH_L:.6f}",
               axis="0 1 0", limit=LIMITS["knee"])

        # spherical foot, rigid to the calf, the only intended ground contact
        link = _sub(root, "link", name=f"foot_{leg}_Link")
        _inertial(link, 0.020000, 0.000004, 0.000004, 0.000004)
        _shape(link, "sphere", {"radius": f"{FOOT_R:.6f}"},
               "0 0 0", "0 0 0", ("foot", "0.05 0.05 0.05 1"))
        _joint(root, f"foot_{leg}_joint", "fixed", f"knee_{leg}_Link",
               f"foot_{leg}_Link", f"0.000000 0.000000 {-CALF_L:.6f}")

    return root


if __name__ == "__main__":
    import sys
    xml = minidom.parseString(ET.tostring(build())).toprettyxml(indent="  ")
    open(sys.argv[1], "w").write(xml)
```

Sub-task 2.4, write the conversion document. `context/quadruped_xml_to_urdf_conversion.md` in the simulation repository records the whole of sections 4.1 to 4.5 as established fact rather than as proposal, including the four corrections of sub-task 2.2 and the reasoning that produced each, and additionally serves as the general procedure for any future MuJoCo to URDF conversion in this project. Its prescribed structure is a statement of the primitive conversions, the half extent to full extent rule for boxes and the radius with half length rule for cylinders foremost, the absence of a capsule primitive from URDF and the consequences of rendering one as a cylinder, the rule that a spherical or capsular foot places its frame at the centre and not at the sole so that the standing height exceeds the kinematic drop by a radius, the inertia formulae and the validation table, the axis and frame conventions with the observation that MuJoCo body positions and URDF joint origins both express a child frame in its parent and therefore transcribe directly, the joint semantic audit of section 4.1 which is the step most easily skipped and most costly to skip, the kinematic verification of the standing pose, the clearance audit, and a closing checklist. It is placed in the simulation repository rather than at the workspace level because its subject is confined to that repository, per clause 25 of `../CLAUDE.md`.

## 5. Task 4, the joint control analysis for the quadruped

### 5.1 Scope of the document

`context/joint_control_analysis_quadruped.md` in the simulation repository derives the gains and the velocity ranges for this robot alone. It does not restate the theory, which `context/joint_control_analysis.md` already establishes, namely the equivalence of the implicit proportional derivative actuator to a forced mass spring damper, the characteristic equation and its roots, the natural frequency, the damping ratio and the critical damping coefficient, the behaviour in each damping regime, and the causal chain from an excessive action scale through velocity saturation to divergence. The new document opens by referring the reader there for all of it, and then applies the results.

It adds two bodies of theory the existing document lacks. The first is a treatment of how a proportional gain is chosen when no gain is given, which section 5.3 sets out and which the existing document does not need because the biped's gains arrived identified from hardware. The second is a treatment of how a joint velocity range is chosen, which section 5.4 sets out.

### 5.2 Effective inertias

The effective inertia at a joint is the inertia of everything distal to it about that joint's axis, obtained by the parallel axis theorem with the distal joints held locked, plus the armature. The armature is the rotor inertia reflected through the square of the gear ratio and is added directly to the diagonal of the mass matrix by both MuJoCo and PhysX, so it participates in the joint's second order dynamics exactly as the link inertia does [11]. The MuJoCo model declares 0.01 kilogramme metres squared on every joint, which section 4.3 records as high by roughly a factor of six for this motor, and that figure is nonetheless the single most consequential number in this section, dominating the knee whose entire distal chain amounts to 0.0023.

Two configurations are computed, and the distinction was elided in an earlier draft of this plan. The nominal crouch is the operating point, the pose the robot stands in and the pose about which the policy's actions are offsets, and it is the configuration at which the gains are set. Full extension is the worst case, the pose in which each distal chain reaches its greatest lever arm, and it is the configuration against which the resulting damping ratio is checked for the excursion it suffers. All distances are taken from the chain of section 4.2 and all link inertias from the corrected table of section 4.3.

| Joint | Distal chain | Effective inertia at the crouch | Effective inertia extended |
|---|---|---|---|
| `abad_*_Joint` | hip housing, knee housing, thigh, calf, foot, about the roll axis | 0.024292 | 0.033342 |
| `hip_*_Joint` | knee housing, thigh, calf, foot, about the pitch axis | 0.020572 | 0.027728 |
| `knee_*_Joint` | calf, foot, about the pitch axis | 0.012305 | 0.012305 |

Three observations follow. The abduction joint carries the largest effective inertia, not the smallest, because the whole leg swings about a roll axis at a lever arm of 0.080 metres laterally and up to 0.426 metres vertically, and because the two actuator housings that dominate the leg's mass both sit on that lever. This inverts the intuition carried over from the biped, where the abduction joint is the lightest, and it is the reason the abduction gains cannot simply be copied down from the hip. The knee's inertia does not vary with configuration, its distal chain containing no joint. And the armature is 41 percent of the abduction figure, 49 percent of the hip figure and 81 percent of the knee figure, so the gains derived below are, to a large extent, gains for a rotor rather than for a limb.

### 5.3 The proportional and derivative gains

An earlier draft of this plan took the proportional gain of 100 newton metres per radian from the MuJoCo position actuators unaltered. That is now rejected. A MuJoCo `position` actuator gain is a modelling convenience with no hardware behind it, 100 is a round number rather than an identified one, and it is four times what Isaac Lab's own A1 and Go2 configurations use at `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:92` and `:175`, six times the Mini Cheetah's 17 [13] and nearly twice the rapid motor adaptation work's 55 on A1 [14]. Both gains are therefore derived here, and the derivation is what the analysis document contributes to this project's practice.

The proportional gain is fixed by what the joint must hold, not by what response it should have. This is the point the existing analysis document does not make, and it is the reason the reference configurations carry nearly uniform stiffness across joints of very different inertia. A proportional law produces a steady state error of `tau / Kp` under a constant load, so the standing pose the robot must hold sets a floor on `Kp` for a tolerated sag. At the nominal crouch each leg carries a quarter of 155.36 newtons and the stance torques are as follows.

| Joint | Lever at the crouch | Stance torque | Kp floor at a sag of 0.125 rad |
|---|---|---|---|
| `abad_*_Joint` | 0.080 m laterally | 4.046 N m | 32.4 |
| `hip_*_Joint` | The foot sits beneath the axis | 0.226 N m | 1.8 |
| `knee_*_Joint` | 0.165 m in the fore and aft sense | 6.506 N m | 52.1 |

The tolerance of 0.125 radians is half the action scale of 0.25. Beyond that the policy spends more than half its per step authority merely cancelling the sag rather than shaping the gait, which is the practical failure this floor exists to prevent. The hip's floor is vacuous at the nominal crouch because the keyframe places each foot directly beneath its pitch axis, so the hip is sized with the abduction joint instead, the two sharing a motor.

Two ceilings then bound the gain from above. The integration ceiling is not binding, PhysX evaluating the implicit proportional derivative law at 200 hertz and integrating it implicitly, so a natural frequency of even 70 radians per second gives `omega_n dt` of 0.35 and no stiff integration problem. The observability ceiling is binding. The policy acts at 50 hertz, and a response whose natural frequency exceeds a fifth of that rate rings and settles within a single control step, so the policy cannot observe the transient it caused and cannot learn to shape it. The ceiling is therefore 10 hertz.

The gains adopted follow the torque ceilings of the three motors, on the reasoning that a joint sized for more torque was sized for more load and should be correspondingly stiffer. The abduction and hip joints share the 23.622511 newton metre motor and take 40, which clears the abduction floor of 32.4. The knee's 35.238000 newton metre motor stands in a ratio of 1.4917 to the others, which would give 60, and 60 places the knee at 11.11 hertz, above the observability ceiling. The knee is therefore trimmed to 50, which lands at 10.15 hertz and accepts a stance sag of 0.130 radians against the 0.125 target. Both constraints are near binding at that value and neither is violated by more than four percent, which is the correct place to sit when two criteria disagree.

The derivative gain then follows from the damping ratio, `Kd = 2 zeta sqrt(Kp I)`, evaluated at the crouch and rounded to two significant figures.

| Joint | Kp | Effective inertia | Natural frequency | Fraction of the control rate | Critical damping | Kd at zeta 0.6 | Damping ratio extended | Steady state sag |
|---|---|---|---|---|---|---|---|---|
| `abad_*_Joint` | 40.0 | 0.024292 | 40.58 rad/s, 6.46 Hz | 12.9 percent | 1.971 | 1.20 | 0.520 | 0.101 rad |
| `hip_*_Joint` | 40.0 | 0.020572 | 44.10 rad/s, 7.02 Hz | 14.0 percent | 1.814 | 1.10 | 0.522 | 0.006 rad |
| `knee_*_Joint` | 50.0 | 0.012305 | 63.75 rad/s, 10.15 Hz | 20.3 percent | 1.569 | 0.90 | 0.574 | 0.130 rad |

The target of 0.6 sits just below the 0.7 to 1.0 band NVIDIA recommends for a joint drive [10], deliberately, since a legged joint that must absorb an impact benefits from a little compliance and since the surveyed locomotion configurations run nearer 0.4. It gives a peak overshoot of 9.5 percent on a step and a two percent settling time of 0.105 seconds at the knee, which is 5.2 control steps, so the policy sees the whole of the transient it commands. The damping ratio falls no lower than 0.520 anywhere in the configuration space, the excursion arising because `Kd` is fixed while the effective inertia grows toward extension, and 0.520 remains comfortably underdamped without ringing.

Three comparisons close the section. The MuJoCo file's own joint damping of 1.0, 2.0 and 2.0 would give damping ratios of 0.507, 1.102 and 1.275 against these proportional gains, so it is not absurd but it is inconsistent, overdamping the two pitch joints while leaving the roll joint at half their damping. Isaac Lab's Go2 configuration uses 25 and 0.5 uniformly for the same motor family and very nearly the same link lengths, which would give this robot damping ratios of 0.32, 0.35 and 0.45 and a knee sag of 0.260 radians, so the scheme derived here is stiffer and better damped than the canonical one rather than a departure from it in kind. And the biped's leg joints in this repository run at 55, 80 and 60 against effective inertias near 0.10, placing them between 3.7 and 4.5 hertz, so the quadruped sits one octave higher, which is what a robot with a third of the limb inertia should do.

### 5.4 Joint velocity ranges

The theory this section contributes concerns what a velocity limit is and what bounds it. In Isaac Lab the `velocity_limit` of a `DCMotorCfg`, from which `IdentifiedActuatorCfg` derives at `environments/environments/actuators/actuator_cfg.py:15`, is not a clamp. It is the no load speed of a linear torque speed characteristic, the available torque being `saturation_effort` times `(1 - q_dot / velocity_limit)` and then clipped to `effort_limit`. Two consequences follow. The quantity that actually bounds useful motion is the corner speed at which the ramp meets the effort ceiling, `velocity_limit (1 - effort_limit / saturation_effort)`, and the ratio of saturation to effort is therefore a design choice, not a detail. This repository's TRON1 configuration sets that ratio near 6.7, giving constant torque up to 85 percent of the no load speed, whereas Isaac Lab's Unitree configurations set saturation equal to effort, giving a pure ramp with no constant region at all.

The choice adopted is the TRON1 ratio, because the MuJoCo model declares a constant `forcerange` with no speed dependence whatever, so a characteristic that is flat over the operating range is the one that keeps the two models interchangeable. The Unitree alternative is recorded as the more conservative option should hardware measurements later contradict it.

Four bounds then determine the numbers.

The motor bound. The design's torque ceilings of 23.622511 and 35.238000 newton metres match the Unitree Go1 GO-M8010-6 joint ratings of 23.7 and 35.55 newton metres to within 0.4 and 0.9 percent respectively, and its thigh and shank of 0.213 metres match the Go2's exactly, so the design is almost certainly derived from that actuator and that leg, and the corresponding rated joint speeds of 30.1 and 20.06 radians per second are the defensible no load figures [12]. Isaac Lab's own Go2 configuration independently uses 30.0 for the same motor family at `unitree.py:174`.

The slew bound. The policy may change its setpoint by the action scale on every control step, so the actuator must sustain `alpha / dt_ctrl`, which at an action scale of 0.25 and a control period of 0.02 seconds is 12.5 radians per second.

The transient bound. A step of amplitude `A` into a second order system of natural frequency `omega_n` and damping ratio `zeta` produces a peak velocity found by maximising `A omega_n exp(-zeta omega_n t) sin(omega_d t) / sqrt(1 - zeta^2)`. Evaluated numerically at `A` equal to 0.25 and at the gains of section 5.3 this gives 5.02 radians per second at the abduction joint, 5.47 at the hip and 8.14 at the knee. Note that the closed form upper bound `A omega_n / sqrt(1 - zeta^2)` used at section 6.5 of `joint_control_analysis.md` overestimates these by a factor near 2.5, since it ignores the exponential decay before the sine reaches its first peak, and the analysis document should record the correction.

The kinematic bound. A trot at two hertz with a duty factor of one half gives a swing duration of 0.25 seconds. A step length of 0.20 metres on a 0.270 metre standing leg requires a hip sweep of `2 arcsin(0.10 / 0.270)`, which is 0.759 radians, a mean rate of 3.04 radians per second and a half sine peak of 4.77. Knee retraction of about a radian over the same swing gives a peak near 6.3. Both are below the other bounds, which is the expected result, gait rates lying well inside actuator capability except during recovery.

The resulting parameterisation takes the motor bound as the no load speed and verifies that the corner speed exceeds every demand.

| Joint | No load speed | Saturation effort | Effort limit | Corner speed | Largest demand | Margin |
|---|---|---|---|---|---|---|
| `abad_*_Joint` | 30.0 | 158.0 | 23.622511 | 25.51 | 12.5, the slew bound | 2.04 times |
| `hip_*_Joint` | 30.0 | 158.0 | 23.622511 | 25.51 | 12.5, the slew bound | 2.04 times |
| `knee_*_Joint` | 20.0 | 236.0 | 35.238000 | 17.01 | 12.5, the slew bound | 1.36 times |

### 5.5 The action scale

Three bounds constrain the action scale and the biped's inherited value of 0.25 clears all three, which the earlier draft of this plan could not say because it carried the proportional gain of 100.

Velocity saturation through the transient. The peak velocity scales linearly in the action scale, so the scale at which each joint's peak reaches its corner speed is 1.27 at the abduction joint, 1.16 at the hip and 0.52 at the knee.

Velocity saturation through the slew. Requiring `alpha / dt_ctrl` to stay inside the corner speed gives 0.51 at the abduction and hip joints and 0.34 at the knee.

The torque ceiling. A step of amplitude `alpha` demands `Kp alpha` at the instant it is applied, which is 10.00 newton metres at the abduction and hip joints against a ceiling of 23.622511, and 12.50 at the knee against 35.238000. The scale at which a single full step reaches the ceiling is 0.59 and 0.70 respectively.

The tightest of the nine figures is the knee's slew bound of 0.34, so the action scale of 0.25 is safe by a factor of 1.36 and the torque ceiling, which bound the design at 0.236 under the rejected gain of 100, no longer binds at all. This is the clearest single benefit of deriving the stiffness rather than inheriting it.

## 6. Task 3, the actuator and articulation configuration

Create `environments/environments/assets/config/quadruped_identified_cfg.py` with the following content. It mirrors `solefoot_identified_cfg.py` in structure, differing in that it declares no USD variant, the quadruped having no USD asset, and in that its solver position iteration count is raised from 4 to 8, following the recommendation of section 7.2 of `joint_control_analysis.md` for underdamped joints undergoing contact transitions on rough terrain. Every stiffness and damping figure below is derived in section 5.3 and none is taken from the MuJoCo file, whose position actuator gain of 100 this plan rejects.

```python
import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from environments.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../urdf/quadruped/quadruped.urdf")

# Abduction and adduction, the roll axis. MuJoCo names this joint FR_hip_joint.
# The effort limit is the MJCF forcerange and the velocity limit the Go1 GO-M8010-6
# rated joint speed, this design's torque ceiling matching that motor to within 0.4
# percent. Kp is DERIVED, not taken from the MJCF position actuator gain of 100, which
# is a modelling convenience with no hardware behind it. It is set by the stance torque
# of 4.05 N m at a tolerated sag of 0.125 rad, half the action scale, and lands at
# 6.46 Hz, well inside the observability ceiling of a fifth of the 50 Hz control rate.
# Kd gives a damping ratio of 0.6 against an effective inertia of 0.024292 kg m^2 at
# the nominal crouch. See joint_control_analysis_quadruped.md section 5.3 throughout.
QUADRUPED_ABAD_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["abad_.._Joint"],
    effort_limit=23.622511,
    velocity_limit=30.0,
    saturation_effort=158.0,
    stiffness={".*": 40.0},
    damping={".*": 1.2},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Hip flexion and extension, the pitch axis. MuJoCo names this joint FR_thigh_joint.
# It shares the abduction joint's motor and takes the same Kp, its own stance torque
# being near zero at the crouch, where the keyframe places each foot directly beneath
# its pitch axis. Kd is set against an effective inertia of 0.020572 kg m^2.
QUADRUPED_HIP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["hip_.._Joint"],
    effort_limit=23.622511,
    velocity_limit=30.0,
    saturation_effort=158.0,
    stiffness={".*": 40.0},
    damping={".*": 1.1},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Knee flexion and extension. MuJoCo names this joint FR_calf_joint. Its motor stands
# in a ratio of 1.4917 to the other two, which would give a Kp of 60, but 60 places the
# joint at 11.11 Hz, above the observability ceiling. It is trimmed to 50, landing at
# 10.15 Hz and accepting a stance sag of 0.130 rad against the 0.125 target. Kd is set
# against an effective inertia of 0.012305 kg m^2, which does not vary with pose.
QUADRUPED_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["knee_.._Joint"],
    effort_limit=35.238000,
    velocity_limit=20.0,
    saturation_effort=236.0,
    stiffness={".*": 50.0},
    damping={".*": 0.9},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

rigid_props = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True,
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
)
activate_contact_sensors = True

# 0.292 m is the height at which the foot sphere rests on the ground in the nominal
# crouch, being the 0.270 m kinematic drop plus the 0.022 m foot radius. The spawn
# adds 28 mm of clearance so that a rough terrain tile cannot capture the foot at reset.
init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),
    joint_pos={
        "abad_.._Joint": 0.0,
        "hip_.._Joint": 0.884337,
        "knee_.._Joint": -1.768673,
    },
    joint_vel={".*": 0.0},
)
soft_joint_pos_limit_factor = 0.9
actuators = {
    "abad": QUADRUPED_ABAD_ACTUATOR_CFG,
    "hip": QUADRUPED_HIP_ACTUATOR_CFG,
    "knee": QUADRUPED_KNEE_ACTUATOR_CFG,
}

spawn = sim_utils.UrdfFileCfg(
    asset_path=urdf_path,
    fix_base=False,
    merge_fixed_joints=False,
    self_collision=True,
    joint_drive=None,
    rigid_props=rigid_props,
    articulation_props=articulation_props,
    activate_contact_sensors=activate_contact_sensors,
)

QUADRUPED_IDENTIFIED_CFG = ArticulationCfg(
    spawn=spawn,
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)
```

Three details of the above warrant note. The joint name expressions use `.._Joint` with two wildcards rather than `.*_Joint`, which matters because `.*` would also match the fixed joints, and because `hip_.._Joint` must not capture `hip_FR_thigh_joint`. The three expressions were checked against the twelve generated joint names and are disjoint and complete. The initial joint positions are the MuJoCo keyframe verbatim and are symmetric across all four legs, where Isaac Lab's A1 and Go2 configurations instead use an abduction offset of 0.1 radians and a rear hip angle of 1.0 against a front angle of 0.8 at `unitree.py:76` and `:159`. The symmetric pose is retained so that the URDF and the MJCF describe the same standing robot, and the asymmetric alternative is recorded as a tuning option. The dynamic friction coefficient of 0.02 has no MuJoCo counterpart, the model declaring only a Coulomb `frictionloss` of 0.2 which maps to `friction_static`, and it is set to the biped's value so that the two robots share an actuator friction model.

## 7. Task 5, what the Markov decision process package offers the quadruped

The package at `environments/environments/tasks/locomotion/mdp/` is star imported over Isaac Lab's own velocity locomotion module at `mdp/__init__.py:1`, so a configuration addressing `mdp.name` reaches this repository's definition where one exists and Isaac Lab's otherwise. The inventory below classifies every term of both layers. Nothing in it is edited, per clause 3 of the change discipline, the quadruped selecting from what exists.

### 7.1 Rewards

| Term | Source | Transfers | What must be supplied |
|---|---|---|---|
| `stay_alive` | `rewards.py:23` | Yes, unaltered | Nothing |
| `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` | Isaac Lab | Yes, unaltered | Nothing |
| `lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2` | Isaac Lab | Yes, unaltered | Nothing |
| `joint_torques_l2`, `joint_acc_l2`, `joint_vel_l2`, `action_rate_l2`, `joint_pos_limits` | Isaac Lab | Yes, unaltered | Nothing |
| `undesired_contacts` | Isaac Lab | Yes | Quadruped body expressions |
| `joint_deviation_l1` | Isaac Lab | Yes, newly used | Abduction joint expression |
| `base_height_rough_l2` | `rewards.py:1103` | Yes | Target of 0.292 |
| `base_com_height` | `rewards.py:1142` | Yes | Alternative to the above, no height scanner needed |
| `ActionSmoothnessPenalty` | `rewards.py:1303` | Yes, unaltered | Nothing |
| `JointTorqueRatePenalty` | `rewards.py:1375` | Yes, unaltered | Optional |
| `feet_air_time` | `rewards.py:196` | Yes | Foot sensor expression, retuned thresholds |
| `feet_slide` | `rewards.py:225` | Yes | Foot sensor and asset expressions |
| `foot_clearance_reward` | `rewards.py:234` | Reusable, but not used, see section 8.9 | Foot expressions, world frame target, contact gate |
| `feet_impact_force` | `rewards.py:165` | Yes | Foot sensor expression, threshold scaled to a 159 N robot |
| `foot_landing_vel` | `rewards.py:27` | Yes | Foot expressions, foot radius 0.022 |
| `joint_powers_l1` | `rewards.py:482` | Yes, unaltered | Optional, an alternative to torque plus joint velocity |
| `joint_torque_tiredness` | `rewards.py:612` | Yes, unaltered | Optional, normalises torque by its own limit |
| `unbalance_feet_air_time` | `rewards.py:491` | Yes, it is a variance over the feet axis and generalises to four | Optional |
| `unbalance_feet_height` | `rewards.py:499` | Yes, same reason | Optional |
| `no_contact` | `rewards.py:990` | Yes | Foot sensor expression |
| `stand_still` | `rewards.py:1007` | Yes | Nothing robot specific |
| `foot_clearance_reward_v2` | `rewards.py:287` | No | Requires a sole point polygon, meaningless for a sphere |
| `foot_clearance_reward_v3`, `GaitReward` | `rewards.py:403`, `:1167` | No | Require a registered gait command, which no configuration here declares |
| `foot_landing_vel_v2` | `rewards.py:99` | No | Sole polygon again |
| `feet_distance` | `rewards.py:533` | No | Written for exactly two feet |
| `no_fly`, `NoFlyWithGrace` | `rewards.py:691`, `:730` | No | Encode the single support condition of a biped |
| `keep_ankle_pitch_zero_in_air` | `rewards.py:824` | No | The robot has no ankle |
| `feet_regulation`, `nominal_foot_position` | `rewards.py:1064`, `:636` | No | Read `env._foot_radius`, set only by the TRON1 startup event |
| `leg_symmetry`, `same_feet_x_position`, `feet_yaw_alignment` | `rewards.py:656`, `:674`, `:579` | No | Two leg constructions, and a point foot has no meaningful yaw |
| `knee_flexion_in_swing`, `knee_flexion_in_swing_v2` | `rewards.py:897`, `:934` | No | Written against the biped's knee and ankle pair |

### 7.2 Observations

Every term in `observations.py` is robot agnostic in its implementation, returning a whole tensor of the articulation rather than a named subset, with three exceptions that take a body selector, namely `feet_lin_vel`, `robot_contact_force` and `robot_feet_contact_force`, and one that takes a pair of body name lists, namely `robot_link_lengths`. All four transfer on a change of body names alone. `get_gait_phase` and `get_gait_command` require a gait command and are not used. `joint_pos_rel_exclude_wheel` is for the wheelfoot variant.

The link length observation deserves a note, since it is the input the co-optimisation policy reads. Its class at `observations.py:229` takes `parent_body_names` and `child_body_names` and returns the norm of the difference of their world frame link origins, which is invariant under joint angle for a revolute chain. Traversing the quadruped from the base to a foot gives the chain of section 4.2, whose two variable segments are the thigh, from `hip_FR_thigh_Link` to `knee_FR_Link`, and the shank, from `knee_FR_Link` to `foot_FR_Link`. Over four legs that is eight values against the biped's four.

### 7.3 Events

| Term | Source | Transfers |
|---|---|---|
| `randomize_rigid_body_mass` | Isaac Lab | Yes, body name `base_Link` |
| `randomize_rigid_body_material` | Isaac Lab | Yes, unaltered |
| `randomize_actuator_gains` | Isaac Lab | Yes, one instance per joint group, three rather than four |
| `randomize_joint_default_pos` | `events.py:247` | Yes, unaltered |
| `randomize_joint_friction_model` | `events.py:289` | Yes, unaltered |
| `randomize_joint_parameters` | Isaac Lab | Yes, unaltered |
| `reset_root_state_uniform`, `reset_joints_by_offset` | Isaac Lab | Yes, unaltered |
| `push_by_setting_velocity` | Isaac Lab | Yes, magnitude halved |
| `randomize_rigid_body_coms` | `events.py:130` | Yes, unaltered, presently commented out for the biped |
| `prepare_quantity_for_tron` | `events.py:12` | No, it exists to serve `feet_regulation` and `nominal_foot_position`, neither of which the quadruped uses |
| `apply_external_force_torque_stochastic` | `events.py:22` | Yes, unaltered, presently commented out for the biped |

### 7.4 Curriculums and terminations

Every curriculum transfers unaltered. `terrain_levels_vel_delayed`, `modify_push_force`, `modify_command_velocity_x`, `modify_command_velocity_y`, `modify_command_velocity_angular` and `reduce_tracking_rewards_std` all address the command manager, the event manager or a reward term's own parameters, and none reads a body or joint name. Their reward thresholds are normalised by the term weight, as section 3.6 establishes, so they survive the reward rebalancing.

Both terminations transfer. `time_out` is unconditional and `illegal_contact` takes a sensor body name. `root_height_below_minimum_rough` at `terminations.py:15` is available and unused, and is a reasonable addition for the quadruped given how much crouch reserve the leg has.

### 7.5 Symmetry

`mdp/symmetry/brs.py` is written against SD_BRS1's joint and body layout, its maps being built at `symmetry/brs.py:78` and `:91` from that robot's names. It does not transfer, and no quadruped configuration in this plan requests symmetry augmentation. Were one added later, section 4.2 records the complication, that the abduction axes are unmirrored in this URDF and a left to right map must therefore negate the abduction states rather than merely permute them.

## 8. Tasks 6 and 7, the quadruped environment configuration

### 8.1 The naming scheme, and the collision it avoids

The specification originally asked for the quadruped's classes to be named `PFSceneCfg`, `PFEnvCfg`, `PFHIMEnvCfg` and `PFCoptEnvCfg`. Every one of those names is already taken by the LimX TRON1 pointfoot biped. `cfg/PF/limx_base_env_cfg.py:31` defines `PFSceneCfg` and `:455` defines `PFEnvCfg`, and `robots/limx_pointfoot_env_cfg.py` defines `PFBaseEnvCfg`, `PFBaseEnvCfg_PLAY`, `PFBlindFlatEnvCfg` and `PFBlindRoughEnvCfg`, which are exactly the names the hierarchy of section 9 requires. The collision matters because `cfg/__init__.py` reads `from .PF import *` followed by the same for `SF` and `WF`, flattening every public name into the `cfg` namespace, so a quadruped module joining that star import would shadow the biped's `PFEnvCfg` and the failure would surface as a biped task spawning a quadruped rather than as an import error.

The scheme adopted removes the hazard rather than containing it. Every quadruped class carries a `Quadruped` prefix ahead of the biped's name, so `PFSceneCfg` becomes `QuadrupedPFSceneCfg`, `PFBaseEnvCfg` becomes `QuadrupedPFBaseEnvCfg`, and the same throughout the twenty two class names of sections 8 and 9. The `PF` element is retained because the quadruped is a point foot machine and the correspondence with the biped hierarchy it mirrors is worth keeping legible. No name is then shared with any existing module, and the star import may be used exactly as the other three families use it.

Two consequences follow, both of which the implementation must carry out. `cfg/quadruped/__init__.py` mirrors `cfg/PF/__init__.py` and `cfg/SF/__init__.py`, which each read two star imports and nothing else.

```python
from .base_env_cfg import *
from .terrains_cfg import *
```

And `cfg/__init__.py` gains a fourth line, placed BEFORE the existing three rather than after, which is the one detail of the file that is not arbitrary.

```python
from .quadruped import *
from .PF import *
from .SF import *
from .WF import *
```

The ordering matters because the `Quadruped` prefix protects the four names that identify a robot, and only those. Nine further classes in each module carry generic names, `CommandsCfg`, `ActionsCfg`, `ObservationsCfg`, `EventsCfg`, `RewardsCfg`, `TerminationsCfg`, `CurriculumCfg` and the two co-optimisation and hybrid internal model observation classes, and those already collide three ways among the biped families, `WF` shadowing `SF` shadowing `PF` in the present file. The collision is harmless because nothing consumes those names through the `cfg` namespace, every environment class referring to its own module's declarations at class definition time. Placing the quadruped first nonetheless leaves every existing resolution bit for bit as it stands today, so the fourth star import adds four names and displaces none, which is what clause 2 of the change discipline requires of an edit to a shared module.

Consumers may then address the quadruped classes either through the flattened `cfg` namespace or module qualified as `from environments.tasks.locomotion.cfg.quadruped.base_env_cfg import QuadrupedPFEnvCfg`. The module qualified form is used throughout section 9, being the form `robots/limx_solefoot_env_cfg.py` already uses, and it remains the safer habit even where no collision exists.

### 8.2 A terrain configuration scaled to the robot

Isaac Lab's own Go2 port scales the terrain generator's difficulty down explicitly because the robot is small, and this quadruped stands at 0.292 metres where the biped stands at 0.75 [2]. Reusing `BERKELEY_MIMIC_TERRAINS_CFG` would put 0.10 metre stairs and 0.20 metre wave amplitudes under a robot a third the biped's height. Editing that configuration in place is forbidden, four biped tasks reading it, so a new file is created.

Create `environments/environments/tasks/locomotion/cfg/quadruped/terrains_cfg.py`.

```python
from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
)

# The sub-terrain proportions are those of BERKELEY_MIMIC_TERRAINS_CFG, so that the
# curriculum presents the same mixture to both robots. Only the amplitudes differ, each
# scaled toward the fraction of standing height the biped configuration presents, the
# quadruped standing at 0.292 m against the biped's 0.75 m.
QUADRUPED_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=5.0,
    num_rows=6,
    num_cols=64,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    color_scheme="height",
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.00, 0.3),
            platform_width=2.0, border_width=0.25,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.00, 0.3),
            platform_width=2.0, border_width=0.25,
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.05, step_height_range=(0.00, 0.08), step_width=0.25,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05, step_height_range=(0.00, 0.08), step_width=0.25,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.00, 0.12),
            num_waves=4, border_width=0.25,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.00, 0.05),
            noise_step=0.01, border_width=0.25,
        ),
    },
    curriculum=True,
)

QUADRUPED_ROUGH_TERRAINS_PLAY_CFG = QUADRUPED_ROUGH_TERRAINS_CFG.copy()
QUADRUPED_ROUGH_TERRAINS_PLAY_CFG.num_rows = 5
QUADRUPED_ROUGH_TERRAINS_PLAY_CFG.num_cols = 5
QUADRUPED_ROUGH_TERRAINS_PLAY_CFG.curriculum = False
```

The horizontal scale is halved from 0.1 to 0.05 metres, since a 0.1 metre height field cell is a third of this robot's foot to foot stance and would quantise the terrain into features the foot cannot resolve. The stair width is narrowed from 0.3 to 0.25 metres for the same reason.

### 8.3 Constructing the base environment configuration

`environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py` is built by transforming `cfg/SF/limx_base_env_cfg.py`, since the specification requires the two to mirror one another and since the three observation classes differ only in body names. The transformation was executed and its result validated, producing a 1434 line module that parses and declares exactly the thirteen classes `QuadrupedPFSceneCfg`, `CommandsCfg`, `ActionsCfg`, `ObservationsCfg`, `CoptObservationsCfg`, `HIMObservationsCfg`, `EventsCfg`, `RewardsCfg`, `TerminationsCfg`, `CurriculumCfg`, `QuadrupedPFEnvCfg`, `QuadrupedPFHIMEnvCfg` and `QuadrupedPFCoptEnvCfg`, with no residual reference to the biped's ankle.

Step one, copy the file.

```bash
cd /ws/tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/cfg
mkdir -p quadruped
printf 'from .base_env_cfg import *\nfrom .terrains_cfg import *\n' > quadruped/__init__.py
sed -i '1i from .quadruped import *' __init__.py     # FIRST, see section 8.1
cp SF/limx_base_env_cfg.py quadruped/base_env_cfg.py
```

Step two, apply the mechanical substitutions. Four are required and no more.

```bash
cd quadruped
# class names, SF prefix to PF. Run before any other substitution.
sed -i 's/\bSFSceneCfg\b/QuadrupedPFSceneCfg/g; s/\bSFEnvCfg\b/QuadrupedPFEnvCfg/g; s/\bSFHIMEnvCfg\b/QuadrupedPFHIMEnvCfg/g; s/\bSFCoptEnvCfg\b/QuadrupedPFCoptEnvCfg/g' base_env_cfg.py
# the package rename of section 2, if it has not already been applied
sed -i 's/\bfrom bipedal_locomotion\./from environments./g' base_env_cfg.py
# the terrain configuration of section 8.2
sed -i 's/\bBERKELEY_MIMIC_TERRAINS_CFG\b/QUADRUPED_ROUGH_TERRAINS_CFG/g' base_env_cfg.py
# the feet, in feet_lin_vel and robot_contact_force, six occurrences across three classes
sed -i 's/body_names="ankle_\.\*"/body_names="foot_.*_Link"/g' base_env_cfg.py
```

Step three, replace the three link length blocks. The pattern occurs three times, in `CoptObservationsCfg.MorphologyCfg`, in `CoptObservationsCfg.PredictedMorphologyCfg` and in `CoptObservationsCfg.CriticCfg`, and each is replaced by the eight pair form below. The traversal that produces it runs from `base_Link` to a foot through the chain of section 4.2, and the two variable segments are the thigh and the shank.

```python
                "parent_body_names": [
                    "hip_FR_thigh_Link",
                    "hip_FL_thigh_Link",
                    "hip_RR_thigh_Link",
                    "hip_RL_thigh_Link",
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                ],
                "child_body_names": [
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                    "foot_FR_Link",
                    "foot_FL_Link",
                    "foot_RR_Link",
                    "foot_RL_Link",
                ],
```

Step four, replace `EventsCfg` and `RewardsCfg` wholesale with the text of sections 8.7 and 8.9 below. These are the only two classes requiring judgement, and after their replacement the file contains no reference to `ankle`, which is the check that the substitution is complete.

### 8.4 The classes carried across unaltered

`QuadrupedPFSceneCfg` differs from `SFSceneCfg` in name alone. Its terrain importer, its dome light, its `robot: ArticulationCfg = MISSING`, its height scanner mounted on `base_Link` with a 0.1 metre grid over 1.6 by 1.0 metres, and its contact sensor over `{ENV_REGEX_NS}/Robot/.*` with a history of four and air time tracking, all apply to the quadruped without change. The scanner is retained at the biped's extent rather than shrunk, both because the canonical quadruped configurations use the same extent and because `base_height_rough_l2` averages over the whole patch and wants a broad sample of the local terrain.

`CommandsCfg`, `ActionsCfg`, `TerminationsCfg` and `CurriculumCfg` are carried across verbatim, as the specification requires and as section 7 confirms is sound. The command ranges sit inside the band the survey establishes for this class of robot. The action configuration selects `joint_names=[".*"]` and a scale of 0.25, and Isaac Lab derives the twelve dimensional action space from the articulation, so nothing states the dimension. The terminations select `base_Link`, which the quadruped also has. The curricula read no body or joint name.

### 8.5 The three observation classes

All three are carried across under the substitutions of step two and step three. `ObservationsCfg` supplies the vanilla runner, with a policy group of ten step history and a privileged critic. `HIMObservationsCfg` supplies the hybrid internal model runner, omitting the base linear velocity from the actor and adding the `obsHistory`, `estimatorGT` and `targetEnc` groups the estimator requires. `CoptObservationsCfg` supplies the co-optimisation runner, adding `morphologyObs`, `predictedMorphologyObs`, `predictedPrivilegedObs` and `obsHistory`.

The resulting dimensions, useful as a validation check at first run, are as follows. The velocity command term returns three values, the heading being consumed internally by the command generator. The policy group of `ObservationsCfg` is therefore 3 plus 3 plus 3 plus 12 plus 12 plus 12 plus 3, which is 48 per step and 480 over the ten step history. The `HIMObservationsCfg` policy group omits the base linear velocity and is 45 per step, unflattened over a ten step history. The morphology group is eight link lengths plus twenty five body masses plus two hundred and twenty five inertia components, which is 258. None of these is asserted anywhere in the configuration, every network being sized from the observation at construction, so a mismatch surfaces as a shape error at the first forward pass rather than as silent misbehaviour.

### 8.6 Compatibility with the three runners

`HIMManagerBasedRLEnv` at `tasks/locomotion/envs/him_env.py:6` overrides `step` alone and reads no body name, joint name or robot property, so it is robot agnostic and the quadruped needs nothing from it.

`CoptOnPolicyRunner` takes its observation group mapping from the agent configuration, at `co_optimisation/runners/copt_on_policy_runner.py:483`, so the quadruped's groups reach it through `PFQuadrupedCoptPPORunnerCfg.obs_groups` and require no runner change. The runner's design generator does require change, and that work is deferred in its entirety per the review, section 13 recording it as a defect left standing. The environment side is complete, which is the condition this plan undertakes to establish, so that the only outstanding work for a quadruped co-optimisation run is the generator itself.

The vanilla runner reads only the `policy` and `critic` groups and needs nothing.

### 8.7 The events configuration

Two of the biped's twelve terms are dropped, as the specification directs. `prepare_quantity_for_tron` exists solely to populate `env._foot_radius` for `feet_regulation` and `nominal_foot_position`, neither of which the quadruped uses, and `robot_joint_stiffness_and_damping_ankle` addresses a joint that does not exist.

One term is changed and one is added, and the reason is a finding rather than a preference. The biped's `add_base_mass` scales `base_Link` by 0.65 to 1.35, which on a biped whose trunk is 5 kilogrammes of a 9.6 kilogramme robot perturbs the total by roughly a fifth. This quadruped's trunk is 0.674397 kilogrammes of 15.837077, 4.3 percent of the machine, so the same scale perturbs the total by under two percent and the randomisation is very nearly inert. The obvious repair, an absolute addition of plus or minus 1.5 kilogrammes, is wrong and must not be made, since `randomize_rigid_body_mass` validates a scale range but not an additive one, at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:321`, so a sample of minus 1.5 would give the trunk a mass of minus 0.826 kilogrammes. The prescription is instead a small asymmetric payload on the trunk, guarded by `min_mass`, together with a second term scaling every body by 0.90 to 1.10, which delivers the plus or minus ten percent of TOTAL mass the references randomise over. The biped sketched exactly such a term, `add_link_mass`, and left it commented out.

The three gain randomisations are re-expressed over the quadruped's three joint groups with uniform ranges, the biped's asymmetric per group figures having been derived from absolute ranges that do not apply here. The push magnitude is halved, following the Go2 precedent in part, as section 3.9 explains.

```python
@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    # A payload on the trunk. The range is an ABSOLUTE add and is deliberately small and
    # asymmetric, the trunk weighing only 0.674397 kg, so that the sampled mass cannot go
    # negative. randomize_rigid_body_mass does not guard against a negative mass unless
    # min_mass is given, and min_mass is given here as a second line of defence.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-0.3, 0.6),
            "operation": "add",
            "min_mass": 0.2,
        },
    )
    # The trunk is only 4.3 percent of this robot's 15.837077 kg, so perturbing it alone
    # cannot deliver the plus or minus ten percent of TOTAL mass the reference
    # configurations randomise over. This term supplies that, scaling every body.
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    robot_joint_stiffness_and_damping_knee = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["knee_.._Joint"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_hip = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.._Joint"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_abad = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["abad_.._Joint"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_offsets = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_friction_model,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_all_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # interval. Halved against the biped, Isaac Lab's own Go2 rough configuration
    # disabling the push entirely on the ground that a light robot on randomised
    # terrain is already perturbed enough.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
        },
    )

```

### 8.8 The terminations and the curriculum

Carried across verbatim. One optional addition is recorded, `mdp.root_height_below_minimum_rough` at `terminations.py:15`, which would end an episode in which the trunk has collapsed onto the terrain. The biped does not use it and the quadruped need not, `illegal_contact` on `base_Link` catching the same event through a different mechanism.

### 8.9 The rewards configuration

The reasoning for every weight is in section 3. Eight of the biped's terms are dropped, being `rew_keep_ankle_pitch_zero_in_air`, `rew_no_fly`, `pen_feet_distance` and `pen_feet_regulation` outright, and the four already commented out. One is added, the abduction deviation penalty, on the survey's recommendation. The result is the sixteen terms the specification enumerates plus that one, seventeen in all, and every weight in the block below is the weight section 3 derives.

The swing clearance reward the survey also recommends is deliberately not included, and the reason is stated here because the omission is a decision rather than an oversight. `foot_clearance_reward` at `rewards.py:234` measures `body_pos_w[..., 2]`, a WORLD frame height, against a fixed target. On flat ground that is exactly the sole clearance for a spherical foot, the frame origin being the sphere's centre, and the term is correct. On generated terrain the ground is not at zero, so a foot lifted correctly over a raised tile reads as far above target and earns nothing, while a foot dragged along the floor of a pit reads as at target and earns fully. A term that is right on one variant and wrong on the other would have to be enabled on the flat tasks and zeroed on the rough ones, which makes the two families no longer comparable and adds a tuning surface before the configuration has produced a single gait. The immediate objective is a working locomotion policy under the vanilla runner, so the term is left out of every variant, and a terrain relative variant measuring against the height scanner is recorded as defect 6 of section 13 and scheduled there.

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # termination related rewards
    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.05)

    # tracking rewards. The 2 to 1 ratio of linear to angular is the one every
    # surveyed quadruped configuration uses.
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=12.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )

    # penalisations
    # 0.292 m is the standing height at which the 0.022 m foot sphere rests on the
    # ground given the 0.270 m kinematic drop of the nominal crouch.
    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            "target_height": 0.292,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-5.0,
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-05)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.075)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Every body except the four feet. The feet are named foot_<LEG>_Link and are
    # therefore disjoint from all three expressions below.
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_Link", "abad_.*", "hip_.*", "knee_.*"],
            ),
            "threshold": 10.0,
        },
    )

    # The abduction joints are the degrees of freedom a policy widens first, to buy a
    # larger support polygon, and this robot's 0.264 m stance is narrow against its
    # 0.426 m leg. Follows the hip_pos term of Extreme Parkour.
    pen_abad_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["abad_.._Joint"])},
    )

    # gait shaping
    # threshold_min is half the stride period of a 2 Hz trot at a duty factor of one
    # half, which is what the canonical 0.5 s is for ANYmal's 1 Hz trot.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
            "command_name": "base_velocity",
            "threshold_min": 0.25,
            "threshold_max": 0.40,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link"),
        },
    )
```

### 8.10 The three environment classes

`QuadrupedPFEnvCfg`, `QuadrupedPFHIMEnvCfg` and `QuadrupedPFCoptEnvCfg` are carried across from their `SF` counterparts under the rename alone. Each aggregates the scene at four thousand and ninety six environments with a spacing of 2.5, its own observation class, and the shared actions, commands, rewards, terminations, events and curriculum. Each sets a decimation of 4, an episode length of 20 seconds and a simulation step of 0.005, giving the 50 hertz control and 200 hertz physics that section 5 assumes throughout. `QuadrupedPFCoptEnvCfg` additionally raises `gpu_max_rigid_patch_count` to 2 to the 19th, which it needs for the same reason the biped does, the population wide reset at every design swap overflowing the default patch buffer.

## 9. Tasks 8 and 9, the configuration hierarchy and the task registrations

### 9.1 The hierarchy

The specification asks for a two level hierarchy beneath each base class, in place of the biped's four, on the ground that the quadruped has no USD variant and therefore needs no layer to switch between URDF and USD. That is correct, and the result is eighteen classes across three families. The file below was generated and validated, and declares exactly those eighteen.

Create `environments/environments/tasks/locomotion/robots/quadruped_pointfoot_env_cfg.py`.

```python
from isaaclab.utils import configclass

from environments.assets.config.quadruped_identified_cfg import QUADRUPED_IDENTIFIED_CFG
from environments.tasks.locomotion.cfg.quadruped.base_env_cfg import (
    QuadrupedPFCoptEnvCfg,
    QuadrupedPFEnvCfg,
    QuadrupedPFHIMEnvCfg,
)
from environments.tasks.locomotion.cfg.quadruped.terrains_cfg import (
    QUADRUPED_ROUGH_TERRAINS_CFG,
    QUADRUPED_ROUGH_TERRAINS_PLAY_CFG,
)

# The nominal crouch of the MuJoCo keyframe. The knee is the negation of twice the hip,
# which places each foot exactly beneath its hip pitch axis at a drop of 0.270 m.
QUADRUPED_INIT_JOINT_POS = {
    "abad_.._Joint": 0.0,
    "hip_.._Joint": 0.884337,
    "knee_.._Joint": -1.768673,
}


######################
# Quadruped Base Environments
######################


@configclass
class QuadrupedPFBaseEnvCfg(QuadrupedPFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = QUADRUPED_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = dict(QUADRUPED_INIT_JOINT_POS)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # The viewer offsets are scaled to a 0.292 m robot, the biped's (-2.5, 0, 2.5)
        # framing a body three times this one's standing height.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class QuadrupedPFBaseEnvCfg_PLAY(QuadrupedPFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomisation for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.terrain_levels = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.curriculum.modify_linear_tracking_reward_std = None
        self.curriculum.modify_angular_tracking_reward_std = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)

######################
# Quadruped Base Environments
######################


@configclass
class QuadrupedPFHIMBaseEnvCfg(QuadrupedPFHIMEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = QUADRUPED_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = dict(QUADRUPED_INIT_JOINT_POS)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # The viewer offsets are scaled to a 0.292 m robot, the biped's (-2.5, 0, 2.5)
        # framing a body three times this one's standing height.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class QuadrupedPFHIMBaseEnvCfg_PLAY(QuadrupedPFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomisation for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.terrain_levels = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.curriculum.modify_linear_tracking_reward_std = None
        self.curriculum.modify_angular_tracking_reward_std = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)

######################
# Quadruped Base Environments
######################


@configclass
class QuadrupedPFCoptBaseEnvCfg(QuadrupedPFCoptEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.replicate_physics = False

        self.scene.robot = QUADRUPED_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = dict(QUADRUPED_INIT_JOINT_POS)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # The viewer offsets are scaled to a 0.292 m robot, the biped's (-2.5, 0, 2.5)
        # framing a body three times this one's standing height.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class QuadrupedPFCoptBaseEnvCfg_PLAY(QuadrupedPFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomisation for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.terrain_levels = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.curriculum.modify_linear_tracking_reward_std = None
        self.curriculum.modify_angular_tracking_reward_std = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


@configclass
class QuadrupedPFBlindFlatEnvCfg(QuadrupedPFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the scene terrain is already a plane, only the level curriculum must go
        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFBlindFlatEnvCfg_PLAY(QuadrupedPFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFBlindRoughEnvCfg(QuadrupedPFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


@configclass
class QuadrupedPFBlindRoughEnvCfg_PLAY(QuadrupedPFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_PLAY_CFG


@configclass
class QuadrupedPFHIMBlindFlatEnvCfg(QuadrupedPFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the scene terrain is already a plane, only the level curriculum must go
        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFHIMBlindFlatEnvCfg_PLAY(QuadrupedPFHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFHIMBlindRoughEnvCfg(QuadrupedPFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


@configclass
class QuadrupedPFHIMBlindRoughEnvCfg_PLAY(QuadrupedPFHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_PLAY_CFG


@configclass
class QuadrupedPFCoptBlindFlatEnvCfg(QuadrupedPFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the scene terrain is already a plane, only the level curriculum must go
        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFCoptBlindFlatEnvCfg_PLAY(QuadrupedPFCoptBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFCoptBlindRoughEnvCfg(QuadrupedPFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


@configclass
class QuadrupedPFCoptBlindRoughEnvCfg_PLAY(QuadrupedPFCoptBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_PLAY_CFG
```

Four points of departure from the biped's file are deliberate. The initial joint positions are lifted into a module level dictionary rather than repeated eighteen times, the biped's file repeating its six entry dictionary twenty six times and being the longer for it. The viewer offsets are scaled to the robot, the biped's eye at 2.5 metres framing a body three times this one's height. The `_PLAY` classes disable all seven curriculum terms rather than the biped's two, since the biped's `_PLAY` sets `modify_command_velocity`, an attribute no `CurriculumCfg` in the tree declares, and leaves the five that do exist running. The `_PLAY` classes clear `add_link_mass` as well as `add_base_mass`, the quadruped declaring both where the biped leaves the second commented out, so that an evaluation sees the nominal mass on every body and not merely on the trunk.

The name collision of section 8.1 recurs here, `QuadrupedPFBaseEnvCfg`, `QuadrupedPFBaseEnvCfg_PLAY`, `QuadrupedPFBlindFlatEnvCfg`, `QuadrupedPFBlindFlatEnvCfg_PLAY`, `QuadrupedPFBlindRoughEnvCfg` and `QuadrupedPFBlindRoughEnvCfg_PLAY` all existing already in `robots/limx_pointfoot_env_cfg.py`. It is harmless here and only here, because `robots/__init__.py` imports those modules as modules and addresses every class module qualified, so the two sets never meet in one namespace. Do not add a star import of either module.

### 9.2 The agent configuration

Create `environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py`. The specification names `PFQuadrupedPPORunnerCfg` and requires it to carry the same hyperparameters as `SF_TRON1AFlatPPORunnerCfg`, which it does. Two further classes are added because the co-optimisation runner cannot function without an `obs_groups` mapping, exactly as `SFCoptPPORunnerCfg` supplies one for the biped.

```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

from environments.utils.wrappers.rsl_rl.rl_mlp_cfg import (
    DecoderCfg,
    EncoderCfg,
    RslRlPpoAlgorithmMlpCfg,
)


# Mirrors SF_TRON1AFlatPPORunnerCfg exactly, differing only in experiment_name. The
# policy and algorithm hyperparameters are deliberately identical to the biped's so
# that a difference between the two robots' learning curves is attributable to the
# robot and the reward set rather than to the optimiser.
@configclass
class PFQuadrupedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 25
    max_iterations = 30000
    save_interval = 500
    experiment_name = "quadruped_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=19,
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        orthogonal_init=False,
    )


# The co-optimisation runner reads obs_groups to decide which observation groups feed
# the actor and which the critic, at copt_on_policy_runner.py:483.
@configclass
class PFQuadrupedCoptPPORunnerCfg(PFQuadrupedPPORunnerCfg):
    experiment_name: str = "quadruped_copt"
    max_iterations: int = 45000
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "morphologyObs"],
        "critic": ["critic"],
    }


@configclass
class PFQuadrupedCoptLearnedModelPPORunnerCfg(PFQuadrupedCoptPPORunnerCfg):
    experiment_name: str = "quadruped_copt_learned"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    decoder = DecoderCfg(
        output_detach=False,
        num_output_dim=3,
        hidden_dims=[128, 256, 512],
        activation="elu",
        orthogonal_init=False,
    )
```

### 9.3 The task registrations

Extend `environments/environments/tasks/locomotion/robots/__init__.py`. The specification names `Isaac-Quadruped-Blind-Rough-v0`, and the eleven further identifiers below complete the matrix so that each of the three runners has a flat variant, a rough variant and an evaluation variant, matching what the biped offers. The registrations are written as twelve explicit `gym.register` blocks under banner comments, which is the form the file already uses for every biped task, and no loop is introduced. The style is deliberate. A reader who greps the file for a task identifier lands on the whole registration rather than on a tuple whose meaning depends on a loop body elsewhere, and a single task that must diverge later, in its entry point or its keyword arguments, diverges by editing its own block rather than by breaking the table apart.

```python
# --- add to the import block at the head of robots/__init__.py ---
from environments.tasks.locomotion.agents.quadruped_rsl_rl_ppo_cfg import (
    PFQuadrupedCoptLearnedModelPPORunnerCfg,
    PFQuadrupedCoptPPORunnerCfg,
    PFQuadrupedPPORunnerCfg,
)

# --- extend the existing `from . import (...)` tuple with ---
#     quadruped_pointfoot_env_cfg,

# --- add beside the other runner instances ---
quadruped_runner_cfg = PFQuadrupedPPORunnerCfg()

quadruped_him_runner_cfg = PFQuadrupedPPORunnerCfg()

quadruped_copt_runner_cfg = PFQuadrupedCoptPPORunnerCfg()

quadruped_copt_learned_runner_cfg = PFQuadrupedCoptLearnedModelPPORunnerCfg()

# --- append after the last existing gym.register block ---
##################################
# Quadruped Blind Flat Environment
##################################
gym.register(
    id="Isaac-Quadruped-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

###################################
# Quadruped Blind Rough Environment
###################################
gym.register(
    id="Isaac-Quadruped-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

######################################
# Quadruped HIM Blind Flat Environment
######################################
gym.register(
    id="Isaac-Quadruped-HIM-Blind-Flat-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-HIM-Blind-Flat-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

#######################################
# Quadruped HIM Blind Rough Environment
#######################################
gym.register(
    id="Isaac-Quadruped-HIM-Blind-Rough-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-HIM-Blind-Rough-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

#################################
# Quadruped Copt Flat Environment
#################################
gym.register(
    id="Isaac-Quadruped-Copt-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

##################################
# Quadruped Copt Rough Environment
##################################
gym.register(
    id="Isaac-Quadruped-Copt-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Copt-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

################################################
# Quadruped Copt Learned Model Rough Environment
################################################
gym.register(
    id="Isaac-Quadruped-Copt-Learned-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_learned_runner_cfg,
    },
)
```

Two points of correspondence with the existing file are worth stating. `HIMManagerBasedRLEnv` is passed as a class rather than as an import string, exactly as the biped's hybrid internal model registrations do at `robots/__init__.py:162` and `:531`, because the class is already imported at the head of the file. And `quadruped_him_runner_cfg` is a second instance of `PFQuadrupedPPORunnerCfg` rather than of a distinct class, mirroring `limx_sf_him_blind_flat_runner_cfg` at `:32`, the hybrid internal model runner taking its policy and algorithm class names from `train.py` rather than from the configuration.

## 10. Task 10, the container tooling and the training scripts

### 10.1 `../djinn`

Beyond the installation path of section 2.3, four task modes are added to the `start train` dispatch and four to `start play`, in the same `elif` chain the biped modes use.

```bash
                elif [[ "$3" == "quadruped" ]]; then
                    task="Isaac-Quadruped-Blind-Rough-v0"
                elif [[ "$3" == "quadruped-flat" ]]; then
                    task="Isaac-Quadruped-Blind-Flat-v0"
                elif [[ "$3" == "quadruped-him" ]]; then
                    task="Isaac-Quadruped-HIM-Blind-Rough-v0"
                    policy_type="HIMPPO"
                elif [[ "$3" == "quadruped-copt" ]]; then
                    task="Isaac-Quadruped-Copt-Rough-v0"
                    policy_type="COPT"
```

The `start play` chain takes the corresponding `-Play-v0` identifiers, with `quadruped-him` mapping to `Isaac-Quadruped-HIM-Blind-Rough-Play-v0` rather than to the training identifier, which is what the biped's `him` branch mistakenly does at `djinn:175`. That is an existing defect of the biped path, recorded here and not repaired, since repairing it would change the behaviour of an existing caller.

### 10.2 `scripts/rsl_rl/train.py`

The specification judges that no change is needed. For the two runners in the scope of this plan that judgement is correct and is confirmed by inspection. The vanilla path constructs `OnPolicyRunner` from the task identifier and the agent configuration alone, at `train.py:190` and `:249`, and reads no robot property. The hybrid internal model path adds only the two class name assignments at `:192` to `:194`. Neither touches an asset path, a joint name or a body name, so both accept `Isaac-Quadruped-Blind-Rough-v0` and its siblings unmodified.

The co-optimisation path is different, and its remedy is deferred with the rest of the co-optimisation work. Lines 198 to 210 hardcode the biped URDF as `_base_urdf` and hardcode `param_ranges` to the biped's parameter names, so a quadruped co-optimisation run would rewrite the biped's URDF and spawn the wrong robot. This is recorded in section 13 together with the design generator it belongs to, and until both are addressed the four co-optimisation identifiers registered in section 9.3 must not be launched. Registering them now costs nothing, `gym.register` constructing no environment, and it keeps the task matrix complete against the day the generator work lands.

## 11. Validation

No simulator is available in the development container, so every check below is a code review or a static execution. They are ordered so that a failure stops the sequence at the cheapest point.

### 11.1 Static checks

1. The tree parses. `python -c "import ast,pathlib;[ast.parse(p.read_text()) for p in pathlib.Path('environments').rglob('*.py')]"` from the repository root.
2. No reference to the old package survives. `grep -rn 'bipedal_locomotion' --include='*.py' --include='*.toml' . | grep -v build | grep -v egg-info` returns nothing.
3. The URDF is a valid tree. Parse it, confirm twenty five links and twenty four joints of which twelve are revolute, confirm a single root at `base_Link`, confirm every link reaches the root without a cycle, confirm the summed mass equals 15.837077 to six decimal places, and confirm that `abad_FR_Joint` and `abad_RR_Joint` carry the axis `-1 0 0` while `abad_FL_Joint` and `abad_RL_Joint` carry `1 0 0`. This was executed against the generator of sub-task 2.3 and passed.
4. The three joint name expressions of section 6 are disjoint and together match exactly twelve joints, and none matches a fixed joint. Executed and passed.
5. Every body expression used in a reward or a sensor resolves. `foot_.._Link` matches the four feet. The union of `base_Link`, `abad_.*`, `hip_.*` and `knee_.*` matches the remaining twenty one bodies and no foot.
6. `base_env_cfg.py` contains no occurrence of `ankle`, which is the single check that the substitution of section 8.3 is complete. Executed and passed.
7. The eighteen classes of section 9.1 and the three of section 9.2 are declared. Executed and passed.
8. No quadruped class name collides with an existing one. Every class the two new modules declare carries the `Quadruped` prefix of section 8.1, so `grep -rn 'class Quadruped' environments/` returns exactly the twenty two names and `python -c "from environments.tasks.locomotion.cfg import *"` resolves the biped's `PFEnvCfg` to `cfg/PF/limx_base_env_cfg.py` as it does today. This is the check that the fourth star import added to `cfg/__init__.py` shadows nothing.

### 11.2 Consistency checks against the model

9. The nominal joint angles lie inside the soft limits. The soft factor is 0.9, giving abduction from -0.9 to 0.9 against a nominal of 0, hip from -0.965 to 3.265 against 0.884337, and knee from -2.78 to -0.62 against -1.768673. All three clear, and all three still clear after the reset offset of plus or minus 0.2 and the startup offset of plus or minus 0.05.
10. The base height target of 0.292 equals the kinematic drop of 0.270 plus the foot radius of 0.022, and the articulation spawns 28 millimetres above it.
11. The action scale of 0.25 sits below all nine bounds of section 5.5, the tightest being the knee's slew bound of 0.34, so neither velocity saturation nor the torque ceiling binds.
12. The derived gains reproduce section 5.3. For each joint, `sqrt(Kp / I_eff)` divided by two pi is the tabulated natural frequency, `Kd / (2 sqrt(Kp I_eff))` is 0.6 to two significant figures at the nominal crouch, and the stance torque divided by `Kp` is the tabulated sag. The three natural frequencies are 6.46, 7.02 and 10.15 hertz and none exceeds the observability ceiling of 10 hertz by more than two percent.

### 11.3 Runner compatibility, by inspection

13. The vanilla runner reads the `policy` and `critic` groups, both of which `ObservationsCfg` declares.
14. The hybrid internal model runner additionally reads `obsHistory`, `estimatorGT` and `targetEnc`, all three of which `HIMObservationsCfg` declares, and `HIMManagerBasedRLEnv` reads no robot property.
15. The co-optimisation runner reads the groups named by `PFQuadrupedCoptPPORunnerCfg.obs_groups`, which are `policy`, `morphologyObs` and `critic`, all three of which `CoptObservationsCfg` declares. This is the environment level compatibility the review asks be confirmed, and it holds. `CoptOnPolicyRunner` resolves those groups at `co_optimisation/runners/copt_on_policy_runner.py:483` and reads no body name, joint name or asset path of its own, so nothing outside the design generator stands between this configuration and a quadruped co-optimisation run. That generator does not yet accept the robot, per defect 7 of section 13, so the four co-optimisation identifiers must not be launched before that work lands.
16. `gym.make` succeeds for each of the twelve registered identifiers, which can be checked without the simulator only as far as the registration itself, so this reduces to confirming that every `env_cfg_entry_point` names a class the module declares.

### 11.4 Downstream consumers

17. Evaluation is out of scope, `scripts/rsl_rl/play.py` and the statistics pipeline behind it being deferred entire per defect 8 of section 13. No check is prescribed here beyond confirming that nothing in the training path imports either, which it does not, `train.py` sharing only `cli_args` with them.
18. `experiment_params.apply_reward_cfg` resolves each reward term of a dumped `params/env.yaml` by import. Every function the quadruped configuration names already exists in the tree, so no term will fail to import.

## 12. Documents to write and to update

Three documents are written. `../tron1-rl-isaaclab-cozum/context/quadruped_xml_to_urdf_conversion.md` records sections 4.1 to 4.5 as fact, including the four corrections applied to the MuJoCo model and the reasoning behind each, and additionally serves as the general procedure for a MuJoCo to URDF conversion, per section 4.6. `../tron1-rl-isaaclab-cozum/context/joint_control_analysis_quadruped.md` records section 5, referring to `joint_control_analysis.md` for the theory it does not restate and contributing two treatments that document lacks, the derivation of a proportional gain from a stance load in section 5.3 and the derivation of a velocity range in section 5.4. A short `../tron1-rl-isaaclab-cozum/context/quadruped.md` records the reward, event and curriculum configuration as built, in the manner of `tron1.md`, and is written on completion rather than in advance.

Five documents are updated. `../tron1-rl-isaaclab-cozum/context/README.md` registers the two new documents, and separately loses its closing pointer to `base_robot.md`, a file the rename deleted. `../tron1-rl-isaaclab-cozum/ARCHITECTURE.md` gains the quadruped's class hierarchy and its task registry. `../ARCHITECTURE.md` gains the same at the workspace level, together with the directory rename. `../context/literature.md` gains a new cluster for the canonical quadruped reward configurations, since nine of the works cited here are not yet registered there. `../plans/README.md` registers this plan with its status.

One correction is made to `../CLAUDE.md`. Clause 3 names `bipedal_locomotion` as the shared module whose callers must be enumerated before an edit, and must now name `environments`.

Three bodies of work are deliberately excluded from this plan and are to be planned independently, each having more nuance than a section of this document could carry. The co-optimisation design generator and the training script branch that feeds it are defect 7 below. The evaluation script and the statistics pipeline behind it are defect 8, the rename of section 2 being the only edit this plan makes to either. A terrain relative swing clearance reward, which would let the quadruped gain the term the survey recommends without the flat ground assumption that makes it wrong on generated terrain, is defect 6. None blocks the objective this plan serves, which is a quadruped locomotion policy trained under the vanilla runner.

## 13. Defects recorded and left standing

Per clause 6 of the change discipline, the following are recorded rather than repaired, so that a later implementation may opt into a remedy deliberately. Four defects carried by earlier drafts of this plan have since been repaired rather than recorded, on review, and are named here so that a reader of both drafts is not left searching for them. The calf's inertia and the burial of the foot sphere are corrected in `my_design.xml` and in the URDF together, per section 4.3 and section 4.4. The unmirrored abduction axes are mirrored, per section 4.2. And the knee's natural frequency, which stood at 27 percent of the control rate under the rejected proportional gain of 100, stands at 20.3 percent under the derived gain of 50, per section 5.3.

1. `my_design.xml` declares an armature of 0.01 kilogramme metres squared identically on all twelve joints, which is high by roughly a factor of six for a GO-M8010-6 with a reduction of 6.33, per section 4.3. It is carried across unaltered so that the two models remain interchangeable, and the consequence, that it supplies between 41 and 81 percent of every effective inertia and therefore inflates the gains derived in section 5.3, is recorded rather than corrected because no hardware identification is available to replace it with.
2. `context/joint_control_analysis.md` section 6.5 bounds the peak transient velocity by `A omega_n / sqrt(1 - zeta^2)`, which overestimates by a factor near 2.5 because it ignores the exponential decay before the sine reaches its first peak, per section 5.4. The correction belongs in that document and is not applied here.
3. `environments/setup.py` lists only the top level package, so a non editable install would ship an empty package, per sub-task 1.6. Inert under `pip install -e`.
4. `djinn:175` maps the biped's `play him` mode to a training identifier rather than an evaluation one, per section 10.1. Not repaired, an existing caller depending on it.
5. `randomize_rigid_body_mass` validates a scale range for positivity but performs no such check on an additive one, at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:321`, so an additive range wider than a body's own mass silently yields a negative mass unless `min_mass` is passed. This is an upstream behaviour and is worked around rather than repaired, per section 8.7. Any future additive mass randomisation in this project should pass `min_mass`.
6. `mdp.foot_clearance_reward` at `rewards.py:234` measures `body_pos_w[..., 2]`, a world frame height, and is therefore correct only where the ground lies at zero, per section 8.9. The quadruped configuration omits the term entirely rather than enabling it on the flat variants alone, so that the flat and the rough families remain comparable. The remedy is a variant measuring the foot height against the height scanner already mounted on `base_Link`, which would make the term correct on both families and would also repair it for the biped configurations that carry it today. This is new work on a shared module and is not written here.
7. The co-optimisation design generator does not accept this robot, and the whole of that work is deferred. Section 8.6 establishes that the environment side is complete, `CoptOnPolicyRunner` taking its observation group mapping from `PFQuadrupedCoptPPORunnerCfg.obs_groups` and reading no robot property of its own, so the generator is the only outstanding piece. Its scope is as follows, recorded so that the later plan need not rediscover it. `co_optimisation/co_optimisation/runners/usd_generator.py` is keyed to the biped's names at six places, line 73 importing the four TRON1 actuator configurations, lines 108 to 115 mapping the biped's eight joint names to four groups, lines 120 to 123 listing the scalable links, line 127 naming the inertial measurement link skipped during mass scaling, lines 167 to 170 mapping a link to the joint whose z offset encodes its length, and lines 175 to 178 mapping that link to its scale parameter. None may be edited in place, three biped co-optimisation tasks reading them and runs being in flight, so the clause 4 remedy is a robot specification object carrying the six tables with an optional `robot_spec` argument on `RandomDesignGenerator.__init__` defaulting to the present values. The quadruped's tables are the twelve joints grouped by name prefix, the three actuator configurations of section 6, the three cylinder housings per leg as the scalable links, an empty skip list since the robot has no inertial measurement link, and the thigh and shank lengths carried by the z offsets of `knee_<LEG>_Joint` and `foot_<LEG>_joint`. Two further items belong to the same work. `scripts/rsl_rl/train.py` lines 198 to 210 hardcode the biped URDF and the biped parameter ranges and must become task conditional, per section 10.2. And the generator emits absolute extents rounded to two decimal places, which quantises this robot's 0.213 metre segments at 4.7 percent against a parameter range of plus or minus 25 percent, a defect already recorded as outstanding in `COPT_INVESTIGATION_PLAN.md` and materially worse for this robot than for the biped, whose segments are 0.3 and 0.35 metres.
8. `scripts/rsl_rl/play.py` does not evaluate this robot, and that work is deferred entire, the script being non functional for TRON1 today and its repair requiring an independent analysis rather than a patch. Its scope against the quadruped is as follows. Line 112 declares `FEET_LINK_NAMES = "Link6[LR]"`, which is SD_BRS1's foot naming, as a module level constant, and line 141 sets `SOLE_OFFSETS` to SD_BRS1's twelve point sole polygon, so a quadruped checkpoint resolves zero feet. Line 364 then computes `feet_pos[:, 0, :] - feet_pos[:, 1, :]` and raises an index error, and the episode dump is lost. Even with the names corrected that line assumes exactly two feet and would report the separation of the first two of four, and a spherical foot has no sole polygon at all, its frame origin lying a radius above its lowest point. `scripts/analysis/stats.py` belongs to the same work, having been written against bipeds throughout, and must tolerate a dump lacking `feet_distance` and carrying four feet where it has previously seen two. One qualification is needed so that this defect is not read as excluding the file altogether. Sub-tasks 1.1 and 1.2 of section 2 do edit `play.py`, at lines 84, 85, 658 and 692, but only to rewrite the package name and the four absolute URDF paths that the directory rename invalidated. Without those the module does not import at all, for any robot. Nothing in its evaluation logic is touched.

## 14. Bibliography

1. Rudin, N., Hoeller, D., Reist, P., Hutter, M. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Conference on Robot Learning, 2021. arXiv:2109.11978. Reward scales, the terrain curriculum and the command curriculum were read from the `leggedrobotics/legged_gym` repository, files `legged_gym/envs/base/legged_robot_config.py`, `legged_gym/envs/base/legged_robot.py` and `legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`, retrieved 2026-08-19.
2. Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., and others. Orbit, A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE Robotics and Automation Letters 8(6), 3740 to 3747, 2023. arXiv:2301.04195. The author list was not verified beyond the six named and the short form is used. Configuration values were read from the `isaac-sim/IsaacLab` repository, files `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`, `mdp/rewards.py`, `mdp/curriculums.py`, `config/anymal_c/rough_env_cfg.py`, `config/anymal_c/flat_env_cfg.py`, `config/go2/rough_env_cfg.py` and `config/go2/flat_env_cfg.py`, retrieved 2026-08-19, and from the local checkout at `/ws/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`.
3. Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., Pang, J. Hybrid Internal Model, Learning Agile Legged Locomotion with Simulated Robot Response. International Conference on Learning Representations, 2024. arXiv:2312.11460. Reward weights cross verified against the `InternRobotics/HIMLoco` repository, files `legged_gym/legged_gym/envs/a1/a1_config.py` and `envs/base/legged_robot_config.py`, retrieved 2026-08-19.
4. Nahrendra, I. M. A., Yu, B., Myung, H. DreamWaQ, Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning. IEEE International Conference on Robotics and Automation, 2023. arXiv:2301.10602. The reward table was retrieved from the article's HTML rendering and no official repository was located to corroborate it, so its weights carry the caution due a single source.
5. Unitree Robotics. `unitreerobotics/unitree_rl_gym` repository, files `legged_gym/envs/go2/go2_config.py`, `legged_gym/envs/g1/g1_config.py` and `legged_gym/envs/base/legged_robot_config.py`, retrieved 2026-08-19. A software artefact with no accompanying article identified.
6. Margolis, G. B., Agrawal, P. Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior. Conference on Robot Learning, 2022. arXiv:2212.03238. Reward functions read from the `Improbable-AI/walk-these-ways` repository, files `go1_gym/envs/go1/go1_config.py` and `go1_gym/envs/rewards/corl_rewards.py`, retrieved 2026-08-19. The retrieved source did not confirm whether the author list extends beyond the two named.
7. Cheng, X., Shi, K., Agarwal, A., Pathak, D. Extreme Parkour with Legged Robots. IEEE International Conference on Robotics and Automation, 2024. arXiv:2309.14341. Reward scales read from the `chengxuxin/extreme-parkour` repository, files `legged_gym/legged_gym/envs/a1/a1_parkour_config.py`, `envs/base/legged_robot.py` and `envs/base/legged_robot_config.py`, retrieved 2026-08-19.
8. Siekmann, J., Green, K., Warila, J., Fern, A., Hurst, J. Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. IEEE International Conference on Robotics and Automation, 2021. arXiv:2011.01387.
9. van Marum, B., and others. Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking. 2024. arXiv:2404.19173. The author list was not verified in this pass and is reused from the existing entry in `../context/literature.md` cluster 7.
10. NVIDIA. Tutorial 6, Joint Gains Tuning. Isaac Sim OpenUSD Tuning Tutorials documentation, `docs.isaacsim.omniverse.nvidia.com`, retrieved 2026-08-19.
11. Guan, N., Yu, S., Zhu, S., Kim, D. Impedance Matching, Enabling an RL-Based Running Jump in a Quadruped Robot. Ubiquitous Robots, 2024. arXiv:2404.15096.
12. Unitree Robotics. GO Motor product specification, `unitree.com`, and Go1 Datasheet EN v3.0, together with the `unitreerobotics/unitree_ros` repository file `go2_description/urdf/go2_description.urdf`, retrieved 2026-08-19. Source of the GO-M8010-6 ratings of 23.7 newton metres at 30.1 radians per second for the hip and thigh and 35.55 newton metres at 20.06 radians per second for the calf.
13. Ji, G., Mun, J., Kim, H., Hwangbo, J. Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. IEEE Robotics and Automation Letters 7(2), 2022. arXiv:2202.05481.
14. Kumar, A., Fu, Z., Pathak, D., Malik, J. RMA, Rapid Motor Adaptation for Legged Robots. Robotics, Science and Systems, 2021. arXiv:2107.04034.
