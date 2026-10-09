# Visualisation Integration

## Merging the Two Dashboards into One Robot Agnostic `scripts/analysis/dashboard.py` Driven by a `--robot` Flag

Status, written 2026-09-03, IMPLEMENTED IN FULL on 2026-09-03. Every edit below stands in the source tree, `tron1-rl-isaaclab-cozum/scripts/analysis/dashboard.py` having been deleted, `dashboard-brs.py` renamed onto that path, and section 4 applied to the survivor, which now runs to 1211 lines against the 1097 it carried before. Section 6 records the seven checks, six passed and one deferred for want of a surviving TRON1 dump, and section 10 records the two divergences from the specification. Its factual grounding is the environment configurations under `tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/cfg/`, the four robot URDFs under `tron1-rl-isaaclab-cozum/environments/environments/assets/urdf/`, and the dump writer in `tron1-rl-isaaclab-cozum/scripts/rsl_rl/play.py`, each read on 2026-09-03.

This document sits at the workspace level rather than in the simulation repository, although every edit it prescribes falls inside that repository, because two further copies of both dashboards survive under `/ws/IsaacLab/logs/rsl_rl/` and because the pipeline the dashboards terminate is recorded at the workspace level in [../context/gait_metrics.md](../context/gait_metrics.md). The tie break of `/ws/CLAUDE.md` rule 25 prefers the workspace level where the scope is unclear, and it is unclear here.

## 1. The problem

The repository carries two dashboards that read the same dumps and disagree about the robot those dumps describe. `scripts/analysis/dashboard.py` was written for the TRON1 solefoot and fixes its articulation at eight joints, testing `metric_data.shape[1] != 8` at `scripts/analysis/dashboard.py:167` and iterating `range(8)` at `:175` and `:361` against the eight name list at `:38`. `scripts/analysis/dashboard-brs.py` was written for the SD_BRS1 and has since received every improvement the gait work stream produced, the feet window, the base position window, the reward window, the ankle separation window and the statistics comparison table, while retaining four SD_BRS1 constants at `scripts/analysis/dashboard-brs.py:56`, `:75`, `:78` and `:79`.

The divergence is not symmetric. Every view the TRON1 file offers, the base and commanded velocity grid, the five joint metric grids and the torque against velocity scatter, is present in the SD_BRS1 file in a strictly more capable form, since the latter derives its joint count from the resolved name list at `scripts/analysis/dashboard-brs.py:246` rather than fixing it at eight. The TRON1 file therefore contributes exactly one thing the other lacks, its fallback joint name list at `scripts/analysis/dashboard.py:38`, and the merge consists of carrying that list across and deleting the rest.

The hard coding is likewise not uniform in severity. Two of the four constants are already demoted to fallbacks, `JOINT_NAMES` at `scripts/analysis/dashboard-brs.py:56` and `FEET_NAMES` at `:75` being consulted only where a dump carries no recorded order, since `play.py` has recorded `joint_names`, `body_names` and `feet_names` from the live articulation at `scripts/rsl_rl/play.py:368` through `:371` since 2026-07-31 and `_names_from_dumps` at `scripts/analysis/dashboard-brs.py:86` prefers what is recorded. The other two are consulted unconditionally. `SOLE_CLEARANCE_TARGET` at `:78` is drawn as a dotted reference line on every sole clearance panel at `:572`, and `CONTACT_FORCE_THRESHOLD` at `:79` on every contact force panel at `:568`, so a KScale dump is at present read against a clearance target 0.03 m too high and a quadruped dump against a clearance channel it does not possess.

## 2. What the four robots actually present

The four robots differ on the four axes the dashboard must know about, the number of legs, the number of feet the dump records, whether a sole clearance channel exists at all, and what reference level that channel should be read against. The table below states each from the sources rather than from recall.

| Robot | `--robot` value | Legs | Joints | Feet regex resolved by `play.py` | Sole table | Clearance reference | Contact threshold |
|---|---|---|---|---|---|---|---|
| SD_BRS1 | `brs` | 2 | 10 | `Link6[LR]` at `scripts/rsl_rl/play.py:174` | `SD_BRS1_SOLE_OFFSETS`, twelve points | 0.08 m, `swing_height` at `cfg/SF/brs_base_env_cfg.py:117` | 1.0 N at `cfg/SF/brs_base_env_cfg.py:853` |
| KScale | `kscale` | 2 | 12 | `foot_6061.*` at `scripts/rsl_rl/play.py:175` | `KSCALE_SOLE_OFFSETS`, twelve points | 0.05 m, `swing_height` at `cfg/SF/kscale_base_env_cfg.py:136` | 1.0 N at `cfg/SF/kscale_base_env_cfg.py:849` |
| TRON1 solefoot | `tron` | 2 | 8 | `ankle_.*` at `scripts/rsl_rl/play.py:176` | none | none, the gait command is commented out at `cfg/SF/limx_base_env_cfg.py:102` | 1.0 N, hard coded at `mdp/rewards.py:259` |
| Quadruped | `quadruped` | 4 | 12 | `foot_.._Link` at `scripts/rsl_rl/play.py:177` | none | none, the gait command is commented out at `cfg/quadruped/base_env_cfg.py:102` | 1.0 N, hard coded at `mdp/rewards.py:259` |

The joint counts are read from the dumps themselves and matter to the grid geometry, since the leg count alone does not determine it. The KScale carries six joints per leg against the SD_BRS1's five, its hip yaw being revolute where the SD_BRS1's is fixed, so a two column grid gives it six rows against the SD_BRS1's five, and the quadruped carries three joints per leg for twelve in total, which is three rows of four.

Two orderings recorded in those dumps bear directly on the design. The KScale records its feet as `['foot_6061_2', 'foot_6061']`, so the body index order REVERSES the numeric suffix and a fallback list written from the URDF or from the regex would have been transposed. The quadruped records its feet as `['foot_FL_Link', 'foot_FR_Link', 'foot_RL_Link', 'foot_RR_Link']` and its joints grouped by depth rather than by leg, all four abduction joints, then all four hips, then all four knees. Neither order is derivable from the asset, which is why section 3 declines to write a fallback for either robot.

Three consequences follow that the present dashboard does not encode.

The clearance reference is not a constant of the dashboard but a parameter of the gait command, and it is not the `target_height` that the constant's own comment cites. That comment at `scripts/analysis/dashboard-brs.py:77` names `rew_foot_clearance` and `foot_clearance_reward_v2`, and the term at `cfg/SF/brs_base_env_cfg.py:842` has since been repointed at `foot_clearance_reward_v3`, whose reference is the raised cosine `swing_height * sin^2(pi * phi)` formed at `mdp/rewards.py:498` from the gait command's fourth channel read at `:495`. The line is therefore the peak of a phase dependent reference rather than a set point, and it happens to remain 0.08 m for the SD_BRS1 because `swing_height` at `cfg/SF/brs_base_env_cfg.py:117` carries that value, which is why the present figure has never looked wrong. It is 0.05 m for the KScale and undefined for the other two.

The sole clearance channel exists only where a sole table exists. `play.py` appends `feet_sole_clearances` only under `if SOLE_OFFSETS is not None` at `scripts/rsl_rl/play.py:431`, and the profiles for the TRON1 and the quadruped carry `None` at `scripts/rsl_rl/play.py:176` and `:177`, so their dumps have no such key and the corresponding row of the feet window is empty. A reference line drawn on an empty panel is worse than no line, since it invites the reading that the clearance is zero.

The number of feet is not the number of legs for the TRON1. The pattern `ankle_.*` matches `ankle_L_Link` and `ankle_R_Link`, and also `ankle_L_actuator_Link` and `ankle_R_actuator_Link`, all four of which are declared in `environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf`, whereas the environment configuration resolves its feet with the narrower `ankle_[RL]_Link` at `cfg/SF/limx_base_env_cfg.py:1203`. A TRON1 dump therefore records four entries in `feet_names` where the reward terms see two. This is a defect of `play.py` and not of the dashboard, and section 8 records it and proposes its repair separately, but the dashboard must be built so that the column count of the feet window follows the recorded names rather than the leg count, or a TRON1 dump will silently lose half its columns.

## 3. The design

The change introduces one new object, a frozen record naming everything about a robot that a dump cannot state for itself, and one new command line flag selecting among four instances of it. Everything else is the threading of that record and a leg count through the four figure builders that need them.

The record carries a fallback joint name list, a fallback feet name list, a leg count, an optional clearance reference and a contact threshold. The two fallback lists are populated only for the SD_BRS1 and the TRON1, because those are the only robots for which a dump predating the name recording at `scripts/rsl_rl/play.py:368` can exist, the KScale configuration having entered the tree on 2026-08-24 and the quadruped on 2026-08-19, both after 2026-07-31. Where a profile carries no fallback and a dump carries no recorded names, the builders synthesise positional labels rather than inventing an order, since the articulation order is not derivable from the URDF, a finding proved three ways in the twentieth pass of [../context/brs_gait.md](../context/brs_gait.md) and recorded in the correction comment at `scripts/analysis/dashboard-brs.py:48`.

The leg count reaches the figure builders as the parameter `num_legs`, which the requirements name, and its only effect is the column count of the grids, four where the count is four and two otherwise. For the joint grids the column count is the whole of the effect, twelve quadruped joints becoming three rows of four rather than six rows of two. For the feet window the column count already follows the number of feet the dump records, at `scripts/analysis/dashboard-brs.py:518`, so `num_legs` there governs only the width of the synthesised fallback, and the quadruped obtains its four columns from its four recorded feet whether or not the parameter is passed.

Every new parameter is optional and every default reproduces the present behaviour exactly, as `/ws/CLAUDE.md` rule 4 requires. `num_legs` defaults to 2, the profile defaults to the SD_BRS1 instance whose fields are the four constants the module carries today, and `--robot` defaults to `brs`, so the invocation `python dashboard.py <log_dir>` after the merge draws precisely the figures that `python dashboard-brs.py <log_dir>` draws before it. The four module level constants are retained as names bound to the SD_BRS1 profile's fields, so any consumer importing them is unaffected.

## 4. The changes

### 4.1 The profile registry, replacing lines 38 to 79

The block from the `JOINT_NAMES` comment at `scripts/analysis/dashboard-brs.py:38` through `CONTACT_FORCE_THRESHOLD` at `:79` is replaced by the following. The provenance comments of the two existing lists are carried across verbatim, since the correction they record is the reason the lists read left before right and a later reader who loses that comment will restore the transposition.

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotProfile:
    """Everything about a robot that a dump cannot state for itself.

    A dump records the articulation's joint, body and feet names, so the only facts a
    dashboard must be told are the ones that live in the environment configuration rather
    than in the articulation, namely the reference levels the reward terms key on, and the
    fallback name lists for dumps written before play.py began recording the order.
    """

    key: str
    title: str
    num_legs: int
    # Fallback lists, consulted ONLY where a dump carries no recorded names 
    joint_names: tuple = ()
    feet_names: tuple = ()
    # Peak of the raised cosine clearance reference, which is the gait command's
    # swing_height read at mdp/rewards.py:495 and formed into a reference at :498. None
    # where the robot declares no gait command,
    sole_clearance_target: float = None

    contact_force_threshold: float = 1.0



_BRS_JOINT_NAMES = (
    "HipRollL", "HipRollR", "HipPitchL", "HipPitchR", "KneePitchL",
    "KneePitchR", "AnkleRollL", "AnkleRollR", "AnklePitchL", "AnklePitchR",
)

_BRS_FEET_NAMES = ("Link6L", "Link6R")

_TRON_JOINT_NAMES = (
    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
    "knee_L_Joint", "knee_R_Joint", "ankle_L_Joint", "ankle_R_Joint",
)

_TRON_FEET_NAMES = ("ankle_L_Link", "ankle_R_Link")

ROBOT_PROFILES = {
    "brs": RobotProfile(
        key="brs", title="SD_BRS1", num_legs=2,
        joint_names=_BRS_JOINT_NAMES, feet_names=_BRS_FEET_NAMES,
        # cfg/SF/brs_base_env_cfg.py:117, and the contact gate at :853
        sole_clearance_target=0.08, contact_force_threshold=1.0,
    ),
    "kscale": RobotProfile(
        key="kscale", title="KScale", num_legs=2,
        # Every KScale dump postdates play.py:368, so no fallback can be reached.
        # cfg/SF/kscale_base_env_cfg.py:136, and the contact gate at :849
        sole_clearance_target=0.05, contact_force_threshold=1.0,
    ),
    "tron": RobotProfile(
        key="tron", title="TRON1 SoleFoot", num_legs=2,
        joint_names=_TRON_JOINT_NAMES, feet_names=_TRON_FEET_NAMES,
        # No gait command, it is commented out at cfg/SF/limx_base_env_cfg.py:102, so no
        # clearance reference exists and play.py logs no sole clearance for this robot.
        sole_clearance_target=None, contact_force_threshold=1.0,
    ),
    "quadruped": RobotProfile(
        key="quadruped", title="Quadruped", num_legs=4,
        # Every quadruped dump postdates play.py:368, so no fallback can be reached, and
        # the feet order is deliberately NOT asserted here since it is not derivable from
        # quadruped.urdf. cfg/quadruped/base_env_cfg.py:102 leaves the gait command
        # commented out, so there is no clearance reference and no clearance channel.
        sole_clearance_target=None, contact_force_threshold=1.0,
    ),
}

DEFAULT_PROFILE = ROBOT_PROFILES["brs"]
```

### 4.2 The name resolvers, replacing lines 108 to 115

The two wrappers gain the profile as an optional argument whose default is the SD_BRS1 instance, and a third helper is added that reports the width of a channel so that a builder may synthesise positional labels where neither a recorded list nor a fallback exists.

```python
def resolve_joint_names(experiments_dict, profile=DEFAULT_PROFILE):
    """Joint names in articulation order, preferring what the dump itself recorded."""
    return _names_from_dumps(experiments_dict, "joint_names", profile.joint_names)


def resolve_feet_names(experiments_dict, profile=DEFAULT_PROFILE):
    """Feet body names in the order the DataLogger resolved them."""
    return _names_from_dumps(experiments_dict, "feet_names", profile.feet_names)


def channel_width(experiments_dict, metric_key, axis=2):
    """Width of a dump channel along one axis, for labelling a nameless articulation.

    A dump that records no names and belongs to a robot whose profile carries no fallback
    can still be plotted, since the array itself states how many joints or feet it holds.
    Positional labels are given in that case rather than a borrowed order, because the
    articulation order is not derivable from the asset and a wrong label is worse than an
    uninformative one.
    """
    for data in experiments_dict.values():
        if not isinstance(data, dict) or metric_key not in data:
            continue
        array = np.asarray(data[metric_key])
        if array.ndim > axis:
            return int(array.shape[axis])
    return 0
```

### 4.3 `create_joint_plot`, at line 239

The signature gains `num_legs`, the column count is derived from it, and the name list falls back to positional labels. The changed lines are the signature, the block that builds the grid, and the two lines computing the row and the column inside the per joint loop.

```python
def create_joint_plot(experiments_dict, metric_key, env_id, zoom_range=None, smoothing=0.0,
                      hidden_experiments=None, num_legs=2, profile=DEFAULT_PROFILE):
    """Creates a grid for a SINGLE joint metric, comparing all experiments.

    The grid is two columns wide for a biped and four for a quadruped, since a quadruped
    carries three joints per leg and twelve panels stacked two abreast is six rows of
    scrolling where three rows of four is one screen.
    """
    hidden_experiments = hidden_experiments or []

    joint_names = resolve_joint_names(experiments_dict, profile)
    if not joint_names:
        joint_names = [f"joint {i}" for i in range(channel_width(experiments_dict, metric_key))]
    num_joints = len(joint_names)
    if num_joints == 0:
        return go.Figure().update_layout(title=f"No {metric_key} in any dump for this seed")
    num_cols = 4 if num_legs >= 4 else 2
    grid_rows = (num_joints + num_cols - 1) // num_cols
    clean_titles = [pretty_joint_name(name) for name in joint_names]

    fig = make_subplots(
        rows=grid_rows, cols=num_cols,
        subplot_titles=clean_titles,
        shared_xaxes="all",
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
```

and, within the per joint loop that begins at `scripts/analysis/dashboard-brs.py:281`,

```python
        for j in range(num_joints):
            row = (j // num_cols) + 1
            col = (j % num_cols) + 1
            show_leg = (j == 0)
```

The height at `scripts/analysis/dashboard-brs.py:301` is left at `500 * grid_rows`, which gives the SD_BRS1 its present 2500 and the quadruped 1500 over three rows, each row being of the same height as it is today.

### 4.4 `create_torque_velocity_plot`, at line 588

The same three changes, since this figure is the second joint grid and the requirement to widen the joint statistics applies to it identically.

```python
def create_torque_velocity_plot(experiments_dict, env_id, smoothing=0.0,
                                hidden_experiments=None, num_legs=2, profile=DEFAULT_PROFILE):
    """Creates a grid Scatter plot, Joint Torque (Y) against Joint Velocity (X)."""
    hidden_experiments = hidden_experiments or []

    joint_names = resolve_joint_names(experiments_dict, profile)
    if not joint_names:
        joint_names = [f"joint {i}" for i in range(channel_width(experiments_dict, "joint_torques"))]
    num_joints = len(joint_names)
    if num_joints == 0:
        return go.Figure().update_layout(title="No joint torque or velocity in any dump for this seed")
    num_cols = 4 if num_legs >= 4 else 2
    grid_rows = (num_joints + num_cols - 1) // num_cols
    clean_titles = [pretty_joint_name(name) for name in joint_names]

    fig = make_subplots(
        rows=grid_rows, cols=num_cols,
        subplot_titles=clean_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
```

and, in the loop at `scripts/analysis/dashboard-brs.py:635`,

```python
        for j in range(num_joints):
            row = (j // num_cols) + 1
            col = (j % num_cols) + 1
            show_leg = (j == 0)
```

### 4.5 `create_feet_plot`, at line 485

Three changes. The feet name list falls back to positional labels of width `num_legs`, the clearance row is suppressed where the profile declares no reference, and the two reference levels are read from the profile rather than from the module constants.

```python
def create_feet_plot(experiments_dict, env_id, zoom_range=None, smoothing=0.0,
                     hidden_experiments=None, num_legs=2, profile=DEFAULT_PROFILE):
```

The name resolution at `scripts/analysis/dashboard-brs.py:508` becomes

```python
    feet_names = resolve_feet_names(experiments_dict, profile)
    if not feet_names:
        width = channel_width(experiments_dict, "feet_frame_heights") or num_legs
        feet_names = [f"foot {i}" for i in range(width)]
    num_feet = len(feet_names)
    num_rows = len(channels)
```

and the reference level block at `scripts/analysis/dashboard-brs.py:566` through `:575` becomes

```python
    # Reference levels, the contact threshold the gait terms key on, and the clearance
    # reference where the robot declares one. A robot with no gait command carries no
    # clearance channel either, play.py gating that append on the sole table at
    # play.py:431, so the line is suppressed rather than drawn across an empty panel.
    for foot_idx in range(num_feet):
        fig.add_hline(
            y=profile.contact_force_threshold, line=dict(color='grey', dash='dot', width=1),
            row=1, col=foot_idx + 1
        )
        if profile.sole_clearance_target is not None:
            fig.add_hline(
                y=profile.sole_clearance_target, line=dict(color='grey', dash='dot', width=1),
                row=5, col=foot_idx + 1
            )
```

The column count itself needs no change, `cols=num_feet` at `scripts/analysis/dashboard-brs.py:518` already widening the window to four for a quadruped whose dump records four feet, which is the outcome the requirement asks for and the reason `num_legs` governs only the fallback here.

### 4.6 `create_feet_plot_2`, at line 744

This window plots the signed separation of two feet, and the quantity it reads is formed in `play.py` as the difference of exactly two bodies, `feet_pos[:, 0, :] - feet_pos[:, 1, :]` at `scripts/rsl_rl/play.py:437`. For a quadruped that difference is the separation of whichever two feet happen to occupy indices zero and one, which is a pair of the four and not a meaningful gait quantity. The window is therefore refused rather than drawn.

```python
def create_feet_plot_2(experiments_dict, env_id, zoom_range=None, smoothing=0.0,
                       hidden_experiments=None, num_legs=2, profile=DEFAULT_PROFILE):
    """Signed per-axis distance between the two feet, over time."""
    hidden_experiments = hidden_experiments or []

    if num_legs != 2:
        # feet_distance is formed in play.py:437 as the difference of feet indices 0 and 1
        # alone, so on a quadruped it is one arbitrary pair of six and not a gait quantity.
        return go.Figure().update_layout(
            title="Feet distance is defined for two feet only. play.py logs the separation "
                  "of feet 0 and 1, which on a quadruped is one arbitrary pair."
        )
```

The remainder of the function is unchanged.

### 4.7 `main`, at line 905

The parser gains the flag, the profile is selected from it, the titles are taken from it, and the leg count and the profile are threaded through the figure callback.

```python
def main():
    parser = argparse.ArgumentParser(description="RSL-RL Experiment Dashboard")
    parser.add_argument("log_dir", type=str, help="Path to the directory containing all experiments")
    parser.add_argument(
        "--robot", type=str, default="brs", choices=sorted(ROBOT_PROFILES),
        help="Robot the dumps under log_dir belong to. Selects the leg count, the grid "
             "width, the reference levels and the fallback name lists. Defaults to brs, "
             "which reproduces the behaviour this script had before the flag existed.",
    )
    args = parser.parse_args()

    profile = ROBOT_PROFILES[args.robot]
    print(f"[INFO] Robot profile {profile.key}, {profile.num_legs} legs, "
          f"clearance reference "
          f"{'none' if profile.sole_clearance_target is None else profile.sole_clearance_target}")
```

The heading at `scripts/analysis/dashboard-brs.py:936` becomes

```python
        html.H1(f"RSL-RL Experiment Dashboard — {profile.title}", style={'textAlign': 'center'}),
```

and the dispatch at `scripts/analysis/dashboard-brs.py:1056` through `:1070` becomes

```python
        if metric_type == 'base':
            return create_base_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'base_position':
            return create_base_position_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'feet':
            return create_feet_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments,
                                    num_legs=profile.num_legs, profile=profile)
        elif metric_type == 'torque_velocity':
            # Do not pass Time-based zoom range to Velocity axis
            return create_torque_velocity_plot(experiments_dict, env_id, smoothing, hidden_experiments,
                                               num_legs=profile.num_legs, profile=profile)
        elif metric_type == 'rewards':
            return create_rewards_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'feet_distance':
            return create_feet_plot_2(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments,
                                      num_legs=profile.num_legs, profile=profile)
        elif metric_type in JOINT_METRICS:
            return create_joint_plot(experiments_dict, metric_type, env_id, zoom_range, smoothing,
                                     hidden_experiments, num_legs=profile.num_legs, profile=profile)
        else:
            return go.Figure().update_layout(title="Unknown Metric Selected")
```

The three windows that read neither a joint nor a foot, the base velocity grid, the base position grid and the reward grid, take no new argument, since nothing in them depends on the morphology. The statistics table likewise takes none, `stats.py` already guarding its one biped specific family behind `if data.num_feet == 2` at `scripts/analysis/stats.py:1184` and labelling the rest from `data.feet_names` at `:698`.

## 5. Application instructions

Perform the merge before the edits, so that the history of the surviving file is the history of the file that carries the work.

```bash
cd /ws/tron1-rl-isaaclab-cozum
git rm scripts/analysis/dashboard.py
git mv scripts/analysis/dashboard-brs.py scripts/analysis/dashboard.py
```

Then apply section 4 to `scripts/analysis/dashboard.py` in the order the subsections are given, 4.1 first because every later subsection refers to `DEFAULT_PROFILE`. Add `from dataclasses import dataclass` to the import block at the head of the file, `field` being unnecessary since no profile field is mutable.

No launcher change is required. The `djinn start visualise-evaluation` command invokes `scripts/rsl_rl/visualise.py` at `djinn:273` and not either dashboard, so the rename is invisible to it, and no other file in the workspace references either dashboard by path outside the documents listed in section 9.

The two further copies at `/ws/IsaacLab/logs/rsl_rl/dashboard-brs.py` and `/ws/IsaacLab/logs/rsl_rl/dashboard.py` are deliberately left alone. That tree is transient under `/ws/CLAUDE.md` rule 20 and its runs are deleted as disk fills, so the repository copy is the canonical one and duplicating this work into a directory that will be removed is waste.

## 6. Verification, and what it established on 2026-09-03

The code blocks of section 4 were exercised against the applied file, and before that against a working copy of it, against three stored dumps, `sd_brs1_flat/2026-07-31_10-21-10`, `kscale_flat/2026-09-02_06-59-31` and `quadruped_flat/2026-09-02_08-53-04`. The quadruped run originally offered for the test, `quadruped_flat/2026-09-02_11-30-45`, carries checkpoints and TensorBoard events but no `data/` directory, no play having been run against it, so the sibling run of the preceding day was substituted. No TRON1 dump survives anywhere under `/ws/IsaacLab/logs/rsl_rl/`, so every TRON1 check below is deferred rather than passed. The results follow, each check stating the observation that discharged it.

Check 1, the default reproduces the present behaviour, PASSED and on the strongest available evidence. Eight figures were built twice from the same SD_BRS1 dump, once by the unmodified `dashboard-brs.py` and once by the merged file called with no new argument, and the two were compared as serialised JSON rather than by eye. The joint torque grid, the joint position grid, the torque against velocity scatter, the feet window, the ankle separation window, the base velocity grid, the base position grid and the reward grid were byte identical in every case, and the four module constants retained their former values. The default path is therefore not merely equivalent in appearance but identical in output.

Check 2, the quadruped widens, PASSED. The joint grids present three rows of four against the twelve recorded joints, at a figure height of 1500 against the SD_BRS1's 2500, and the feet window presents six rows of four columns titled with the four recorded foot names.

Check 3, the clearance line follows the robot, PASSED. The KScale feet window carries four reference lines, two per foot, at 1.0 N and 0.05 m. The quadruped feet window carries four, one per foot, at 1.0 N alone, and its sole clearance row is empty because the dump has no `feet_sole_clearances` key at all, which confirms the gating at `scripts/rsl_rl/play.py:431` from the consumer's side. Twenty traces are drawn across the quadruped's five populated channels and four feet, against thirty for a robot carrying all six.

Check 4, the feet distance window refuses rather than misleads, PASSED. The quadruped selection returns the explanatory figure with no traces, and the KScale selection returns the five trace plot unchanged.

Check 5, the statistics window is unaffected, PASSED. The comparison table rendered 3585 rows for the KScale and 4166 for the quadruped, the difference being the reward and per foot families rather than any failure, and no branch of `stats.py` raised on either robot.

Check 6, the fallback path, PASSED under simulation. No dump predating the name recording at `scripts/rsl_rl/play.py:368` survives in the tree, so the path was exercised by stripping the recorded names from a dump in memory. The SD_BRS1 profile then yields its ten names, the TRON1 profile its eight, and the KScale and quadruped profiles yield positional labels of the width the arrays declare, `joint 0` through `joint 11` and `foot 0` through `foot 3`, which is the intended behaviour of a profile that deliberately asserts no order.

Check 7, an absent channel does not raise, PASSED. A joint grid requested against an empty experiment returns a titled empty figure stating that the channel is absent, rather than raising on a zero row grid.

One observation is recorded that is not a check. Where a profile's fallback list is reached against a dump of a DIFFERENT robot, the width mismatch is caught by the existing guard at `scripts/analysis/dashboard-brs.py:273` and the series is skipped with a warning, so a `--robot` value misapplied to a nameless dump produces an empty panel and a console line rather than a wrong plot. This behaviour predates the change and is unaltered by it.

## 7. Rationale for the shape of the change

The requirement could have been met by a second family of module constants selected by an `if` at the head of the file, and that form is rejected for three reasons. It reintroduces the coupling the name resolvers already removed, since a constant consulted unconditionally overrides a name the dump recorded, and the transposition of 2026-07-31 is exactly the defect that arrangement produced. It scales by editing rather than by extension, a fifth robot requiring a fifth branch in every function that reads a constant rather than a fifth entry in one table. And it cannot express the absence of a quantity, whereas the profile carries `None` for a clearance reference and the figure builder tests it, which is the distinction between a robot whose target is elsewhere and a robot that has no target at all.

The choice to thread the profile explicitly rather than to assign a module global once at startup follows the same reasoning. A global would make every figure builder impure and would make two dashboards in one process impossible, and it would place the profile outside the signature where a reader of the function cannot see what governs its output. An explicit argument whose default is the SD_BRS1 instance costs one keyword at each of six call sites and preserves the present behaviour of every direct caller of these functions, which `/ws/CLAUDE.md` rule 4 requires.

The choice to derive the feet column count from the recorded names rather than from `num_legs` is forced by the TRON1, whose dump records four feet against two legs for the reason section 2 gives. A design that trusted `num_legs` there would drop two columns silently, which is the class of failure this whole change exists to remove.

## 8. Defects left standing, recorded deliberately

`/ws/CLAUDE.md` rule 6 requires that a defect left in place be recorded rather than repaired silently, so that another implementation may opt into the remedy deliberately.

The first is the TRON1 foot pattern. `_ROBOT_PROFILES` in `scripts/rsl_rl/play.py:176` carries `ankle_.*`, which resolves four bodies on the TRON1 solefoot, `ankle_L_Link`, `ankle_R_Link`, `ankle_L_actuator_Link` and `ankle_R_actuator_Link`, all declared in `environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf`, whereas every reward term resolves the narrower `ankle_[RL]_Link` at `cfg/SF/limx_base_env_cfg.py:1203`. Every TRON1 dump therefore carries two spurious feet whose contact forces and heights describe an actuator housing, and its `feet_distance` at `scripts/rsl_rl/play.py:437` is the separation of whichever two of the four occupy indices zero and one. The repair is one character class, `ankle_[LR]_Link` in place of `ankle_.*`, and it is not proposed here because it changes what a TRON1 replay writes and therefore breaks the comparability of a new dump against every TRON1 dump already stored, which is precisely the consequence `/ws/CLAUDE.md` rule 3 exists to force a deliberate choice about. The dashboard is built to be correct under either.

The second is the feet distance channel itself. It is defined for two feet by construction at `scripts/rsl_rl/play.py:437` and no quadruped quantity replaces it. A quadruped's stance geometry is a four point support polygon and its natural summaries are the polygon's area and the diagonal separations, none of which the dump carries. The window is refused for the quadruped rather than filled with a misleading pair, and the channel is left as it is.

The third is the clearance reference for the TRON1 and the quadruped. Neither declares a gait command, both being commented out at `cfg/SF/limx_base_env_cfg.py:102` and `cfg/quadruped/base_env_cfg.py:102`, so neither has a clearance target to draw and neither carries a sole table in `scripts/rsl_rl/play.py`, which means the sole clearance row of the feet window is empty for both. The remedy is a measured sole table for each robot in the manner of `scripts/analysis/kscale_sole_analysis.py`, and it is a separate piece of work with its own validation, so the profiles carry `None` and the row is left empty and unmarked.

## 9. Documents to update on completion

On applying this plan, set its status banner to implemented with the date, update the register row in [README.md](README.md) to match, and record the outcome together with any divergence in [../context/gait_metrics.md](../context/gait_metrics.md), which is the context document that owns the dump and dashboard pipeline and whose section 11 already names the four live dashboards. Where the merge changes the count of those consumers from four to three, amend that sentence rather than appending to it, the count being a fact of the tree rather than a finding of a pass.

`ARCHITECTURE.md` and `README.md` at the workspace root need no change, neither naming either dashboard, their dashboard references at `ARCHITECTURE.md:934` and `README.md:139` describing the `djinn start visualise-evaluation` command which invokes `scripts/rsl_rl/visualise.py` instead.

## 10. Divergences from the specification, recorded on implementation

Two, both cosmetic, neither touching a line of executable code.

The comment density of section 4.1 was reduced on application, at the instruction of the session that applied it, to the principle that a comment is warranted where it records provenance a reader cannot recover from the code and is not warranted where it restates what the code says. The provenance retained is the 2026-07-31 side ordering correction on the SD_BRS1 lists, the citation of the configuration line behind each clearance reference and contact threshold, the note that the TRON1 fallback was carried across unverified from the deleted file, the note that `play.py` resolves four TRON1 feet against the two the configuration uses, and the note that the KScale and quadruped profiles carry no fallback deliberately. Everything the field names already state was dropped, as were three explanatory comments inside the figure builders, retaining in each only the sentence citing `play.py`.

The docstrings of `channel_width`, `create_joint_plot` and `create_feet_plot_2` were shortened on the same principle, each retaining the reason for its behaviour and losing the restatement of it.
