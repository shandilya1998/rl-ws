# repo_restoration.md — Recovering the working tree lost to the changelog_creator.py reset of 2026-09-17

Status, IMPLEMENTED 2026-09-17, following the user's review of the draft written the same day. Investigation drew on four kinds of source, the working tree of `/ws` and `/ws/tron1-rl-isaaclab-cozum` as they stand after the incident, the dumped run configuration at `/ws/IsaacLab/logs/rsl_rl/quadruped_flat/2026-09-11_11-24-52/`, every context and plan document the incident's work streams touch, and, decisively for most of what follows, the raw session transcripts this tool records for every past conversation, stored as JSON lines under `/root/.claude/projects/` and addressable by the working directory each session was opened in. That fourth source is what turned most of this document from a plausible reconstruction into a verbatim recovery, and section 2 explains the method before the findings are given, since a reader checking this document's claims needs to know where to look.

Implementation note, 2026-09-17. Every edit proposed in sections 5 through 8 was applied as written, with one correction and one addition found necessary during implementation, both recorded here rather than in the sections above, per the append convention of rule 17 of `/ws/CLAUDE.md`. The correction, in section 7, `Isaac-Quadruped-Go1Asset-Blind-Rough-v0`'s `entry_point` was quoted from the transcript as `HIMManagerBasedRLEnv`, but the surviving, authoritative `context/go1_quadruped_throughput.md` component comparison table states the entry point class as `ManagerBasedRLEnv` for all three debug tasks, matching `Isaac-Quadruped-Blind-Rough-v0`'s own registration and the ablation's premise of holding the environment and MDP stack fixed, so the plain class was used instead. The addition, also in section 7, section 10's own instruction to check `scripts/rsl_rl/train.py` and `cli_args.py` against the pattern quoted there found that the `DebugOnPolicyRunner` selection branch and the `--policy-type` help string entry were themselves missing, not merely unconfirmed, and both were added. Item 3's re-execution of `plans/co_optimisation_cleanup.md` found that the outer workspace documents its own section 11 lists, `ARCHITECTURE.md`, `CO_OPTIMISATION.md`, and `context/copt.md`, already reflect the new architecture in full, including line citations matching the restored code closely, corroborating the restoration independently, so none of the three required a further edit. Verification performed, without a simulator being reachable from this session either, is recorded in `co_optimisation_cleanup.md`'s own section 9 static checks and in this document's section 10, `py_compile` and a full `compileall` sweep across every touched package, a grep sweep confirming no reference to a retired name or task identifier survives, and the four surviving test and self-test suites of section 5 and 8 run and passing in full.

Five work streams are addressed rather than the three first named, because auditing the transcripts surfaced two more that the incident also destroyed. Each section below is self-contained.

## 1. What happened, established from the tooling itself rather than from recollection

`changelog_creator.py` read a branch's commits through `get_branch_commits` at `changelog_creator.py:151-179`. Before the edit the user made, that function set `repo.head.reference = target_branch` and then called `repo.head.reset(index=True, working_tree=True)` with no commit argument, which is GitPython's spelling of `git reset --hard HEAD`, resetting the index and the working tree of the currently checked out branch to its own tip. Because `target_branch` was resolved from `repo.heads[branch_name]` when a local branch of that name already existed, and the branch in force was `shreyas/kscale_drdo`, the branch the repository was already on, the call reset the current branch onto itself, discarding every staged and unstaged change to every tracked file without moving `HEAD` at all. `LOCAL_REPO_PATH` is fixed at `/ws/tron1-rl-isaaclab-cozum/`, so the outer `/ws` repository, a separate git repository in its own right with `tron1-rl-isaaclab-cozum` checked in as a submodule, was never touched by this call at all. Every uncommitted change sitting in `/ws` itself, `CLAUDE.md`, the context and plan documents, `djinn`, survived intact for this reason alone, and this is confirmed directly, `/ws`'s own `git status` at the start of this investigation still shows them modified and unstaged.

The reflog of `/ws/tron1-rl-isaaclab-cozum` corroborates the mechanism precisely. It carries two entries reading `reset: moving to HEAD`, at `07:23:43` and `07:42:35` on 2026-09-17, both immediately following the amendment of commit `e770a7e`, `KBot Training Ablation Changes`, at `07:21:12` the same morning, which followed the original commit of `bee8475` at `05:49:46`. A `reset: moving to HEAD` reflog line is produced exactly by resetting a branch to its own tip, never by resetting to a different commit, which a `reset: moving to <sha>` line would instead read. This settles what the rest of this document depends on. No commit was ever rewound or discarded, `HEAD` sits at `e770a7e` throughout, and everything the incident destroyed was uncommitted working tree state that existed on top of `e770a7e` at some point between `07:21:12` and the second reset at `07:42:35`, bounding the loss to the single most recent working session rather than to an unbounded span of history.

A `git reset --hard` of this kind never touches an untracked file, and a plain unstaged edit to a tracked file creates no git object at any point, a blob being written only when the file is staged. `git fsck --dangling` was run against the repository and every dangling object it returned was inspected by date and message and found to predate 2026-09-04, arising from ordinary historical amends and rebases unconnected to this incident. Nothing lost here is recoverable from git's own object store, which is why sections 4 through 8 lean on the three other sources instead, the dumped run configuration, the surviving plan and context documents, and the session transcripts of section 2.

## 2. The session transcripts as a source of record, and why they can be trusted

This tool retains a full, line-by-line JSON transcript of every past session, keyed by the directory the session was opened in, under paths such as `/root/.claude/projects/-ws/<session-id>.jsonl` for a session opened at `/ws` and `/root/.claude/projects/-ws-tron1-rl-isaaclab-cozum-environments-environments-tasks-locomotion/<session-id>.jsonl` for one opened four levels deeper. A transcript records every message and every tool call verbatim, including the exact text a `Write`, `Edit`, or `Bash` heredoc call sent to the filesystem, so where a past session authored a function that this incident later destroyed, the transcript holds the same bytes that once sat in the working tree, not a paraphrase of them.

Thirty-one sessions dated after 2026-09-07, the last real commit before today's, were enumerated by file modification time and searched by the exact symbol names this investigation needed, `def feet_air_time_v2`, `def foot_clearance_reward_v4`, `def foot_landing_vel_v3`, `def _terrain_height_under_points`, `def feet_distance_v2`, `class Go1NativeMatchedEnvCfg`, and `gtEncoderOut`. Where a match sat inside a short `grep -n "^def "` style listing it was set aside as a table of contents rather than a source. Where it sat inside a large block, several thousand characters, emitted by a `Write` call, an `Edit` call's `new_string`, or a `Bash` heredoc, it was extracted and is quoted in the sections below essentially unchanged, and in three cases, `feet_air_time_v2`, `foot_landing_vel_v3`, and `feet_distance_v2`, the extracted block is the literal Python heredoc a session ran against the live file, closing with its own `ast.parse` or pytest confirmation that the insertion succeeded, which is as strong a guarantee of correctness as this investigation could obtain short of the file itself surviving.

This is why the confidence language of section 4 changed between the first draft of this document and this one. What began as a reconstruction graded by how well a description in `plans/QUADRUPED_DEGENERATE_GAIT.md` constrained the code is now, for every function but one, a verbatim quotation of the code a past session actually wrote and verified, recovered from a source this incident could not touch, since the transcripts live outside the git working tree entirely.

## 3. The five work streams, and the state each was actually found in

| Item | Subject | State found | Confidence of restoration |
|---|---|---|---|
| 1 | Co-optimisation non-stationarity fix | Not lost. Present at `e770a7e`, safe | N/A, no action needed |
| 2 | Quadruped reward and hyperparameter changes | Lost from tracked source, recovered verbatim from transcripts and the 2026-09-11 dump | High, four of five functions are literal quotations |
| 3 | Co-optimisation cleanup | Lost from tracked source, fully specified by a surviving untracked plan document | High, the plan is complete and was verified once already |
| 4 | Go1 debug environment and `DebugOnPolicyRunner` | Partially lost. The runner survived untracked. The two comparison environment configurations and their registration were lost, recovered verbatim from a transcript | High |
| 5 | `feet_distance_v2`, a footprint-aware feet separation penalty for KScale and SD_BRS1 | Lost from tracked source, recovered verbatim from a transcript. Its two companion analysis and self-test scripts survived, already committed | High |

Sections 4 to 8 take these in turn. Section 9 records what the transcript audit screened and found not to require restoration, and one item it found that is not a casualty of this incident at all but is worth the user's attention regardless.

## 4. Item one, the non-stationarity fix, is present and was never lost

The user's description names a specific code snippet, re-randomising `episode_length_buf` after a design reset, and a specific refactor, streamlining `_respawn_population` to reset environments the way `_update_morphology` does. Both are in the tree today, at commit `e770a7e`, which the incident never disturbed.

The re-randomisation is at `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:424-427`, immediately following the call to `_update_morphology` inside the main training loop.

```python
with torch.inference_mode():
    self._update_morphology(it + 1, respawn=late_start_toggled)
    self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
    )
```

This is exactly the remedy `../../context/copt_ppo_nonstationarity.md` section 12 prescribes as its first remedy, restoring at the design reset the same randomisation that `learn` performs once at start up at `copt_on_policy_runner.py:264-266`, so that the episode phase lock the design swap otherwise imposes on every one of four thousand and ninety six environments simultaneously is broken after one episode rather than persisting for the whole of a generation.

The refactor is likewise present. `_respawn_population` is defined once, at `copt_on_policy_runner.py:550-577`, and is called from both `_reload_morphology` at `:600` and from `_update_morphology` at `:695` when its `respawn` argument is true, so the two pathways share one reset sequence rather than each performing its own.

No action is proposed for this item. It is recorded here so the user is not asked to re-derive or re-review work that was never in fact lost.

## 5. Item two, the quadruped reward and hyperparameter changes

### 5.1 What the dumped run and the transcripts together establish

The run at `2026-09-11_11-24-52` was launched six days before the incident, and its `params/env.yaml` and `params/agent.yaml` are the full resolved configuration under which it trained, recording every reward weight, parameter, and function import path exactly as configured, per `../../context/gait_metrics.md` section 9. Six reward terms in that dump reference functions or a helper the post-reset `environments/environments/tasks/locomotion/mdp/rewards.py` does not define, `feet_air_time_v2`, `foot_clearance_reward_v4`, `foot_landing_vel_v3`, and `_terrain_height_under_points`. A fifth, `feet_hold_penalty`, is likewise absent, its weight and gate confirmed by the dump. A sixth, `feet_impact_force`, is not missing at all, unchanged at `rewards.py:165-190`.

The session transcripts supply the bodies. `_terrain_height_under_points` and `foot_clearance_reward_v4` were recovered from an auto-generated compaction summary embedded in session `29c8d5d6-f603-4b28-9218-a9a0293dc796` (2026-09-10), which records that both were inserted at `rewards.py:524` and `:623`, "verified byte-identical to the validated draft," with thirteen tests passing. `feet_air_time_v2` was recovered twice over, once from the same session's later record of the exact line quoted from the live file, `torch.where(air_time > cap, cap - air_time, air_time)`, and again independently from session `b40fca2a-7b50-4f95-b7c0-75edc07d1f65` (2026-09-17, this morning, before the incident), which quotes the identical body. `foot_landing_vel_v3` was recovered from the same session as the literal Python heredoc that inserted it, closing with an AST parse check. `feet_hold_penalty` was recovered from `plans/QUADRUPED_DEGENERATE_GAIT.md`, an untracked file that survived the reset and quotes the function in full as its own specification, corroborated by the untracked, previously-executed test suite at `environments/tests/rewards_feet_hold_penalty_test.py`.

The dump also shows five hyperparameters differ from the values the tree carries today, all of them exactly `../../plans/QUADRUPED_DEGENERATE_GAIT.md` Phase 1's targets.

| Quantity | Current tree | Dumped run | Source |
|---|---|---|---|
| `entropy_coef` | 0.005 | 0.001 | `agent.yaml:43`, Phase 1 Change 1.1 |
| push force ceiling | (3.0, 3.0) | (1.0, 1.0) | `env.yaml:1788-1790`, Phase 1 Change 1.2 |
| `pen_ang_vel_xy` weight | -0.5 | -0.05 | `env.yaml:1345`, Phase 1 Change 1.3 |
| `pen_action_rate` weight | -0.05 | -0.01 | `env.yaml:1357`, Phase 1 Change 1.3 |
| `pen_action_smoothness` weight | -0.075 | -0.01 | `env.yaml:1369`, Phase 1 Change 1.3 |

`pen_abad_deviation` should be checked against `env.yaml:1406-1420` against the Phase 1 target of -0.5, the current tree carrying -2.0.

### 5.2 Restoring the Phase 1 hyperparameters

```python
# environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py
entropy_coef = 0.001          # was 0.005

# environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py, CurriculumCfg.modify_push_force
"max_velocity": (1.0, 1.0),   # was (3.0, 3.0)

# RewardsCfg
pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)                  # was -0.5
pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)                # was -0.05
pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01) # was -0.075
pen_abad_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["abad_.._Joint"])})  # verify against env.yaml:1406-1420, was -2.0
```

One further pinning is recorded in the same transcript and should be restored alongside these five, so that Phase 1's `entropy_coef` change does not silently reach the co-optimisation quadruped task, whose exploration dynamics under a non-stationary morphology are a separate question governed by `../../context/copt_ppo_nonstationarity.md`.

```python
# environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py
class PFQuadrupedCoptPPORunnerCfg(PFQuadrupedPPORunnerCfg):
    experiment_name: str = "quadruped_copt"
    max_iterations: int = 45000
    # Pinned so that the Phase 1 entropy_coef change to the parent does not silently alter
    # the co-optimisation runs, whose exploration dynamics under a non-stationary
    # morphology are a separate question, recorded in context/copt_ppo_nonstationarity.md.
    algorithm: RslRlPpoAlgorithmMlpCfg = PFQuadrupedPPORunnerCfg.algorithm.replace(
        entropy_coef=0.01
    )
```

### 5.3 `feet_air_time_v2`, recovered verbatim

```python
def feet_air_time_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel, penalising a step that overruns.

    Identical to :func:`feet_air_time` below the cap, since both reward
    ``last_air_time - threshold_min`` on first contact. Where that function clamps the
    reward at ``threshold_max - threshold_min`` so an overlong step earns no more but no
    less than the cap, this one continues past the cap with a NEGATIVE slope, so a step
    that ran on twice as long as intended is charged rather than merely capped.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # negative penalty for large steps: reward grows up to the cap, then
    # falls below zero the further air_time exceeds threshold_max
    cap = threshold_max - threshold_min
    air_time = torch.where(air_time > cap, cap - air_time, air_time)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
```

Wired as follows, matching the dump exactly. The original `feet_air_time` is unchanged and untouched, so nothing calling it elsewhere in the tree is affected.

```python
feet_air_time = RewTerm(
    func=mdp.feet_air_time_v2,
    weight=10.0,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
        "command_name": "base_velocity",
        "threshold_min": 0.15,
        "threshold_max": 0.45,
    },
)
```

The weight of 10.0 is the dump's own figure and is higher than the 5.0 that `plans/QUADRUPED_DEGENERATE_GAIT.md` Phase 1 first proposed, recording a further manual increase made after Phase 1 that no surviving document explains. It should be treated as the last known working value rather than re-derived.

### 5.4 `_terrain_height_under_points` and `foot_clearance_reward_v4`, recovered verbatim

```python
def _terrain_height_under_points(
    points_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    num_neighbours: int = 4,
    max_horizontal_dist: float | None = None,
    reduction: str = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the terrain height directly beneath a set of world frame points.

    For every query point this selects the ``num_neighbours`` ray hits nearest to it in the
    horizontal plane and reduces their world heights, which gives a LOCAL terrain reference
    rather than the single mean over all rays that :func:`base_height_rough_l2` takes. The
    distinction matters on generated terrain, where the four feet of a quadruped may stand
    on four different surfaces, a mean beneath the base being an average of all of them and
    therefore correct for none of them.

    The estimate is valid because the ray caster's directions are NOT rotated under a yaw
    alignment, only its start positions are, so every ray descends vertically in the world
    frame and ``ray_hits_w[..., 2]`` is the ground height vertically below that grid point.

    Args:
        points_w: World frame query points, shape (N, P, 3) or (N, P, 2). Only the first two
            components are read, the height of the point itself being irrelevant to where
            the ground beneath it lies.
        ray_hits_w: World frame ray hit positions from the ray caster, shape (N, R, 3).
        num_neighbours: How many nearest hits to reduce over. Clamped to the ray count. At
            the 0.1 m grid resolution this configuration uses, 4 brackets a point between
            the surrounding grid cells.
        max_horizontal_dist: Optional radius (m) beyond which a neighbour is rejected. A
            point with no accepted neighbour is reported invalid rather than silently
            resolved against a distant grid edge, which is the failure a query point outside
            the scanner's footprint would otherwise produce. Defaults to None, no limit.
        reduction: ``"max"``, the default, for the highest accepted neighbour, the
            conservative choice near a step edge where a foot must clear the tallest nearby
            ground, resolving a rising tread from the moment a foot's horizontal position
            clears its edge at the cost of a bias of about half a grid cell diagonal on a
            slope. ``"mean"`` for an inverse distance weighted mean instead, smooth in the
            point position and therefore well conditioned for a reward gradient, but wrong
            by a reversal of the clearance signal, not merely an offset, at a riser.

    Returns:
        A tuple of the estimated terrain height, shape (N, P), and a boolean validity mask
        of the same shape. The height at an invalid entry is zero and must not be read.
    """
    if reduction not in ("mean", "max"):
        raise ValueError(f"reduction must be 'mean' or 'max'. Received: '{reduction}'.")

    hits_xy = ray_hits_w[..., :2]  # (N, R, 2)
    hits_z = ray_hits_w[..., 2]  # (N, R)
    # A ray that struck no geometry reports a non finite hit. Such a ray is excluded from
    # SELECTION by an infinite distance, rather than having its height substituted as
    # base_height_rough_l2 does, since a substituted height adjacent to a foot would corrupt
    # that foot's local estimate in a way it cannot corrupt a mean over the whole grid.
    finite = torch.isfinite(ray_hits_w).all(dim=-1)  # (N, R)

    # squared horizontal distance from every query point to every ray hit, (N, P, R)
    delta = points_w[..., :2].unsqueeze(2) - hits_xy.unsqueeze(1)
    dist_sq = delta.pow(2).sum(dim=-1)
    dist_sq = torch.where(finite.unsqueeze(1), dist_sq, torch.inf)

    k = min(int(num_neighbours), hits_xy.shape[1])
    near_dist_sq, near_idx = torch.topk(dist_sq, k, dim=-1, largest=False)  # (N, P, k)
    near_z = torch.gather(hits_z.unsqueeze(1).expand(-1, points_w.shape[1], -1), 2, near_idx)

    accepted = torch.isfinite(near_dist_sq)
    if max_horizontal_dist is not None:
        accepted = accepted & (near_dist_sq <= max_horizontal_dist**2)
    valid = accepted.any(dim=-1)  # (N, P)

    if reduction == "max":
        # -inf so a rejected neighbour can never win the maximum
        height = torch.where(accepted, near_z, torch.full_like(near_z, -torch.inf)).max(dim=-1)[0]
    else:
        # inverse distance weighting, the epsilon bounding the weight of a point sitting
        # exactly on a grid node rather than letting it diverge
        weights = torch.where(accepted, 1.0 / (near_dist_sq + 1.0e-6), torch.zeros_like(near_dist_sq))
        total = weights.sum(dim=-1)
        height = (weights * near_z).sum(dim=-1) / total.clamp(min=1.0e-12)

    return torch.where(valid, height, torch.zeros_like(height)), valid


def foot_clearance_reward_v4(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
    sole_offsets: list[list[float]] | None = None,
    num_neighbours: int = 4,
    max_horizontal_dist: float | None = 0.5,
    reduction: str = "max",
) -> torch.Tensor:
    """Reward the swinging feet for clearing a height above the LOCAL terrain.

    This is the terrain referenced form of :func:`foot_clearance_reward`, and it exists to
    close the defect that variant and :func:`foot_clearance_reward_v2` both carry, that
    their clearance is an absolute world height and therefore equals the height above the
    ground only on flat terrain. Here each foot is referenced to the terrain beneath ITSELF,
    estimated from the ray caster hits nearest that foot, rather than to the mean terrain
    height beneath the base that :func:`base_height_rough_l2` uses, because on a generated
    terrain carrying stairs, slopes and waves the four feet may stand on four different
    surfaces and a single mean is correct for none of them.

    The clearance is a world frame vertical difference, foot height minus ground height,
    which is the physically correct quantity because the ray caster descends vertically in
    the world frame even under a yaw alignment. Projecting it into the body frame would
    scale it by the cosine of the trunk tilt and couple the reward to pitch and roll, so a
    robot on a slope would be charged for a clearance it in fact has.

    ``target_height`` here is a clearance ABOVE THE TERRAIN and is a different physical
    quantity from v1's absolute body frame height and v2's absolute sole height, so a value
    tuned against either of those must not be carried across without re-tuning.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset, resolving the feet bodies.
        height_sensor_cfg: Configuration for the ray caster supplying the terrain heights,
            the same sensor :func:`base_height_rough_l2` reads.
        target_height: Desired clearance of the foot above the local terrain (m).
        std: Width of the Gaussian clearance kernel (m).
        tanh_mult: Scaling applied to the horizontal foot speed inside the tanh gate.
        sensor_cfg: Optional contact sensor configuration. When given, a foot in contact
            earns nothing whatever its measured clearance. Defaults to None.
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.
            Only used when ``sensor_cfg`` is given.
        sole_offsets: Optional sole points in the foot frame, as v2 takes them. When given,
            the clearance is the lowest sole point above the terrain rather than the body
            frame origin above the terrain, closing the tilt exploit on a soled foot. Leave
            as None on a point foot, where the two agree and the machinery is inert.
        num_neighbours: Ray hits reduced over per foot, passed through to the lookup.
        max_horizontal_dist: Radius (m) beyond which a ray hit is not accepted as a
            neighbour of a foot. Defaults to 0.5, half the scanner's 1.0 m width.
        reduction: ``"mean"`` or ``"max"``, passed through to the lookup.

    Returns:
        The computed reward tensor, summed over the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]

    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (N, F, 3)

    if sole_offsets is None:
        query_xy = foot_pos
        foot_z = foot_pos[..., 2]
    else:
        # lowest sole point, as v2 computes it, so the clearance is invariant to foot tilt
        foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, F, 4)
        num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]
        offsets = torch.as_tensor(sole_offsets, dtype=foot_pos.dtype, device=foot_pos.device)
        num_pts = offsets.shape[0]
        quat = foot_quat.unsqueeze(2).expand(num_envs, num_feet, num_pts, 4)
        pts = offsets.view(1, 1, num_pts, 3).expand(num_envs, num_feet, num_pts, 3)
        pts_w = math_utils.quat_apply(quat.reshape(-1, 4), pts.reshape(-1, 3)).view(num_envs, num_feet, num_pts, 3)
        pts_w = pts_w + foot_pos.unsqueeze(2)
        lowest = pts_w[..., 2].argmin(dim=2, keepdim=True)  # (N, F, 1)
        foot_z = torch.gather(pts_w[..., 2], 2, lowest).squeeze(2)
        # the ground is looked up beneath the lowest sole point, not beneath the ankle
        query_xy = torch.gather(pts_w, 2, lowest.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)

    terrain_z, valid = _terrain_height_under_points(
        query_xy, sensor.data.ray_hits_w, num_neighbours, max_horizontal_dist, reduction
    )

    clearance = foot_z - terrain_z  # (N, F), world frame vertical
    clearance_error = torch.square(clearance - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.exp(-clearance_error / std**2) * foot_velocity_tanh
    # a foot the scanner cannot resolve earns nothing rather than being scored against a
    # terrain height that was never measured beneath it
    reward = reward * valid

    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_contact = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
            > force_threshold
        )
        reward = reward * ~in_contact
    return torch.sum(reward, dim=1)
```

Wired as follows, matching the dump exactly.

```python
rew_foot_clearance = RewTerm(
    func=mdp.foot_clearance_reward_v4,
    weight=0.5,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link"),
        "height_sensor_cfg": SceneEntityCfg("height_scanner"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
        "target_height": 0.10,
        "std": 0.05,
        "tanh_mult": 2.0,
    },
)
```

Docstring notes recording the absolute-world-height defect and pointing at v4 should be added to `foot_clearance_reward` and `foot_clearance_reward_v2`, per rule 6 of `../CLAUDE.md`. Neither function's own code changes. `environments/tests/rewards_foot_clearance_v4_test.py`, thirteen checks, survived as an untracked file and previously passed in full against this exact implementation, and should be re-run to confirm the restoration before it is trusted.

### 5.5 `foot_landing_vel_v3`, recovered verbatim

```python
def foot_landing_vel_v3(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    about_landing_threshold: float,
    force_threshold: float = 1.0,
    sole_offsets: list[list[float]] | None = None,
    foot_radius: float = 0.0,
    num_neighbours: int = 4,
    max_horizontal_dist: float | None = 0.5,
    reduction: str = "max",
) -> torch.Tensor:
    """Penalise the approach velocity of a foot about to land on the LOCAL terrain.

    This is to :func:`foot_landing_vel_v2` what :func:`foot_clearance_reward_v4` is to
    :func:`foot_clearance_reward_v2`. Both earlier variants gate on an ABSOLUTE world
    height, correct only on flat terrain, so on generated terrain a foot descending toward a
    raised tread is judged against the plane the terrain was built upon rather than the
    surface it is about to strike, opening the gate too late on rising ground and too early
    on falling ground. Here the clearance is referenced to the terrain beneath each foot
    individually via :func:`_terrain_height_under_points`, the same estimator and defaults
    that :func:`foot_clearance_reward_v4` uses, so the clearance this term gates on and the
    clearance that term rewards cannot drift apart.

    The term serves a sole footed and a point footed robot from one implementation, and a
    biped and a quadruped alike, the foot count entering only as the axis the penalty is
    summed over. Which branch is taken is decided by ``sole_offsets``.

    On a sole foot the contact point is the lowest of the sole points, its clearance is that
    point's height above the terrain beneath ITSELF rather than beneath the ankle, and the
    penalised quantity is the vertical component of ``v_link + omega x r`` evaluated there,
    so a foot rotating its sole into the ground is charged for the rotation. On a point foot
    the contact point is ``foot_radius`` directly below the frame origin along the WORLD
    vertical, whatever the foot's orientation, because the lowest point of a sphere does not
    move in the body frame as the body turns. Its lever arm therefore has no horizontal
    component, ``omega_x r_y - omega_y r_x`` vanishes identically, and the approach velocity
    is the frame's vertical velocity exactly.

    This function reads ``body_link_lin_vel_w`` and ``body_link_pos_w`` throughout, as
    :func:`foot_landing_vel_v2` does, rather than the mismatched centre-of-mass velocity and
    link frame position that :func:`foot_landing_vel` pairs. A negative clearance, a foot
    that has penetrated the estimated terrain, is retained and charged rather than clipped.

    The term remains a TIME INTEGRAL of the squared approach velocity over the gate, and is
    therefore still reducible in principle by descending slowly through the window rather
    than by arriving softly. The remedy adopted is v2's, sizing ``about_landing_threshold``
    to the terminal approach rather than the whole descent, so the free fall velocity from
    the threshold height bounds what an unpowered descent can deliver.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies.
        sensor_cfg: Contact sensor configuration resolving the same feet, in the same order.
            A foot already carrying load is exempt.
        height_sensor_cfg: Configuration for the ray caster supplying the terrain heights,
            the same sensor :func:`foot_clearance_reward_v4` and :func:`base_height_rough_l2`
            read.
        about_landing_threshold: Clearance (m) above the LOCAL terrain below which a
            descending, unloaded foot is charged. Agrees with v2's argument of the same name
            on flat ground and diverges from it elsewhere, so a value tuned on a plane
            carries across unchanged while a value tuned against v1's frame proxy does not.
        force_threshold: Contact force magnitude (N) above which a foot counts as landed and
            is exempt. Defaults to 1.0, matching ``rew_foot_clearance``.
        sole_offsets: Points on the sole in the foot body frame whose lowest world height is
            the clearance. Leave as None on a point foot, where ``foot_radius`` supplies the
            offset instead.
        foot_radius: Radius (m) of the contact sphere of a point foot, read only when
            ``sole_offsets`` is None. Defaults to 0.0.
        num_neighbours: Ray hits reduced over per foot, passed through to the lookup.
        max_horizontal_dist: Radius (m) beyond which a ray hit is not accepted as a
            neighbour of a foot. Defaults to 0.5, half the scanner's 1.0 m width.
        reduction: ``"max"`` or ``"mean"``, passed through to the lookup.

    Returns:
        The computed penalty tensor, summed over the feet.

    Note:
        DEFECT LEFT STANDING. A foot for which the ray caster resolves no accepted
        neighbour is NOT charged, mirroring the choice :func:`foot_clearance_reward_v4`
        makes in refusing to pay such a foot. Refusing to pay is conservative for a reward;
        refusing to charge is permissive for a penalty, leaving a nominal exploit in which a
        policy escapes the term by placing a foot beyond half the scanner width from the
        base. On the configurations in this workspace the scanner spans 1.6 m by 1.0 m about
        the base and no reachable foot placement leaves it, so the exploit is not available
        in practice.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]

    # link frame throughout, so that the position and the velocity belong to the same point
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]        # (N, F, 3)
    lin_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]     # (N, F, 3)

    if sole_offsets is None:
        # a sphere's lowest point sits foot_radius below the origin along the world vertical
        # whatever the foot's orientation, so the lever arm is purely vertical and the
        # rotational contribution to the vertical velocity is identically zero
        query_pts = foot_pos
        contact_z = foot_pos[..., 2] - foot_radius
        approach_vel = lin_vel[..., 2]
    else:
        pts_w, r_w = _sole_points_world(asset, asset_cfg.body_ids, sole_offsets)
        contact_z, lowest = pts_w[..., 2].min(dim=2)                    # (N, F), (N, F)
        idx = lowest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)   # (N, F, 1, 3)
        r_low = torch.gather(r_w, 2, idx).squeeze(2)                    # (N, F, 3)
        # the ground is looked up beneath the lowest sole point, not beneath the ankle
        query_pts = torch.gather(pts_w, 2, idx).squeeze(2)              # (N, F, 3)
        ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
        approach_vel = lin_vel[..., 2] + (
            ang_vel[..., 0] * r_low[..., 1] - ang_vel[..., 1] * r_low[..., 0]
        )

    terrain_z, valid = _terrain_height_under_points(
        query_pts, sensor.data.ray_hits_w, num_neighbours, max_horizontal_dist, reduction
    )
    # world frame vertical difference, for the reason foot_clearance_reward_v4 records. A
    # negative value, the contact point below the estimated terrain, is retained rather than
    # clipped, so a foot that has penetrated is charged rather than hidden.
    clearance = contact_z - terrain_z                                   # (N, F)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # history max, as in feet_slide, so contact chatter cannot flicker a loaded foot back
    # into the gate between two control steps
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > force_threshold
    )

    about_to_land = (
        (clearance < about_landing_threshold) & (~in_contact) & (approach_vel < 0.0) & valid
    )
    landing_vel = torch.where(about_to_land, approach_vel, torch.zeros_like(approach_vel))
    return torch.sum(torch.square(landing_vel), dim=1)
```

Wired as follows, matching the dump exactly.

```python
pen_foot_landing_vel = RewTerm(
    func=mdp.foot_landing_vel_v3,
    weight=-0.3,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
        "height_sensor_cfg": SceneEntityCfg("height_scanner"),
        "foot_radius": 0.022,
        "about_landing_threshold": 0.04,
        "force_threshold": 1.0,
    },
)
```

The `foot_landing_vel` docstring should record a fifth defect and a pointer to v3, per the same session's own note, which read in full "FIVE DEFECTS are left standing here. Use `foot_landing_vel_v2` on a sole footed robot on flat ground, and `foot_landing_vel_v3` on any robot, soled or point footed, on terrain that is not flat. This function is retained unchanged for the callers that already read it." `environments/tests/rewards_foot_landing_vel_v3_test.py`, eleven checks, survived as an untracked file and should be re-run to confirm the restoration.

### 5.6 `pen_feet_hold` and `feet_hold_penalty`, recovered from a surviving plan document

```python
# environments/environments/tasks/locomotion/mdp/rewards.py, appended, no existing function edited
def feet_hold_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    air_time_ceiling: float,
) -> torch.Tensor:
    """Penalise a foot held in the air beyond a ceiling.

    Unlike feet_air_time and feet_air_time_v2, which are evaluated only at touchdown and
    therefore pay a foot that never lands exactly zero, this term reads the RUNNING air
    time every step, so a retracted limb accrues without bound. Deliberately unclipped,
    since a clip would remove the gradient exactly where a runaway retraction should be
    punished more, which is the defect this term exists to avoid. Gated on the command
    norm so that a standing robot is not charged for holding still.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(current_air_time - air_time_ceiling, min=0.0)
    reward = torch.sum(excess, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
```

```python
pen_feet_hold = RewTerm(
    func=mdp.feet_hold_penalty,
    weight=-0.1,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
        "command_name": "base_velocity",
        "air_time_ceiling": 0.45,
    },
)
```

`environments/tests/rewards_feet_hold_penalty_test.py`, fourteen checks, survived as an untracked file together with its compiled `__pycache__` artefact, evidence that this suite was executed against the real implementation before the loss, and should be re-run first of the four suites since a failure here would indicate a transcription error rather than an uncertain reconstruction.

### 5.7 `pen_feet_impact`, wiring only

`feet_impact_force` already exists, unchanged, at `rewards.py:165-190`. Only the wiring is missing.

```python
pen_feet_impact = RewTerm(
    func=mdp.feet_impact_force,
    weight=-5.0e-4,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
        "force_threshold": 160.0,
    },
)
```

`plans/QUADRUPED_DEGENERATE_GAIT.md` records that at this weight the term is very nearly inert, and reserves raising it to roughly -5.0e-3 for after Phase 2's retraction fix has closed the escape route by which a foot avoids the term entirely by not landing. No change to the weight is proposed beyond restoring the dump's own value.

## 6. Item three, the co-optimisation cleanup

`plans/co_optimisation_cleanup.md` states its own status as "IMPLEMENTED on 2026-09-10," with section 11 recording fourteen source files changed and a validation pass performed. The present tree contradicts this at every point checked, `copt_actor_critic.py`, `copt_ppo.py`, and `robots/__init__.py` all still carry the pre-cleanup architecture, `predictedPrivilegedObs`, the two learned model classes, and the seven task identifiers the plan retires. No grep for `gtEncoderOut`, `ENCODER_OUTPUT_KEY`, `SFCoptMorphologyRunnerCfg`, or `Isaac-Limx-SF-Copt-MoRAL` returns any match anywhere in the tree.

Since no commit exists between `a79a603`, dated 2026-09-07, and today's, and the plan's own implementation date of 2026-09-10 falls inside that gap, the entire implementation was carried out as uncommitted working tree state and was wiped in full. The plan document itself is intact and untracked, and was written and verified against the very sources that now again need the edit, so the correct action is to re-execute `plans/co_optimisation_cleanup.md` sections 5.1 to 5.8 exactly as written, not repeated here, and then to re-apply the four divergences its own section 11 records, since those were themselves part of what was implemented and lost. Before re-executing, the plan's status banner must be corrected from "IMPLEMENTED" to a state recording that the implementation did not survive, so a future reader is not again misled by the banner alone, following the correction convention of rule 17 of `../CLAUDE.md`, which asks that a contradiction be appended rather than that the earlier entry be edited away.

## 7. Item four, the Go1 debug environment and `DebugOnPolicyRunner`

The user's mid-turn addition named a timing-instrumented on-policy runner and a Go1-asset ablation environment, both underlying `/ws/context/go1_quadruped_throughput.md`, and wired into `/ws/djinn`. Because `/ws/djinn` lives in the outer, untouched repository, its wiring for `quadruped-debug`, `go1-debug`, and `go1-default-debug` survived intact and needs no restoration, confirmed present at `djinn:148-169,192`. `DebugOnPolicyRunner` itself also survived, as an untracked file at `rsl_rl/rsl_rl_debug/runners/debug_on_policy_runner.py`, 179 lines, complete. What did not survive is the pair of environment configuration classes that `go1-debug` and `go1-default-debug` name, `QuadrupedPFGo1AssetBlindRoughEnvCfg` and `Go1NativeMatchedEnvCfg`, together with their registration, all three living in tracked files. Both were recovered verbatim from session `7f6bad51-9657-4ba0-82a7-6754795f6b71` (2026-09-16), the same session `go1_quadruped_throughput.md` itself records as their author.

```python
# environments/environments/tasks/locomotion/cfg/quadruped/quadruped_pointfoot_env_cfg.py
# added import
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG


@configclass
class QuadrupedPFGo1AssetBaseEnvCfg(QuadrupedPFEnvCfg):
    """The quadruped task's own environment and MDP stack, unchanged, with the Unitree
    Go1's shipped USD in place of QUADRUPED_IDENTIFIED_CFG. Built to isolate whether this
    task's slower throughput against IsaacLab's shipped Go1 rough task, recorded in
    context/go1_quadruped_throughput.md, traces to the asset (solver iterations, self
    collision, USD versus URDF spawn, actuator model) or to the environment and MDP
    stack (the doubled observation-manager pass HIMManagerBasedRLEnv.step performs, the
    asymmetric privileged critic ObservationsCfg builds, the terrain size). The terrain,
    the observation groups, the reward, curriculum and event term counts, and the entry
    point class are left exactly as QuadrupedPFEnvCfg leaves them, since those are what
    this ablation holds fixed while swapping the asset.

    Reward magnitudes tuned to the quadruped's own geometry (pen_base_height's
    target_height=0.33, the 0.022 m foot radius in the landing and clearance terms) are
    left untouched and are therefore wrong for Go1's 0.4 m standing height, this task
    exists to measure throughput, not to converge a gait, and retuning them was out of
    scope here.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"

        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        for term in (
            self.rewards.feet_air_time,
            self.rewards.pen_feet_hold,
            self.rewards.feet_slide,
            self.rewards.rew_foot_clearance,
            self.rewards.pen_foot_landing_vel,
            self.rewards.pen_feet_impact,
        ):
            if "sensor_cfg" in term.params:
                term.params["sensor_cfg"].body_names = ".*_foot"
            if "asset_cfg" in term.params:
                term.params["asset_cfg"].body_names = ".*_foot"
        self.observations.critic.feet_lin_vel.params["asset_cfg"].body_names = ".*_foot"
        self.observations.critic.feet_contact_force.params["sensor_cfg"].body_names = ".*_foot"

        # Go1's own hip_joint/thigh_joint/calf_joint are respectively the abduction, hip
        # flexion, and knee axes, see the derivation comments in quadruped_identified_cfg.py.
        self.rewards.pen_abad_deviation.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.events.robot_joint_stiffness_and_damping_abad.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.events.robot_joint_stiffness_and_damping_hip.params["asset_cfg"].joint_names = [".*_thigh_joint"]
        self.events.robot_joint_stiffness_and_damping_knee.params["asset_cfg"].joint_names = [".*_calf_joint"]

        # No verified Go1 equivalent for the quadruped's four-name non-foot contact list.
        # Go1's own rough_env_cfg.py:54 disables this term rather than guess one, followed
        # here.
        self.rewards.pen_undesired_contacts = None

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.4, 0.0, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.4)


@configclass
class QuadrupedPFGo1AssetBlindRoughEnvCfg(QuadrupedPFGo1AssetBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


###############################################
# Go1 Native, Population and Terrain Matched, Debug
###############################################


@configclass
class Go1NativeMatchedEnvCfg(UnitreeGo1RoughEnvCfg):
    """IsaacLab's own shipped Go1 rough task, `UnitreeGo1RoughEnvCfg`, completely
    unchanged except for the population and the terrain generator, both overridden here
    to match this repository's quadruped task, `QuadrupedPFBlindRoughEnvCfg`, and its Go1
    asset ablation, `QuadrupedPFGo1AssetBlindRoughEnvCfg` above, exactly. With the asset,
    the population and the terrain all held fixed across this task and the ablation, the
    only remaining difference between the two is the environment and MDP stack itself,
    one observation group against three, a symmetric actor and critic against the
    asymmetric split `ObservationsCfg` produces, ten reward terms against twenty, one
    curriculum term against seven, three startup events against nine, letting that
    difference be measured on its own rather than inferred. See
    context/go1_quadruped_throughput.md.

    `self.scene.terrain.terrain_generator` is replaced wholesale rather than edited, so
    the `grid_height_range` and `noise_range` scaling `UnitreeGo1RoughEnvCfg.__post_init__`
    applies to Go1's own smaller terrain, `rough_env_cfg.py:24-27`, is discarded along
    with the terrain it was scaled for, `QUADRUPED_ROUGH_TERRAINS_CFG` needs no equivalent
    scaling, it already carries `curriculum=True` and its own tile dimensions as part of
    its own definition, `terrains_cfg.py`. `max_init_terrain_level`, `self.scene.terrain`'s
    other terrain related field, is left at Go1's own default of 5 rather than matched to
    the quadruped's 0, since the request this task was built for named the terrain
    generator and the population, not the curriculum's starting difficulty, and changing
    it would alter curriculum dynamics beyond the terrain's shape.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 7000
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG
```

Registered in `robots/__init__.py`, next to `Isaac-Quadruped-Blind-Rough-v0`.

```python
gym.register(
    id="Isaac-Quadruped-Go1Asset-Blind-Rough-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFGo1AssetBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

go1_native_matched_runner_cfg = UnitreeGo1RoughPPORunnerCfg()

gym.register(
    id="Isaac-Quadruped-Go1Native-Matched-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.Go1NativeMatchedEnvCfg,
        "rsl_rl_cfg_entry_point": go1_native_matched_runner_cfg,
    },
)
```

The exact module path from which `UnitreeGo1RoughEnvCfg` and `UnitreeGo1RoughPPORunnerCfg` are imported was not captured in the extracted transcript block and should be confirmed by `grep -rn "class UnitreeGo1RoughEnvCfg\|class UnitreeGo1RoughPPORunnerCfg" /ws/IsaacLab` before this is applied, a check that costs one command. `train.py` and `cli_args.py` need no restoration beyond what section 7 records for `djinn`, since `/ws/djinn`'s wiring already resolves both new task identifiers and both `train.py`'s runner selection and `cli_args.py`'s help string live in the tracked, reset repository and should be checked against the pattern of `scripts/rsl_rl/train.py:197` quoted in the throughput document, `elif args_cli.policy_type in ("quadruped-debug", "go1-debug", "go1-default-debug"): runner_cls = DebugOnPolicyRunner`.

## 8. Item five, `feet_distance_v2`, a footprint-aware feet separation penalty

This item was not named by the user and was found only by the transcript audit of section 9. Sessions `1e78dbf9-032f-4c71-ae95-d89694bc4938` and `69f84f8e-247c-4653-af45-762464786e57`, both opened 2026-09-11 at the `environments/environments` directory, developed a generalisation of `feet_distance` that additionally penalises the separation of the two feet's SOLE POLYGONS, projected onto the ground, falling below a floor, closing a gap the existing frame-origin measure cannot see, that on the KScale at its nominal pose the origin measure exceeds the true footprint separation by 84.6 mm and reads a penalty of exactly zero while the footprints are in fact touching at 0.60 rad of symmetric toe-in. The function is confirmed absent from the current `rewards.py`, and its two companion files, `scripts/analysis/kscale_feet_distance_analysis.py` and `scripts/analysis/kscale_feet_distance_v2_selftest.py`, are confirmed present and already committed at `e770a7e`, so they need no restoration.

The session's own closing statement is important and bounds this item precisely. "`environments/environments/tasks/locomotion/mdp/rewards.py` gains `feet_distance_v2` with the two private helpers `_footprint_separation` and `_perimeter_order`, and `feet_distance` gains a note recording the defect left standing... Nothing existing was altered and no configuration was touched, so no run changes behaviour until you wire it." No `RewTerm` for this function was ever written, so there is no wiring to restore, only the function itself.

```python
def _footprint_separation(pts_w: torch.Tensor) -> torch.Tensor:
    """Minimum horizontal distance between two footprints, vertex against edge, both directions.

    Args:
        pts_w: (N, 2, P, 3) world points of the two feet, the P points of each foot given in
            perimeter order so that consecutive entries bound an edge.

    Returns:
        (N,) the separation of the two convex footprints, which is exact while they are disjoint.

    For two disjoint convex polygons the closest approach is realised at a vertex of one against an
    edge of the other, so sweeping every such pair is exact rather than sampled. Comparing instead
    the P squared VERTEX pairs errs by up to 3.42 mm over the KScale pinned envelope, the closest
    approach falling on the interior of the sole's long edge, which
    scripts/analysis/kscale_feet_distance_analysis.py measures in its section 5b.

    The measure saturates at zero rather than going negative once the footprints interpenetrate,
    and that regime is deliberately left to `filtered_contacts` on the foot pair sensors, which
    reports the true contact this geometry could only approximate.
    """
    best = None
    for near, far in ((pts_w[:, 0, :, :2], pts_w[:, 1, :, :2]),
                      (pts_w[:, 1, :, :2], pts_w[:, 0, :, :2])):
        edge = torch.roll(far, -1, dims=1) - far                          # (N, Q, 2)
        delta = near.unsqueeze(2) - far.unsqueeze(1)                      # (N, P, Q, 2)
        scale = (delta * edge.unsqueeze(1)).sum(-1) / (edge * edge).sum(-1).clamp_min(1.0e-12).unsqueeze(1)
        foot = delta - scale.clamp(0.0, 1.0).unsqueeze(-1) * edge.unsqueeze(1)
        span = torch.linalg.vector_norm(foot, dim=-1).amin(dim=(1, 2))    # (N,)
        best = span if best is None else torch.minimum(best, span)
    return best


def _perimeter_order(sole_offsets: list[list[float]]) -> list[list[float]]:
    """The sole table sorted around its own perimeter, which an angular sort gives for a convex set.

    The configuration is not obliged to declare the points in traversal order, the SD_BRS1 table at
    cfg/SF/brs_base_env_cfg.py:644 being grouped in mirrored pairs, so the order is imposed here
    rather than required of the caller. The sole normal is taken as the axis of least spread, which
    needs no robot specific knowledge of which axis that is, and that matters because the KScale
    foot frame is a full axis permutation away from the SD_BRS1 convention.
    """
    offsets = np.asarray(sole_offsets, dtype=np.float64)
    plane = [i for i in range(3) if i != int(np.argmin(offsets.std(axis=0)))]
    planar = offsets[:, plane] - offsets[:, plane].mean(axis=0)
    return offsets[np.argsort(np.arctan2(planar[:, 1], planar[:, 0]))].tolist()


def feet_distance_v2(env: ManagerBasedRLEnv,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     feet_links_name: list[str] = ["foot_[RL]_Link"],
                     min_feet_distance: float = 0.1,
                     max_feet_distance: float = 1.0,
                     lateral_only: bool = False,
                     sole_offsets: list[list[float]] | None = None,
                     min_sole_distance: float = 0.0,) -> torch.Tensor:
    """Penalise the separation of the feet, measured at the frame origins and at the soles.

    :func:`feet_distance` measures the separation of the two foot LINK FRAME ORIGINS, which on a
    sole footed robot is the ankle and not the foot. The quantity that decides whether the feet
    foul one another is the separation of the two SOLE POLYGONS projected onto the ground, and the
    two diverge the moment a foot yaws, the toe swinging laterally about an origin that need not
    move. On the KScale at its nominal pose the origin measure exceeds the true footprint
    separation by 84.6 mm, being the width of a sole the original cannot see, and with the ankles
    held at the configured threshold of 0.24 m the two footprints touch at 0.60 rad of symmetric
    toe in while the original reads a penalty of exactly zero throughout. Both figures are
    reproduced by scripts/analysis/kscale_feet_distance_analysis.py.

    This term therefore carries two hinges rather than one. The first is the frame origin hinge of
    :func:`feet_distance`, reproduced bit for bit, which regulates the stance width. The second
    charges the footprint separation falling below ``min_sole_distance``, and it is silent unless
    both ``sole_offsets`` and a positive floor are supplied, so that every existing caller of this
    signature is unaffected. Give the two hinges independent weights by registering the function
    twice, the second term setting ``min_feet_distance`` to zero so that only the sole hinge speaks.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset.
        feet_links_name: Name patterns resolving the two feet bodies.
        min_feet_distance: Frame origin separation (m) below which the lower hinge becomes active.
        max_feet_distance: Frame origin separation (m) above which the upper hinge becomes active.
        lateral_only: As :func:`feet_distance`, and it governs both hinges. When False the sole
            hinge measures the true planar distance between the footprints, which is a genuine
            clearance and stays silent for feet that are merely separated fore and aft. When True
            it measures the SIGNED lateral gap in the base yaw frame, which goes negative when the
            footprints overlap in the lateral band, and which is the stance width proper.
        sole_offsets: Points on the sole in the foot body frame, the same table
            :func:`foot_clearance_reward_v3` consumes, ordered around the perimeter internally.
            Leave unset to obtain :func:`feet_distance` exactly.
        min_sole_distance: Footprint separation (m) below which the sole hinge becomes active.
            Leave at zero to obtain :func:`feet_distance` exactly.

    Returns:
        The computed penalty tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:, feet_links_idx]
    yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w) if lateral_only else None

    if lateral_only:
        # rotate the planar separation into the base frame and keep the lateral component
        diff_w = feet_pos[:, 0, :3] - feet_pos[:, 1, :3]
        diff_b = math_utils.quat_apply_inverse(yaw_quat, diff_w)
        feet_distance = torch.abs(diff_b[:, 1])
    else:
        # feet distance on x-y plane
        feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)

    if sole_offsets is None or min_sole_distance <= 0.0:
        return reward

    pts_w, _ = _sole_points_world(asset, feet_links_idx, _perimeter_order(sole_offsets))
    if lateral_only:
        # the signed lateral gap between the two footprints, negative where their bands overlap
        num_envs, num_feet, num_pts, _ = pts_w.shape
        rel = (pts_w - asset.data.root_link_pos_w[:, None, None, :]).reshape(-1, 3)
        quat = yaw_quat[:, None, None, :].expand(num_envs, num_feet, num_pts, 4).reshape(-1, 4)
        lateral = math_utils.quat_apply_inverse(quat, rel).view(num_envs, num_feet, num_pts, 3)[..., 1]
        sole_distance = torch.maximum(lateral[:, 0].amin(-1) - lateral[:, 1].amax(-1),
                                      lateral[:, 1].amin(-1) - lateral[:, 0].amax(-1))
    else:
        sole_distance = _footprint_separation(pts_w)
    reward += torch.clip(min_sole_distance - sole_distance, 0, 1)
    return reward
```

This is inserted immediately before `feet_yaw_alignment`, following `feet_distance`, and no existing function is edited beyond a docstring note added to `feet_distance` itself recording the defect and pointing at v2, per rule 6 of `../CLAUDE.md`. `scripts/analysis/kscale_feet_distance_v2_selftest.py`, already in the tree, runs seventeen checks against exactly this implementation extracted from the live source, and confirms at its defaults that the function reproduces `feet_distance` bit for bit, so restoring it and running that suite is the correct order of operations. Whether and how to wire this term into any `RewardsCfg` was left to the user in the original session and remains so here, this document restoring only what existed in the working tree.

## 9. Audit of session transcripts for other missed work

Every session transcript dated after the last real commit, 2026-09-07, was surveyed by title, opening request, and, where substantial, its closing summary, specifically to find work beyond the five items above. The following were screened and found not to require restoration.

Session `c960f51a-3d5f-4c31-b90a-c8cee6258305` (2026-09-11) opened by asking for an undesired-contacts collision penalty between the KScale's foot links, investigating whether the URDF's foot bodies could be merged to simplify it, and concluded that no fixed joints exist to merge, the apparent multiplicity being separate collision shapes on one rigid body that an upstream export tool had already consolidated. No code change resulted, only an answer, so nothing from this session needs restoring.

Session `9cf03661-5022-4f9b-a79f-b0558e74d299` (2026-09-17, today) verified two already-existing commits, `d6208a5` and `4ad0bd1` from 2026-08-26, extracting the SD_BRS1 and KScale asset trees into two private submodules, and confirmed via `git submodule status` that both remain correctly pinned. This is a status check against history that already stands committed, not new work at risk.

Session `58c1bf13-e702-468a-bc16-f9fe99c9b8bb` asked for a context document section recording a more crouched initial pose for a biped. This was not investigated to the same depth as the other five items, its scope being a documentation update rather than a code change, and the user should check whether the relevant context document, most likely `context/brs_gait.md` or `plans/NATURAL_GAIT_PLAN.md` given the session's working directory, already carries the section, since several dated additions of exactly this kind were found already present in `context/KScale.md` during this investigation and this may be one more.

One further item was found that is not a casualty of this incident and is recorded here because the audit was asked to find missed work generally, not only work this incident destroyed. Session `ff145cdb-4b7b-4cb4-8eb4-99eda25f7f83` (2026-09-17, this morning, before the incident) carried out a substantial investigation into a quadruped training failure, establishing by cross-referencing every run's TensorBoard record against its dumped configuration that the failure is terrain-specific rather than asset-specific, refuting a hypothesis the user had proposed. Its own last words are "Checkpointing here. The investigation is complete and the evidence is preserved; the document itself is still to be written." Nothing here was lost by the reset, since nothing was yet written to a file the reset could reach, but the evidence and its analysis exist only inside that session's transcript, which is not the durable record `../CLAUDE.md` rule 17 calls for. Whether to have that investigation written up as a proper context document is a decision for the user, and is outside the scope of restoring what the incident destroyed, so it is flagged here rather than acted upon.

## 10. Verification, in the order that catches the cheapest mistakes first

Compile every file this restoration touches with `python3 -m py_compile` before anything else, exactly as `plans/co_optimisation_cleanup.md` section 9 already prescribes for its own scope, and apply the same first step to the reward module.

Run the four surviving test files before wiring any new term into a `RewardsCfg`.

```bash
python3 -m pytest environments/tests/rewards_feet_hold_penalty_test.py
python3 -m pytest environments/tests/rewards_foot_clearance_v4_test.py
python3 -m pytest environments/tests/rewards_foot_landing_vel_v3_test.py
```

`scripts/analysis/kscale_feet_distance_v2_selftest.py` should be run the same way once `feet_distance_v2` is restored, since it is already in the tree and already exercises exactly this implementation.

Once the reward module compiles and its tests pass, wire the quadruped's terms in the order given in section 5, launch `Isaac-Quadruped-Blind-Rough-v0` for a handful of iterations, and confirm from the startup output that every reward term resolves and that `Episode_Reward/pen_feet_hold`, `Episode_Reward/rew_foot_clearance`, `Episode_Reward/pen_foot_landing_vel`, and `Episode_Reward/pen_feet_impact` all appear among the logged scalars.

For the co-optimisation cleanup, follow `plans/co_optimisation_cleanup.md` section 9 in full. For the Go1 debug environment, confirm the import paths named in section 7 by grep against the installed `IsaacLab` tree, then invoke `djinn start train go1-debug` and `djinn start train go1-default-debug` far enough to see both resolved task identifiers and the `[debug]` timing line the runner prints every iteration.

## 11. What this document does not attempt

No attempt is made to recover the exact wording of every docstring note added to an untouched sibling function, such as the rule-6 pointers on `foot_clearance_reward`, `foot_clearance_reward_v2`, `foot_landing_vel`, and `feet_distance`, beyond what the transcripts happened to quote, and a reader who later obtains the original wording from some other channel should prefer it. The `changes` and `changes_nithin` files at the workspace and repository roots were searched for every function name this document restores and matched only the already present `feet_impact_force`, confirming they predate the September work entirely. No training run has been launched by this investigation, so every weight and threshold restored here carries only the authority of the 2026-09-11 dump, the surviving plan documents, and the session transcripts, not of a fresh confirmation that the restored configuration trains as the original did.
