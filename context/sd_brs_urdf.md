# sd_brs_urdf.md — Simplified primitive URDF for the SD_BRS1 biped

Created 2026-07-10. This file records how `exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/SD_BRS1/SD_BRS.urdf` was derived from `SD_BRS1_Assembly2.urdf`, so future sessions can regenerate or adjust it without repeating the mesh investigation.

## What SD_BRS.urdf is

It is a primitive only replica of `SD_BRS1_Assembly2.urdf` in which every STL mesh (visual and collision) is replaced by boxes and cylinders, while the kinematic tree, joint types, joint origins, joint axes, joint limits, link masses, centres of mass, and inertia tensors are copied verbatim from the source, the dynamics are therefore bit identical and only the collision and visual geometry is simplified.

## Design rules used

1. Each link is approximated by one box for the body plus cylinders at the revolute joints, every joint cylinder is coaxial with its joint axis, which makes its swept volume invariant under that joint's motion and keeps clearances constant.
2. Geometry around a shared joint cluster is owned by one link only, for example the hip rotor belongs to Link3 and the roll housing to Link1, which avoids spurious static overlap between non adjacent links.
3. Link5 (the 72 g ankle cross bar, radius 9.5 mm) keeps its inertial block and a visual cylinder but has no collision geometry, it lives entirely inside the foot yoke and any solid for it would permanently interpenetrate the foot drum.
4. Parent child pairs overlap at the joint clusters exactly as the nested real hardware does, simulators (PhysX and Isaac Lab articulations, MuJoCo) filter direct parent child contacts by default, all other pairs were verified collision free, see validation below.

## Primitive dimensions (right leg, link frames, metres, left leg is the y mirror)

| Link | Primitive | Size or radius x length | Origin (xyz) | Axis |
|---|---|---|---|---|
| Part_Torso | box | 0.1622 0.4604 0.1550 | -0.0186 0 0.01467 | |
| Link1R | box (yaw arm) | 0.0928 0.0850 0.0948 | -0.2024 0 0.0049 | |
| Link2R | cylinder (pitch stub) | r 0.0505 l 0.0428 | 0 0.0906 0 | y |
| Link3R | cylinder (hip rotor) | r 0.0680 l 0.1396 | 0 0.0041 0 | y |
| Link3R | box (thigh) | 0.1382 0.2100 0.2900 | 0.0011 0 -0.2300 | |
| Link3R | cylinder (knee fork) | r 0.0650 l 0.2285 | 0 0.0032 -0.4400 | y |
| Link4R | cylinder (knee clevis) | r 0.0500 l 0.1206 | 0 0 0 | y |
| Link4R | box (shank) | 0.1425 0.1250 0.3060 | 0.01125 0 -0.2210 | |
| Link6R | cylinder (ankle drum) | r 0.0500 l 0.1400 | 0 0 -0.0320 | y |
| Link6R | box (sole plate) | 0.3156 0.1900 0.0460 | 0.0202 0 -0.1010 | |

Constraint notes worth keeping. The Link1 housing cylinder must end at x >= -0.106 and x <= -0.071 because the Link3 rotor occupies |x| <= 0.068 under every roll and pitch combination. The thigh box top sits at z = -0.085 with x half width 0.068 so its back top corner clears the Link1 housing across the full hip pitch range. The sole box y half width is 0.095, two millimetres under the mesh, because sharp box corners otherwise clip the opposite foot at simultaneous double extreme ankle roll and pitch where the real meshes clear by only 0.5 mm. The shank ankle motor housing (mesh x down to -0.142 near the ankle) was deliberately not modelled, adding it introduced a right shank versus left foot collision where meshes clear by 23 mm.

## Validation performed (all passing on 2026-07-10)

1. Structure, joints, limits, inertials parsed from both URDFs and compared numerically, identical, total mass 59.8523 kg both.
2. Coverage, fraction of mesh vertices inside the primitive union with 2 mm tolerance, torso 100 %, Link2 95 %, Link3 86 to 90 %, Link4 83 %, Link5 87 %, Link1 69 %, Link6 65 % (the foot ankle motor bulge is uncovered by design, the sole contact surface is exact).
3. Self collision sweep, forward kinematics over 3973 configurations (per leg 3^5 limit grids, mirrored and anti mirrored double leg extremes, 3000 random interior samples), fcl distance checks on all 47 simulator unfiltered link pairs, zero introduced collisions, every residual contact pair (thigh thigh, knee knee, shank opposite foot, foot foot, Link1 Link3 at combined extremes) was cross checked against the original STL meshes and the meshes also collide there, these contacts are inherent to the hardware geometry and identical in kind to the source model.
4. Independent parse with yourdfpy, 13 links, 12 joints, 10 actuated joints, validate() true.

The analysis, builder, and validation scripts lived in the session scratchpad (`analyze_stl.py`, `slice_stl.py`, `brs_spec.py`, `build_urdf.py`, `validate.py`), the primitive table above plus the constraint notes are sufficient to reconstruct them, key libraries were trimesh, python-fcl, and yourdfpy installed with pip --user --break-system-packages.

## IsaacLab DOF ordering (added 2026-07-14)

Both URDFs declare the right leg chain first (Link1R through Link6R) and then the left, and each chain runs HipYaw (fixed), HipRoll, HipPitch, KneePitch, AnkleRoll, AnklePitch. IsaacLab, through the PhysX articulation importer, enumerates degrees of freedom breadth first over the articulation tree, so the two symmetric legs interleave at each tree depth with the right joint preceding its left counterpart, and the fixed HipYaw joints contribute no DOF (the config `sd_brs1_identified_cfg.py` spawns with `merge_fixed_joints=False`, which retains the Link1 bodies but adds no joint). The resulting 10 element DOF order, which is the column order of every joint quantity dumped by `DataLogger` in `scripts/rsl_rl/play.py` (joint_velocities, joint_torques, joint_powers, joint_positions, joint_accelerations), is HipRollR, HipRollL, HipPitchR, HipPitchL, KneePitchR, KneePitchL, AnkleRollR, AnkleRollL, AnklePitchR, AnklePitchL. This ordering is derived from the documented breadth first rule and the URDF declaration order, it has not yet been confirmed against a live `robot.joint_names` printout, a session with the IsaacLab python available should verify it once and note the result here. The dashboard at `/ws/IsaacLab/logs/rsl_rl/dashboard-brs.py` uses this order in its `JOINT_NAMES` list, plots the 10 joints on a 5 by 2 grid, and derives all loop bounds from `NUM_JOINTS = len(JOINT_NAMES)`.
