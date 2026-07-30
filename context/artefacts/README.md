# Artefacts Index

This directory holds the debugging artefacts of the workspace, the exported TensorBoard panels, evaluation plots, and dashboards that the context and plan documents cite as evidence. Their purpose here is twofold, to keep the workspace root clear of loose exports, and to preserve a durable experimental memory, so that a claim made in a document about a training run can still be checked against the figure that produced it long after the run itself has been deleted from the log tree.

The distinction from the training log tree matters. A run's own outputs, its checkpoints, its dumped configuration, its videos, and its numpy dumps, remain where the trainer writes them under `/ws/IsaacLab/logs/rsl_rl/<experiment>/<timestamp>/`, and that tree is transient, runs are deleted as disk fills. This directory holds only the artefacts that were exported out of that tree for analysis, and those are permanent.

Two conditions govern admission, and both must hold. An artefact must be cited by a context or a plan document, whether individually or as a set the document reasons about, and it must be attributable to the run that produced it. An artefact that no document cites is not evidence of anything and does not belong here, and an artefact whose provenance is unknown cannot be quoted against a run. Both conditions are enforced by deletion rather than by annotation, so the presence of a file in this directory is itself the assurance that some document depends upon it.

## Register

| Directory | Contents | Attributed run | Provenance of the attribution |
|---|---|---|---|
| `plots/` | 14 TensorBoard panels, `reward1` to `reward7`, `loss1`, `loss2`, `curriculum1` to `curriculum3`, `noise`, `termination` | `sf_copt/2026-06-26_05-26-28`, 30000 iterations | Certain, the panels carry the run label, recorded in `../task_plots.md` section 1 |
| `plots-latest/` | 10 TensorBoard panels, `reward1` to `reward7`, `loss1`, `loss2`, `metrics` | `sf_copt/2026-07-03_08-16-11`, 45000 iterations over 3.377 days | Certain, recorded in `../literature.md` and in the LIPM section of `../knowledge_base.md` |
| `plots-latest-2/` | 1 DataLogger figure, `clearance_and_forces.png` | Environment 0 of `sd_brs1_flat/2026-07-17_11-01-29` | Certain, named in the ninth pass of `../brs_gait.md` |
| `plots_latest_play/` | 8 evaluation plots from the `play.py` DataLogger, `base_vel`, `feet_forces_clearance_vel`, `joint_accel`, `joint_position`, `joint_power`, `joint_torque`, `joint_vel`, `torque_vs_joint_vel` | Play evaluation of `sd_brs1_flat/2026-07-22_11-36-53` | Certain, recorded in `../brs_gait.md` sixteenth pass and in `../../plans/NATURAL_GAIT_PLAN.md` section 1 |

Three of the four directories are cited as complete sets rather than file by file. The fourteen panels of `plots/` are enumerated in the file map of `../knowledge_base.md` and reasoned about individually in section 4 of `../../plans/COPT_INVESTIGATION_PLAN.md`. The eight plots of `plots_latest_play/` are cited as the eight diagnostic plots of that evaluation in both documents that use them. The ten panels of `plots-latest/` carry three individual embeds in `../../plans/LIPM_REWARD.md`, namely `reward7`, `loss2`, and `metrics`, and the remaining seven supply the per component reward bands and the explained variance trace that the same passage and the LIPM section of `../knowledge_base.md` reason about collectively, so the directory is retained whole.

## Removals

The register above is the result of a cull performed on 2026-07-30, recorded here so that a reader who recalls a figure and cannot find it learns why rather than assuming an oversight.

Two collections were deleted for want of any citation. `plots_eval/` held seventeen Plotly exports named `newplot.png` through `newplot(16).png`, and `report.html` was an evaluation dashboard from the same period, both dating to March 2026 and both unattributable to any run. No context or plan document referenced either, so neither could serve as evidence for a claim.

The fourteen TensorBoard panels that stood alongside `clearance_and_forces.png` in `plots-latest-2/` were deleted for the same want of citation, compounded by an unresolvable provenance. They belonged to the SD_BRS1 arm trained on 2026-07-17, but three runs share that date and no document records which produced them, so they could not have been quoted against a run even had one been cited. The single figure that survives is the one that settled the mechanism of the forged foot contact and refuted the earlier claim that the two feet pressed against one another.

Two panels that the documents once cited were never present to begin with. `../../plans/LIPM_REWARD.md` embedded a termination figure and a curriculum figure from run `sf_copt/2026-07-03_08-16-11`, but `plots-latest/` contains neither, so those two embeds were removed on the same date and the measurements they illustrated were retained as prose.

## Conventions

A figure referenced from a context document is addressed as `artefacts/<directory>/<file>`, and one referenced from a plan document as `../context/artefacts/<directory>/<file>`, which is the form the markdown embeds in `../../plans/LIPM_REWARD.md` use.

The directory names are historical and deliberately unaltered, since every citation in the documents uses them, and renaming would divorce the prose from the path. Their meaning is supplied by the register above rather than by the names themselves, which is why any new directory should be named for its run rather than for its recency.

When a new debugging artefact arrives, place it here, attribute it in the register, and cite it from the document that reasons about it. This obligation is codified in `../../CLAUDE.md`.
