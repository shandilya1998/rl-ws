# Context Index

This directory holds the accumulated factual record of the workspace, the documents that establish what the codebase does, what the experiments measured, and what the literature prescribes. A context document is descriptive rather than prescriptive, it records findings that have been verified against the live sources, the dumped run configurations, the TensorBoard logs, or the published literature, and it carries file and line citations so that a later reader need not repeat the investigation. Documents that propose changes rather than record facts live in [../plans](../plans/README.md) instead.

The purpose of this index is to let a reader, whether human or agent, determine which documents bear on a question and in what order to read them, without opening any of them speculatively. Each entry below states the subject, the currency of the contents, and the specific questions the document answers.

## Reading protocol

Begin with [knowledge_base.md](knowledge_base.md), which is the hub of the co-optimisation investigation and carries the objectives, the entry point file map, and the consolidated findings of every investigation wave. It also records the later work streams in summary, so it will indicate which of the specialised documents a given question requires.

Thereafter read only the documents named in the register below as bearing on the subject at hand. The specialised documents were written by parallel investigation agents against disjoint areas, so they overlap very little, and reading one rarely obliges reading another. Where two documents disagree, the later dated entry governs, and the currency column records which documents contain claims that the codebase has since overtaken.

Before modifying any shared module, consult the document covering that module rather than re-reading the source, and record any newly discovered fact in the relevant document so that the next session inherits it. This obligation is codified in [../CLAUDE.md](../CLAUDE.md).

## Document register

| Document | Subject | Last revised | Currency |
|---|---|---|---|
| [knowledge_base.md](knowledge_base.md) | Investigation hub, objectives, file map, consolidated findings, work stream log | 2026-07-29 | Current, early sections describe the pre-remediation pipeline |
| [brs_gait.md](brs_gait.md) | SD_BRS1 gait quality investigation across nineteen passes and eight training runs | 2026-07-29 | Current, the most active record in the workspace |
| [copt.md](copt.md) | The `co_optimisation` package, runner, generator, utilities, policy | 2026-07-03 | Current in sections 8 and 9, which supersede the earlier bug register |
| [rsl_rl.md](rsl_rl.md) | The vendored `rsl_rl` library internals, runner, PPO, storage, normalisation, symmetry contract | 2026-07-22 | Current |
| [isaaclab_env.md](isaaclab_env.md) | Isaac Lab environment and manager state inventory, reset semantics, physics tensor layouts | 2026-07-22 | Current |
| [literature.md](literature.md) | Verified literature survey in twelve thematic clusters | 2026-07-27 | Current |
| [sd_brs_urdf.md](sd_brs_urdf.md) | The simplified primitive URDF for SD_BRS1, dimensions, validation, DOF ordering | 2026-07-23 | Current |
| [mjlab.md](mjlab.md) | Survey of mjlab as a candidate replacement simulation backend | 2026-07-30 | Current, every citation re-verified against the restored clone, see section 15 |
| [cmaes.md](cmaes.md) | CMA-ES theory, pycma behaviour, statistics of the recorded design trajectory | 2026-06-29 | Diagnosis stands, the hyperparameters it criticises have since been changed |
| [task_plots.md](task_plots.md) | COPT task chain, hyperparameter inventory, per-plot variance characterisation | 2026-06-29 | Analysis stands, several inventoried values have since been changed |
| [RESET.md](RESET.md) | Why the episode termination plots collapse to flat lines under the COPT runner | 2026-06-12 | Historical, analysed a morphology interval of 10 against the present 240 |
| [RL_Environment_Comparison_Report.md](RL_Environment_Comparison_Report.md) | Component comparison of the TRON1 HIM environment against the Berkeley Humanoid | 2026-03-02 | Historical, predates the SD_BRS1 and co-optimisation work streams |

One document that formerly sat here now lives in the simulation repository. The analysis of proportional derivative joint control, action scaling, and velocity saturation is at [../tron1-rl-isaaclab-cozum/context/joint_control_analysis.md](../tron1-rl-isaaclab-cozum/context/joint_control_analysis.md), a byte identical copy of it having been kept in both places until 2026-07-30, when the workspace copy was deleted and the repository copy made canonical. The subject is the parameterisation of a robot in that repository, so it belongs beside the SD_BRS1 gain derivation in `BRS.md` rather than here.

## Document summaries

### knowledge_base.md

The shared knowledge base of the design and policy co-optimisation investigation, and the correct point of entry to this directory. It states the three investigation objectives, namely the persistently high training variance, the redundant repetition of the random design phase on every run, and the biased low variety sampling of the design optimiser. It then carries the key entry point file map, the facts established by the orchestrator on first reading, and the consolidated cross-validated findings of the first investigation wave, which resolved each objective to a root cause. Three later work streams are recorded in summary at the end, the LIPM guided reward of 2026-07-10, the learned model architecture extension proposed on 2026-07-02 and implemented on 2026-07-03, and the SD_BRS1 gait quality stream opened on 2026-07-17 and last updated on 2026-07-29. Read this document to learn what has been investigated and which specialised document holds the detail.

### brs_gait.md

The running record of the SD_BRS1 gait quality investigation, and the most frequently amended document in the workspace. It advances in numbered passes, each pass analysing a training run through video frames, TensorBoard scalars, and the run's own dumped configuration, and each pass superseding the narrative of its predecessors where the evidence demanded. Its principal results are the discovery that the foot clearance reward measured the ankle frame height and was therefore farmed by tilting a grounded foot, the two successive remedies of a contact gate and then a true sole clearance term, the identification of a permanent shank into foot self penetration in the URDF as the root cause of hopping and dragging, the reward budget inversions computed as rates per second divided by weights, and the finding that the all zero nominal pose rested on the knee hard limit at the straight leg singularity. It closes with an experiment history table covering eight runs from 2026-07-15 to 2026-07-27, which is the fastest way to learn what each configuration achieved.

### copt.md

The complete reading of the `co_optimisation` package, covering the runner, the design generators, the utilities, and the policy. It enumerates the member state of `CoptOnPolicyRunner`, traces the rollout and bookkeeping of `learn`, gives the fitness accumulation arithmetic, and distinguishes the `_update_morphology` pathway from `_reload_morphology`. It documents the generator class hierarchy, the ask and tell cycle, the design resolution floor imposed by rounding to two decimal places, and the growing distribution schedule. It answers decisively that the value function does observe the morphology, since the link lengths, body masses, and inertias are present in the critic group. Sections 8 and 9 are the current statements, section 8 records the code state as of 2026-07-02 and supersedes the earlier bug register, and section 9 is the implementation record of the learned model extension including the two ambiguities that were resolved during implementation.

### rsl_rl.md

The internals of the vendored reinforcement learning library, read at the granularity of individual methods with line citations. It covers the constructor, learning loop, logging, save, load, and algorithm construction of `OnPolicyRunner`, the exact differences of the COPT override, the hyperparameters and losses of `PPO` with its adaptive schedule, generalised advantage estimation and advantage normalisation in `RolloutStorage`, the treatment of truncation against termination, the mini-batch generator, and the empirical observation normaliser. It enumerates every stateful member that a complete checkpoint must persist, which is the foundation of the save and load work. Two later additions matter for current work, section 7 records the groundwork for the learned model architecture, and the final two sections give the symmetry augmentation consumer contract together with the requirement that the critic group be mirrored for value invariance.

### isaaclab_env.md

The state inventory of the Isaac Lab environment and its managers, written to determine what a resume must restore. It walks `ManagerBasedEnv` and `ManagerBasedRLEnv` member by member, gives the exact semantics of `reset` and of both reset index paths, and enumerates the state held by the curriculum, command, reward, termination, observation, action, and event managers. Its central artefact is a categorised table dividing state into three classes, the cheaply serialisable tensors and scalars that must be saved for an identical resume, the quantities that are re-derivable and need no persistence, and the state bound to USD or PhysX that cannot be pickled and must instead be extracted and reapplied. A later addition documents the rigid body physics tensor layouts and how a morphology change is reflected in them, which is the mechanism any per environment design update must use.

### literature.md

The verified literature survey supporting every design decision in the workspace, organised into twelve thematic clusters, each entry carrying a citation and a note on its specific relevance. Clusters one through five support the learned model co-optimisation architecture, covering privileged learning and estimator architectures, auxiliary and representation losses, dynamics information for sample efficiency, morphology and control co-optimisation, and co-design with learned latent representations. The later clusters were added as work streams opened, cluster six for the LIPM guided reward, cluster seven for gait quality and posture shaping, cluster eight for cadence and contact scheduling, cluster nine for symmetry enforcement, cluster ten for natural gait and knee flexion, cluster eleven for reward kernel failure and nominal pose shaping, and cluster twelve for the provenance of the contact history index and the ancestry of this codebase. Consult this document before citing any external result.

### sd_brs_urdf.md

The specification of the simplified primitive URDF built for the SD_BRS1 biped, the shortest document in the directory and the authoritative source for that robot's geometry. It states what `SD_BRS.urdf` is and why a primitive rebuild was preferred to the mesh assembly, gives the design rules that governed the rebuild, tabulates the primitive dimensions of the right leg in link frame metres with the left leg as the lateral mirror, and records the validation performed. A later addition gives the Isaac Lab degree of freedom ordering, which any observation mirror, reward term, or plotting script indexing joints by position must respect.

### mjlab.md

An investigation of whether the mjlab framework could replace Isaac Lab as the simulation backend for co-optimisation. It documents the entity, actuator, and scene abstractions, the world dimension and the shared model with selective per world expansion, zero copy interoperability between PyTorch and Warp, CUDA graph capture, and the contact and constraint memory model. Its decisive finding is that mjlab offers two per world design mechanisms, a static variant system and dynamic domain randomisation implemented as per world model field writes, and that only the latter admits the runtime morphology changes co-optimisation requires, because MuJoCo Warp performs no collision cooking and therefore needs no respawn. The document closes with an implementation blueprint, a point by point answer to the four questions it was charged with, open verification items, and a file and symbol reference map. Note that the mjlab clone it cites at `/ws/mjlab` is no longer present in the workspace, so its line citations cannot presently be re-verified.

### cmaes.md

The theory of covariance matrix adaptation and the behaviour of the pycma implementation, written to explain why the recorded design trajectory collapsed. It establishes what an initial step size means on a unit box and why the value then in use was extreme, exposes the hidden per coordinate standard deviation cap that pycma silently applies from the bound range, describes the saturating boundary handler and the resulting bias toward the upper bound, and enumerates the relevant options and termination criteria. It then analyses the recorded design trajectory statistically, showing the spread growing through the random phase and collapsing to zero thereafter with both parameters pinned to their rounded upper bounds, and closes with a recommended initialisation. The diagnosis remains sound, and the hyperparameters it criticises have since been changed in the manner it recommended, so read it for the reasoning rather than for the present configuration.

### task_plots.md

The configuration inventory of the co-optimisation task together with a characterisation of every diagnostic plot. It resolves the full task chain from the registered identifier through the environment and agent configuration classes, inventories the environment, agent, encoder, and evolutionary hyperparameters, and explains the interaction between the environment count and the bounded reward buffer that lies at the centre of the training variance question. It establishes the contents of each observation group, which is what settles the question of morphology visibility, describes the curriculum terms and their schedules, and then characterises each reward, loss, curriculum, and termination panel individually with the observed ranges and trends. A closing section lists the discrepancies it found against the architecture document. Several inventoried values have since been changed, so treat the inventory as a record of the run it describes rather than as the present configuration.

### RESET.md

An investigation of why every episode termination curve collapses to a flat horizontal line under the co-optimisation runner while the same metrics evolve normally under the standard runner. Its conclusion is that the flat lines are a correct statistic of a degenerate event distribution rather than a logging fault, because the full environment reset performed at every morphology update made timeouts physically unreachable. The document traces all three reset pathways end to end, base contact, timeout, and design update, explains how the termination statistic is computed and logged, correlates the mechanism with the observed plots, audits the design update pathway for secondary defects, and proposes five remedies. It analysed a morphology update interval of ten iterations, whereas the present interval is 240 and episodes now complete, so the specific pathology is resolved and the document is retained for its authoritative trace of the reset machinery.

### RL_Environment_Comparison_Report.md

A component by component comparison of the TRON1 environment against the Berkeley Humanoid implementation, undertaken to explain why the former tracked commanded velocities poorly while the latter tracked them well. It compares the two implementations across observations, actions, curriculum, rewards, and agent configuration, performs a root cause analysis within each category, and closes with a combined table of every difference, a summary of root causes, and actionable recommendations. It is the earliest document in the workspace and predates both the SD_BRS1 robot and the co-optimisation pipeline, so it is retained as the origin of several reward and observation choices that later documents assume rather than as a statement of the present configuration.

## Debugging artefacts

The figures these documents reason about are held in [artefacts/](artefacts/README.md), which carries its own register attributing each directory of plots to the run that produced it. The distinction from the training log tree is deliberate, a run's own outputs remain under `/ws/IsaacLab/logs/rsl_rl/` where the trainer writes them and are deleted as disk fills, whereas anything exported for analysis is moved into `artefacts/` and kept permanently. A figure cited from a context document is addressed as `artefacts/<directory>/<file>`.

Of the fifteen training runs cited across this directory only `sd_brs1_flat/2026-07-27_07-19-39` still exists in the log tree, so the videos and numpy dumps of every earlier run are gone. Their pointers were removed on 2026-07-30 and the observations they supported retained, each marked as no longer retained at the point of use.

## Conventions

Codebase citations take the form `path:line`, resolved against the workspace root unless the path is already relative to the simulation repository. Line numbers were correct when written and drift as the sources change, so a citation should be treated as a pointer to a symbol rather than to a position, and the symbol should be located by name when the line no longer matches.

Cross references within this directory use the bare filename, since the documents are co-located. References to the plan documents take the form `../plans/NAME.md`, and references to the two architecture documents at the workspace root take the form `../NAME.md`.

Document names are lowercase and descriptive of their subject. The specialised documents carried a `CONTEXT_` prefix until 2026-07-30, when it was dropped as redundant within a directory named `context`, and the hub was renamed from `CONTEXT.md` to `knowledge_base.md` in the same pass. Two documents retain their original uppercase names, `RESET.md` and `RL_Environment_Comparison_Report.md`, because they are cited under those names in the older record.

Dated section headings record when a finding was established. Where two sections of one document conflict, the later date governs, and the earlier section is retained because the reasoning that produced it remains instructive.

## Related documentation

[../ARCHITECTURE.md](../ARCHITECTURE.md) describes the software architecture of the framework, the task registry, and the training and evaluation pipelines. [../CO_OPTIMISATION.md](../CO_OPTIMISATION.md) describes the design and controller co-optimisation architecture in full. Both remain at the workspace root because they describe the project rather than a single investigation.

The simulation repository keeps its own factual record, indexed by [../tron1-rl-isaaclab-cozum/context/README.md](../tron1-rl-isaaclab-cozum/context/README.md), holding what is specific to that repository alone, the SD_BRS1 actuator gain derivation in `BRS.md`, the hybrid internal model mathematics in `HIM.md`, and a copy of the joint control analysis. Its plan directory, [../tron1-rl-isaaclab-cozum/plans/README.md](../tron1-rl-isaaclab-cozum/plans/README.md), is presently empty, since every plan so far spans the workspace rather than that repository alone. Its architecture document `../tron1-rl-isaaclab-cozum/ARCHITECTURE.md` covers task integration and remains at that repository's root.
