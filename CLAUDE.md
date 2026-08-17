# CLAUDE.md

## Code change instructions

These rules govern all changes Claude makes to code in this workspace.

1. Evaluate the available options before making a change, not after. The goal is to write as little code as possible while preserving backwards compatibility absolutely.
2. Backwards compatibility must always be maintained. No change may alter the behaviour of an existing caller that did not ask for the change.
3. Before editing anything in a shared module, notably the `bipedal_locomotion` mdp package, enumerate its callers across every configuration, since SD_BRS1 and the TRON1 SF and PF tasks share those functions and an in place edit silently changes the others and invalidates comparison against in flight and historical runs.
4. Where an optional argument can carry the new behaviour without altering the method's existing logic, prefer an optional argument, or a similarly small measure, whose default reproduces the old behaviour exactly. Set it explicitly only in the calling configuration, which also records the choice in that run's dumped `params/env.yaml`.
5. Only where no optional argument can preserve the old behaviour should a v2 of the method be created, for example `feet_regulation_v2`, with the calling configuration repointed at it and the original left untouched.
6. Record the defect left standing in the untouched original or the unset default, so that other implementations may opt in deliberately.

## Writing instructions

These rules govern all documents, reports, plans, and context files written by Claude in this workspace. They apply to newly authored prose, pre-existing documents keep their original formatting unless a rewrite is requested.

### Line formatting

1. Never hard-wrap prose at a character column. Every sentence, paragraph, and list item must occupy one complete logical line, however long, with line breaks only at genuine boundaries, the end of a paragraph, a list item, a heading, or a table row. Partial truncation of lines and breaking of sentences across lines is not acceptable.
2. Code blocks, tables, and shell commands are exempt, format those according to their own conventions.

### Punctuation and emphasis

3. Do not use the hyphen, the semicolon, or the colon to break sentences. Only the comma may break a sentence. The three symbols may appear solely in contexts where they are not sentence breaks, for example code, file paths, list markers, table syntax, ratios, and compound words.
4. Do not italicise or bold text unless explicitly asked to.

### Structure and style

5. Ensure all writing follows a high level coherent structure.
6. Ensure all writing is grounded in fact and evidence, from papers, the codebase, the internet, or the dumped context files, never from unverified recall.
7. When writing documents ensure a flow to the document while slowly building up to the ideas to be delivered.
8. Do not use overtly long or short sentences.
9. Write in a formal and academic register with a succinct and regal flair.
10. Mimic the writing style of the user's prompts.

### Literature and citation

11. Any document that draws on a literature survey must carry its references explicitly within the prose. Every statement of non-novel information taken from a published source must be followed at its point of use by a numbered marker in square brackets, in the manner of an academic paper, so that a reader establishes the origin of a claim where it is made rather than inferring it from a bibliography that lists sources without attaching them to anything.
12. Every document carrying such markers must close with a numbered bibliography section, its entries ordered by first appearance in the text, each giving the authors, the title, the venue, the year, and an identifier such as an arXiv number or a DOI. A marker without a matching entry is a broken pointer, and an entry that no marker cites is evidence of nothing, so treat both as defects to be repaired before the document is considered finished.
13. Where a document is asked for a literature survey summary or a related works section, the academic register governs it entirely. Each idea, concept, or reported result attributed to a source carries its marker, related sources are grouped so that the section builds toward the position the document takes, and the survey is never allowed to become a list of summaries that the surrounding argument does not use.
14. The requirement to attribute holds for the codebase exactly as it holds for papers, a claim about the behaviour of this workspace carrying its `file:line` citation and a claim drawn from the literature carrying its bibliography number, so that every substantive sentence traces either to the source tree or to a published work. This is the operative form of the grounding rule above, and it applies to every document written here, not only to those that announce a survey.
15. Before citing a work, consult `context/literature.md`, which registers the surveyed literature in thematic clusters with the relevance of each entry already established. Where a document cites a work that the survey does not yet record, add it to the appropriate cluster in the same pass, and never expand an author list, a venue, or a year from recall, giving the short form and noting the omission where the retrieved source did not state them.

### Context discipline

16. Always refer to the dumped context files in `context/` when writing, to avoid doubts and hallucinations and to minimise redundant code investigation. Begin with `context/README.md`, which registers every context document with a summary and a currency note, and read only the documents that index names as bearing on the matter at hand.
17. Any new information discovered about a topic must be added to its relevant context file so that future sessions can rely on it without re-spending tokens. Where the discovery contradicts an existing entry, append a dated correction rather than editing the original in place, so that the reasoning which produced the earlier claim survives.
18. Implementation plans live in `plans/`, indexed by `plans/README.md`, which records the implementation status of each. Read that status before acting on a plan, since a plan whose proposals have already landed must not be applied a second time. On completing a plan, update its status banner and record the outcome, together with any divergence from the plan, in the corresponding context file.
19. The two documents at the workspace root, `ARCHITECTURE.md` and `CO_OPTIMISATION.md`, describe the project as a whole rather than a single investigation. Keep them at the root, and when a change alters the task registry, the training modes, or the co-optimisation configuration, update them in the same pass.

### Debugging artefacts

20. Every debugging artefact supplied or produced during an investigation, meaning any exported plot, TensorBoard panel, evaluation dashboard, contact sheet, video, or numpy dump that a document reasons about, must be moved into `context/artefacts/` for safekeeping. Never leave such a file loose at the workspace root, and never cite one from a location outside that directory, since the training log tree under `IsaacLab/logs/` is transient and its runs are deleted as disk fills, whereas this directory is permanent.
21. The directory admits only artefacts that a context or a plan document cites, whether individually or as a set the document reasons about. An artefact that no document references is evidence of nothing and must be deleted rather than kept against a future use that may never come, and the same applies to an artefact whose provenance cannot be established, since it could not be quoted against a run even if it were cited. Enforce both conditions by deletion rather than by annotation, so that the presence of a file is itself the assurance that some document depends upon it.
22. On placing an artefact there, register it in `context/artefacts/README.md` against the run that produced it, stating how the attribution was established. Record every deletion in the same file with the reason, so that a reader who recalls a figure and cannot find it learns why rather than assuming an oversight.
23. Address an artefact as `artefacts/<directory>/<file>` from a context document and as `../context/artefacts/<directory>/<file>` from a plan document. Name any new artefact directory for the run that produced it rather than for its recency, the existing directories carrying names such as `plots-latest` being a historical accident that the register exists to compensate for.
24. Before citing an artefact, confirm the file is present. Where a document cites an artefact that has since been deleted, remove the dead pointer and retain the observation it supported, marking the artefact as no longer retained, so that the finding survives without inviting a reader to chase a file that is gone.

### Repository specific context

25. The simulation repository carries its own `tron1-rl-isaaclab-cozum/context/` and `tron1-rl-isaaclab-cozum/plans/`, governed by these same rules and indexed the same way. A document belongs there when its subject is confined to that repository, the physical parameterisation of a robot or the mathematics of an algorithm it implements, and at the workspace level when it spans the repository, the vendored libraries, and the container tooling together. Where the distinction is unclear, prefer the workspace level, since a document is more useful over-scoped and findable than correctly scoped and overlooked.
26. Never keep the same document in both places. Where a subject appears to belong to both, choose the narrower home and delete the other copy, recording the move in both indices, since two copies of one document diverge silently and a reader cannot then tell which governs.
