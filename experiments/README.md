# Experiments

Each experiment lives in its own subdirectory and must state, before any
results are recorded:

- **`C`** — the matched experimental conditions (what is held fixed, what
  varies between control and target).
- **`M`** — the measured feature vector, and the exact procedure used to
  compute it.
- **Test procedure** — the statistical test used to compare `M` under
  control vs. target (per `docs/RESEARCH_THESIS.md`), specified in advance.

Results and conclusions from an experiment must be labeled per
`docs/EVIDENCE_LEVELS.md`. No experiment is present yet.
