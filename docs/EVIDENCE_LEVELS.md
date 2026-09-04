# Evidence Levels

Every claim made in this repository — in code comments, docstrings,
experiment write-ups, or results — must be labeled with exactly one of the
following levels. No other evidence labels are permitted.

## Labels

- **PROVEN** — Established by a mathematical proof, or a direct, necessary
  consequence of definitions already in this repository. Not an empirical
  claim.
- **DERIVED** — Follows deductively from PROVEN or DERIVED statements plus
  explicitly stated assumptions. The assumptions must be listed alongside
  the claim.
- **MEASURED** — Obtained from a specific, reproducible experiment: the
  data source, measurement procedure, sample size, and statistical test are
  recorded, and the experiment can be re-run. A MEASURED claim describes
  what was observed under the stated conditions, not what it means.
- **HEURISTIC** — A rule of thumb or design choice adopted for practical
  reasons (e.g. a default threshold, a convenient parameterization) that is
  not derived or validated, but is useful and explicitly flagged as such.
- **CONJECTURAL** — A hypothesis under active investigation, not yet
  supported by MEASURED evidence. Stated so it is falsifiable.
- **REJECTED** — A claim that was considered (in this project or in prior
  work this project descends from) and has since been shown false,
  unsupported, or is being explicitly excluded as an assumption. See
  `docs/RESEARCH_THESIS.md` for the current REJECTED list.

## Rules

1. **Stable output is not evidence of meaningful output.** A metric that
   produces consistent, low-variance numbers across runs has only
   demonstrated stability. Stability says nothing about whether the metric
   tracks anything semantically real. Consistency is not validation.

2. **A metric earns semantic interpretation only after validation against
   independently defined ground truth.** A measurement (`M`) may be
   labeled MEASURED once it is reproducibly computed. It may not be
   described as meaning anything about model behavior, quality, or health
   until that interpretation has been tested against ground truth defined
   independently of the metric itself (e.g. human judgment, task accuracy,
   a held-out behavioral label) — and that validation result is itself
   labeled MEASURED, DERIVED, or REJECTED based on its own outcome.

3. Every MEASURED claim must be traceable to code and data that can
   reproduce it. A MEASURED claim without a reproducible procedure should be
   downgraded to CONJECTURAL until reproducibility is restored.

4. Claims should be downgraded, not deleted, when evidence weakens. A claim
   that fails replication moves from MEASURED to REJECTED (or CONJECTURAL,
   if the failure is inconclusive rather than a clear disproof) with a note
   explaining why, rather than being silently removed.
