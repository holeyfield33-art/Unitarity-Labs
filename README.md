# Unitarity Labs

An experimental research framework for model-state integrity, representation
telemetry, change detection, and error-correction research.

This project does not assume that GUE statistics, spectral rigidity, or any
single geometric quantity constitutes model health. Every claim made in this
repository is labeled per `docs/EVIDENCE_LEVELS.md`, and the hypothesis this
project actually tests is stated precisely in `docs/RESEARCH_THESIS.md`.

## What this is

The working hypothesis: internal model-state geometry may contain measurable
changes associated with externally defined model behavior. That's a
hypothesis to test (`H0` vs `H1` over matched conditions `C` and a measured
feature vector `M` — see `docs/RESEARCH_THESIS.md`), not a conclusion this
codebase starts from.

This repository is a clean research reset, not a fork of the prior
`unitarity-lab`, `geometric-brain-mcp`, `VAR`, or `insideai` codebooks. Those
remain active codebooks; anything reused from them requires a provenance
record with an exact source commit SHA (`docs/CODEBOOK_PROVENANCE.md`) and
is not assumed to carry forward any of its prior theoretical claims.

## Research before intervention

This project studies model-state integrity before it attempts to act on it.
Active model correction — anything that modifies model state based on a
measured signal — will not be implemented until:

1. an error model, syndrome, and decoder are defined per the
   `(C, E, s, d, D)` tuple in `docs/ECC_RESEARCH.md`, and
2. correction under that decoder is independently tested against ground
   truth, not just observed to be stable.

Until then, this repository only measures, compares, and reports — it does
not intervene. See `docs/ECC_RESEARCH.md` for the current status
(CONJECTURAL: no tuple has been defined or tested yet).

## Structure

```text
Unitarity-Labs/
├── docs/                        Research contract: thesis, evidence levels,
│                                 provenance rules, ECC research plan
├── src/unitarity_labs/
│   ├── evidence.py               EvidenceLevel enum
│   ├── instrumentation/          model-state instrumentation (placeholder)
│   ├── statistics/                change-detection statistics (placeholder)
│   ├── ecc/                       error-correcting-code research (placeholder)
│   └── provenance/                ProvenanceRecord schema
├── experiments/                  one subdirectory per experiment
├── tests/
└── results/                      experiment outputs
```

## Setup

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Reading order

1. `docs/RESEARCH_THESIS.md` — the hypothesis, the statistical frame, and
   what is explicitly *not* assumed.
2. `docs/EVIDENCE_LEVELS.md` — how every claim in this repo must be labeled.
3. `docs/CODEBOOK_PROVENANCE.md` — the provenance bar for reusing any code
   from a prior codebook.
4. `docs/ECC_RESEARCH.md` — what "error correction" would have to mean here
   before the term is used.
