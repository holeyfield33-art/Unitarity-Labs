# Codebook Provenance

This repository is a clean research reset. It is **not** a fork of, and does
not bulk-copy from, any of the following prior codebooks:

- `holeyfield33-art/unitarity-lab`
- `holeyfield33-art/geometric-brain-mcp`
- `holeyfield33-art/VAR`
- `holeyfield33-art/insideai`

Those repositories remain active codebooks and may contain useful, working
implementations. They are not treated as sources of established theory —
any theoretical claims, terminology, or framing from them must be
re-evaluated under `docs/EVIDENCE_LEVELS.md` before being reused here, and
several specific claims and code paths are explicitly excluded (see
"Prohibited carryover" below, and `docs/RESEARCH_THESIS.md`).

## Provenance record requirement

Nothing may be imported from a prior codebook into this repository without a
provenance record. **Do not claim provenance unless the exact source commit
SHA is known.** If the SHA is not known, the import does not happen until it
is.

Every future imported implementation must have a record of this form (add
one entry per import below, or alongside the imported code):

```text
Source repo:
Source path:
Source commit SHA:
Original purpose:
What was retained:
What was removed:
Why retained:
Validation status:
```

A machine-checkable version of this schema is implemented in
`src/unitarity_labs/provenance/` (see `ProvenanceRecord`).

## Prohibited carryover

The following are not to be ported into this repository, in any form,
without a full re-derivation and independent validation from first
principles (not merely copied and relabeled):

- `GUELoss`
- `compute_correction` GUE controller
- `CasimirOptimizer`
- Hawking flux code
- k=1 sigma bridge
- raw-text spectral health scoring
- "entanglement" terminology applied to model internals
- "Page-time" terminology
- "hallucination = Poisson" claims

See `docs/RESEARCH_THESIS.md` for the full list of rejected assumptions this
project does not carry forward.

## Import log

No implementations have been imported from prior codebooks yet. Entries will
be appended here, one per import, using the schema above, as imports occur.
