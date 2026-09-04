# Error-Correcting-Code Research

## Question

Whether model-state or distributed-model integrity can support a legitimate
error-correcting-code (ECC) formulation is an open research question. This
document defines the terms that a proposed formulation must specify before
it can be evaluated, and what "correction" would mean in that context. It
does not propose or implement an answer.

## Required tuple

Any candidate ECC formulation over model state must define all five of the
following objects:

`(C, E, s, d, D)`

- **`C` — code / state space.** The set of valid ("codeword") states the
  mechanism is defined over, and how a raw model state maps into it.
- **`E` — error model.** A specified distribution or adversary model over
  the perturbations the code is meant to detect or correct (e.g. bit flips,
  numerical noise, dropped messages, adversarial edits). "Error" must mean a
  concrete, named perturbation — not an unspecified notion of "unhealthy"
  state.
- **`s` — syndrome.** A function of a (possibly corrupted) state that reveals
  information about whether, and how, it deviates from `C`, without
  requiring the correct codeword to already be known.
- **`d` — distance.** A metric on `C` (or on the ambient space `C` sits in)
  that determines how many/which errors under `E` are distinguishable or
  correctable, in the standard coding-theory sense (e.g. minimum distance
  bounds on detectable/correctable error weight).
- **`D` — decoder.** A concrete procedure that maps a (possibly corrupted)
  state and its syndrome back to a state in `C`, together with a defined
  notion of decoding success or failure.

## Hard rule

**No model-state mechanism may be called "error correction" unless all five
objects — `C`, `E`, `s`, `d`, `D` — are explicitly defined, and correction is
independently tested** (i.e. validated against a ground truth of what the
"correct" state was, under a stated error model, with measured
success/failure rates — not merely observed to produce stable or
low-variance output; see `docs/EVIDENCE_LEVELS.md`, rule 1).

Until that bar is met, any related work is CONJECTURAL or HEURISTIC at best,
and must be labeled as such.

## Conceptual positive control

The old Unitarity Lab codebase contains a working Reed-Solomon transport
code. Classical Reed-Solomon coding is a legitimate, well-understood
instance of the `(C, E, s, d, D)` tuple over symbol sequences:

- `C`: Reed-Solomon codewords over a finite field
- `E`: symbol erasures/errors up to a bounded weight
- `s`: syndrome computed from the received word via the generator
  polynomial
- `d`: the code's minimum (Singleton) distance
- `D`: syndrome decoding (e.g. Berlekamp-Massey / Euclidean decoding)

It is referenced here **only as a conceptual positive control** — an
existence proof that the five-object tuple is satisfiable and testable in a
domain where "correction" is unambiguous — so that any future model-state
proposal can be compared against a known-good instance of the same
structure. Per the scope of this task, that implementation is **not**
being copied into this repository yet. Any future import must follow the
provenance requirements in `docs/CODEBOOK_PROVENANCE.md`, including the
exact source commit SHA.

## Status

CONJECTURAL. No `(C, E, s, d, D)` tuple over model state has been defined or
tested in this repository. No ECC implementation exists here yet.
