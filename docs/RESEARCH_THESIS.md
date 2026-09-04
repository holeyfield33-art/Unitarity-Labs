# Research Thesis

## Hypothesis

Internal model-state geometry may contain measurable changes associated with
externally defined model behavior.

This is a hypothesis, not a finding. It is stated in the weakest form that is
still testable: *may contain measurable changes associated with*, not
*determines*, *causes*, or *explains*. No stronger claim is assumed anywhere
in this repository.

## Formal research model

Let `M` be a measured feature vector extracted from model internal state
(e.g. activation statistics, representation geometry, spectral features of a
weight or activation matrix). Let `C` be a set of matched experimental
conditions (prompt distribution, model checkpoint, decoding parameters,
sampling seed, hardware/precision, and any other variable that must be held
fixed or explicitly controlled for the comparison to be valid).

We test:

- **H0**: `P(M | C, control) = P(M | C, target)`
- **H1**: `P(M | C, control) != P(M | C, target)`

where `control` and `target` are two externally defined conditions of
interest (for example: a baseline prompt set vs. a prompt set associated with
a specific behavior, or two checkpoints being compared under otherwise
identical `C`).

A result is only meaningful when:

1. `C` is actually matched between control and target (confounds are named
   and controlled, not assumed away);
2. `M` is defined by a specific, reproducible measurement procedure before
   the comparison is run;
3. the statistical test used to compare the distributions of `M` under
   control and target is specified in advance, along with its assumptions.

Rejecting H0 shows an association between `C`'s target/control distinction
and the distribution of `M`. It does not, by itself, establish what `M`
means, whether the association is causal, or whether it generalizes beyond
the tested conditions.

## Explicitly rejected assumptions

The following are **not** assumed to be true anywhere in this codebase.
Where prior work (inside or outside this project) treated them as given,
this project treats them as unproven claims requiring independent
validation, if they are used at all:

- GUE statistics == healthy reasoning
- Poisson statistics == hallucination
- a "k=1" invariant bridge between any two measured quantities
- a universal spectral-health number that applies across models, layers, or
  tasks
- Hawking / Page / Bell / ER=EPR interpretations of model internals
- Casimir or Kolmogorov-complexity based "optimization" claims about model
  behavior
- raw text spectral score as a measure of semantic quality

Any of these terms reappearing in future work must be re-derived and
independently validated under the evidence framework in
`docs/EVIDENCE_LEVELS.md`, not imported as established fact.

## Scope

This document defines the hypothesis and the statistical frame this project
operates under. It does not itself claim any result. Individual experiments
in `experiments/` are responsible for stating their own `C`, `M`, and test
procedure, and for labeling their conclusions per `docs/EVIDENCE_LEVELS.md`.
