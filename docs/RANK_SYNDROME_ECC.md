# Rank-Syndrome ECC Synthetic Prototype

## Scope

This document specifies a classical, controlled low-rank matrix corruption
experiment. It answers only whether redundant linear measurements can correct
controlled low-rank matrix corruption. It does not test model states,
transformers, learned decoders, spectral quantities, or quantum systems.

## Mathematical Contract

### State

**DERIVED.** The clean state is arbitrary:

$$X \in \mathbb{R}^{d \times T}.$$

No low-rank restriction is imposed on $X$. The experiment uses iid Gaussian,
full-rank structured, and deliberately low-rank clean-state families.

### Parity map

**DERIVED.** Define $P: \mathbb{R}^{d \times T} \rightarrow \mathbb{R}^m$ by

$$[P(Z)]_i = \langle A_i, Z \rangle_F = \operatorname{tr}(A_i^T Z).$$

The matrices $A_1,\ldots,A_m$ are generated once from a public deterministic
NumPy `default_rng` seed. The experiment uses iid Gaussian entries with
distribution $\mathcal{N}(0, 1/m)$; this normalization keeps expected
measurement energy comparable while $m$ varies. The seed and normalization are
recorded in each result file.

The implemented adjoint is

$$P^*(z) = \sum_i z_i A_i,$$

and its Frobenius inner-product identity is numerically tested to tolerance
$10^{-12}$.

### Encoded state, error, and syndrome

**DERIVED.** The encoded state and controlled corruption are

$$\operatorname{Enc}(X) = (X,p), \qquad p=P(X),$$
$$Y=X+E, \qquad \operatorname{rank}(E)\le t.$$

The syndrome is

$$s=P(Y)-p=P(Y)-P(X)=P(E).$$

Errors are independently generated as $E=UV^T$, where seeded Gaussian
$U\in\mathbb{R}^{d\times t}$ and $V\in\mathbb{R}^{T\times t}$ are checked to
produce numerical rank exactly $t$.

### Exact uniqueness condition

**PROVEN.** Exact identification of rank-$\le t$ errors requires

$$\ker P \cap \{Z: \operatorname{rank}(Z)\le 2t\}=\{0\}.$$

Proof: suppose distinct $E_1,E_2$ each have rank at most $t$ and the same
syndrome. Then $P(E_1-E_2)=0$. Rank subadditivity gives
$\operatorname{rank}(E_1-E_2)\le2t$. The stated condition therefore implies
$E_1-E_2=0$, contradicting distinctness. Hence no two distinct rank-$\le t$
errors have the same syndrome. $\square$

### Decoder

**PROVEN UNDER APPROPRIATE MEASUREMENT CONDITIONS; EMPIRICALLY TESTED HERE.**
The primary reference decoder solves

$$\widehat E = \arg\min_Z \|Z\|_* \quad \text{subject to}\quad P(Z)=s,$$

then reconstructs $\widehat X=Y-\widehat E$. It uses CVXPY as the optional
`ecc-research` dependency and raises an actionable install error when absent;
no unrelated heuristic is substituted. This is not a claim of universal
nuclear-norm recovery.

## Pre-Registered Measurements

**MEASURED** values are recorded for every trial:

$$R_E=\frac{\|\widehat E-E\|_F}{\max(\|E\|_F,\epsilon)}, \qquad
R_X=\frac{\|\widehat X-X\|_F}{\max(\|X\|_F,\epsilon)},$$
$$R_s=\|P(\widehat E)-s\|_2,$$

along with numerical rank of $\widehat E$. The fixed success predicate is

```text
relative_error_recovery < 1e-5
AND relative_state_recovery < 1e-5
AND syndrome_residual < 1e-7
```

These thresholds are not tuned per result. The zero-error control is evaluated
as an exact no-op, avoiding a $0/0$ relative-error convention.

## Theory Reference

**DERIVED.** The algebraic variety of $d\times T$ matrices with rank at most
$r$ has dimension $r(d+T-r)$. To generically avoid nonzero kernel matrices of
rank at most $2t$, the dimension-count reference is approximately

$$m \gtrsim 2t(d+T-2t).$$

This is a generic algebraic dimension criterion, not a guarantee that the
nuclear-norm decoder succeeds at the threshold. It is reported without fitting
it to the data.

## Controls and Interpretation

**MEASURED.** The runner includes zero-error, rank-1, rank-2, and full
observation positive controls, plus deliberately low measurement, rank-beyond-
design, and wrong-parity negative controls. Conventional Reed-Solomon remains
the external positive-control standard for the logical pipeline `encode ->
corrupt -> syndrome -> decode -> exact reconstruction`; this independent
matrix construction must satisfy that same pipeline.

Evidence classifications:

- $P(E)$ syndrome identity: **DERIVED**.
- Rank-error uniqueness condition: **PROVEN**.
- Gaussian-map recovery rate at a particular $m$: **MEASURED**.
- Universal nuclear-norm success: **REJECTED**.
- Model hidden-state applicability: **CONJECTURAL / NOT TESTED**.

## Kill Criteria and Promotion Gate

The prototype is rejected if recovery does not materially improve with more
measurements, rank-1 fails in highly overdetermined regimes, success depends on
clean-state rank, small syndrome residual consistently accompanies wrong
reconstruction, or identical seeds are not reproducible. Do not tune around a
failure.

Synthetic success permits only: `RANK-SYNDROME CONSTRUCTION VALIDATED ON
CONTROLLED MATRICES`. It does not establish model-state ECC; a later directive
must establish a low-rank hidden-state corruption model.