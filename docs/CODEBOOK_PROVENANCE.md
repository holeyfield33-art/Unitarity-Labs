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

### 1. `mean_gap_ratio`

```text
Source repo:        holeyfield33-art/unitarity-lab
Source path:         unitarity_labs/core/spectral_monitor.py (get_r_ratio)
Source commit SHA:   66c004b5f37a5f4a129761ef1d7f87f306b100ac
Original purpose:    mean consecutive-eigenvalue-gap ratio, one input to a
                      composite "stability" score framed around GOE/GUE
                      "manifold health" vs. Poisson "collapse"
What was retained:   the gap-ratio formula only; explicit shape/finite
                      validation added (source had none)
What was removed:    GOE_R_MEAN/GUE_R_MEAN/POISSON_R_MEAN constants,
                      R_RATIO_FLOOR "collapse" threshold, StabilityBreak
                      exception, all "healthy/unhealthy" framing
Why retained:        mathematically well-defined level-spacing statistic,
                      independent of any health interpretation
Validation status:   MEASURED (implementation verified against GOE/GUE/
                      Poisson reference-ensemble asymptotics in
                      tests/test_gap_ratio.py); no claim about model
                      behavior is made or tested
```

Ported to `src/unitarity_labs/statistics/gap_ratio.py`.

### 2. `singular_value_spectrum`

```text
Source repo:        holeyfield33-art/unitarity-lab
Source path:         unitarity_labs/core/spectral_monitor.py
                      (TransportEvaluator.svd)
Source commit SHA:   66c004b5f37a5f4a129761ef1d7f87f306b100ac
Original purpose:    full SVD of a square "transport" matrix, one input to
                      the same composite "stability" score as above
What was retained:   the SVD computation; generalized from square-only to
                      any real 2-D matrix; explicit shape/finite
                      validation added
What was removed:    TransportEvaluator.stability(),
                      frobenius_distance_from_identity(), all
                      "stability"/"collapse" framing
Why retained:        SVD is a standard, well-defined decomposition
Validation status:   MEASURED (implementation verified against hand-
                      computed examples in tests/test_svd_spectrum.py)
```

Ported to `src/unitarity_labs/statistics/svd_spectrum.py`.

### 3. `matched_null_zscore`

```text
Source repo:        holeyfield33-art/unitarity-lab
Source path:         unitarity_labs/core/metrics.py
                      (length_matched_null_zeta, manifold_coherence_zeta)
Source commit SHA:   66c004b5f37a5f4a129761ef1d7f87f306b100ac
Original purpose:    "Manifold Coherence zeta" -- cosine similarity between
                      a matched activation pair, tested against a
                      cross-sample control null
What was retained:   the statistic (matched cosine similarity vs. null
                      mean/std, gap, z-score) and the length-matched
                      (truncation, not zero-padding) variant only;
                      reimplemented in NumPy (source used PyTorch); the
                      null_std == 0 & gap == 0 edge case was corrected
                      from the source's unconditional +inf to 0.0
What was removed:    "Manifold Coherence" / "zeta" naming and framing as
                      cross-layer semantic coherence; the buggy
                      zero-padding cross_sample_null_zeta variant
                      (source's own "HONESTY-1" bug) was not ported
Why retained:        cross-sample matched-null significance testing is a
                      generically useful, mathematically defensible
                      pattern independent of what "zeta" was claimed to
                      measure
Validation status:   MEASURED (implementation verified against hand-
                      computed examples in tests/test_matched_null.py)
```

Ported to `src/unitarity_labs/statistics/matched_null.py`.

### 4. `RollingCovariance` / `TopKEigenTracker` / `leading_eigengap`

```text
Source repo:        holeyfield33-art/var
Source path:         var/spectral.py (IncrementalTopKEigen)
Source commit SHA:   31234551e524249a5e81453ec851c98ec8836fb7
Original purpose:    incrementally-updated top-k eigen decomposition of a
                      sliding-window sample covariance, one stage of a
                      streaming spectral-drift pipeline
What was retained:   the online covariance accumulator (split out as
                      RollingCovariance) and the subspace-iteration top-k
                      eigenvalue tracker (TopKEigenTracker), composed
                      rather than duplicated; explicit shape/finite
                      validation added. leading_eigengap is new code (not
                      ported), directly derived from the tracker's output.
What was removed:    Merkle-linked snapshot hashing, sliding-window buffer
                      management, and the RollingW2SpectralMonitor
                      pipeline that coupled this class to var's detection
                      stack
Why retained:        standard online covariance and subspace-iteration
                      eigenvalue tracking; no health framing was present
                      in this class to begin with
Validation status:   MEASURED (RollingCovariance verified against a
                      hand-computed example; TopKEigenTracker's full-decomp
                      path cross-checked against numpy.linalg.eigh; see
                      tests/test_covariance.py)
```

Ported to `src/unitarity_labs/statistics/covariance.py`.

### 5. `wasserstein_spectrum_distance`

```text
Source repo:        holeyfield33-art/var
Source path:         var/spectral.py (wasserstein2_spectrum)
Source commit SHA:   31234551e524249a5e81453ec851c98ec8836fb7
Original purpose:    W2 distance between two eigenvalue spectra, the
                      primary drift signal in var's spectral-anomaly
                      pipeline
What was retained:   the W2 formula only; explicit shape/finite validation
                      added
What was removed:    Merkle-linked snapshot hashing and the streaming
                      RollingW2SpectralMonitor pipeline it fed;
                      wasserstein1_spectrum (SciPy-based) was not ported
                      -- this phase is numpy-only
Why retained:        standard, well-defined distance between sorted
                      discrete distributions
Validation status:   MEASURED (implementation verified against hand-
                      computed examples in tests/test_wasserstein_distance.py)
```

Ported to `src/unitarity_labs/statistics/wasserstein_distance.py`.

### 6. `MedianMADChangeDetector`

```text
Source repo:        holeyfield33-art/var
Source path:         var/detector.py (SpectralRuptureDetector, RuptureState)
Source commit SHA:   31234551e524249a5e81453ec851c98ec8836fb7
Original purpose:    calibrated median/MAD threshold detector with
                      hysteresis on a streaming scalar drift signal, framed
                      as detecting "ruptures" against a "healthy baseline"
What was retained:   the full calibration and detection method (warmup,
                      robust median/MAD baseline with std fallback,
                      threshold, hysteresis-gated state commit); explicit
                      scalar-shape/finite validation added to `update`
What was removed:    "rupture" / "healthy-baseline" naming, renamed to
                      neutral CALIBRATING/QUIET/CHANGED states and
                      "change_detected"/"change_cleared" events
Why retained:        a detector reports a statistically-defined change,
                      not a health judgment -- the calibration and
                      hysteresis method is retained exactly, only the
                      naming changed
Validation status:   MEASURED (stationary-control-quiet, injected-shift-
                      detected, pre-calibration-silent, and calibration-
                      state-exposure behaviors verified in
                      tests/test_change_detection.py)
```

Ported to `src/unitarity_labs/statistics/change_detection.py`.

`MeasurementVector` (`src/unitarity_labs/statistics/measurement_vector.py`) is
new code, not ported from any prior codebook -- it is the result type defined
by `docs/RESEARCH_THESIS.md`'s `M` vector, with only the components backed by
a ported primitive above populated; all other components are `None`.

No implementations have been imported from `holeyfield33-art/geometric-brain-mcp`
or `holeyfield33-art/insideai` yet.
