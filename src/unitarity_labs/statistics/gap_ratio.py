"""
Source repository: holeyfield33-art/unitarity-lab
Source commit: 66c004b5f37a5f4a129761ef1d7f87f306b100ac
Source file: unitarity_labs/core/spectral_monitor.py (get_r_ratio)
Original implementation: mean consecutive-eigenvalue-gap ratio, used as
    one input to a composite "stability" score framed around GOE/GUE
    "manifold health" vs. Poisson "collapse".
Retained behavior: the gap-ratio formula itself -- for sorted eigenvalues
    lambda_1 <= ... <= lambda_n with gaps d_i = lambda_{i+1} - lambda_i,
    r_i = min(d_i, d_{i+1}) / max(d_i, d_{i+1}), mean_gap_ratio = mean(r_i),
    with r_i defined as 0 where both gaps are 0. Explicit input-shape
    validation and NaN/Inf rejection were added; the source did not
    validate for non-finite input.
Removed interpretation: GOE_R_MEAN / GUE_R_MEAN / POISSON_R_MEAN reference
    constants, the R_RATIO_FLOOR "collapse" threshold, the StabilityBreak
    exception, and TransportEvaluator.stability()'s "healthy/unhealthy"
    framing. This function reports a number only -- see
    docs/EVIDENCE_LEVELS.md before assigning it any interpretation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_gap_ratio(eigenvalues: NDArray[np.floating]) -> float:
    """Mean ratio of consecutive eigenvalue gaps (raw statistic).

    For an ordered sequence of eigenvalues, gaps ``d_i = lambda_{i+1} -
    lambda_i`` and ratios ``r_i = min(d_i, d_{i+1}) / max(d_i, d_{i+1})``;
    this returns ``mean(r_i)``. This is a level-spacing statistic. It
    carries no semantic interpretation on its own -- see
    docs/EVIDENCE_LEVELS.md.

    Parameters
    ----------
    eigenvalues : 1-D array of real numbers, at least 3 elements.
        Need not be pre-sorted.

    Returns
    -------
    float in [0, 1].

    Raises
    ------
    ValueError
        If fewer than 3 eigenvalues are given, if the input is not 1-D,
        or if it contains NaN/Inf.
    """
    arr = np.asarray(eigenvalues, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"eigenvalues must be 1-D, got shape {arr.shape}")
    if arr.size < 3:
        raise ValueError("Need at least 3 eigenvalues to compute gap ratios.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("eigenvalues must not contain NaN or Inf.")

    arr = np.sort(arr)
    gaps = np.diff(arr)
    numerator = np.minimum(gaps[:-1], gaps[1:])
    denominator = np.maximum(gaps[:-1], gaps[1:])
    # Both branches of np.where are evaluated eagerly, so a 0/0 gap pair
    # raises a spurious "invalid value" warning even though the result is
    # discarded in favor of the 0.0 fallback; suppress it (matches source).
    with np.errstate(invalid="ignore"):
        ratios = np.where(denominator > 0, numerator / denominator, 0.0)
    return float(np.mean(ratios))
