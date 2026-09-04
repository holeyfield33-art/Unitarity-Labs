"""
Source repository: holeyfield33-art/unitarity-lab
Source commit: 66c004b5f37a5f4a129761ef1d7f87f306b100ac
Source file: unitarity_labs/core/metrics.py (length_matched_null_zeta,
    manifold_coherence_zeta)
Original implementation: "Manifold Coherence zeta" -- cosine similarity
    between a matched pair of activation tensors -- tested for significance
    against a null distribution of cosine similarities to unrelated
    control tensors. The source's plain cross_sample_null_zeta zero-pads
    length-mismatched tensors, which the source's own "HONESTY-1" comment
    documents as deflating the null and inflating the z-score;
    length_matched_null_zeta is the source's own corrected variant, which
    truncates instead of padding.
Retained behavior: the statistic itself -- cosine similarity between a
    matched pair vs. the mean/std of cosine similarities to N >= 1
    controls, gap = matched - null_mean, z = gap / null_std when
    null_std > 0. Length-matching by truncation along a chosen axis (no
    zero-padding) is retained as the only variant ported; the buggy
    zero-padding cross_sample_null_zeta was not ported. Reimplemented in
    NumPy (source used PyTorch) since this phase's dependencies are
    numpy-only; the formula and length-matching logic are unchanged. The
    z-score's zero-variance edge case was corrected: the source returned
    +inf whenever null_std == 0, even when gap == 0 (matched score exactly
    equal to a degenerate, zero-variance null); this returns 0.0 in that
    case and a signed infinity only when gap != 0.
Removed interpretation: "Manifold Coherence", the "zeta" name, and any
    framing of this statistic as measuring cross-layer semantic coherence.
    This function reports a z-score against a matched null only -- see
    docs/EVIDENCE_LEVELS.md before assigning it any interpretation.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray


def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if a_flat.shape != b_flat.shape:
        raise ValueError(
            "vectors must have equal length after flattening, got "
            f"{a_flat.shape} vs {b_flat.shape}"
        )
    if not (np.all(np.isfinite(a_flat)) and np.all(np.isfinite(b_flat))):
        raise ValueError("inputs must not contain NaN or Inf.")
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compute cosine similarity of a zero vector.")
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def matched_null_zscore(
    matched_a: NDArray[np.floating],
    matched_b: NDArray[np.floating],
    control_bs: Sequence[NDArray[np.floating]],
    seq_axis: int = 0,
) -> Dict[str, float]:
    """Matched-pair cosine similarity vs. a cross-sample control null.

    Compares cosine_similarity(matched_a, matched_b) against the
    distribution of cosine_similarity(matched_a, control) for each control
    in ``control_bs``. All inputs are truncated to the shortest length
    along ``seq_axis`` before comparison, so every cosine is computed
    between equal-length vectors (no zero-padding).

    Parameters
    ----------
    matched_a, matched_b : array_like
        The matched pair.
    control_bs : sequence of array_like, non-empty
        Controls compared against ``matched_a``.
    seq_axis : int
        Axis to truncate to a common length across all inputs.

    Returns
    -------
    dict with keys:
        matched, null_mean, null_std, gap, z_score, n_controls, matched_len

    Raises
    ------
    ValueError
        If ``control_bs`` is empty, if any input is not finite, or if
        shapes remain mismatched after truncation.
    """
    if len(control_bs) == 0:
        raise ValueError("control_bs must be non-empty")

    a = np.asarray(matched_a, dtype=np.float64)
    b = np.asarray(matched_b, dtype=np.float64)
    controls = [np.asarray(c, dtype=np.float64) for c in control_bs]

    common_len = min(t.shape[seq_axis] for t in [a, b] + controls)
    if common_len < 1:
        raise ValueError("all inputs must have at least one element along seq_axis")

    def _truncate(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.take(t, range(common_len), axis=seq_axis)

    a_t = _truncate(a)
    b_t = _truncate(b)
    controls_t = [_truncate(c) for c in controls]

    matched = _cosine_similarity(a_t, b_t)
    null_scores = np.array(
        [_cosine_similarity(a_t, c) for c in controls_t], dtype=np.float64
    )

    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores)) if null_scores.size > 1 else 0.0
    gap = matched - null_mean

    if null_std > 0.0:
        z_score = gap / null_std
    elif gap == 0.0:
        z_score = 0.0
    else:
        z_score = float(np.inf) if gap > 0.0 else float(-np.inf)

    return {
        "matched": matched,
        "null_mean": null_mean,
        "null_std": null_std,
        "gap": gap,
        "z_score": float(z_score),
        "n_controls": len(control_bs),
        "matched_len": common_len,
    }
