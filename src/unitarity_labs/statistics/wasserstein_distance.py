"""
Source repository: holeyfield33-art/var
Source commit: 31234551e524249a5e81453ec851c98ec8836fb7
Source file: var/spectral.py (wasserstein2_spectrum)
Original implementation: W2 distance between two sorted eigenvalue arrays,
    used as the primary drift signal in a rolling spectral-anomaly
    detection pipeline (var/spectral.py::RollingW2SpectralMonitor).
Retained behavior: the W2 formula for two sorted, equal-mass 1-D discrete
    distributions -- sqrt(mean((a_i - b_i)^2)) over the top
    min(len(a), len(b)) values of each. Explicit input-shape and
    finite-value validation was added; the source did not validate for
    empty or non-finite input.
Removed interpretation: the Merkle-linked snapshot hashing, sliding-window
    buffer management, and streaming pipeline it fed
    (RollingW2SpectralMonitor) are not ported here -- this is the distance
    formula only. wasserstein1_spectrum (the source's SciPy-based W1
    cross-check) was not ported: this phase's dependencies are numpy-only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wasserstein_spectrum_distance(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> float:
    """W2 distance between two sorted, non-negative eigenvalue spectra.

    For 1-D discrete distributions supported on sorted values with equal
    mass at each point, ``W2^2 = (1/k) * sum((a_i - b_i)^2)`` over the top
    ``k = min(len(a), len(b))`` values of each input.

    Parameters
    ----------
    a, b : 1-D arrays of real numbers, non-empty.

    Returns
    -------
    float, >= 0.

    Raises
    ------
    ValueError
        If either input is not 1-D, is empty, or contains NaN/Inf.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    for name, arr in (("a", aa), ("b", bb)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
        if arr.size < 1:
            raise ValueError(f"{name} must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must not contain NaN or Inf.")

    k = min(aa.size, bb.size)
    top_a = np.sort(aa)[-k:]
    top_b = np.sort(bb)[-k:]
    return float(np.sqrt(np.mean((top_a - top_b) ** 2)))
