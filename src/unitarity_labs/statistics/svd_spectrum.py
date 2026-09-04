"""
Source repository: holeyfield33-art/unitarity-lab
Source commit: 66c004b5f37a5f4a129761ef1d7f87f306b100ac
Source file: unitarity_labs/core/spectral_monitor.py (TransportEvaluator.svd)
Original implementation: full SVD of a square matrix, computed inside a
    class that also produced a composite "stability" score folding in the
    gap ratio and a "manifold collapse" framing.
Retained behavior: the SVD computation itself -- singular values of a real
    matrix, sorted descending. Explicit input-shape and finite-value
    validation was added as a standalone function (the source validated
    squareness only inside the class constructor). Generalized from
    square-only (the source's TransportEvaluator required a square
    "transport" matrix) to any real 2-D matrix, since SVD is defined for
    non-square matrices and the source's square restriction was an
    artifact of its class context, not of the SVD computation itself.
Removed interpretation: TransportEvaluator.stability() (the composite
    metric combining this with the gap ratio),
    frobenius_distance_from_identity(), and all "stability"/"collapse"
    framing. This function reports singular values only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def singular_value_spectrum(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """Singular values of a real matrix, sorted descending (raw statistic).

    Parameters
    ----------
    matrix : 2-D array_like.
        Need not be square.

    Returns
    -------
    ndarray, shape (min(m, n),)
        Singular values, sorted descending.

    Raises
    ------
    ValueError
        If the input is not 2-D, is empty, or contains NaN/Inf.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("matrix must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix must not contain NaN or Inf.")

    singular_values = np.linalg.svd(arr, compute_uv=False)
    return np.sort(singular_values)[::-1]
