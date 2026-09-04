"""
Source repository: holeyfield33-art/var
Source commit: 31234551e524249a5e81453ec851c98ec8836fb7
Source file: var/spectral.py (IncrementalTopKEigen)
Original implementation: an incrementally-updated top-k eigen
    decomposition of a sliding-window sample covariance matrix, used as
    one stage of a streaming spectral-drift / anomaly-detection pipeline.
Retained behavior: the online covariance accumulator (running sum and
    sum-of-outer-products, decremented on ``remove``) is split out as its
    own ``RollingCovariance`` class so it is independently testable; the
    top-k eigenvalue tracker (``TopKEigenTracker``) keeps the source's
    subspace iteration with oversampling and periodic full ``eigh``
    recomputation for numerical drift control, delegating covariance
    bookkeeping to ``RollingCovariance`` by composition instead of
    duplicating its state. Explicit input-shape and finite-value
    validation was added to both classes; the source did not validate
    ``add``/``remove`` inputs.
Removed interpretation: no "spectral health" or "anomaly" framing existed
    in this specific class -- it is a numerical utility -- so nothing
    beyond the surrounding pipeline was removed: the Merkle-linked
    snapshot hashing and ``RollingW2SpectralMonitor`` that coupled this
    class to the rest of var's detection stack are not ported.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RollingCovariance:
    """Online (running) covariance accumulator over a set of vectors.

    Supports incremental ``add``/``remove`` of observations so a caller
    can maintain a sliding-window covariance without recomputing from
    scratch.

    Parameters
    ----------
    dim : int
        Dimensionality of each observation vector.
    """

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self._sum: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
        self._sq: NDArray[np.float64] = np.zeros((dim, dim), dtype=np.float64)
        self._n: int = 0

    def _validate(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("observation must not contain NaN or Inf.")
        return arr

    def add(self, x: NDArray[np.floating]) -> None:
        """Add one observation to the running statistics."""
        arr = self._validate(x)
        self._sum += arr
        self._sq += np.outer(arr, arr)
        self._n += 1

    def remove(self, x: NDArray[np.floating]) -> None:
        """Remove one previously-added observation from the running statistics."""
        arr = self._validate(x)
        if self._n < 1:
            raise ValueError("cannot remove from an empty accumulator")
        self._sum -= arr
        self._sq -= np.outer(arr, arr)
        self._n -= 1

    @property
    def n(self) -> int:
        return self._n

    def covariance(self) -> NDArray[np.float64]:
        """Current (population) covariance matrix.

        Raises
        ------
        ValueError
            If no observations have been added.
        """
        if self._n < 1:
            raise ValueError("covariance is undefined with zero observations")
        mean = self._sum / self._n
        return self._sq / self._n - np.outer(mean, mean)


class TopKEigenTracker:
    """Incrementally track the top-k eigenvalues of a rolling covariance.

    Uses subspace (orthogonal) iteration to update the top-k eigenpairs in
    O(k^2 d) per step, with a full ``numpy.linalg.eigh`` recomputation
    every ``recalc_period`` steps to bound accumulated numerical drift.

    Parameters
    ----------
    dim : int
        Dimensionality of incoming vectors.
    k : int
        Number of eigenvalues to track (clipped to ``dim``).
    recalc_period : int
        Full recomputation interval.
    oversample : int
        Extra basis columns used for numerical stability during subspace
        iteration.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        recalc_period: int = 100,
        oversample: int = 2,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if k < 1:
            raise ValueError("k must be >= 1")
        if recalc_period < 1:
            raise ValueError("recalc_period must be >= 1")
        if oversample < 0:
            raise ValueError("oversample must be >= 0")

        self.dim = dim
        self.k = min(k, dim)
        self.recalc_period = recalc_period
        self.oversample = min(oversample, max(0, dim - self.k))

        self._cov = RollingCovariance(dim)
        self._step = 0

        self.eigvals: NDArray[np.float64] = np.zeros(self.k)
        self.eigvecs: NDArray[np.float64] = np.zeros((dim, self.k))

    def add(self, x: NDArray[np.floating]) -> None:
        self._cov.add(x)
        self._step += 1

    def remove(self, x: NDArray[np.floating]) -> None:
        self._cov.remove(x)

    def update_spectrum(self) -> NDArray[np.float64]:
        """Recompute the top-k eigenvalues (full or subspace-update path).

        Returns the current top-k eigenvalues, sorted ascending.
        """
        if self._cov.n < 1:
            self.eigvals = np.zeros(self.k)
            return self.eigvals

        cov = self._cov.covariance()

        if self._step % self.recalc_period == 0 or self._step <= 2:
            return self._full_decomp(cov)
        return self._subspace_update(cov)

    def _full_decomp(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        all_vals, all_vecs = np.linalg.eigh(cov)
        idx = np.argsort(all_vals)[-self.k :]
        self.eigvals = all_vals[idx]
        self.eigvecs = all_vecs[:, idx]
        return self.eigvals.copy()

    def _subspace_update(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        V = self.eigvecs

        if self.oversample > 0 and self.dim > self.k:
            rng = np.random.default_rng(self._step)
            extra = rng.standard_normal((self.dim, self.oversample))
            extra -= V @ (V.T @ extra)
            extra, _ = np.linalg.qr(extra)
            V_ext = np.hstack([V, extra[:, : self.oversample]])
        else:
            V_ext = V

        Y = cov @ V_ext
        Q, _ = np.linalg.qr(Y)

        H = Q.T @ cov @ Q
        vals, vecs = np.linalg.eigh(H)
        idx = np.argsort(vals)[-self.k :]
        self.eigvals = vals[idx]
        self.eigvecs = Q @ vecs[:, idx]

        self.eigvecs, _ = np.linalg.qr(self.eigvecs)
        return self.eigvals.copy()


def leading_eigengap(eigenvalues: NDArray[np.floating]) -> float:
    """Gap between the largest and second-largest eigenvalue (lambda_gap).

    New in Unitarity Labs (not ported from a prior codebook): a direct
    reading of the two largest values in an eigenvalue list -- the
    ``lambda_gap`` component of the measurement vector defined in
    docs/RESEARCH_THESIS.md, directly supported by the ported
    ``TopKEigenTracker`` above. It carries no interpretation on its own --
    see docs/EVIDENCE_LEVELS.md.

    Parameters
    ----------
    eigenvalues : 1-D array of real numbers, at least 2 elements.

    Returns
    -------
    float
        largest - second_largest, >= 0.

    Raises
    ------
    ValueError
        If fewer than 2 eigenvalues are given, if the input is not 1-D,
        or if it contains NaN/Inf.
    """
    arr = np.asarray(eigenvalues, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"eigenvalues must be 1-D, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError("Need at least 2 eigenvalues to compute a gap.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("eigenvalues must not contain NaN or Inf.")

    sorted_desc = np.sort(arr)[::-1]
    return float(sorted_desc[0] - sorted_desc[1])
