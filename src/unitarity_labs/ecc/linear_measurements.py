"""Deterministic linear maps for controlled matrix experiments."""

from __future__ import annotations

import numpy as np


class GaussianLinearMap:
    """A seeded Gaussian linear map on matrices.

    Measurement entries have variance ``1 / measurements`` so expected map
    energy remains comparable as the number of measurements changes.
    """

    def __init__(self, rows: int, cols: int, measurements: int, seed: int) -> None:
        if rows < 1 or cols < 1 or measurements < 1:
            raise ValueError("rows, cols, and measurements must all be positive")
        self.rows = rows
        self.cols = cols
        self.measurements = measurements
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.matrices = rng.standard_normal((measurements, rows, cols)) / np.sqrt(measurements)

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Return the vector of Frobenius inner-product measurements."""
        matrix = self._validate_matrix(matrix)
        return np.einsum("ijk,jk->i", self.matrices, matrix)

    def adjoint(self, vector: np.ndarray) -> np.ndarray:
        """Return the Frobenius adjoint of ``apply``."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.measurements,):
            raise ValueError(f"vector must have shape ({self.measurements},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("vector must contain only finite values")
        return np.einsum("i,ijk->jk", vector, self.matrices)

    def _validate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        expected_shape = (self.rows, self.cols)
        if matrix.shape != expected_shape:
            raise ValueError(f"matrix must have shape {expected_shape}, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must contain only finite values")
        return matrix


class FixedScaleNestedGaussianLinearMap(GaussianLinearMap):
    """Seeded Gaussian map with rows fixed at variance ``1 / (rows * cols)``.

    For a fixed seed, an operator with more measurements contains the prior
    operator's rows unchanged. This condition separates added information from
    measurement-count-dependent row rescaling.
    """

    condition = "fixed_scale_nested"

    def __init__(self, rows: int, cols: int, measurements: int, seed: int) -> None:
        if rows < 1 or cols < 1 or measurements < 1:
            raise ValueError("rows, cols, and measurements must all be positive")
        self.rows = rows
        self.cols = cols
        self.measurements = measurements
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.matrices = rng.standard_normal((measurements, rows, cols)) / np.sqrt(rows * cols)


class IdentityLinearMap:
    """Full-observation vectorization map used as a decoder plumbing control."""

    def __init__(self, rows: int, cols: int) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must both be positive")
        self.rows = rows
        self.cols = cols
        self.measurements = rows * cols

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        matrix = self._validate_matrix(matrix)
        return matrix.reshape(-1)

    def adjoint(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.measurements,):
            raise ValueError(f"vector must have shape ({self.measurements},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("vector must contain only finite values")
        return vector.reshape(self.rows, self.cols)

    def _validate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        expected_shape = (self.rows, self.cols)
        if matrix.shape != expected_shape:
            raise ValueError(f"matrix must have shape {expected_shape}, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must contain only finite values")
        return matrix