"""Reference generation, decoding, and metrics for rank-syndrome ECC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LinearMap(Protocol):
    rows: int
    cols: int
    measurements: int

    def apply(self, matrix: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RecoveryMetrics:
    """Measured recovery metrics with the pre-registered success predicate."""

    relative_error_recovery: float
    relative_state_recovery: float
    syndrome_residual: float
    recovered_rank: int
    success: bool


def generate_rank_error(rows: int, cols: int, rank: int, seed: int) -> np.ndarray:
    """Generate a seeded Gaussian matrix of exact numerical rank ``rank``."""
    if not 1 <= rank <= min(rows, cols):
        raise ValueError("rank must be between 1 and min(rows, cols)")
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((rows, rank))
    right = rng.standard_normal((cols, rank))
    error = left @ right.T
    if numerical_rank(error) != rank:
        raise RuntimeError("seeded Gaussian factors did not produce the requested numerical rank")
    return error


def generate_clean_state(rows: int, cols: int, family: str, seed: int) -> np.ndarray:
    """Generate an arbitrary clean state independently from the corruption."""
    rng = np.random.default_rng(seed)
    if family == "gaussian":
        return rng.standard_normal((rows, cols))
    if family == "structured_full_rank":
        diagonal = np.arange(1, min(rows, cols) + 1, dtype=float)
        matrix = np.zeros((rows, cols))
        matrix[np.arange(diagonal.size), np.arange(diagonal.size)] = diagonal
        return matrix + 0.01 * rng.standard_normal((rows, cols))
    if family == "low_rank":
        return generate_rank_error(rows, cols, rank=1, seed=seed)
    raise ValueError("family must be gaussian, structured_full_rank, or low_rank")


def decode_nuclear_norm(
    operator: LinearMap,
    syndrome: np.ndarray,
    *,
    solver: str = "CLARABEL",
    feasibility_tolerance: float = 1e-8,
) -> np.ndarray:
    """Solve the convex nuclear-norm recovery problem with CVXPY.

    CVXPY is deliberately an optional ``ecc-research`` dependency.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "Nuclear-norm decoding requires the optional research dependency. "
            "Install it with: python -m pip install -e '.[ecc-research]'"
        ) from exc

    syndrome = np.asarray(syndrome, dtype=float)
    if syndrome.shape != (operator.measurements,):
        raise ValueError(
            f"syndrome must have shape ({operator.measurements},), got {syndrome.shape}"
        )
    if not np.all(np.isfinite(syndrome)):
        raise ValueError("syndrome must contain only finite values")
    variable = cp.Variable((operator.rows, operator.cols))
    if hasattr(operator, "matrices"):
        measurement_matrix = operator.matrices.reshape(operator.measurements, -1)
    else:
        measurement_matrix = np.eye(operator.measurements)
    problem = cp.Problem(cp.Minimize(cp.normNuc(variable)), [measurement_matrix @ cp.vec(variable, order="C") == syndrome])
    problem.solve(solver=solver, tol_gap_abs=feasibility_tolerance, tol_feas=feasibility_tolerance)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or variable.value is None:
        raise RuntimeError(f"nuclear-norm decoder failed with solver status {problem.status!r}")
    return np.asarray(variable.value, dtype=float)


def numerical_rank(matrix: np.ndarray, relative_tolerance: float = 1e-8) -> int:
    """Return numerical rank using a scale-relative singular-value tolerance."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    return int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))


def recovery_metrics(
    operator: LinearMap,
    error: np.ndarray,
    estimated_error: np.ndarray,
    clean_state: np.ndarray,
    syndrome: np.ndarray,
    epsilon: float = 1e-12,
) -> RecoveryMetrics:
    """Calculate the pre-registered metrics and fixed success criterion."""
    estimated_state = clean_state + error - estimated_error
    error_relative = float(np.linalg.norm(estimated_error - error) / max(np.linalg.norm(error), epsilon))
    state_relative = float(np.linalg.norm(estimated_state - clean_state) / max(np.linalg.norm(clean_state), epsilon))
    syndrome_residual = float(np.linalg.norm(operator.apply(estimated_error) - syndrome))
    return RecoveryMetrics(
        relative_error_recovery=error_relative,
        relative_state_recovery=state_relative,
        syndrome_residual=syndrome_residual,
        recovered_rank=numerical_rank(estimated_error),
        success=(error_relative < 1e-5 and state_relative < 1e-5 and syndrome_residual < 1e-7),
    )