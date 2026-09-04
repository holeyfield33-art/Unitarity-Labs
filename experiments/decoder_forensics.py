"""Forensic audit of frozen nuclear-norm recovery reversals.

The nonconvex routines in this file are research-only heuristic diagnostics.
They are not production decoders or proofs of identifiability.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.optimize import least_squares, minimize

from unitarity_labs.ecc import (
    FixedScaleNestedGaussianLinearMap,
    GaussianLinearMap,
    generate_clean_state,
    generate_rank_error,
    recovery_metrics,
)
from unitarity_labs.ecc.rank_syndrome import numerical_rank

ROWS = COLS = 12
SOLVERS = ("CLARABEL", "SCS")
TOLERANCES = (1e-6, 1e-8, 1e-10)
ALPHAS = tuple(np.linspace(-0.5, 1.5, 9))
ORACLE_STARTS = 12
KERNEL_STARTS = 16
KERNEL_WARNING_RESIDUAL = 1e-6
OBJECTIVE_ZERO_TOLERANCE = 1e-10
RESULTS = Path(__file__).resolve().parents[1] / "results" / "rank_syndrome" / "decoder_forensics.json"

# The last success and first later failure from normalization_audit.json.
REVERSALS = (
    {"condition": "m_normalized", "rank": 1, "seed": 9, "success_m": 100, "failure_m": 120},
    {"condition": "m_normalized", "rank": 1, "seed": 18, "success_m": 60, "failure_m": 80},
    {"condition": "fixed_scale_nested", "rank": 1, "seed": 0, "success_m": 100, "failure_m": 120},
    {"condition": "fixed_scale_nested", "rank": 3, "seed": 2, "success_m": 100, "failure_m": 120},
    {"condition": "fixed_scale_nested", "rank": 3, "seed": 8, "success_m": 100, "failure_m": 120},
)
MAPS = {
    "m_normalized": GaussianLinearMap,
    "fixed_scale_nested": FixedScaleNestedGaussianLinearMap,
}


def _family(seed: int) -> str:
    return ("gaussian", "structured_full_rank", "low_rank")[seed % 3]


def _sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _solve(problem: cp.Problem, solver: str, tolerance: float) -> None:
    if solver == "CLARABEL":
        problem.solve(solver=solver, tol_gap_abs=tolerance, tol_feas=tolerance)
    else:
        problem.solve(solver=solver, eps=tolerance)


def _case_data(case: dict[str, object], measurements: int) -> dict[str, Any]:
    operator = MAPS[str(case["condition"])](ROWS, COLS, measurements, int(case["seed"]))
    clean = generate_clean_state(ROWS, COLS, _family(int(case["seed"])), 10_000 + int(case["seed"]))
    error = generate_rank_error(ROWS, COLS, int(case["rank"]), 20_000 + int(case["seed"]))
    syndrome = operator.apply(clean + error) - operator.apply(clean)
    return {"operator": operator, "clean": clean, "error": error, "syndrome": syndrome}


def _solver_run(data: dict[str, Any], solver: str, tolerance: float) -> dict[str, Any]:
    operator = data["operator"]
    error = data["error"]
    clean = data["clean"]
    syndrome = data["syndrome"]
    matrix = operator.matrices.reshape(operator.measurements, -1)
    variable = cp.Variable((ROWS, COLS))
    constraint = matrix @ cp.vec(variable, order="C") == syndrome
    problem = cp.Problem(cp.Minimize(cp.normNuc(variable)), [constraint])
    try:
        _solve(problem, solver, tolerance)
        estimate = None if variable.value is None else np.asarray(variable.value, dtype=float)
        if estimate is None:
            return {"solver": solver, "tolerance": tolerance, "status": str(problem.status), "objective": problem.value, "error": "no primal value"}
        metrics = asdict(recovery_metrics(operator, error, estimate, clean, syndrome))
        stats = problem.solver_stats
        dual = None if constraint.dual_value is None else np.asarray(constraint.dual_value, dtype=float)
        return {
            "solver": solver,
            "tolerance": tolerance,
            "status": str(problem.status),
            "objective": float(problem.value),
            "true_nuclear_norm": float(np.linalg.norm(error, ord="nuc")),
            "recovered_nuclear_norm": float(np.linalg.norm(estimate, ord="nuc")),
            "nuclear_norm_delta": float(np.linalg.norm(estimate, ord="nuc") - np.linalg.norm(error, ord="nuc")),
            "iterations": stats.num_iters,
            "solve_time_seconds": stats.solve_time,
            "syndrome_residual": metrics["syndrome_residual"],
            "relative_syndrome_residual": metrics["relative_syndrome_residual"],
            "relative_error_recovery": metrics["relative_error_recovery"],
            "relative_state_recovery": metrics["relative_state_recovery"],
            "recovered_rank": metrics["recovered_rank"],
            "dual_vector": None if dual is None else dual.tolist(),
            "adjoint_dual_frobenius_norm": None if dual is None else float(np.linalg.norm(operator.adjoint(dual))),
            "estimate": estimate,
        }
    except Exception as exc:
        return {"solver": solver, "tolerance": tolerance, "status": "solver_error", "error": str(exc)}


def _feasible_path(operator: Any, error: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    direction = estimate - error
    return {
        "kernel_residual": float(np.linalg.norm(operator.apply(direction))),
        "direction_rank": numerical_rank(direction),
        "values": [{"alpha": float(alpha), "nuclear_norm": float(np.linalg.norm(error + alpha * direction, ord="nuc"))} for alpha in ALPHAS],
    }


def _certificate_check(operator: Any, error: np.ndarray) -> dict[str, Any]:
    left, _, right_t = np.linalg.svd(error, full_matrices=False)
    rank = numerical_rank(error)
    left = left[:, :rank]
    right = right_t[:rank].T
    dual = cp.Variable(operator.measurements)
    remainder = cp.Variable((ROWS, COLS))
    subgradient = left @ right.T + remainder
    constraints = [left.T @ remainder == 0, remainder @ right == 0, cp.norm(remainder, 2) <= 1]
    adjoint_matrix = operator.matrices.reshape(operator.measurements, -1).T
    problem = cp.Problem(cp.Minimize(cp.norm(adjoint_matrix @ dual - cp.vec(subgradient, order="C"), 2)), constraints)
    attempts = []
    for solver, tolerance in (("CLARABEL", 1e-8), ("SCS", 1e-8)):
        try:
            _solve(problem, solver, tolerance)
            attempts.append({
                "solver": solver,
                "tolerance": tolerance,
                "status": str(problem.status),
                "certificate_distance": None if problem.value is None else float(problem.value),
            })
        except cp.error.SolverError as exc:
            attempts.append({"solver": solver, "tolerance": tolerance, "status": "solver_error", "error": str(exc)})
    successful = [attempt for attempt in attempts if attempt["status"] in ("optimal", "optimal_inaccurate")]
    return {
        "attempts": attempts,
        "best_certificate_distance": min((attempt["certificate_distance"] for attempt in successful), default=None),
        "interpretation": "Numerical subgradient-range distance only; not a proof of certificate existence.",
    }


def _factor_matrix(parameters: np.ndarray, rank: int) -> np.ndarray:
    split = ROWS * rank
    return parameters[:split].reshape(ROWS, rank) @ parameters[split:].reshape(COLS, rank).T


def _rank_oracle(operator: Any, error: np.ndarray, syndrome: np.ndarray, rank: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(40_000 + seed)
    matrix = operator.matrices.reshape(operator.measurements, -1)
    best: tuple[float, np.ndarray] | None = None
    for _ in range(ORACLE_STARTS):
        start = rng.standard_normal((ROWS + COLS) * rank)
        result = least_squares(lambda value: matrix @ _factor_matrix(value, rank).reshape(-1) - syndrome, start, max_nfev=2_000)
        residual = float(np.linalg.norm(result.fun))
        if best is None or residual < best[0]:
            best = (residual, _factor_matrix(result.x, rank))
    assert best is not None
    estimate = best[1]
    return {
        "label": "HEURISTIC ORACLE",
        "starts": ORACLE_STARTS,
        "best_syndrome_residual": best[0],
        "relative_error_recovery": float(np.linalg.norm(estimate - error) / np.linalg.norm(error)),
        "recovered_rank": numerical_rank(estimate),
    }


def _kernel_search(operator: Any, rank_bound: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(50_000 + seed)
    matrix = operator.matrices.reshape(operator.measurements, -1)
    best: tuple[float, np.ndarray] | None = None
    for _ in range(KERNEL_STARTS):
        start = rng.standard_normal((ROWS + COLS) * rank_bound)
        def objective(value: np.ndarray) -> float:
            candidate = _factor_matrix(value, rank_bound)
            norm = np.linalg.norm(candidate)
            if norm == 0:
                return 1e12
            return float(np.linalg.norm(matrix @ (candidate / norm).reshape(-1)) ** 2)
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 1_000})
        candidate = _factor_matrix(result.x, rank_bound)
        candidate /= np.linalg.norm(candidate)
        residual = float(np.linalg.norm(matrix @ candidate.reshape(-1)))
        if best is None or residual < best[0]:
            best = (residual, candidate)
    assert best is not None
    return {
        "label": "HEURISTIC KERNEL SEARCH",
        "starts": KERNEL_STARTS,
        "rank_bound": rank_bound,
        "best_unit_frobenius_kernel_residual": best[0],
        "candidate_rank": numerical_rank(best[1]),
        "finding": "IDENTIFIABILITY WARNING" if best[0] < KERNEL_WARNING_RESIDUAL else "NO NUMERICAL LOW-RANK KERNEL FOUND",
    }


def _classify(failure: dict[str, Any], true_feasibility: float, oracle: dict[str, Any], kernel: dict[str, Any]) -> str:
    delta = failure["nuclear_norm_delta"]
    if true_feasibility > 1e-10:
        return "UNRESOLVED"
    if kernel["finding"] == "IDENTIFIABILITY WARNING":
        return "IDENTIFIABILITY WARNING"
    if delta > OBJECTIVE_ZERO_TOLERANCE:
        return "SOLVER FAILURE — true E has lower objective"
    if delta < -OBJECTIVE_ZERO_TOLERANCE:
        return "NUCLEAR-NORM FAILURE — alternate feasible solution has lower objective"
    if abs(delta) <= 1e-6:
        return "NEAR-DEGENERATE OPTIMUM"
    return "UNRESOLVED"


def _serializable(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "estimate"}


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for reversal in REVERSALS:
        measurements = (int(reversal["success_m"]), int(reversal["failure_m"]))
        snapshots = []
        failure_record: dict[str, Any] | None = None
        for measurement_count in measurements:
            data = _case_data(reversal, measurement_count)
            true_feasibility = float(np.linalg.norm(data["operator"].apply(data["error"]) - data["syndrome"]))
            runs = [_solver_run(data, solver, tolerance) for solver in SOLVERS for tolerance in TOLERANCES]
            default = next(item for item in runs if item["solver"] == "CLARABEL" and item["tolerance"] == 1e-8)
            snapshot = {
                "measurements": measurement_count,
                "hashes": {key: _sha256(data[key]) for key in ("clean", "error", "syndrome")} | {"measurement_bank": _sha256(data["operator"].matrices)},
                "true_error_feasibility_residual": true_feasibility,
                "true_nuclear_norm": float(np.linalg.norm(data["error"], ord="nuc")),
                "solver_matrix": [_serializable(item) for item in runs],
            }
            if measurement_count == int(reversal["failure_m"]):
                if "estimate" not in default:
                    raise RuntimeError(f"primary failure solve did not return an estimate: {default}")
                path = _feasible_path(data["operator"], data["error"], default["estimate"])
                certificate = _certificate_check(data["operator"], data["error"])
                oracle = _rank_oracle(data["operator"], data["error"], data["syndrome"], int(reversal["rank"]), int(reversal["seed"]))
                kernel = _kernel_search(data["operator"], 2 * int(reversal["rank"]), int(reversal["seed"]))
                snapshot |= {"feasible_direction": path, "subgradient_certificate": certificate, "rank_oracle": oracle, "kernel_search": kernel}
                failure_record = {"failure": default, "true_feasibility": true_feasibility, "oracle": oracle, "kernel": kernel}
            snapshots.append(snapshot)
        assert failure_record is not None
        records.append({**reversal, "snapshots": snapshots, "classification": _classify(**failure_record)})
    payload = {
        "evidence": "MEASURED",
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "dependencies": {"numpy": np.__version__, "cvxpy": cp.__version__},
        "solvers": list(SOLVERS),
        "tolerance_ladder": list(TOLERANCES),
        "reversals": records,
    }
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        print(record["condition"], record["rank"], record["seed"], record["classification"])
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
