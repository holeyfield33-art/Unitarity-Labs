"""Reproducible synthetic sweep for the rank-syndrome ECC construction."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cvxpy as cp
import numpy as np

from unitarity_labs.ecc import (
    GaussianLinearMap,
    IdentityLinearMap,
    decode_nuclear_norm,
    generate_clean_state,
    generate_rank_error,
    recovery_metrics,
)


ROWS = COLS = 12
RANKS = (1, 2, 3)
MEASUREMENTS = (20, 40, 60, 80, 100, 120)
SEEDS = tuple(range(20))
SOLVER = "CLARABEL"
SOLVER_TOLERANCE = 1e-8
CLEAN_FAMILIES = ("gaussian", "structured_full_rank", "low_rank")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "rank_syndrome"


def run_trial(rank: int, measurements: int, seed: int, family: str) -> dict[str, object]:
    """Run one corruption/recovery trial with independently seeded components."""
    operator = GaussianLinearMap(ROWS, COLS, measurements, seed=seed)
    clean_state = generate_clean_state(ROWS, COLS, family, seed=10_000 + seed)
    error = generate_rank_error(ROWS, COLS, rank, seed=20_000 + seed)
    syndrome = operator.apply(clean_state + error) - operator.apply(clean_state)
    estimate = decode_nuclear_norm(
        operator, syndrome, solver=SOLVER, feasibility_tolerance=SOLVER_TOLERANCE
    )
    metrics = recovery_metrics(operator, error, estimate, clean_state, syndrome)
    return {
        "rank": rank,
        "measurements": measurements,
        "seed": seed,
        "clean_family": family,
        "measurement_seed": seed,
        "error_seed": 20_000 + seed,
        "clean_seed": 10_000 + seed,
        **metrics.__dict__,
    }


def run_controls() -> dict[str, dict[str, object]]:
    """Run pre-registered positive and negative controls."""
    clean = generate_clean_state(ROWS, COLS, "gaussian", seed=31)
    zero_operator = GaussianLinearMap(ROWS, COLS, 120, seed=32)
    zero = np.zeros((ROWS, COLS))
    zero_metrics = recovery_metrics(zero_operator, zero, zero, clean, zero_operator.apply(zero))

    full_operator = IdentityLinearMap(ROWS, COLS)
    full_error = generate_rank_error(ROWS, COLS, 2, seed=33)
    full_syndrome = full_operator.apply(full_error)
    full_estimate = decode_nuclear_norm(full_operator, full_syndrome, solver=SOLVER)
    full_metrics = recovery_metrics(full_operator, full_error, full_estimate, clean, full_syndrome)

    rank_one = run_trial(rank=1, measurements=120, seed=2, family="gaussian")
    rank_two = run_trial(rank=2, measurements=120, seed=2, family="structured_full_rank")
    too_few = run_trial(rank=1, measurements=20, seed=0, family="gaussian")
    beyond_design = run_trial(rank=3, measurements=40, seed=1, family="gaussian")

    wrong_operator = IdentityLinearMap(ROWS, COLS)
    wrong_error = generate_rank_error(ROWS, COLS, 1, seed=34)
    other_clean = generate_clean_state(ROWS, COLS, "structured_full_rank", seed=35)
    wrong_syndrome = wrong_operator.apply(clean + wrong_error) - wrong_operator.apply(other_clean)
    wrong_estimate = decode_nuclear_norm(wrong_operator, wrong_syndrome, solver=SOLVER)
    wrong_metrics = recovery_metrics(wrong_operator, wrong_error, wrong_estimate, clean, wrong_syndrome)

    return {
        "zero_error": {**zero_metrics.__dict__, "expected_success": True},
        "full_observation": {**full_metrics.__dict__, "expected_success": True},
        "rank_1_overdetermined": {**rank_one, "expected_success": True},
        "rank_2_easy_regime": {**rank_two, "expected_success": True},
        "too_few_measurements": {**too_few, "expected_success": False},
        "rank_beyond_design_target": {**beyond_design, "expected_success": False},
        "wrong_parity": {**wrong_metrics.__dict__, "expected_success": False},
    }


def summarize(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    """Calculate unfit success-rate summaries for every (rank, measurements) pair."""
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        grouped[(int(trial["rank"]), int(trial["measurements"]))].append(trial)
    return [
        {
            "rank": rank,
            "measurements": measurements,
            "theory_dimension_reference": 2 * rank * (ROWS + COLS - 2 * rank),
            "successes": sum(bool(trial["success"]) for trial in group),
            "trials": len(group),
            "success_rate": sum(bool(trial["success"]) for trial in group) / len(group),
        }
        for (rank, measurements), group in sorted(grouped.items())
    ]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials = [
        run_trial(rank, measurements, seed, CLEAN_FAMILIES[seed % len(CLEAN_FAMILIES)])
        for rank in RANKS
        for measurements in MEASUREMENTS
        for seed in SEEDS
    ]
    payload = {
        "evidence": "MEASURED",
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "dependencies": {"numpy": np.__version__, "cvxpy": cp.__version__},
        "dimensions": {"rows": ROWS, "cols": COLS},
        "measurement_counts": list(MEASUREMENTS),
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
        "measurement_distribution": "iid Gaussian N(0, 1/m)",
        "solver": SOLVER,
        "solver_tolerances": {"tol_gap_abs": SOLVER_TOLERANCE, "tol_feas": SOLVER_TOLERANCE},
        "success_thresholds": {"relative_error": 1e-5, "relative_state": 1e-5, "syndrome_residual": 1e-7},
        "trials": trials,
        "summary": summarize(trials),
        "controls": run_controls(),
    }
    output = RESULTS_DIR / "synthetic_sweep.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("rank  m    success rate  theory reference")
    for result in payload["summary"]:
        print(
            f"{result['rank']:>4}  {result['measurements']:>3}  "
            f"{result['success_rate']:>11.0%}  {result['theory_dimension_reference']:>16}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()