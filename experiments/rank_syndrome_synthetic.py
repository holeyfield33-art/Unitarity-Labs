"""Reproducible numerical audit for the rank-syndrome ECC construction."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cvxpy as cp
import numpy as np

from unitarity_labs.ecc import (
    GaussianLinearMap,
    IdentityLinearMap,
    decode_nuclear_norm,
    decode_nuclear_norm_noisy,
    decode_nuclear_norm_with_diagnostics,
    generate_clean_state,
    generate_rank_error,
    recovery_metrics,
)

ROWS = COLS = 12
RANKS = (1, 2, 3)
MEASUREMENTS = (20, 40, 60, 80, 100, 120)
SEEDS = tuple(range(20))
SOLVER = "CLARABEL"
RETRY_SOLVER = "SCS"
SOLVER_TOLERANCE = 1e-8
CLEAN_FAMILIES = ("gaussian", "structured_full_rank", "low_rank")
NOISE_LEVELS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
QUANTIZATION_DTYPES = ("float64", "float32", "float16")
SCALING_DIMENSIONS = (8, 12, 16)
SCALING_RANKS = (1, 2)
SCALING_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.75)
SCALING_SEEDS = tuple(range(10))
NOISE_AWARE_RECOVERY_THRESHOLD = 1e-3
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "rank_syndrome"


def _is_success(trial: dict[str, object], threshold: float = 1e-5) -> bool:
    return (
        float(trial["relative_error_recovery"]) < threshold
        and float(trial["relative_state_recovery"]) < threshold
        and float(trial["syndrome_residual"]) < 1e-7
    )


def _trial(rows: int, cols: int, rank: int, measurements: int, seed: int, family: str, solver: str) -> dict[str, object]:
    operator = GaussianLinearMap(rows, cols, measurements, seed=seed)
    clean = generate_clean_state(rows, cols, family, seed=10_000 + seed)
    error = generate_rank_error(rows, cols, rank, seed=20_000 + seed)
    syndrome = operator.apply(clean + error) - operator.apply(clean)
    try:
        estimate, diagnostics = decode_nuclear_norm_with_diagnostics(
            operator, syndrome, solver=solver, feasibility_tolerance=SOLVER_TOLERANCE
        )
        metrics = asdict(recovery_metrics(operator, error, estimate, clean, syndrome))
        singular_values = np.linalg.svd(estimate, compute_uv=False)
        return {
            "rank": rank, "measurements": measurements, "redundancy": measurements / (rows * cols),
            "seed": seed, "clean_family": family, "measurement_seed": seed,
            "error_seed": 20_000 + seed, "clean_seed": 10_000 + seed,
            "solver_diagnostics": asdict(diagnostics),
            "smallest_singular_value": float(singular_values[-1]),
            "largest_singular_value": float(singular_values[0]), **metrics,
        }
    except Exception as exc:
        return {
            "rank": rank, "measurements": measurements, "redundancy": measurements / (rows * cols),
            "seed": seed, "clean_family": family, "measurement_seed": seed,
            "error_seed": 20_000 + seed, "clean_seed": 10_000 + seed,
            "solver_diagnostics": {"status": "solver_error", "objective_value": None,
                "primal_residual": float("inf"), "feasibility_residual": float("inf"),
                "solver": solver, "error": str(exc)},
            "smallest_singular_value": None, "largest_singular_value": None,
            "relative_error_recovery": float("inf"), "relative_state_recovery": float("inf"),
            "syndrome_residual": float("inf"), "recovered_rank": None, "success": False,
        }


def run_trial(rank: int, measurements: int, seed: int, family: str) -> dict[str, object]:
    """Run one primary 12x12 trial with independently seeded components."""
    return _trial(ROWS, COLS, rank, measurements, seed, family, SOLVER)


def _summarize(trials: list[dict[str, object]], rows: int, cols: int) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        grouped[(int(trial["rank"]), int(trial["measurements"]))].append(trial)
    return [{
        "rank": rank, "measurements": measurements, "redundancy": measurements / (rows * cols),
        "theory_dimension_reference": 2 * rank * (rows + cols - 2 * rank),
        "successes": sum(bool(trial["success"]) for trial in group), "trials": len(group),
        "success_rate": sum(bool(trial["success"]) for trial in group) / len(group),
    } for (rank, measurements), group in sorted(grouped.items())]


def _monotonicity_violations(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        groups[(int(trial["rank"]), int(trial["seed"]), str(trial["clean_family"]))].append(trial)
    findings = []
    for group in groups.values():
        group.sort(key=lambda trial: int(trial["measurements"]))
        for earlier in group:
            for later in group:
                if int(later["measurements"]) > int(earlier["measurements"]) and bool(earlier["success"]) and not bool(later["success"]):
                    findings.append({"rank": earlier["rank"], "seed": earlier["seed"], "clean_family": earlier["clean_family"], "success_measurements": earlier["measurements"], "failure_measurements": later["measurements"], "finding": "NONMONOTONE_RECOVERY"})
    return findings


def _retry_audit(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    audits = []
    for trial in trials:
        if bool(trial["success"]):
            continue
        retry = _trial(ROWS, COLS, int(trial["rank"]), int(trial["measurements"]), int(trial["seed"]), str(trial["clean_family"]), RETRY_SOLVER)
        status = str(trial["solver_diagnostics"]["status"])
        residual = float(trial["solver_diagnostics"]["feasibility_residual"])
        classification = "LIKELY_SOLVER_TOLERANCE" if bool(retry["success"]) else "UNRESOLVED_NUMERICAL" if status == "optimal_inaccurate" or residual >= 1e-7 else "LIKELY_INFORMATION_FAILURE"
        audits.append({"primary": trial, "retry": retry, "classification": classification})
    return audits


def _threshold_sensitivity(trials: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    output = {}
    for label, threshold in (("strict", 1e-6), ("current", 1e-5), ("relaxed", 1e-4)):
        rows = []
        for row in _summarize(trials, ROWS, COLS):
            group = [trial for trial in trials if trial["rank"] == row["rank"] and trial["measurements"] == row["measurements"]]
            successes = sum(_is_success(trial, threshold) for trial in group)
            rows.append({**{key: row[key] for key in ("rank", "measurements", "redundancy")}, "successes": successes, "trials": len(group), "success_rate": successes / len(group)})
        output[label] = rows
    return output


def _redundancy(summary: list[dict[str, object]]) -> dict[str, object]:
    milestones = {}
    for rank in RANKS:
        rank_rows = [row for row in summary if row["rank"] == rank]
        milestones[str(rank)] = {f"{int(target * 100)}_percent": next(({"measurements": row["measurements"], "redundancy": row["redundancy"]} for row in rank_rows if row["success_rate"] >= target), None) for target in (0.90, 0.95, 1.00)}
    return {
        "ratios": [{"measurements": m, "redundancy": m / (ROWS * COLS)} for m in MEASUREMENTS],
        "milestones": milestones,
        "bit_costs": [{"measurements": m, "precision_bits": bits, "parity_bits": m * bits, "state_bits": ROWS * COLS * bits, "parity_to_state_bits": m / (ROWS * COLS)} for m in MEASUREMENTS for bits in (64, 32, 16)],
        "duplicate_storage_reference": {"redundancy": 1.0, "parity_to_state_bits": 1.0},
    }


def _noisy_trial(kind: str, level: float | str, seed: int) -> dict[str, object]:
    operator = GaussianLinearMap(ROWS, COLS, 100, seed=seed)
    family = CLEAN_FAMILIES[seed % len(CLEAN_FAMILIES)]
    clean = generate_clean_state(ROWS, COLS, family, seed=10_000 + seed)
    error = generate_rank_error(ROWS, COLS, 1, seed=20_000 + seed)
    parity = operator.apply(clean)
    if kind == "quantization":
        stored_parity = parity.astype(np.dtype(str(level))).astype(float)
    else:
        rng = np.random.default_rng(30_000 + seed)
        direction = rng.standard_normal(operator.measurements)
        stored_parity = parity + direction / np.linalg.norm(direction) * float(level) * np.linalg.norm(parity)
    noisy_syndrome = operator.apply(clean + error) - stored_parity
    noise_bound = max(float(np.linalg.norm(noisy_syndrome - operator.apply(error))), SOLVER_TOLERANCE)
    estimate = decode_nuclear_norm_noisy(operator, noisy_syndrome, noise_bound, solver=SOLVER)
    metrics = asdict(recovery_metrics(operator, error, estimate, clean, noisy_syndrome))
    noise_aware_success = float(metrics["relative_error_recovery"]) < NOISE_AWARE_RECOVERY_THRESHOLD and float(metrics["relative_state_recovery"]) < NOISE_AWARE_RECOVERY_THRESHOLD and float(metrics["syndrome_residual"]) <= noise_bound + 1e-7
    return {"kind": kind, "level": level, "seed": seed, "clean_family": family, "noise_bound": noise_bound, "noise_aware_success": noise_aware_success, **metrics}


def _noisy_summary(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[object, list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        grouped[trial["level"]].append(trial)
    return [{"level": level, "successes": sum(bool(trial["noise_aware_success"]) for trial in group), "trials": len(group), "success_rate": sum(bool(trial["noise_aware_success"]) for trial in group) / len(group), "median_relative_error_recovery": float(np.median([trial["relative_error_recovery"] for trial in group])), "median_relative_state_recovery": float(np.median([trial["relative_state_recovery"] for trial in group]))} for level, group in grouped.items()]


def _scaling() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trials = []
    for dimension in SCALING_DIMENSIONS:
        for rank in SCALING_RANKS:
            for fraction in SCALING_FRACTIONS:
                for seed in SCALING_SEEDS:
                    trials.append(_trial(dimension, dimension, rank, round(fraction * dimension * dimension), seed, CLEAN_FAMILIES[seed % 3], SOLVER) | {"dimension": dimension, "measurement_fraction": fraction})
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        groups[(int(trial["dimension"]), int(trial["rank"]), int(trial["measurements"]))].append(trial)
    summary = [{"dimension": dim, "rank": rank, "measurements": measurements, "redundancy": group[0]["redundancy"], "successes": sum(bool(trial["success"]) for trial in group), "trials": len(group), "success_rate": sum(bool(trial["success"]) for trial in group) / len(group)} for (dim, rank, measurements), group in sorted(groups.items())]
    return trials, summary


def run_controls() -> dict[str, dict[str, object]]:
    clean = generate_clean_state(ROWS, COLS, "gaussian", seed=31)
    zero_operator = GaussianLinearMap(ROWS, COLS, 120, seed=32)
    zero = np.zeros((ROWS, COLS))
    full_operator = IdentityLinearMap(ROWS, COLS)
    full_error = generate_rank_error(ROWS, COLS, 2, seed=33)
    full_syndrome = full_operator.apply(full_error)
    full_estimate = decode_nuclear_norm(full_operator, full_syndrome, solver=SOLVER)
    wrong_error = generate_rank_error(ROWS, COLS, 1, seed=34)
    other_clean = generate_clean_state(ROWS, COLS, "structured_full_rank", seed=35)
    wrong_syndrome = full_operator.apply(clean + wrong_error) - full_operator.apply(other_clean)
    wrong_estimate = decode_nuclear_norm(full_operator, wrong_syndrome, solver=SOLVER)
    return {
        "zero_error": {**asdict(recovery_metrics(zero_operator, zero, zero, clean, zero_operator.apply(zero))), "expected_success": True},
        "full_observation": {**asdict(recovery_metrics(full_operator, full_error, full_estimate, clean, full_syndrome)), "expected_success": True},
        "rank_1_overdetermined": {**run_trial(1, 120, 2, "gaussian"), "expected_success": True},
        "rank_2_easy_regime": {**run_trial(2, 120, 2, "structured_full_rank"), "expected_success": True},
        "too_few_measurements": {**run_trial(1, 20, 0, "gaussian"), "expected_success": False},
        "rank_beyond_design_target": {**run_trial(3, 40, 1, "gaussian"), "expected_success": False},
        "wrong_parity": {**asdict(recovery_metrics(full_operator, wrong_error, wrong_estimate, clean, wrong_syndrome)), "expected_success": False},
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials = [run_trial(rank, m, seed, CLEAN_FAMILIES[seed % 3]) for rank in RANKS for m in MEASUREMENTS for seed in SEEDS]
    summary = _summarize(trials, ROWS, COLS)
    quantization = [_noisy_trial("quantization", dtype, seed) for dtype in QUANTIZATION_DTYPES for seed in SCALING_SEEDS]
    additive_noise = [_noisy_trial("additive_noise", level, seed) for level in NOISE_LEVELS for seed in SCALING_SEEDS]
    scaling_trials, scaling_summary = _scaling()
    payload = {
        "evidence": "MEASURED", "git_sha": _git_sha(), "python_version": platform.python_version(), "dependencies": {"numpy": np.__version__, "cvxpy": cp.__version__}, "dimensions": {"rows": ROWS, "cols": COLS}, "measurement_counts": list(MEASUREMENTS), "ranks": list(RANKS), "seeds": list(SEEDS), "measurement_distribution": "iid Gaussian N(0, 1/m)", "primary_solver": SOLVER, "retry_solver": RETRY_SOLVER, "solver_tolerances": {"CLARABEL": {"tol_gap_abs": SOLVER_TOLERANCE, "tol_feas": SOLVER_TOLERANCE}, "SCS": {"eps": SOLVER_TOLERANCE}}, "success_thresholds": {"relative_error": 1e-5, "relative_state": 1e-5, "syndrome_residual": 1e-7}, "noise_aware_recovery_threshold": NOISE_AWARE_RECOVERY_THRESHOLD, "trials": trials, "summary": summary, "monotonicity_violations": _monotonicity_violations(trials), "failed_trial_audit": _retry_audit(trials), "threshold_sensitivity": _threshold_sensitivity(trials), "redundancy": _redundancy(summary), "quantization": {"trials": quantization, "summary": _noisy_summary(quantization)}, "additive_noise": {"trials": additive_noise, "summary": _noisy_summary(additive_noise)}, "finite_size_trend": {"trials": scaling_trials, "summary": scaling_summary}, "controls": run_controls(),
    }
    output = RESULTS_DIR / "synthetic_sweep.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("rank  m    success rate  redundancy  theory reference")
    for row in summary:
        print(f"{row['rank']:>4}  {row['measurements']:>3}  {row['success_rate']:>11.0%}  {row['redundancy']:>10.3f}  {row['theory_dimension_reference']:>16}")
    print(f"Nonmonotone findings: {len(payload['monotonicity_violations'])}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
