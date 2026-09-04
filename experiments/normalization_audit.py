"""Paired audit separating measurement information from row rescaling."""

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
    FixedScaleNestedGaussianLinearMap,
    GaussianLinearMap,
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
QUANTIZATION_DTYPES = ("float64", "float32", "float16")
SOLVER = "CLARABEL"
TOLERANCE = 1e-8
RESULTS = Path(__file__).resolve().parents[1] / "results" / "rank_syndrome" / "normalization_audit.json"

CONDITIONS = {
    "m_normalized": GaussianLinearMap,
    "fixed_scale_nested": FixedScaleNestedGaussianLinearMap,
}


def _family(seed: int) -> str:
    return ("gaussian", "structured_full_rank", "low_rank")[seed % 3]


def _trial(condition: str, rank: int, measurements: int, seed: int) -> dict[str, object]:
    operator = CONDITIONS[condition](ROWS, COLS, measurements, seed)
    clean = generate_clean_state(ROWS, COLS, _family(seed), 10_000 + seed)
    error = generate_rank_error(ROWS, COLS, rank, 20_000 + seed)
    syndrome = operator.apply(clean + error) - operator.apply(clean)
    estimate, diagnostics = decode_nuclear_norm_with_diagnostics(
        operator, syndrome, solver=SOLVER, feasibility_tolerance=TOLERANCE
    )
    metrics = asdict(recovery_metrics(operator, error, estimate, clean, syndrome))
    row_norms = np.linalg.norm(operator.matrices.reshape(measurements, -1), axis=1)
    return {
        "condition": condition,
        "rank": rank,
        "measurements": measurements,
        "seed": seed,
        "clean_family": _family(seed),
        "success": metrics["success"],
        "syndrome_norm": float(np.linalg.norm(syndrome)),
        "mean_operator_row_frobenius_norm": float(np.mean(row_norms)),
        "solver_dual_residual": None,
        "solver_dual_residual_note": "CLARABEL CVXPY interface does not expose a dual residual.",
        "solver_diagnostics": asdict(diagnostics),
        **metrics,
    }


def _summary(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        groups[(str(trial["condition"]), int(trial["rank"]), int(trial["measurements"]))].append(trial)
    return [
        {
            "condition": condition,
            "rank": rank,
            "measurements": measurements,
            "successes": sum(bool(item["success"]) for item in group),
            "trials": len(group),
            "success_rate": sum(bool(item["success"]) for item in group) / len(group),
            "median_syndrome_norm": float(np.median([item["syndrome_norm"] for item in group])),
            "median_operator_row_frobenius_norm": float(np.median([item["mean_operator_row_frobenius_norm"] for item in group])),
            "median_primal_residual": float(np.median([item["solver_diagnostics"]["primal_residual"] for item in group])),
            "median_relative_syndrome_residual": float(np.median([item["relative_syndrome_residual"] for item in group])),
        }
        for (condition, rank, measurements), group in sorted(groups.items())
    ]


def _nonmonotone(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        groups[(str(trial["condition"]), int(trial["rank"]), int(trial["seed"]))].append(trial)
    findings = []
    for (condition, rank, seed), group in groups.items():
        group.sort(key=lambda item: int(item["measurements"]))
        for earlier in group:
            for later in group:
                if (
                    int(later["measurements"]) > int(earlier["measurements"])
                    and bool(earlier["success"])
                    and not bool(later["success"])
                ):
                    findings.append({
                        "condition": condition,
                        "rank": rank,
                        "seed": seed,
                        "success_measurements": earlier["measurements"],
                        "failure_measurements": later["measurements"],
                        "finding": "NONMONOTONE_RECOVERY",
                    })
    return findings


def _quantization(seed: int, dtype: str) -> dict[str, object]:
    operator = FixedScaleNestedGaussianLinearMap(ROWS, COLS, 100, seed)
    clean = generate_clean_state(ROWS, COLS, _family(seed), 10_000 + seed)
    error = generate_rank_error(ROWS, COLS, 1, 20_000 + seed)
    parity = operator.apply(clean)
    stored_parity = parity.astype(dtype).astype(float)
    syndrome = operator.apply(clean + error) - stored_parity
    noise_bound = max(float(np.linalg.norm(syndrome - operator.apply(error))), TOLERANCE)
    estimate = decode_nuclear_norm_noisy(operator, syndrome, noise_bound, solver=SOLVER)
    metrics = asdict(recovery_metrics(operator, error, estimate, clean, syndrome))
    return {"precision": dtype, "seed": seed, "noise_bound": noise_bound, **metrics}


def _quantization_summary(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "precision": dtype,
            "trials": len(group),
            "median_relative_error_recovery": float(np.median([item["relative_error_recovery"] for item in group])),
            "median_relative_state_recovery": float(np.median([item["relative_state_recovery"] for item in group])),
            "median_relative_syndrome_residual": float(np.median([item["relative_syndrome_residual"] for item in group])),
        }
        for dtype in QUANTIZATION_DTYPES
        for group in [[item for item in trials if item["precision"] == dtype]]
    ]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    trials = [_trial(condition, rank, measurements, seed) for condition in CONDITIONS for rank in RANKS for measurements in MEASUREMENTS for seed in SEEDS]
    findings = _nonmonotone(trials)
    quantization = [_quantization(seed, dtype) for dtype in QUANTIZATION_DTYPES for seed in range(10)]
    payload = {
        "evidence": "MEASURED",
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "dependencies": {"numpy": np.__version__, "cvxpy": cp.__version__},
        "dimensions": {"rows": ROWS, "cols": COLS},
        "measurement_counts": list(MEASUREMENTS),
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
        "solver": SOLVER,
        "solver_tolerance": TOLERANCE,
        "conditions": {
            "m_normalized": "G_i / sqrt(m)",
            "fixed_scale_nested": "G_i / sqrt(dT)",
        },
        "trials": trials,
        "summary": _summary(trials),
        "nonmonotonicity": findings,
        "nonmonotonicity_counts": {condition: sum(item["condition"] == condition for item in findings) for condition in CONDITIONS},
        "fixed_scale_quantization": {"trials": quantization, "summary": _quantization_summary(quantization)},
    }
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("condition             rank  m    successes")
    for row in payload["summary"]:
        print(f"{row['condition']:<21} {row['rank']:>4} {row['measurements']:>3} {row['successes']:>3}/{row['trials']}")
    print(f"Nonmonotonicity counts: {payload['nonmonotonicity_counts']}")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()