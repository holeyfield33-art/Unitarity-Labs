"""Controlled low-rank matrix error-correction experiments.

This package implements only the synthetic rank-syndrome construction
documented in docs/RANK_SYNDROME_ECC.md.
"""

from .linear_measurements import FixedScaleNestedGaussianLinearMap, GaussianLinearMap, IdentityLinearMap
from .rank_syndrome import (
	DecodeDiagnostics,
	RecoveryMetrics,
	decode_nuclear_norm,
	decode_nuclear_norm_noisy,
	decode_nuclear_norm_with_diagnostics,
	generate_clean_state,
	generate_rank_error,
	recovery_metrics,
)

__all__ = [
	"GaussianLinearMap",
	"FixedScaleNestedGaussianLinearMap",
	"IdentityLinearMap",
	"DecodeDiagnostics",
	"RecoveryMetrics",
	"decode_nuclear_norm",
	"decode_nuclear_norm_noisy",
	"decode_nuclear_norm_with_diagnostics",
	"generate_clean_state",
	"generate_rank_error",
	"recovery_metrics",
]
