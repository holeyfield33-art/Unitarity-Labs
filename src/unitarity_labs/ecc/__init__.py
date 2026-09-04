"""Controlled low-rank matrix error-correction experiments.

This package implements only the synthetic rank-syndrome construction
documented in docs/RANK_SYNDROME_ECC.md.
"""

from .linear_measurements import GaussianLinearMap, IdentityLinearMap
from .rank_syndrome import (
	RecoveryMetrics,
	decode_nuclear_norm,
	generate_clean_state,
	generate_rank_error,
	recovery_metrics,
)

__all__ = [
	"GaussianLinearMap",
	"IdentityLinearMap",
	"RecoveryMetrics",
	"decode_nuclear_norm",
	"generate_clean_state",
	"generate_rank_error",
	"recovery_metrics",
]
