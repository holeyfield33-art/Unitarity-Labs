"""Tests for controlled rank-syndrome ECC components."""

import numpy as np
import pytest

from unitarity_labs.ecc.linear_measurements import IdentityLinearMap
from unitarity_labs.ecc.rank_syndrome import (
    decode_nuclear_norm,
    decode_nuclear_norm_noisy,
    decode_nuclear_norm_with_diagnostics,
    generate_clean_state,
    generate_rank_error,
    numerical_rank,
    recovery_metrics,
)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_seeded_error_has_exact_requested_rank(rank):
    assert numerical_rank(generate_rank_error(7, 8, rank, seed=4)) == rank


def test_clean_state_families_do_not_control_error_generation():
    assert numerical_rank(generate_clean_state(6, 6, "gaussian", seed=0)) == 6
    assert numerical_rank(generate_clean_state(6, 6, "structured_full_rank", seed=0)) == 6
    assert numerical_rank(generate_clean_state(6, 6, "low_rank", seed=0)) == 1


def test_full_observation_decoder_recovers_error():
    operator = IdentityLinearMap(rows=3, cols=3)
    error = generate_rank_error(3, 3, rank=1, seed=9)
    clean_state = generate_clean_state(3, 3, "gaussian", seed=10)
    syndrome = operator.apply(error)
    estimate = decode_nuclear_norm(operator, syndrome)
    metrics = recovery_metrics(operator, error, estimate, clean_state, syndrome)
    assert metrics.success


def test_exact_decoder_reports_feasibility_diagnostics():
    operator = IdentityLinearMap(rows=3, cols=3)
    error = generate_rank_error(3, 3, rank=1, seed=14)
    estimate, diagnostics = decode_nuclear_norm_with_diagnostics(operator, operator.apply(error))
    np.testing.assert_allclose(estimate, error, atol=1e-7)
    assert diagnostics.status == "optimal"
    assert diagnostics.feasibility_residual < 1e-7


def test_noisy_decoder_respects_a_known_syndrome_noise_bound():
    operator = IdentityLinearMap(rows=3, cols=3)
    error = generate_rank_error(3, 3, rank=1, seed=15)
    noise = np.full(operator.measurements, 1e-6)
    syndrome = operator.apply(error) + noise
    estimate = decode_nuclear_norm_noisy(operator, syndrome, np.linalg.norm(noise) + 1e-8)
    assert np.linalg.norm(operator.apply(estimate) - syndrome) <= np.linalg.norm(noise) + 1e-7


def test_zero_error_is_a_no_op():
    operator = IdentityLinearMap(rows=3, cols=3)
    clean_state = generate_clean_state(3, 3, "low_rank", seed=10)
    error = np.zeros((3, 3))
    metrics = recovery_metrics(operator, error, error, clean_state, operator.apply(error))
    assert metrics.success

def test_wrong_parity_is_not_reported_as_successful_recovery():
    operator = IdentityLinearMap(rows=3, cols=3)
    clean_state = generate_clean_state(3, 3, "gaussian", seed=11)
    other_clean_state = generate_clean_state(3, 3, "structured_full_rank", seed=12)
    error = generate_rank_error(3, 3, rank=1, seed=13)
    wrong_syndrome = operator.apply(clean_state + error) - operator.apply(other_clean_state)
    estimate = decode_nuclear_norm(operator, wrong_syndrome)
    metrics = recovery_metrics(operator, error, estimate, clean_state, wrong_syndrome)
    assert not metrics.success