"""Tests for unitarity_labs.statistics.gap_ratio.mean_gap_ratio."""

import numpy as np
import pytest

from unitarity_labs.statistics.gap_ratio import mean_gap_ratio


def test_hand_checkable_example():
    # eigenvalues [0, 1, 2, 4] -> gaps [1, 1, 2]
    # ratios: min(1,1)/max(1,1)=1.0, min(1,2)/max(1,2)=0.5 -> mean 0.75
    assert mean_gap_ratio(np.array([0.0, 1.0, 2.0, 4.0])) == pytest.approx(0.75)


def test_unsorted_input_is_sorted_internally():
    assert mean_gap_ratio(np.array([4.0, 0.0, 2.0, 1.0])) == pytest.approx(0.75)


def test_degenerate_equal_eigenvalues_gives_zero():
    # all gaps are 0 -> ratio defined as 0 -> mean 0
    assert mean_gap_ratio(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0


def test_requires_at_least_three_eigenvalues():
    with pytest.raises(ValueError):
        mean_gap_ratio(np.array([1.0, 2.0]))


def test_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mean_gap_ratio(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_rejects_nan():
    with pytest.raises(ValueError):
        mean_gap_ratio(np.array([1.0, float("nan"), 3.0]))


def test_rejects_inf():
    with pytest.raises(ValueError):
        mean_gap_ratio(np.array([1.0, float("inf"), 3.0]))


# --------------------------------------------------------------------------
# Reference-ensemble checks: these validate the *statistic's implementation*
# against known random-matrix-theory asymptotics (Atas et al., 2013). They
# are not a claim about AI behavior or model quality -- see
# docs/EVIDENCE_LEVELS.md. mean_gap_ratio itself makes no such claim either.
# --------------------------------------------------------------------------


def test_reference_ensemble_goe_gap_ratio():
    rng = np.random.default_rng(0)
    n = 400
    a = rng.standard_normal((n, n))
    goe = (a + a.T) / 2.0
    evals = np.linalg.eigvalsh(goe)
    r = mean_gap_ratio(evals)
    assert 0.50 < r < 0.56  # reference: ~0.5307


def test_reference_ensemble_gue_gap_ratio():
    rng = np.random.default_rng(0)
    n = 400
    re = rng.standard_normal((n, n))
    im = rng.standard_normal((n, n))
    a = re + 1j * im
    gue = (a + a.conj().T) / 2.0
    evals = np.linalg.eigvalsh(gue)
    r = mean_gap_ratio(evals)
    assert 0.56 < r < 0.64  # reference: ~0.5996


def test_reference_ensemble_poisson_gap_ratio():
    rng = np.random.default_rng(0)
    evals = rng.uniform(size=4000)
    r = mean_gap_ratio(evals)
    assert 0.35 < r < 0.42  # reference: ~0.3863
