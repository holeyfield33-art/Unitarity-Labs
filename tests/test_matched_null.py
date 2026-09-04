"""Tests for unitarity_labs.statistics.matched_null.matched_null_zscore."""

import numpy as np
import pytest

from unitarity_labs.statistics.matched_null import matched_null_zscore


def test_hand_checkable_example():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])  # cos(a, b) = 1.0
    controls = [np.array([0.0, 1.0]), np.array([-1.0, 0.0])]  # cos = 0.0, -1.0

    result = matched_null_zscore(a, b, controls)

    assert result["matched"] == pytest.approx(1.0)
    assert result["null_mean"] == pytest.approx(-0.5)
    assert result["null_std"] == pytest.approx(0.5)
    assert result["gap"] == pytest.approx(1.5)
    assert result["z_score"] == pytest.approx(3.0)
    assert result["n_controls"] == 2
    assert result["matched_len"] == 2


def test_single_control_zero_variance_nonzero_gap_gives_signed_inf():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])  # cos = 1.0
    controls = [np.array([0.0, 1.0])]  # cos = 0.0, gap = 1.0 != 0

    result = matched_null_zscore(a, b, controls)

    assert result["null_std"] == 0.0
    assert result["gap"] == pytest.approx(1.0)
    assert result["z_score"] == float("inf")


def test_zero_gap_zero_variance_gives_zero_not_inf():
    # Corrected from the source: matched == the (degenerate, single-point)
    # null exactly -> no observed difference -> z = 0.0, not +inf.
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])  # cos(a, b) = 1.0
    controls = [np.array([1.0, 0.0])]  # cos = 1.0 too -> gap = 0

    result = matched_null_zscore(a, b, controls)

    assert result["gap"] == 0.0
    assert result["z_score"] == 0.0


def test_length_matched_by_truncation_not_padding():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0])  # shorter than a; truncates a and controls to 2
    controls = [np.array([0.0, 1.0])]

    result = matched_null_zscore(a, b, controls)
    assert result["matched_len"] == 2


def test_empty_controls_raises():
    a = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        matched_null_zscore(a, a, [])


def test_zero_vector_raises():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        matched_null_zscore(a, b, [np.array([0.0, 1.0])])


def test_nan_raises():
    a = np.array([1.0, float("nan")])
    b = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        matched_null_zscore(a, b, [np.array([0.0, 1.0])])
