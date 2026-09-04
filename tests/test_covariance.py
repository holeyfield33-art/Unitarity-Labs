"""Tests for unitarity_labs.statistics.covariance."""

import numpy as np
import pytest

from unitarity_labs.statistics.covariance import (
    RollingCovariance,
    TopKEigenTracker,
    leading_eigengap,
)


# --------------------------------------------------------------------------
# RollingCovariance
# --------------------------------------------------------------------------


def test_rolling_covariance_hand_checkable_example():
    # x1 = [1, 0], x2 = [-1, 0] -> mean = [0, 0]
    # cov = (outer(x1,x1) + outer(x2,x2)) / 2 - outer(mean, mean)
    #     = [[1, 0], [0, 0]]
    rc = RollingCovariance(dim=2)
    rc.add(np.array([1.0, 0.0]))
    rc.add(np.array([-1.0, 0.0]))
    np.testing.assert_allclose(rc.covariance(), [[1.0, 0.0], [0.0, 0.0]])
    assert rc.n == 2


def test_rolling_covariance_remove_reverses_add():
    rc = RollingCovariance(dim=2)
    x1, x2 = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    rc.add(x1)
    rc.add(x2)
    rc.remove(x2)
    np.testing.assert_allclose(rc.covariance(), [[0.0, 0.0], [0.0, 0.0]])
    assert rc.n == 1


def test_rolling_covariance_empty_raises():
    rc = RollingCovariance(dim=2)
    with pytest.raises(ValueError):
        rc.covariance()


def test_rolling_covariance_remove_below_empty_raises():
    rc = RollingCovariance(dim=2)
    with pytest.raises(ValueError):
        rc.remove(np.array([1.0, 0.0]))


def test_rolling_covariance_rejects_wrong_shape():
    rc = RollingCovariance(dim=2)
    with pytest.raises(ValueError):
        rc.add(np.array([1.0, 0.0, 0.0]))


def test_rolling_covariance_rejects_nan():
    rc = RollingCovariance(dim=2)
    with pytest.raises(ValueError):
        rc.add(np.array([1.0, float("nan")]))


def test_rolling_covariance_rejects_dim_lt_1():
    with pytest.raises(ValueError):
        RollingCovariance(dim=0)


# --------------------------------------------------------------------------
# TopKEigenTracker
# --------------------------------------------------------------------------


def test_top_k_eigen_tracker_matches_direct_eigh():
    # recalc_period=1 forces the full-decomposition path on every step, so
    # the result must match a direct numpy.linalg.eigh on the same
    # covariance exactly (the subspace-iteration path is only approximate).
    dim, k = 3, 2
    rng = np.random.default_rng(0)
    vectors = [rng.standard_normal(dim) for _ in range(5)]

    tracker = TopKEigenTracker(dim=dim, k=k, recalc_period=1)
    rc = RollingCovariance(dim=dim)
    for v in vectors:
        tracker.add(v)
        rc.add(v)
    result = tracker.update_spectrum()

    expected_all, _ = np.linalg.eigh(rc.covariance())
    expected_top_k = np.sort(expected_all)[-k:]

    assert result.shape == (k,)
    np.testing.assert_allclose(np.sort(result), expected_top_k, atol=1e-8)


def test_top_k_eigen_tracker_subspace_update_path_runs():
    # With the default recalc_period=100 and > 2 observations, later steps
    # use the subspace-iteration path rather than full recomputation.
    # Exercise it and check the eigenvalues stay finite and sorted.
    dim, k = 4, 2
    rng = np.random.default_rng(1)
    tracker = TopKEigenTracker(dim=dim, k=k)
    for _ in range(5):
        tracker.add(rng.standard_normal(dim))
        result = tracker.update_spectrum()

    assert result.shape == (k,)
    assert np.all(np.isfinite(result))
    assert result[0] <= result[1]  # ascending, per update_spectrum's contract


def test_top_k_eigen_tracker_zero_observations_gives_zeros():
    tracker = TopKEigenTracker(dim=3, k=2)
    result = tracker.update_spectrum()
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_top_k_eigen_tracker_k_clipped_to_dim():
    tracker = TopKEigenTracker(dim=2, k=10)
    assert tracker.k == 2


def test_top_k_eigen_tracker_rejects_invalid_params():
    with pytest.raises(ValueError):
        TopKEigenTracker(dim=0, k=1)
    with pytest.raises(ValueError):
        TopKEigenTracker(dim=2, k=0)
    with pytest.raises(ValueError):
        TopKEigenTracker(dim=2, k=1, recalc_period=0)
    with pytest.raises(ValueError):
        TopKEigenTracker(dim=2, k=1, oversample=-1)


# --------------------------------------------------------------------------
# leading_eigengap
# --------------------------------------------------------------------------


def test_leading_eigengap_hand_checkable_example():
    assert leading_eigengap(np.array([1.0, 5.0, 3.0])) == pytest.approx(2.0)


def test_leading_eigengap_requires_two_values():
    with pytest.raises(ValueError):
        leading_eigengap(np.array([1.0]))


def test_leading_eigengap_rejects_nan():
    with pytest.raises(ValueError):
        leading_eigengap(np.array([1.0, float("nan")]))


def test_leading_eigengap_rejects_non_1d():
    with pytest.raises(ValueError):
        leading_eigengap(np.array([[1.0, 2.0]]))
