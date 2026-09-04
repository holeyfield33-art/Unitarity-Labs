"""Tests for deterministic matrix measurement maps."""

import numpy as np

from unitarity_labs.ecc.linear_measurements import GaussianLinearMap, IdentityLinearMap


def test_gaussian_map_is_deterministic_and_has_expected_shape():
    first = GaussianLinearMap(rows=3, cols=4, measurements=7, seed=5)
    second = GaussianLinearMap(rows=3, cols=4, measurements=7, seed=5)
    np.testing.assert_array_equal(first.matrices, second.matrices)
    assert first.apply(np.ones((3, 4))).shape == (7,)


def test_gaussian_map_adjoint_satisfies_frobenius_identity():
    operator = GaussianLinearMap(rows=4, cols=3, measurements=8, seed=2)
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((4, 3))
    vector = rng.standard_normal(8)
    np.testing.assert_allclose(
        np.dot(operator.apply(matrix), vector),
        np.sum(matrix * operator.adjoint(vector)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_gaussian_map_rows_are_nested_up_to_measurement_normalization():
    smaller = GaussianLinearMap(rows=4, cols=3, measurements=20, seed=8)
    larger = GaussianLinearMap(rows=4, cols=3, measurements=60, seed=8)
    np.testing.assert_allclose(
        larger.matrices[: smaller.measurements],
        np.sqrt(smaller.measurements / larger.measurements) * smaller.matrices,
        rtol=1e-15,
        atol=1e-15,
    )


def test_identity_map_is_full_observation_and_self_adjoint():
    operator = IdentityLinearMap(rows=2, cols=3)
    matrix = np.arange(6, dtype=float).reshape(2, 3)
    np.testing.assert_array_equal(operator.adjoint(operator.apply(matrix)), matrix)