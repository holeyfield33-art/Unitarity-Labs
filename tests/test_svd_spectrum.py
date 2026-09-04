"""Tests for unitarity_labs.statistics.svd_spectrum.singular_value_spectrum."""

import numpy as np
import pytest

from unitarity_labs.statistics.svd_spectrum import singular_value_spectrum


def test_identity_matrix_gives_unit_singular_values():
    result = singular_value_spectrum(np.eye(3))
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


def test_diagonal_matrix_sorted_descending():
    # diag(3, 1, 2) -> singular values sorted descending [3, 2, 1]
    matrix = np.diag([3.0, 1.0, 2.0])
    result = singular_value_spectrum(matrix)
    np.testing.assert_allclose(result, [3.0, 2.0, 1.0])


def test_rectangular_matrix_returns_min_dim_values():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = singular_value_spectrum(matrix)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [2.0, 1.0])


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        singular_value_spectrum(np.array([1.0, 2.0, 3.0]))


def test_rejects_empty_matrix():
    with pytest.raises(ValueError):
        singular_value_spectrum(np.zeros((0, 0)))


def test_rejects_nan():
    matrix = np.eye(2)
    matrix[0, 0] = float("nan")
    with pytest.raises(ValueError):
        singular_value_spectrum(matrix)


def test_rejects_inf():
    matrix = np.eye(2)
    matrix[0, 0] = float("inf")
    with pytest.raises(ValueError):
        singular_value_spectrum(matrix)
