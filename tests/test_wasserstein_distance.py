"""Tests for unitarity_labs.statistics.wasserstein_distance.wasserstein_spectrum_distance."""

import math

import numpy as np
import pytest

from unitarity_labs.statistics.wasserstein_distance import wasserstein_spectrum_distance


def test_identical_spectra_gives_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert wasserstein_spectrum_distance(a, a) == 0.0


def test_hand_checkable_example():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    # ascending sort: a=[1,2,3], b=[1,2,4]; diffs=[0,0,-1]
    # W2 = sqrt(mean([0, 0, 1])) = sqrt(1/3)
    expected = math.sqrt(1.0 / 3.0)
    assert wasserstein_spectrum_distance(a, b) == pytest.approx(expected)


def test_hand_checkable_unequal_lengths():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    # k = 2; top_a (ascending, last 2) = [1, 2]; top_b (ascending, last 2) = [2, 3]
    # diffs = [-1, -1] -> W2 = sqrt(mean([1, 1])) = 1.0
    assert wasserstein_spectrum_distance(a, b) == pytest.approx(1.0)


def test_rejects_non_1d_input():
    with pytest.raises(ValueError):
        wasserstein_spectrum_distance(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        wasserstein_spectrum_distance(np.array([]), np.array([1.0]))


def test_rejects_nan():
    with pytest.raises(ValueError):
        wasserstein_spectrum_distance(np.array([1.0, float("nan")]), np.array([1.0, 2.0]))


def test_rejects_inf():
    with pytest.raises(ValueError):
        wasserstein_spectrum_distance(np.array([1.0, float("inf")]), np.array([1.0, 2.0]))
