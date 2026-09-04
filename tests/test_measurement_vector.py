"""Tests for unitarity_labs.statistics.measurement_vector.MeasurementVector."""

import dataclasses

import pytest

from unitarity_labs.statistics.measurement_vector import MeasurementVector


def test_defaults_are_all_none():
    m = MeasurementVector()
    assert m.d_spec is None
    assert m.d_subspace is None
    assert m.z_null is None
    assert m.r_eff is None
    assert m.r_gap is None
    assert m.lambda_gap is None


def test_partial_population_leaves_unsupported_fields_none():
    m = MeasurementVector(r_gap=0.75, lambda_gap=2.0)
    assert m.r_gap == 0.75
    assert m.lambda_gap == 2.0
    assert m.d_spec is None
    assert m.d_subspace is None
    assert m.z_null is None
    assert m.r_eff is None


def test_is_immutable():
    m = MeasurementVector(r_gap=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.r_gap = 0.9  # type: ignore[misc]
