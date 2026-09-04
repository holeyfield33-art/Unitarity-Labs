"""Tests for unitarity_labs.statistics.change_detection.MedianMADChangeDetector."""

import pytest

from unitarity_labs.statistics.change_detection import (
    ChangeState,
    MedianMADChangeDetector,
)

# Calibration data with nonzero MAD, chosen so the hand-computed threshold
# is exact: sorted [1, 2, 2, 3, 4] -> median = 2; abs devs sorted
# [0, 1, 1, 1, 2] -> MAD = 1 -> scale = 1.4826; change_sigma = 3.0 ->
# threshold = 2 + 3 * 1.4826 = 6.4478
CALIB_DATA = [1.0, 3.0, 2.0, 4.0, 2.0]
EXPECTED_MEDIAN = 2.0
EXPECTED_THRESHOLD = 2.0 + 3.0 * 1.4826


def _fresh_detector():
    return MedianMADChangeDetector(warmup=0, calib_len=5, change_sigma=3.0, hysteresis=2)


def test_calibration_baseline_hand_checkable():
    det = _fresh_detector()
    for v in CALIB_DATA:
        det.update(v)

    assert det.calibrated is True
    assert det.baseline_median == pytest.approx(EXPECTED_MEDIAN)
    assert det.threshold == pytest.approx(EXPECTED_THRESHOLD)


def test_pre_calibration_never_fires_even_on_extreme_value():
    det = _fresh_detector()
    # An extreme outlier as the very first (still-calibrating) sample must
    # not produce a confident verdict.
    assert det.update(1_000_000.0) is None
    assert det.calibrated is False
    assert det.state is ChangeState.CALIBRATING


def test_stationary_control_remains_quiet():
    det = _fresh_detector()
    for v in CALIB_DATA:
        det.update(v)
    assert det.calibrated is True

    # Feed more samples drawn from the same (stationary) distribution used
    # to calibrate; none should exceed the threshold.
    events = []
    for _ in range(10):
        for v in CALIB_DATA:
            ev = det.update(v)
            if ev is not None:
                events.append(ev)

    assert events == []
    assert det.state is ChangeState.QUIET
    assert det.n_changes_detected == 0


def test_injected_shift_produces_change_detected():
    det = _fresh_detector()
    for v in CALIB_DATA:
        det.update(v)
    assert det.calibrated is True

    # Inject a sustained shift well above threshold; hysteresis=2 means the
    # event commits on the 2nd consecutive elevated sample.
    assert det.update(100.0) is None  # 1st elevated sample, hysteresis not yet met
    assert det.update(100.0) == "change_detected"
    assert det.state is ChangeState.CHANGED
    assert det.n_changes_detected == 1

    # Return to baseline; hysteresis=2 -> clears on the 2nd consecutive
    # below-threshold sample.
    assert det.update(2.0) is None
    assert det.update(2.0) == "change_cleared"
    assert det.state is ChangeState.QUIET
    assert det.n_changes_cleared == 1


def test_degenerate_constant_calibration_never_fires():
    det = MedianMADChangeDetector(warmup=0, calib_len=5, change_sigma=3.0, hysteresis=1)
    for _ in range(5):
        det.update(1.0)
    assert det.calibrated is False  # threshold is +inf -> never "calibrated"
    assert det.threshold == float("inf")
    assert det.update(1_000_000.0) is None


def test_warmup_samples_excluded_from_calibration():
    det = MedianMADChangeDetector(warmup=2, calib_len=5, change_sigma=3.0, hysteresis=1)
    # first 2 samples are warmup (ignored), next 5 are the real calibration data
    for v in [999.0, -999.0] + CALIB_DATA:
        det.update(v)
    assert det.baseline_median == pytest.approx(EXPECTED_MEDIAN)


def test_rejects_non_scalar_value():
    det = _fresh_detector()
    with pytest.raises(ValueError):
        det.update([1.0, 2.0])  # type: ignore[arg-type]


def test_rejects_nan():
    det = _fresh_detector()
    with pytest.raises(ValueError):
        det.update(float("nan"))


def test_rejects_inf():
    det = _fresh_detector()
    with pytest.raises(ValueError):
        det.update(float("inf"))


def test_invalid_construction_params_raise():
    with pytest.raises(ValueError):
        MedianMADChangeDetector(warmup=-1)
    with pytest.raises(ValueError):
        MedianMADChangeDetector(calib_len=1)
    with pytest.raises(ValueError):
        MedianMADChangeDetector(hysteresis=0)
    with pytest.raises(ValueError):
        MedianMADChangeDetector(change_sigma=0.0)
