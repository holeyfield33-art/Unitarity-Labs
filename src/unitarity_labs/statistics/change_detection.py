"""
Source repository: holeyfield33-art/var
Source commit: 31234551e524249a5e81453ec851c98ec8836fb7
Source file: var/detector.py (SpectralRuptureDetector, RuptureState)
Original implementation: a calibrated median/MAD threshold detector with
    hysteresis, applied to a streaming scalar drift signal (W2 spectral
    drift) and framed as detecting "ruptures" against a "healthy-baseline".
Retained behavior: the full calibration and detection method -- skip a
    ``warmup`` prefix, estimate a robust baseline (median, 1.4826*MAD,
    with a std fallback for a degenerate/near-zero MAD) over the next
    ``calib_len`` samples, then flag a sample as elevated when it exceeds
    ``median + change_sigma * scale``, committing a state transition only
    after ``hysteresis`` consecutive samples on one side of the
    threshold. The ``calibrated`` / ``calib_end`` introspection and the
    "stay silent on an under-filled calibration buffer" behavior are
    retained unchanged. Explicit scalar-shape and finite-value validation
    was added to ``update``; the source did not validate its input.
Removed interpretation: "rupture" and "healthy-baseline" framing renamed
    to neutral state/event names (CALIBRATING/QUIET/CHANGED,
    "change_detected"/"change_cleared"); this module reports a
    statistically-defined change in a scalar signal, not a health
    judgment -- see docs/EVIDENCE_LEVELS.md.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class ChangeState(enum.Enum):
    CALIBRATING = "calibrating"
    QUIET = "quiet"
    CHANGED = "changed"


@dataclass
class MedianMADChangeDetector:
    """Detect a sustained departure of a streaming scalar signal from a
    calibrated median/MAD baseline.

    Parameters
    ----------
    warmup : int
        Leading samples excluded from calibration (e.g. a signal's
        window-fill transient).
    calib_len : int
        Number of post-warmup samples used to estimate the baseline.
    change_sigma : float
        Threshold, in robust-sigma units (``1.4826 * MAD``), above the
        baseline median.
    hysteresis : int
        Consecutive samples a threshold crossing must hold before a
        change_detected / change_cleared event is committed.
    """

    warmup: int = 0
    calib_len: int = 100
    change_sigma: float = 6.0
    hysteresis: int = 3

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.calib_len < 2:
            raise ValueError("calib_len must be >= 2")
        if self.hysteresis < 1:
            raise ValueError("hysteresis must be >= 1")
        if self.change_sigma <= 0:
            raise ValueError("change_sigma must be > 0")

        self._calib_buf: deque[float] = deque()
        self.state: ChangeState = ChangeState.CALIBRATING
        self.threshold: Optional[float] = None
        self.baseline_median: Optional[float] = None
        self.baseline_scale: Optional[float] = None

        self._step: int = 0
        self._consec_above: int = 0
        self._consec_below: int = 0
        self.events: List[Tuple[int, str]] = []

    @property
    def calibrated(self) -> bool:
        """True once a finite baseline threshold has been established."""
        return (
            self.state is not ChangeState.CALIBRATING
            and self.threshold is not None
            and bool(np.isfinite(self.threshold))
        )

    @property
    def calib_end(self) -> int:
        """Step index at which calibration finalises."""
        return self.warmup + self.calib_len

    def update(self, value: float) -> Optional[str]:
        """Ingest one scalar sample; return an event string or ``None``.

        Returns ``"change_detected"`` or ``"change_cleared"`` on the
        sample that commits a state transition, otherwise ``None``. While
        the detector is still calibrating, this always returns ``None``,
        regardless of the magnitude of ``value``.

        Raises
        ------
        ValueError
            If ``value`` is not a real scalar, or is NaN/Inf.
        """
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value must be a real scalar, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError("value must not be NaN or Inf.")

        step = self._step
        self._step += 1

        if self.state is ChangeState.CALIBRATING:
            if step >= self.warmup:
                self._calib_buf.append(value)
            if step + 1 >= self.calib_end:
                self._finalize_calibration()
            return None

        assert self.threshold is not None
        event: Optional[str] = None

        if value > self.threshold:
            self._consec_above += 1
            self._consec_below = 0
            if (
                self.state is ChangeState.QUIET
                and self._consec_above >= self.hysteresis
            ):
                self.state = ChangeState.CHANGED
                event = "change_detected"
                self.events.append((step, event))
        else:
            self._consec_below += 1
            self._consec_above = 0
            if (
                self.state is ChangeState.CHANGED
                and self._consec_below >= self.hysteresis
            ):
                self.state = ChangeState.QUIET
                event = "change_cleared"
                self.events.append((step, event))

        return event

    def _finalize_calibration(self) -> None:
        arr = np.asarray(self._calib_buf, dtype=float)
        if arr.size < 2:
            log.warning(
                "MedianMADChangeDetector: only %d calibration sample(s); "
                "detector will not fire.",
                int(arr.size),
            )
            self.baseline_median = float(arr.mean()) if arr.size else 0.0
            self.baseline_scale = 0.0
            self.threshold = float("inf")
        else:
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            scale = 1.4826 * mad
            if scale < 1e-12:
                scale = float(arr.std())
            self.baseline_median = median
            self.baseline_scale = scale
            if scale < 1e-12:
                self.threshold = float("inf")
            else:
                self.threshold = median + self.change_sigma * scale
        self.state = ChangeState.QUIET

    @property
    def n_changes_detected(self) -> int:
        return sum(1 for _, kind in self.events if kind == "change_detected")

    @property
    def n_changes_cleared(self) -> int:
        return sum(1 for _, kind in self.events if kind == "change_cleared")
