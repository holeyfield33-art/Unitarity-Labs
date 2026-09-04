"""Statistical change detection.

Passive measurement primitives extracted from prior codebooks, per
docs/CODEBOOK_PROVENANCE.md, plus the ``MeasurementVector`` result type
defined by docs/RESEARCH_THESIS.md. Every function here reports a number
or a statistically-defined change; none of them assigns a health or
quality interpretation on its own -- see docs/EVIDENCE_LEVELS.md.
"""

from .change_detection import ChangeState, MedianMADChangeDetector
from .covariance import RollingCovariance, TopKEigenTracker, leading_eigengap
from .gap_ratio import mean_gap_ratio
from .matched_null import matched_null_zscore
from .measurement_vector import MeasurementVector
from .svd_spectrum import singular_value_spectrum
from .wasserstein_distance import wasserstein_spectrum_distance

__all__ = [
    "ChangeState",
    "MedianMADChangeDetector",
    "RollingCovariance",
    "TopKEigenTracker",
    "leading_eigengap",
    "mean_gap_ratio",
    "matched_null_zscore",
    "MeasurementVector",
    "singular_value_spectrum",
    "wasserstein_spectrum_distance",
]
