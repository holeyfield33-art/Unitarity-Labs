"""New in Unitarity Labs (not ported from a prior codebook).

Implements the measurement vector M = [D_spec, D_subspace, z_null, r_eff,
r_gap, lambda_gap] defined in docs/RESEARCH_THESIS.md. This phase only
populates the components backed by a ported primitive (see
docs/CODEBOOK_PROVENANCE.md); every other component is left as ``None`` --
no synthetic defaults are used, per the extraction contract for this
phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MeasurementVector:
    """Immutable measurement vector M.

    Fields map to docs/RESEARCH_THESIS.md's
    ``M = [D_spec, D_subspace, z_null, r_eff, r_gap, lambda_gap]``. A field
    is ``None`` when this phase has no ported primitive that supports it.
    A populated field is a number only -- see docs/EVIDENCE_LEVELS.md
    before assigning it any semantic interpretation.
    """

    d_spec: Optional[float] = None  # wasserstein_spectrum_distance(...)
    d_subspace: Optional[float] = None  # not yet supported by a ported primitive
    z_null: Optional[float] = None  # matched_null_zscore(...)["z_score"]
    r_eff: Optional[float] = None  # not yet supported by a ported primitive
    r_gap: Optional[float] = None  # mean_gap_ratio(...)
    lambda_gap: Optional[float] = None  # leading_eigengap(...)
