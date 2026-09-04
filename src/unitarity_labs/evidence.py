"""Evidence-level labeling, per docs/EVIDENCE_LEVELS.md.

Every claim made in this codebase must carry exactly one EvidenceLevel.
No other labels are permitted.
"""

from __future__ import annotations

from enum import Enum


class EvidenceLevel(Enum):
    """The six permitted evidence labels. See docs/EVIDENCE_LEVELS.md."""

    PROVEN = "proven"
    DERIVED = "derived"
    MEASURED = "measured"
    HEURISTIC = "heuristic"
    CONJECTURAL = "conjectural"
    REJECTED = "rejected"

    @classmethod
    def from_str(cls, value: str) -> "EvidenceLevel":
        """Parse a label, case-insensitively. Raises ValueError if invalid."""
        normalized = value.strip().lower()
        for level in cls:
            if level.value == normalized:
                return level
        valid = ", ".join(level.value for level in cls)
        raise ValueError(f"{value!r} is not a valid EvidenceLevel; expected one of: {valid}")
