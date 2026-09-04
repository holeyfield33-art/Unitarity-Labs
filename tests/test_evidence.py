"""EvidenceLevel enum and validation, per docs/EVIDENCE_LEVELS.md."""

import pytest

from unitarity_labs.evidence import EvidenceLevel


def test_exactly_six_levels():
    assert {level.value for level in EvidenceLevel} == {
        "proven",
        "derived",
        "measured",
        "heuristic",
        "conjectural",
        "rejected",
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("measured", EvidenceLevel.MEASURED),
        ("Measured", EvidenceLevel.MEASURED),
        ("  CONJECTURAL  ", EvidenceLevel.CONJECTURAL),
        ("rejected", EvidenceLevel.REJECTED),
    ],
)
def test_from_str_accepts_known_labels(text, expected):
    assert EvidenceLevel.from_str(text) is expected


def test_from_str_rejects_unknown_label():
    with pytest.raises(ValueError):
        EvidenceLevel.from_str("definitely-true")
