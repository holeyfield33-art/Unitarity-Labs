"""ProvenanceRecord schema, per docs/CODEBOOK_PROVENANCE.md."""

import pytest

from unitarity_labs.evidence import EvidenceLevel
from unitarity_labs.provenance import ProvenanceRecord

VALID_SHA = "a" * 40


def make_record(**overrides):
    fields = dict(
        source_repo="holeyfield33-art/unitarity-lab",
        source_path="src/reed_solomon/transport.py",
        source_commit_sha=VALID_SHA,
        original_purpose="Reed-Solomon transport encoding",
        retained="encode/decode functions",
        removed="unrelated CLI wrapper",
        retained_reason="known-good ECC positive control",
        validation_status=EvidenceLevel.MEASURED,
    )
    fields.update(overrides)
    return ProvenanceRecord(**fields)


def test_valid_record_passes_validation():
    make_record().validate()  # should not raise


def test_missing_text_field_is_rejected():
    with pytest.raises(ValueError):
        make_record(original_purpose="   ").validate()


@pytest.mark.parametrize(
    "bad_sha",
    [
        "main",
        "latest",
        "a" * 39,
        "a" * 41,
        "g" * 40,
        "",
    ],
)
def test_non_exact_sha_is_rejected(bad_sha):
    with pytest.raises(ValueError):
        make_record(source_commit_sha=bad_sha).validate()


def test_validation_status_must_be_evidence_level():
    with pytest.raises(ValueError):
        make_record(validation_status="measured").validate()  # type: ignore[arg-type]
