"""ProvenanceRecord: the schema required for any code imported from a
prior codebook (docs/CODEBOOK_PROVENANCE.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from unitarity_labs.evidence import EvidenceLevel

# The prior codebooks this repository is a research reset from. Not a fork
# of, and not bulk-copied from, any of these — see docs/CODEBOOK_PROVENANCE.md.
KNOWN_SOURCE_REPOS = (
    "holeyfield33-art/unitarity-lab",
    "holeyfield33-art/geometric-brain-mcp",
    "holeyfield33-art/VAR",
    "holeyfield33-art/insideai",
)

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ProvenanceRecord:
    """A record of one implementation imported from a prior codebook.

    All fields are required. source_commit_sha must be an exact, full
    40-character git commit hash: provenance may not be claimed against a
    branch name, tag, "latest", or an otherwise unknown SHA.
    """

    source_repo: str
    source_path: str
    source_commit_sha: str
    original_purpose: str
    retained: str
    removed: str
    retained_reason: str
    validation_status: EvidenceLevel

    def validate(self) -> None:
        """Raise ValueError if this record does not meet the provenance bar."""
        required_text_fields = {
            "source_repo": self.source_repo,
            "source_path": self.source_path,
            "original_purpose": self.original_purpose,
            "retained": self.retained,
            "removed": self.removed,
            "retained_reason": self.retained_reason,
        }
        for name, value in required_text_fields.items():
            if not value or not value.strip():
                raise ValueError(f"ProvenanceRecord.{name} must not be empty")

        if not _FULL_SHA_RE.match(self.source_commit_sha):
            raise ValueError(
                "ProvenanceRecord.source_commit_sha must be an exact 40-character "
                f"git commit SHA; got {self.source_commit_sha!r}. Provenance may not "
                "be claimed unless the exact source SHA is known."
            )

        if not isinstance(self.validation_status, EvidenceLevel):
            raise ValueError("ProvenanceRecord.validation_status must be an EvidenceLevel")
