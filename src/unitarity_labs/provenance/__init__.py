"""Provenance record schema, per docs/CODEBOOK_PROVENANCE.md.

Nothing may be imported from a prior codebook into this repository without
a ProvenanceRecord. A record is invalid (see ProvenanceRecord.validate)
unless source_commit_sha is an exact, known commit hash — provenance may
not be claimed against a branch name, "latest", or an unknown SHA.
"""

from .record import ProvenanceRecord

__all__ = ["ProvenanceRecord"]
