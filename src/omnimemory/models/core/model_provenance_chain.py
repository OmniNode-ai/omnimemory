"""
Provenance chain model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .model_provenance_entry import ModelProvenanceEntry


class ModelProvenanceChain(BaseModel):
    """Complete provenance chain for memory data following ONEX standards."""

    # Chain metadata
    chain_id: UUID = Field(
        description="Unique identifier for this provenance chain",
    )
    root_operation_id: UUID = Field(
        description="Identifier of the operation that started this chain",
    )

    # Chain entries
    entries: list[ModelProvenanceEntry] = Field(
        default_factory=list,
        description="Chronological list of provenance entries in this chain",
    )

    # Chain statistics
    total_operations: int = Field(
        default=0,
        description="Total number of operations in this chain",
    )
    chain_started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this provenance chain was started",
    )
    chain_updated_at: datetime | None = Field(
        default=None,
        description="When this provenance chain was last updated",
    )

    # Integrity verification
    chain_hash: str | None = Field(
        default=None,
        description="Hash of the entire chain for integrity verification",
    )
    verified: bool = Field(
        default=False,
        description="Whether this provenance chain has been cryptographically verified",
    )

    def add_entry(self, entry: ModelProvenanceEntry) -> None:
        """Add a new provenance entry to the chain."""
        self.entries.append(entry)
        self.total_operations = len(self.entries)
        self.chain_updated_at = datetime.utcnow()

    def get_latest_entry(self) -> ModelProvenanceEntry | None:
        """Get the most recent provenance entry."""
        return self.entries[-1] if self.entries else None

    def get_entry_by_operation_id(
        self, operation_id: UUID
    ) -> ModelProvenanceEntry | None:
        """Find a provenance entry by operation ID."""
        for entry in self.entries:
            if entry.operation_id == operation_id:
                return entry
        return None
