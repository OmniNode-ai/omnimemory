"""
Memory data content model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .model_memory_data_value import ModelMemoryDataValue


class ModelMemoryDataContent(BaseModel):
    """Memory data content following ONEX standards."""

    content_id: UUID = Field(
        description="Unique identifier for this data content",
    )
    primary_data: ModelMemoryDataValue = Field(
        description="Primary data value",
    )
    metadata: dict[str, ModelMemoryDataValue] = Field(
        default_factory=dict,
        description="Additional metadata as typed data values",
    )
    relationships: dict[str, UUID] = Field(
        default_factory=dict,
        description="Relationships to other data content by UUID",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the data content",
    )
    source_system: str | None = Field(
        default=None,
        description="Source system that generated this data",
    )
    source_reference: str | None = Field(
        default=None,
        description="Reference or identifier in the source system",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the data content was created",
    )
    modified_at: datetime | None = Field(
        default=None,
        description="When the data content was last modified",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this data has been accessed",
    )
    last_accessed_at: datetime | None = Field(
        default=None,
        description="When the data was last accessed",
    )

    def add_metadata(self, key: str, value: ModelMemoryDataValue) -> None:
        """Add metadata to the data content."""
        self.metadata[key] = value
        self.modified_at = datetime.utcnow()

    def get_metadata(self, key: str) -> ModelMemoryDataValue | None:
        """Get metadata by key."""
        return self.metadata.get(key)

    def add_relationship(self, relationship_type: str, target_id: UUID) -> None:
        """Add a relationship to another data content."""
        self.relationships[relationship_type] = target_id
        self.modified_at = datetime.utcnow()

    def record_access(self) -> None:
        """Record an access to this data content."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()

    @property
    def total_size_bytes(self) -> int:
        """Calculate total size including metadata."""
        total = self.primary_data.size_bytes or 0
        for metadata_value in self.metadata.values():
            total += metadata_value.size_bytes or 0
        return total

    @property
    def is_recently_accessed(self, hours: int = 24) -> bool:
        """Check if data was accessed recently."""
        if not self.last_accessed_at:
            return False
        delta = datetime.utcnow() - self.last_accessed_at
        return delta.total_seconds() / 3600 < hours
