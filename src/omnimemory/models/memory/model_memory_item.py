"""
Memory item model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .enum_memory_storage_type import EnumMemoryStorageType


class ModelMemoryItem(BaseModel):
    """A single memory item in the ONEX memory system."""

    # Item identification
    item_id: UUID = Field(
        description="Unique identifier for the memory item",
    )
    item_type: str = Field(
        description="Type or category of the memory item",
    )

    # Content
    content: str = Field(
        description="Main content of the memory item",
    )
    title: str | None = Field(
        default=None,
        description="Optional title for the memory item",
    )
    summary: str | None = Field(
        default=None,
        description="Optional summary of the content",
    )

    # Metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the memory item",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for search and indexing",
    )

    # Storage information
    storage_type: EnumMemoryStorageType = Field(
        description="Type of storage where this item is stored",
    )
    storage_location: str = Field(
        description="Location identifier within the storage system",
    )

    # Versioning
    version: int = Field(
        default=1,
        description="Version number of the memory item",
    )
    previous_version_id: UUID | None = Field(
        default=None,
        description="ID of the previous version if this is an update",
    )

    # Temporal information
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory item was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When the memory item was last updated",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the memory item expires (optional)",
    )

    # Usage tracking
    access_count: int = Field(
        default=0,
        description="Number of times this item has been accessed",
    )
    last_accessed_at: datetime | None = Field(
        default=None,
        description="When the memory item was last accessed",
    )

    # Quality indicators
    importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score for prioritization",
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance score for search ranking",
    )
    quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality score based on content analysis",
    )

    # Relationships
    parent_item_id: UUID | None = Field(
        default=None,
        description="ID of parent item if this is part of a hierarchy",
    )
    related_item_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of related memory items",
    )

    # Processing status
    processing_complete: bool = Field(
        default=True,
        description="Whether processing of this item is complete",
    )
    indexed: bool = Field(
        default=False,
        description="Whether this item has been indexed for search",
    )