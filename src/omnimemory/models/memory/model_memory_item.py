"""
Memory item model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ...enums import EnumMemoryItemType, EnumMemoryStorageType


class ModelMemoryItem(BaseModel):
    """A single memory item in the ONEX memory system."""

    # Item identification
    item_id: UUID = Field(
        description="Unique identifier for the memory item",
    )
    item_type: EnumMemoryItemType = Field(
        description="Type or category of the memory item using standardized enum values",
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

    # Validation using Pydantic v2 syntax
    @field_validator("content")
    @classmethod
    def validate_content_size(cls, v: str) -> str:
        """Validate content size to prevent oversized memory items."""
        MAX_CONTENT_SIZE = 1_000_000  # 1MB max content size
        if len(v.encode("utf-8")) > MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content exceeds maximum size of {MAX_CONTENT_SIZE} bytes"
            )
        return v

    @field_validator("title")
    @classmethod
    def validate_title_length(cls, v: str | None) -> str | None:
        """Validate title length for reasonable limits."""
        if v is not None:
            MAX_TITLE_LENGTH = 500
            if len(v) > MAX_TITLE_LENGTH:
                raise ValueError(
                    f"Title exceeds maximum length of " f"{MAX_TITLE_LENGTH} characters"
                )
        return v

    @field_validator("tags", "keywords")
    @classmethod
    def validate_tag_limits(cls, v: list[str]) -> list[str]:
        """Validate tag and keyword limits to prevent abuse."""
        MAX_TAGS = 100
        MAX_TAG_LENGTH = 100
        if len(v) > MAX_TAGS:
            raise ValueError(f"Cannot have more than {MAX_TAGS} tags/keywords")
        for tag in v:
            if len(tag) > MAX_TAG_LENGTH:
                raise ValueError(
                    f"Tag '{tag}' exceeds maximum length of "
                    f"{MAX_TAG_LENGTH} characters"
                )
        return v
