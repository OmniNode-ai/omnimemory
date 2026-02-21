# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Document changed event model for document ingestion pipeline."""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.crawl.types import TriggerSource


class ModelDocumentChangedEvent(BaseModel):
    """Emitted when a previously crawled document has new content.

    Extends the discovered event shape with previous-version fields so
    downstream processors can perform diff analysis, stat carry-over,
    and staleness transitions atomically.

    Published to: {env}.onex.evt.omnimemory.document-changed.v1

    All fields are frozen; this event is immutable after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    event_type: Literal["DocumentChanged"] = Field(
        default="DocumentChanged",
        description="Discriminator for event routing and deserialization",
    )
    schema_version: Literal["v1"] = Field(
        default="v1",
        description="Schema version for forward-compatibility",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID threaded from the originating crawl tick command",
    )
    emitted_at_utc: datetime = Field(
        ...,
        description="ISO-8601 UTC timestamp when this event was emitted",
    )

    # Crawler metadata
    crawler_type: EnumCrawlerType = Field(
        ...,
        description="Crawler that detected the change",
    )
    crawl_scope: str = Field(
        ...,
        description="Scope string from the originating crawl tick",
    )
    trigger_source: TriggerSource = Field(
        ...,
        description="What triggered the crawl: scheduled, manual, git_hook, or filesystem_watch",
    )

    # Document identity
    source_ref: str = Field(
        ...,
        description="Absolute path, URL, or Linear ID",
    )
    source_type: EnumContextSourceType = Field(
        ...,
        description="Authoritative source classification",
    )
    source_version: str | None = Field(
        default=None,
        description="New version identifier after the change",
    )

    # Current content
    content_fingerprint: str = Field(
        ...,
        description="SHA-256 hex digest of the new content",
    )
    content_blob_ref: str = Field(
        ...,
        description="Pointer to blob storage entry for the new content",
    )
    token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token count of the new content",
    )

    # Scope and classification
    scope_ref: str = Field(
        ...,
        description="Resolved scope assignment",
    )
    detected_doc_type: EnumDetectedDocType = Field(
        ...,
        description="Document type classification",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Classification tags",
    )
    priority_hint: int = Field(
        ...,
        ge=0,
        le=100,
        description="Initial priority score 0-100",
    )

    # Previous version fields (what changed)
    previous_content_fingerprint: str = Field(
        ...,
        description="SHA-256 hex digest of the content before the change",
    )
    previous_source_version: str | None = Field(
        default=None,
        description="Version identifier before the change (git SHA, Linear updatedAt, or None)",
    )

    @field_validator("emitted_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
