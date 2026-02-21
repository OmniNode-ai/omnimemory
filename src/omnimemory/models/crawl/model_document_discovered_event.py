# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Document discovered event model for document ingestion pipeline."""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.crawl.types import TriggerSource


class ModelDocumentDiscoveredEvent(BaseModel):
    """Emitted when a new document is found during crawl with no prior state.

    Published to: {env}.onex.evt.omnimemory.document-discovered.v1

    All fields are frozen; this event is immutable after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    event_type: Literal["DocumentDiscovered"] = Field(
        default="DocumentDiscovered",
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
        description="Crawler that discovered this document",
    )
    crawl_scope: str = Field(
        ...,
        description="Scope string from the originating crawl tick (e.g. 'omninode/omnimemory')",
    )
    trigger_source: TriggerSource = Field(
        ...,
        description="What triggered the crawl: scheduled, manual, git_hook, or filesystem_watch",
    )

    # Document identity
    source_ref: str = Field(
        ...,
        description="Absolute path, URL, or Linear ID that uniquely identifies this document",
    )
    source_type: EnumContextSourceType = Field(
        ...,
        description="Authoritative source classification for bootstrap tier assignment",
    )
    source_version: str | None = Field(
        default=None,
        description="Git SHA, Linear updatedAt, or None for unversioned filesystem files",
    )

    # Content fingerprint (content stored in blob store, not inline)
    content_fingerprint: str = Field(
        ...,
        description="SHA-256 hex digest of the raw document content",
    )
    content_blob_ref: str = Field(
        ...,
        description=(
            "Pointer to blob storage entry containing the raw content. "
            "Format: 'sha256:<hex>' for content-addressed storage"
        ),
    )
    token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token count computed as len(content) // 4",
    )

    # Scope and classification
    scope_ref: str = Field(
        ...,
        description=(
            "Resolved scope assignment for this document "
            "(e.g. 'omninode/omnimemory')"
        ),
    )
    detected_doc_type: EnumDetectedDocType = Field(
        ...,
        description="Document type classification for chunking strategy selection",
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Classification tags including file path, repo name, doc type, "
            "and language tags"
        ),
    )
    priority_hint: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Initial priority score 0-100 based on source pattern. "
            "The scoring system adjusts this over time"
        ),
    )

    @field_validator("emitted_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
