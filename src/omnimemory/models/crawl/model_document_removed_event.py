# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Document removed event model for document ingestion pipeline."""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.models.crawl.types import TriggerSource


class ModelDocumentRemovedEvent(BaseModel):
    """Emitted when a document in the crawl state table is no longer found.

    Downstream processors (DocStalenessDetectorEffect, Stream C) use this
    event to immediately blacklist the corresponding ContextItems.

    Published to: {env}.onex.evt.omnimemory.document-removed.v1

    All fields are frozen; this event is immutable after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    event_type: Literal["DocumentRemoved"] = Field(
        default="DocumentRemoved",
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

    crawler_type: EnumCrawlerType = Field(
        ...,
        description="Crawler that detected the removal",
    )
    crawl_scope: str = Field(
        ...,
        min_length=1,
        description="Scope string from the originating crawl tick (e.g. 'omninode/omnimemory')",
    )
    trigger_source: TriggerSource = Field(
        ...,
        description="What triggered the crawl: scheduled, manual, git_hook, or filesystem_watch",
    )
    source_ref: str = Field(
        ...,
        min_length=1,
        description="Absolute path, URL, or Linear ID of the removed document",
    )
    source_type: EnumContextSourceType = Field(
        ...,
        description="Authoritative source classification of the removed document",
    )
    scope_ref: str = Field(
        ...,
        min_length=1,
        description="Scope assignment of the removed document",
    )
    last_known_content_fingerprint: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex digest of the content at the last successful crawl",
    )
    last_known_source_version: str | None = Field(
        default=None,
        description="Version identifier at the last successful crawl",
    )

    @field_validator("emitted_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
