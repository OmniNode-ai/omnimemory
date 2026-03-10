# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Document indexed event model for document ingestion pipeline.

Emitted after a document has been fully crawled and indexed (either
discovered for the first time or updated after content change).

Consumed by omniintelligence ``node_crawl_scheduler_effect`` to reset
the per-document debounce window, confirming crawl completion.

Related:
    - OMN-2717: CONTRACT_DRIFT gap fix — add document-indexed.v1 producer
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.models.crawl.types import TriggerSource


class ModelDocumentIndexedEvent(BaseModel):
    """Emitted after a document is fully crawled and written to the index.

    Published to: {env}.onex.evt.omnimemory.document-indexed.v1

    Consumed by ``node_crawl_scheduler_effect`` in omniintelligence to
    reset the per-``(source_ref, crawler_type)`` debounce window after a
    successful crawl completion.

    Emitted for both newly-discovered and content-changed documents.
    Not emitted for mtime-only touches (no content change) or removals.

    All fields are frozen; this event is immutable after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    event_type: Literal["DocumentIndexed"] = Field(
        default="DocumentIndexed",
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
        description="Crawler that indexed this document",
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

    # Document identity
    source_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "Absolute path, URL, or Linear ID that uniquely identifies this document. "
            "Used by omniintelligence crawl_scheduler_effect to clear the debounce window."
        ),
    )
    source_type: EnumContextSourceType = Field(
        ...,
        description="Authoritative source classification",
    )

    # Content fingerprint
    content_fingerprint: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex digest of the indexed document content",
    )

    # Scope
    scope_ref: str = Field(
        ...,
        min_length=1,
        description="Resolved scope assignment for this document (e.g. 'omninode/omnimemory')",
    )

    @field_validator("emitted_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
