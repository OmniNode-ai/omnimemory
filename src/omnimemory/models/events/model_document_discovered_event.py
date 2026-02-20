"""
Document discovered event model for the document ingestion pipeline.

Emitted by a crawler Effect when a document is seen for the first time
(no entry in ``omnimemory_crawl_state``). Consumed by the Processing
Layer (Stream B) to fetch, parse, and index the document.

Kafka topic: ``{env}.onex.evt.omnimemory.document-discovered.v1``

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §6
Ticket: OMN-2426
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.enum_context_item_type import EnumContextItemType
from omnimemory.enums.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.enum_detected_doc_type import EnumDetectedDocType


class ModelDocumentDiscoveredEvent(BaseModel):
    """Event emitted when a crawler encounters a new, previously-unseen document.

    All content is stored in blob storage; only the ``content_blob_ref``
    pointer and ``content_fingerprint`` (SHA-256) are included here.
    Raw document text is never inlined into Kafka events.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ------------------------------------------------------------------
    # Event envelope
    # ------------------------------------------------------------------

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance.",
    )
    event_type: Literal["DocumentDiscovered"] = Field(
        default="DocumentDiscovered",
        description="Event type discriminator for Kafka consumer routing.",
    )
    schema_version: Literal["v1"] = Field(
        default="v1",
        description="Schema version. Bump when the payload shape changes.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing across the pipeline.",
    )
    emitted_at_utc: datetime = Field(
        ...,
        description="UTC timestamp when this event was emitted.",
    )

    # ------------------------------------------------------------------
    # Crawler metadata
    # ------------------------------------------------------------------

    crawler_type: EnumCrawlerType = Field(
        ...,
        description="Crawler subsystem that discovered this document.",
    )
    crawl_scope: str = Field(
        ...,
        min_length=1,
        description="Scope string that bounded the crawl run, e.g. 'omninode/omnimemory'.",
    )
    trigger_source: str = Field(
        ...,
        description="Trigger mechanism that initiated the crawl (scheduled, git_hook, etc.).",
    )

    # ------------------------------------------------------------------
    # Document identity
    # ------------------------------------------------------------------

    source_ref: str = Field(
        ...,
        min_length=1,
        description="Unique document identifier: absolute path, URL, or Linear ID.",
    )
    source_type: EnumContextSourceType = Field(
        ...,
        description="Provenance category used for bootstrap tier assignment.",
    )
    source_version: str | None = Field(
        default=None,
        description="Version token from the source system: git file SHA, Linear updatedAt, or None.",
    )

    # ------------------------------------------------------------------
    # Content (blob pointer — not inline text)
    # ------------------------------------------------------------------

    content_fingerprint: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of whitespace-normalized document content.",
    )
    content_blob_ref: str = Field(
        ...,
        min_length=1,
        description="Pointer to blob storage entry holding the raw document text.",
    )
    token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token count: len(content) // 4. No model call required.",
    )

    # ------------------------------------------------------------------
    # Scope and classification
    # ------------------------------------------------------------------

    scope_ref: str = Field(
        ...,
        min_length=1,
        description="Resolved scope string: org/repo/subpath hierarchy.",
    )
    detected_doc_type: EnumDetectedDocType = Field(
        ...,
        description="Structural role of the document (CLAUDE_MD, DESIGN_DOC, etc.).",
    )
    suggested_context_item_type: EnumContextItemType = Field(
        ...,
        description="Suggested ContextItemType for root-level chunks of this document.",
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Tag set for retrieval filtering (repo, doctype, language, etc.).",
    )
    priority_hint: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Initial priority hint (0-100). Used for bootstrap tier assignment. "
            "95 for ~/.claude/CLAUDE.md, 85 for repo CLAUDE.md, etc."
        ),
    )
