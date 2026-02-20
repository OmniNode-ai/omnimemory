"""
Document removed event model for the document ingestion pipeline.

Emitted by a crawler Effect when a known document is no longer present
in the source system (file deleted, Linear issue removed, etc.). Triggers
immediate BLACKLISTED tier assignment in ``ContextItemWriterEffect``.

Kafka topic: ``{env}.onex.evt.omnimemory.document-removed.v1``

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §6, §12
Ticket: OMN-2426
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.enum_crawler_type import EnumCrawlerType


class ModelDocumentRemovedEvent(BaseModel):
    """Event emitted when a crawler detects that a known document no longer exists.

    Per the staleness policy: FILE_DELETED triggers immediate BLACKLISTED
    at any tier. The ``last_known_content_fingerprint`` allows the writer
    to locate the correct ContextItems for blacklisting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ------------------------------------------------------------------
    # Event envelope
    # ------------------------------------------------------------------

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance.",
    )
    event_type: Literal["DocumentRemoved"] = Field(
        default="DocumentRemoved",
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
        description="Crawler subsystem that detected the removal.",
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
        description="Provenance category — used by the writer to route blacklisting.",
    )
    scope_ref: str = Field(
        ...,
        min_length=1,
        description="Scope string at the time of last crawl.",
    )

    # ------------------------------------------------------------------
    # Last known state (for blacklisting lookup)
    # ------------------------------------------------------------------

    last_known_content_fingerprint: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description=(
            "SHA-256 hex digest from the last successful crawl. "
            "Used to locate ContextItems to blacklist."
        ),
    )
    last_known_source_version: str | None = Field(
        default=None,
        description="Version token from the last successful crawl, or None.",
    )
