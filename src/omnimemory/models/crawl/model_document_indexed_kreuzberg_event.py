# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Document indexed kreuzberg event model for kreuzberg parsing pipeline.

Emitted after a document has been successfully parsed by the kreuzberg
service and the extracted text has been stored.

Related:
    - OMN-2733: Adopt kreuzberg as document parsing handler for omnimemory
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelDocumentIndexedKreuzbergEvent(BaseModel):
    """Emitted after a document is successfully parsed by kreuzberg.

    Published to: {env}.onex.evt.omnimemory.document-indexed.v1

    Extended form of the indexed event that carries kreuzberg-specific
    parsing metadata in addition to the standard event envelope.

    All fields are frozen; this event is immutable after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Standard envelope
    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    event_type: Literal["DocumentIndexedKreuzberg"] = Field(
        default="DocumentIndexedKreuzberg",
        description="Discriminator for event routing and deserialization",
    )
    schema_version: Literal["v1"] = Field(
        default="v1",
        description="Schema version for forward-compatibility",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID threaded from the originating event",
    )
    emitted_at_utc: datetime = Field(
        ...,
        description="ISO-8601 UTC timestamp when this event was emitted",
    )

    # Kreuzberg-specific document identity
    document_id: UUID = Field(
        ...,
        description=(
            "Stable document identifier computed as first 16 bytes of "
            "sha256(source_url + content_hash + parser_version)"
        ),
    )
    source_url: str = Field(
        ...,
        min_length=1,
        description="Source reference (path or URL) from the triggering event",
    )
    content_hash: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex digest of the document content (from triggering event)",
    )

    # Kreuzberg parsing output
    extracted_text_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "Inline extracted text if len < inline_text_max_chars, "
            "otherwise a file:// pointer to the stored text file"
        ),
    )
    mime_type: str = Field(
        ...,
        min_length=1,
        description="MIME type of the parsed document",
    )
    parser_version: str = Field(
        ...,
        min_length=1,
        description="Semver string of the kreuzberg parser version used",
    )
    parse_status: Literal["ok"] = Field(
        default="ok",
        description="Parse outcome — always 'ok' for this event type",
    )

    @field_validator("emitted_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
