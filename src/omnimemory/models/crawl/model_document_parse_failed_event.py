# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Document parse failed event model for kreuzberg parsing pipeline.

Emitted when kreuzberg fails to parse a document due to size limits,
request timeouts, or server-side parse errors.

Related:
    - OMN-2733: Adopt kreuzberg as document parsing handler for omnimemory
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelDocumentParseFailedEvent(BaseModel):
    """Emitted when kreuzberg fails to parse a document.

    Published to: {env}.onex.evt.omnimemory.document-parse-failed.v1

    Covers three failure modes:
        - too_large: document exceeds configured max_doc_bytes
        - timeout: kreuzberg HTTP request exceeded timeout_ms
        - parse_error: kreuzberg returned an HTTP error response

    All fields are frozen; this event is immutable after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    event_type: Literal["DocumentParseFailed"] = Field(
        default="DocumentParseFailed",
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
    error_code: Literal["too_large", "timeout", "parse_error"] = Field(
        ...,
        description="Machine-readable failure reason",
    )
    error_detail: str = Field(
        ...,
        description="Human-readable detail about the failure",
    )
    parser_version: str = Field(
        ...,
        min_length=1,
        description="Semver string of the kreuzberg parser version configured",
    )

    @field_validator("emitted_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
