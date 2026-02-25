# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Handler configuration model for KreuzbergParseEffect node."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelKreuzbergParseConfig(  # omnimemory-model-exempt: handler config
    BaseModel
):
    """Configuration for HandlerKreuzbergParse.

    All values may be sourced from environment variables or injected
    directly. Frozen to prevent accidental mutation after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kreuzberg_url: str = Field(
        default="http://localhost:8090",
        description="Base URL of the kreuzberg Docker service",
    )
    text_store_path: str = Field(
        ...,
        description="Filesystem path where extracted text files are stored (KREUZBERG_TEXT_STORE_PATH)",
    )
    document_root: str = Field(
        default="/",
        description=(
            "Root directory that source_ref paths must be confined to. "
            "Defaults to filesystem root (permissive — still prevents .. escapes from relative paths). "
            "Set to a tighter path in production (KREUZBERG_DOCUMENT_ROOT)."
        ),
    )

    @field_validator("document_root")
    @classmethod
    def validate_document_root_exists(cls, v: str) -> str:
        if not Path(v).is_dir():
            raise ValueError(
                f"document_root '{v}' does not exist or is not a directory"
            )
        return v

    parser_version: str = Field(
        ...,
        description="Semver string identifying the kreuzberg parser version (e.g. '1.0.0')",
    )
    max_doc_bytes: int = Field(
        default=50_000_000,
        ge=1,
        description="Maximum document size in bytes before emitting too_large parse failure (default 50 MB)",
    )
    timeout_ms: int = Field(
        default=30_000,
        ge=1,
        description="HTTP request timeout in milliseconds for kreuzberg extract calls (default 30 s)",
    )
    inline_text_max_chars: int = Field(
        default=4096,
        ge=1,
        description=(
            "Maximum character length for storing extracted text inline in the event. "
            "Texts longer than this threshold are written to disk and referenced via file://"
        ),
    )
    publish_topic_indexed: str = Field(
        default="onex.evt.omnimemory.document-indexed.v1",
        description="Topic to publish ModelDocumentIndexedKreuzbergEvent messages to",
    )
    publish_topic_parse_failed: str = Field(
        default="onex.evt.omnimemory.document-parse-failed.v1",
        description="Topic to publish ModelDocumentParseFailedEvent messages to",
    )
