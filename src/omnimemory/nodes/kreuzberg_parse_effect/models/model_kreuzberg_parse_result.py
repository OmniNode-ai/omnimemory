# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Summary result model for a single KreuzbergParseEffect run."""

from pydantic import BaseModel, ConfigDict, Field


class ModelKreuzbergParseResult(  # omnimemory-model-exempt: handler result
    BaseModel
):
    """Summary of documents processed in a single handler invocation.

    Returned by HandlerKreuzbergParse.process_event() aggregation when
    used in batch mode, or inspected by tests for assertion purposes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    indexed_count: int = Field(
        ...,
        ge=0,
        description="Number of document-indexed events successfully emitted",
    )
    failed_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of document-parse-failed events emitted for parse_error or timeout "
            "outcomes only. Does NOT include too_large outcomes, which are tracked "
            "separately in skipped_too_large_count. Callers that need the total number "
            "of document-parse-failed events emitted must sum "
            "failed_count + skipped_too_large_count."
        ),
    )
    skipped_too_large_count: int = Field(
        ...,
        ge=0,
        description="Number of documents rejected before kreuzberg call due to size > max_doc_bytes",
    )
    timeout_count: int = Field(
        ...,
        ge=0,
        description="Number of kreuzberg requests that timed out",
    )
