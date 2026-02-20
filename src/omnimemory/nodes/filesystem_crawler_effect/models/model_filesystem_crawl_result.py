# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Result model for a single filesystem crawl run."""

from pydantic import BaseModel, ConfigDict, Field


class ModelFilesystemCrawlResult(  # omnimemory-model-exempt: handler result
    BaseModel
):
    """Summary of a completed filesystem crawl run.

    Returned by HandlerFilesystemCrawler.crawl() to the caller
    (effect node execute_effect method) for logging and observability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    files_walked: int = Field(
        ...,
        ge=0,
        description="Total number of .md files found during the filesystem walk",
    )
    discovered_count: int = Field(
        ...,
        ge=0,
        description="Number of document-discovered events emitted",
    )
    changed_count: int = Field(
        ...,
        ge=0,
        description="Number of document-changed events emitted",
    )
    unchanged_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of files whose content fingerprint was unchanged "
            "(mtime changed but SHA-256 matched)"
        ),
    )
    skipped_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of files skipped (exceeded max_file_size_bytes or "
            "other non-fatal exclusion)"
        ),
    )
    removed_count: int = Field(
        ...,
        ge=0,
        description="Number of document-removed events emitted",
    )
    error_count: int = Field(
        ...,
        ge=0,
        description="Number of files that caused non-fatal errors during processing",
    )
    mtime_skipped_count: int = Field(
        default=0,
        ge=0,
        description="Files skipped via mtime fast-path (mtime unchanged since last crawl, no content check performed)",
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True if max_files_per_crawl was reached before the walk completed. "
            "Some files may not have been evaluated"
        ),
    )
