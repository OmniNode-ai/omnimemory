# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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
        description=(
            "Total number of .md files encountered by rglob that passed the "
            "symlink guard, before any stat or size filtering. This counter is "
            "incremented immediately after the symlink check. "
            "Holds: files_walked == len(walked_paths) + error_count + size_exceeded_skipped, "
            "where size_exceeded_skipped is the portion of skipped_count due to "
            "max_file_size_bytes (NOT symlink-escaped files, which are rejected "
            "before this counter is incremented and are tracked separately in "
            "skipped_count)."
        ),
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
            "Number of files skipped. Includes two categories: "
            "(1) symlink-escaped files rejected before files_walked is incremented, "
            "(2) files that exceeded max_file_size_bytes (counted in files_walked). "
            "These two categories are not distinguished in this field."
        ),
    )
    indexed_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of document-indexed events emitted. Equals discovered_count + "
            "changed_count (one indexed event per successfully indexed document). "
            "Consumed by omniintelligence crawl_scheduler_effect to reset debounce windows."
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
