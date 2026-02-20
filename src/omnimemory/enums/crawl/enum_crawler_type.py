# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Crawler type enumeration for document ingestion pipeline."""

from enum import Enum


class EnumCrawlerType(str, Enum):
    """Identifies the crawler that produced a crawl event.

    Each crawler type has distinct source characteristics and change
    detection strategies as described in the document ingestion pipeline
    design (DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §5).

    Attributes:
        FILESYSTEM: Walks configured path prefixes for .md files using
            mtime + SHA-256 change detection.
        GIT_REPO: Scans git repositories via git log/diff, versioned by
            commit SHA.
        LINEAR: Fetches Linear issues and documents, deduplicated via
            updatedAt timestamps.
        WATCHDOG: Receives FSEvents/inotify notifications for unversioned
            critical files.
    """

    FILESYSTEM = "filesystem"
    GIT_REPO = "git_repo"
    LINEAR = "linear"
    WATCHDOG = "watchdog"
