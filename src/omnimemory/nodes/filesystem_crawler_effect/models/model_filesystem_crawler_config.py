# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for FilesystemCrawlerEffect handler."""

from pydantic import BaseModel, ConfigDict, Field


# omnimemory-model-exempt: handler config
class ModelFilesystemCrawlerConfig(BaseModel):
    """Configuration for HandlerFilesystemCrawler.

    Declares the filesystem path prefixes to walk, the .md file glob
    pattern, topic routing suffixes, and operational limits.

    Note: Topic suffix defaults MUST match the ``event_bus.publish_topics``
    declared in this node's contract.yaml. The contract is the source of
    truth for topic declarations.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, from_attributes=True
    )

    # Path prefixes to crawl (longest-prefix scope mapping is applied externally)
    path_prefixes: list[str] = Field(
        default_factory=list,
        description=(
            "Absolute path prefixes to walk recursively for .md files. "
            "If empty the crawler is a no-op"
        ),
    )

    # File pattern
    file_glob: str = Field(
        default="*.md",
        description=(
            "Glob pattern passed to Path.rglob(). "
            "Default '*.md' matches all Markdown files recursively"
        ),
    )

    # Published topic suffixes (env prefix added at runtime)
    publish_topic_discovered: str = Field(
        default="onex.evt.omnimemory.document-discovered.v1",
        description="Topic suffix for document-discovered events",
    )
    publish_topic_changed: str = Field(
        default="onex.evt.omnimemory.document-changed.v1",
        description="Topic suffix for document-changed events",
    )
    publish_topic_removed: str = Field(
        default="onex.evt.omnimemory.document-removed.v1",
        description="Topic suffix for document-removed events",
    )

    # Operational limits
    max_file_size_bytes: int = Field(
        default=5_242_880,  # 5 MiB
        ge=1,
        description=(
            "Maximum file size in bytes to index. Files larger than this are "
            "skipped with a warning log"
        ),
    )
    max_files_per_crawl: int = Field(
        default=10_000,
        ge=1,
        description=(
            "Hard limit on files processed per crawl run to prevent runaway walks. "
            "A warning is logged when the limit is reached"
        ),
    )
