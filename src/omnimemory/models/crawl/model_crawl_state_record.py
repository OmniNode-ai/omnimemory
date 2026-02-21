# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Crawl state record model for the omnimemory_crawl_state table."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType


class ModelCrawlStateRecord(BaseModel):
    """Represents a row in the omnimemory_crawl_state table.

    Stores the last known state of each crawled document, enabling the
    two-stage mtime → SHA-256 change detection strategy.

    Primary key: (source_ref, crawler_type, scope_ref).

    The scope_ref is included in the PK so that if scope mapping
    configuration changes (file moves between scopes, or mapping is
    reconfigured), a clean re-index is required. Without scope_ref in
    the PK, stale crawl entries would silently block re-indexing under
    the new scope assignment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source_ref: str = Field(
        ...,
        description=(
            "Absolute path for filesystem crawls, URL for web crawls, "
            "or Linear ID for Linear crawls"
        ),
    )
    crawler_type: EnumCrawlerType = Field(
        ...,
        description="Identifies which crawler produced this state record",
    )
    scope_ref: str = Field(
        ...,
        description=(
            "Scope assignment for this document (e.g. 'omninode/omnimemory'). "
            "Included in PK for scope migration safety"
        ),
    )
    content_fingerprint: str = Field(
        ...,
        description="SHA-256 hex digest of the document content at last crawl",
    )
    source_version: str | None = Field(
        default=None,
        description=(
            "Version identifier: git SHA for repo files, Linear updatedAt for "
            "Linear items, or None for unversioned filesystem files"
        ),
    )
    last_crawled_at_utc: datetime = Field(
        ...,
        description="UTC timestamp of the most recent crawl attempt for this document",
    )
    last_changed_at_utc: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp when content last changed; None if never changed "
            "since first indexing"
        ),
    )
    last_known_mtime: float | None = Field(
        default=None,
        description=(
            "stat.st_mtime value at last crawl for FilesystemCrawler fast-path. "
            "Not persisted by other crawler types"
        ),
    )

    @field_validator("last_crawled_at_utc", mode="after")
    @classmethod
    def _require_tz_crawled(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("last_crawled_at_utc must be timezone-aware (UTC)")
        return v

    @field_validator("last_changed_at_utc", mode="after")
    @classmethod
    def _require_tz_changed(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("last_changed_at_utc must be timezone-aware (UTC)")
        return v
