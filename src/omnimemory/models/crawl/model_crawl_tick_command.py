# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Crawl tick command model — triggers a filesystem crawl run."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType

TriggerSource = Literal["scheduled", "manual", "git_hook", "filesystem_watch"]


class ModelCrawlTickCommand(BaseModel):
    """Kafka command that triggers a crawler node to execute a crawl run.

    Issued by CrawlSchedulerEffect or a manual trigger. Consumed by
    FilesystemCrawlerEffect (and future crawler nodes) via the
    ``onex.cmd.omnimemory.crawl-tick.v1`` topic.

    # strict=True: JSON consumers must pass crawl_type as the enum value string
    # (e.g. "filesystem"), not a dict. UUID fields must be passed as UUID objects
    # or str — Pydantic v2 coerces str→UUID in non-strict mode, which is fine
    # here since strict=True is NOT set on this command model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    crawl_type: EnumCrawlerType = Field(
        ...,
        description=(
            "Identifies which crawler should handle this tick "
            "(e.g. FILESYSTEM, GIT_REPO, LINEAR, WATCHDOG)"
        ),
    )
    crawl_scope: str = Field(
        ...,
        min_length=1,
        description=(
            "Scope identifier for the crawl run. Passed to scope mapping "
            "resolution to determine which scope_ref to assign discovered documents."
        ),
    )
    correlation_id: UUID = Field(
        ...,
        description="Unique identifier for this crawl run, used for event correlation and tracing",
    )
    trigger_source: TriggerSource = Field(
        ...,
        description=(
            "What triggered this crawl run: 'scheduled' (CrawlSchedulerEffect), "
            "'manual' (operator-issued), 'git_hook' (post-commit hook), "
            "or 'filesystem_watch' (FSEvents/inotify)"
        ),
    )
