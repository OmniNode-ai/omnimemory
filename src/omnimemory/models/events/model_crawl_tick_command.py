# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Crawl tick command model for the document ingestion pipeline.

Emitted by ``CrawlSchedulerEffect`` (or a git post-commit hook / MCP tool)
to trigger a crawl run for a specific crawler type and scope. Consumed by
the crawler Effects (FilesystemCrawler, GitRepoCrawler, LinearCrawler,
WatchdogEffect).

Kafka topic: ``{env}.onex.cmd.omnimemory.crawl-tick.v1``

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §4
Ticket: OMN-2426
"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType


class ModelCrawlTickCommand(BaseModel):
    """Command that triggers a crawl run for one crawler type and scope.

    ``event_type`` is a Literal discriminator for Kafka consumer routing.
    ``trigger_source`` records how the crawl was initiated for debounce
    guard and audit purposes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this command instance.",
    )
    event_type: Literal["CrawlTickRequested"] = Field(
        default="CrawlTickRequested",
        description="Event type discriminator for Kafka consumer routing.",
    )
    crawl_type: EnumCrawlerType = Field(
        ...,
        description="Crawler subsystem that should process this tick.",
    )
    crawl_scope: str = Field(
        ...,
        min_length=1,
        description=(
            "Scope string that bounds this crawl run, e.g. 'omninode/omniintelligence'. "
            "The crawler restricts its walk to paths or resources assigned to this scope."
        ),
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing across the pipeline.",
    )
    triggered_at_utc: datetime = Field(
        ...,
        description="UTC timestamp when this command was emitted.",
    )
    trigger_source: Literal["scheduled", "manual", "git_hook", "filesystem_watch"] = (
        Field(
            ...,
            description="Mechanism that initiated this crawl tick.",
        )
    )

    @field_validator("triggered_at_utc", mode="after")
    @classmethod
    def _require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        return v
