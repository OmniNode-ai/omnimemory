"""
Notification event model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.models.foundation.model_semver import ModelSemVer

from .constants import TOPIC_PATTERN, TOPIC_VALIDATION_ERROR
from .model_notification_event_payload import (
    ModelNotificationEventPayload,  # noqa: TC001 - runtime import for Pydantic field type
)


class ModelNotificationEvent(BaseModel):
    """Notification event for memory changes following ONEX standards."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, from_attributes=True
    )

    event_id: str = Field(
        description="Unique event identifier (non-empty string)",
    )
    topic: Annotated[str, Field(min_length=1, max_length=256)] = Field(
        description="Topic this event belongs to (format: memory.<entity>.<event>)",
    )
    payload: ModelNotificationEventPayload = Field(
        description="Structured event payload data",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event was created",
    )
    # NOTE: dict contents remain mutable even on frozen models; this is a known
    # Pydantic v2 limitation.  frozen=True prevents field *reassignment* but not
    # in-place mutation of mutable containers.
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional event metadata for routing or filtering",
    )
    source: str | None = Field(
        default=None,
        description="Source system or service that generated this event",
    )
    # strict=True: JSON consumers must pass version as {"major":N, "minor":N, "patch":N},
    # not as a plain string — str→ModelSemVer coercion is not allowed in strict mode.
    version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Event schema version for forward compatibility",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic_format(cls, v: str) -> str:
        """Validate topic follows memory.<entity>.<event> convention."""
        if not TOPIC_PATTERN.match(v):
            raise ValueError(TOPIC_VALIDATION_ERROR.format(topic=v))
        return v

    @field_validator("event_id")
    @classmethod
    def validate_event_id_non_empty(cls, v: str) -> str:
        """Validate event_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("event_id cannot be empty")
        return v
