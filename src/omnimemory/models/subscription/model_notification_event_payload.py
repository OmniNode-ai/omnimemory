"""
Notification event payload model following ONEX standards.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNotificationEventPayload(BaseModel):
    """Structured payload for notification events following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid", strict=True)

    entity_type: str = Field(
        description="Type of entity that changed (e.g., 'item', 'collection')",
    )
    entity_id: str = Field(
        description="ID of the entity that changed",
    )
    action: str = Field(
        description="Action that occurred (e.g., 'created', 'updated', 'deleted')",
    )
    changes: dict[str, str] | None = Field(
        default=None,
        description="Key-value pairs of fields that changed (for updates)",
    )
    actor_id: str | None = Field(
        default=None,
        description="ID of the agent or user that triggered the change",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing related events",
    )
