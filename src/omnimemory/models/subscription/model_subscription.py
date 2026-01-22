"""
Subscription model following ONEX standards.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...enums.enum_subscription_status import EnumSubscriptionStatus
from .model_subscription_delivery import (
    ModelSubscriptionDeliveryWebhook,  # noqa: TC001 - runtime import for Pydantic field type
)

# Topic pattern: memory.<entity>.<event>
# Examples: memory.item.created, memory.item.updated, memory.item.deleted
TOPIC_PATTERN = re.compile(r"^memory\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class ModelSubscription(BaseModel):
    """Agent subscription for memory change notifications following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    id: str = Field(
        description="Unique subscription ID (UUID format)",
    )
    agent_id: str = Field(
        description="Agent that owns this subscription",
    )
    topic: Annotated[str, Field(min_length=1, max_length=256)] = Field(
        description="Topic pattern (format: memory.<entity>.<event>)",
    )
    delivery: ModelSubscriptionDeliveryWebhook = Field(
        description="Webhook delivery configuration",
    )
    status: EnumSubscriptionStatus = Field(
        default=EnumSubscriptionStatus.ACTIVE,
        description="Subscription status: active, suspended, or deleted",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the subscription was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the subscription was last updated",
    )
    suspended_reason: str | None = Field(
        default=None,
        description="Reason for suspension if status is suspended",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional metadata for the subscription",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic_format(cls, v: str) -> str:
        """Validate topic follows memory.<entity>.<event> convention."""
        if not TOPIC_PATTERN.match(v):
            raise ValueError(
                f"Topic must match pattern 'memory.<entity>.<event>', got: {v}"
            )
        return v

    @field_validator("id", "agent_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate required string fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v
