"""
Subscription model following ONEX standards.

Subscriptions track which agents want to receive notifications for specific
memory topics. Delivery mechanism is through Kafka event bus - agents
consume events via consumer groups.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...enums.enum_subscription_status import EnumSubscriptionStatus
from .constants import TOPIC_PATTERN, TOPIC_VALIDATION_ERROR


class ModelSubscription(BaseModel):
    """Agent subscription for memory change notifications following ONEX standards.

    Agents subscribe to topics and receive notifications via Kafka consumer groups.
    The subscription tracks which agent wants which topics - actual delivery happens
    through the event bus.

    Attributes:
        id: Unique subscription identifier.
        agent_id: Agent that owns this subscription.
        topic: Topic pattern (format: memory.<entity>.<event>).
        status: Subscription status (active, suspended, deleted).
        created_at: When the subscription was created.
        updated_at: When the subscription was last updated.
        suspended_reason: Reason for suspension if status is suspended.
        metadata: Optional metadata for the subscription.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", strict=True)

    id: str = Field(
        description="Unique subscription identifier (non-empty string)",
    )
    agent_id: str = Field(
        description="Agent that owns this subscription",
    )
    topic: Annotated[str, Field(min_length=1, max_length=256)] = Field(
        description="Topic pattern (format: memory.<entity>.<event>)",
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
            raise ValueError(TOPIC_VALIDATION_ERROR.format(topic=v))
        return v

    @field_validator("id", "agent_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate required string fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v
