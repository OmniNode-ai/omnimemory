"""
Notification delivery attempt model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...enums.enum_subscription_status import (
    EnumDeliveryStatus,  # noqa: TC001 - runtime import for Pydantic field type
)


class ModelNotificationDeliveryAttempt(BaseModel):
    """Record of a notification delivery attempt following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid", strict=True)

    delivery_id: str = Field(
        description="Unique delivery attempt identifier (non-empty string)",
    )
    subscription_id: str = Field(
        description="Target subscription ID",
    )
    event_id: str = Field(
        description="Event being delivered",
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Attempt number (1-based, max 100)",
    )
    status: EnumDeliveryStatus = Field(
        description="Delivery status: pending, success, failed, or dlq",
    )
    status_code: int | None = Field(
        default=None,
        ge=100,
        le=599,
        description="HTTP response status code (100-599)",
    )
    error_message: str | None = Field(
        default=None,
        max_length=2048,
        description="Error message if delivery failed",
    )
    response_body: str | None = Field(
        default=None,
        max_length=4096,
        description="Truncated response body for debugging (max 4KB)",
    )
    next_retry_at: datetime | None = Field(
        default=None,
        description="Scheduled time for next retry attempt",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this delivery attempt was made",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When this delivery attempt completed",
    )
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Request latency in milliseconds",
    )

    @field_validator("delivery_id", "subscription_id", "event_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate required ID fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("ID field cannot be empty")
        return v
