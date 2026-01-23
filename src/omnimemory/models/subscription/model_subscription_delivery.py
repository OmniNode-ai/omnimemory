"""
Subscription delivery configuration model following ONEX standards.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ModelSubscriptionDeliveryWebhook(BaseModel):
    """Webhook delivery configuration for subscriptions following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid", strict=True)

    webhook_url: HttpUrl = Field(
        description="URL to POST notifications to (must be HTTPS in production)",
    )
    secret: str | None = Field(
        default=None,
        description="Shared secret for HMAC-SHA256 signature verification",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers to include in webhook requests",
    )
    timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=30000,
        description="Request timeout in milliseconds (100ms to 30s)",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts before moving to DLQ",
    )
    retry_delay_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Initial delay between retries in milliseconds (exponential backoff)",
    )
