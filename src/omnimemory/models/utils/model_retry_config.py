"""
Retry configuration model for OmniMemory ONEX architecture.

This module contains the configuration model for retry behavior.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelRetryConfig",
]


class ModelRetryConfig(BaseModel):
    """Configuration for retry behavior."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum number of retry attempts"
    )
    base_delay_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Base delay between attempts in milliseconds",
    )
    max_delay_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Maximum delay between attempts in milliseconds",
    )
    exponential_multiplier: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Exponential backoff multiplier"
    )
    jitter: bool = Field(
        default=True, description="Whether to add random jitter to delays"
    )
    retryable_exceptions: list[str] = Field(
        default_factory=lambda: [
            "ConnectionError",
            "TimeoutError",
            "HTTPError",
            "TemporaryFailure",
        ],
        description="Exception types that should trigger retries",
    )
