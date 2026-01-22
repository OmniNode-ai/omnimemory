"""
Retry-related Pydantic models for OmniMemory ONEX architecture.

This module contains models for retry configuration, attempt tracking,
and statistics collection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field

__all__ = [
    "ModelRetryAttemptInfo",
    "ModelRetryConfig",
    "ModelRetryStatistics",
    # Backward compatibility aliases
    "RetryAttemptInfo",
    "RetryConfig",
    "RetryStatistics",
]


class ModelRetryConfig(BaseModel):
    """Configuration for retry behavior."""

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


class ModelRetryAttemptInfo(BaseModel):
    """Information about a retry attempt."""

    attempt_number: int = Field(description="Current attempt number (1-indexed)")
    delay_ms: int = Field(description="Delay before this attempt in milliseconds")
    exception: str | None = Field(
        default=None, description="Exception that triggered the retry"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the attempt was made",
    )
    correlation_id: UUID | None = Field(
        default=None, description="Request correlation ID"
    )


class ModelRetryStatistics(BaseModel):
    """Statistics about retry operations."""

    total_operations: int = Field(
        default=0, description="Total number of operations attempted"
    )
    successful_operations: int = Field(
        default=0, description="Number of successful operations"
    )
    failed_operations: int = Field(
        default=0, description="Number of permanently failed operations"
    )
    total_retries: int = Field(default=0, description="Total number of retry attempts")
    average_attempts: float = Field(
        default=0.0, description="Average number of attempts per operation"
    )
    common_exceptions: dict[str, int] = Field(
        default_factory=dict, description="Count of common exceptions encountered"
    )


# Backward compatibility aliases
RetryConfig = ModelRetryConfig
RetryAttemptInfo = ModelRetryAttemptInfo
RetryStatistics = ModelRetryStatistics
