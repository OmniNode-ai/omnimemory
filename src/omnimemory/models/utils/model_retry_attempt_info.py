"""
Retry attempt information model for OmniMemory ONEX architecture.

This module contains the model for tracking individual retry attempts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelRetryAttemptInfo",
]


class ModelRetryAttemptInfo(BaseModel):
    """Information about a retry attempt."""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(ge=1, description="Current attempt number (1-indexed)")
    delay_ms: int = Field(ge=0, description="Delay before this attempt in milliseconds")
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
