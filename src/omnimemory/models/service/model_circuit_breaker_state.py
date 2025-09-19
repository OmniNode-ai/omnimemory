"""
Circuit breaker state model for OmniMemory following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelCircuitBreakerState(BaseModel):
    """Circuit breaker state tracking."""

    is_open: bool = Field(default=False, description="Whether circuit is open")
    failure_count: int = Field(default=0, description="Current failure count")
    last_failure_time: datetime | None = Field(
        default=None, description="Time of last failure"
    )
    next_retry_time: datetime | None = Field(
        default=None, description="When to retry next"
    )
