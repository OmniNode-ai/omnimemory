"""
Circuit breaker statistics model following ONEX standards.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelCircuitBreakerStats(BaseModel):
    """Circuit breaker statistics for a single dependency."""

    state: Literal["closed", "open", "half_open"] = Field(
        description="Current circuit breaker state"
    )
    failure_count: int = Field(ge=0, description="Number of consecutive failures")
    success_count: int = Field(ge=0, description="Total number of successful calls")
    total_calls: int = Field(ge=0, description="Total number of calls made")
    total_timeouts: int = Field(ge=0, description="Total number of timeout failures")
    last_failure_time: Optional[datetime] = Field(
        default=None, description="Timestamp of the last failure"
    )
    state_changed_at: datetime = Field(
        description="When the circuit breaker state last changed"
    )
