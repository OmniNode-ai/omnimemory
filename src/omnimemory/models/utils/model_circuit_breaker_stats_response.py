# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Circuit breaker statistics response Pydantic model for OmniMemory ONEX architecture."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelCircuitBreakerStatsResponse",
]


class ModelCircuitBreakerStatsResponse(BaseModel):
    """Typed response model for circuit breaker statistics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        from_attributes=True,
    )

    state: str = Field(description="Current circuit breaker state")
    failure_count: int = Field(description="Number of failures recorded")
    success_count: int = Field(description="Number of successful calls")
    total_calls: int = Field(description="Total number of calls attempted")
    total_timeouts: int = Field(description="Total number of timeout failures")
    last_failure_time: str | None = Field(description="ISO timestamp of last failure")
    state_changed_at: str = Field(description="ISO timestamp when state last changed")
