"""
Resource management Pydantic models for OmniMemory ONEX architecture.

This module contains models for circuit breaker configuration and statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "ModelCircuitBreakerConfig",
    "ModelCircuitBreakerStatsResponse",
    # Backward compatibility aliases
    "CircuitBreakerConfig",
    "CircuitBreakerStatsResponse",
]


class ModelCircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = Field(
        default=5, description="Number of failures before opening circuit"
    )
    recovery_timeout: int = Field(
        default=60, description="Seconds to wait before trying half-open"
    )
    recovery_timeout_jitter: float = Field(
        default=0.1, description="Jitter factor (0.0-1.0) to prevent thundering herd"
    )
    success_threshold: int = Field(
        default=3, description="Successful calls needed to close circuit"
    )
    timeout: float = Field(default=30.0, description="Default timeout for operations")


class ModelCircuitBreakerStatsResponse(BaseModel):
    """Typed response model for circuit breaker statistics."""

    state: str = Field(description="Current circuit breaker state")
    failure_count: int = Field(description="Number of failures recorded")
    success_count: int = Field(description="Number of successful calls")
    total_calls: int = Field(description="Total number of calls attempted")
    total_timeouts: int = Field(description="Total number of timeout failures")
    last_failure_time: str | None = Field(description="ISO timestamp of last failure")
    state_changed_at: str = Field(description="ISO timestamp when state last changed")


# Backward compatibility aliases
CircuitBreakerConfig = ModelCircuitBreakerConfig
CircuitBreakerStatsResponse = ModelCircuitBreakerStatsResponse
