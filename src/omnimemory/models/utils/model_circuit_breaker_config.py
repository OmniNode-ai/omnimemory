"""
Circuit breaker configuration Pydantic model for OmniMemory ONEX architecture.

This module contains the configuration model for circuit breaker behavior.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelCircuitBreakerConfig",
]


class ModelCircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""

    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = Field(
        default=5, ge=1, description="Number of failures before opening circuit"
    )
    recovery_timeout: int = Field(
        default=60, ge=0, description="Seconds to wait before trying half-open"
    )
    recovery_timeout_jitter: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Jitter factor (0.0-1.0) to prevent thundering herd",
    )
    success_threshold: int = Field(
        default=3, ge=1, description="Successful calls needed to close circuit"
    )
    timeout: float = Field(
        default=30.0, gt=0, description="Default timeout for operations"
    )
