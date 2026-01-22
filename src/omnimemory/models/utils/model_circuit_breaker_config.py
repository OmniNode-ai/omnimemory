"""
Circuit breaker configuration Pydantic model for OmniMemory ONEX architecture.

This module contains the configuration model for circuit breaker behavior.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "ModelCircuitBreakerConfig",
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
