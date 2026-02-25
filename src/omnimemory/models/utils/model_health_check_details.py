# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Health check details model for OmniMemory ONEX architecture.

This module contains the ModelHealthCheckDetails class.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelHealthCheckDetails",
]


class ModelHealthCheckDetails(BaseModel):
    """Strongly typed health check details with rate-limit and circuit tracking."""

    model_config = ConfigDict(extra="forbid")

    message: str | None = Field(
        default=None, description="Human-readable status message"
    )
    error: str | None = Field(default=None, description="Error message if unhealthy")
    version: str | None = Field(default=None, description="Service version")
    connection_url: str | None = Field(default=None, description="Connection URL")
    last_check: datetime | None = Field(
        default=None, description="Last check timestamp"
    )
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="Latency in milliseconds"
    )
    # Rate limiting state
    rate_limit_active: bool = Field(
        default=False, description="Whether rate limiting is currently active"
    )
    rate_limit_remaining: int | None = Field(
        default=None, description="Remaining requests in current window"
    )
    rate_limit_reset_time: float | None = Field(
        default=None, description="Time when rate limit resets (epoch)"
    )
    # Circuit breaker state
    circuit_open: bool = Field(
        default=False, description="Whether circuit breaker is open"
    )
    circuit_state: str | None = Field(
        default=None, description="Current circuit breaker state"
    )
    circuit_failure_count: int | None = Field(
        default=None, description="Number of failures recorded"
    )
    # Result details
    result_type: str | None = Field(
        default=None, description="Type of result (success/error/timeout)"
    )
    extra: dict[str, str] = Field(
        default_factory=dict, description="Additional string details"
    )
