# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Health response model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.foundation.model_semver import ModelSemVer  # noqa: TC001


class ModelDependencyStatus(BaseModel):
    """Status of a system dependency."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str = Field(description="Name of the dependency")
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Health status of the dependency"
    )
    latency_ms: float = Field(ge=0.0, description="Response latency in milliseconds")
    last_check: datetime = Field(description="When the dependency was last checked")
    error_message: str | None = Field(
        default=None, description="Error message if unhealthy"
    )


class ModelResourceMetrics(BaseModel):
    """System resource utilization metrics."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    cpu_usage_percent: float = Field(
        ge=0.0, le=100.0, description="CPU usage percentage"
    )
    memory_usage_mb: float = Field(ge=0.0, description="Memory usage in megabytes")
    memory_usage_percent: float = Field(
        ge=0.0, le=100.0, description="Memory usage percentage"
    )
    disk_usage_percent: float = Field(
        ge=0.0, le=100.0, description="Disk usage percentage"
    )
    network_throughput_mbps: float = Field(
        ge=0.0, description="Network throughput in megabits per second"
    )


class ModelHealthResponse(BaseModel):
    """Health check response following ONEX standards."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall system health status"
    )
    latency_ms: float = Field(
        ge=0.0, description="Health check response time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the health check was performed",
    )
    resource_usage: ModelResourceMetrics = Field(
        description="Current resource utilization"
    )
    dependencies: list[ModelDependencyStatus] = Field(
        default_factory=list, description="Status of system dependencies"
    )
    uptime_seconds: int = Field(ge=0, description="System uptime in seconds")
    version: ModelSemVer = Field(description="System version information")
    environment: str = Field(description="Deployment environment")


class ModelCircuitBreakerStats(BaseModel):
    """Circuit breaker statistics for a single dependency."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    state: Literal["closed", "open", "half_open"] = Field(
        description="Current circuit breaker state"
    )
    failure_count: int = Field(ge=0, description="Number of consecutive failures")
    success_count: int = Field(ge=0, description="Total number of successful calls")
    total_calls: int = Field(ge=0, description="Total number of calls made")
    total_timeouts: int = Field(ge=0, description="Total number of timeout failures")
    last_failure_time: datetime | None = Field(
        default=None, description="Timestamp of the last failure"
    )
    state_changed_at: datetime = Field(
        description="When the circuit breaker state last changed"
    )


class ModelCircuitBreakerStatsCollection(BaseModel):
    """Collection of circuit breaker statistics for all dependencies."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    circuit_breakers: dict[str, ModelCircuitBreakerStats] = Field(
        description="Circuit breaker statistics keyed by dependency name"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the statistics were collected",
    )


class ModelRateLimitedHealthCheckResponse(BaseModel):
    """Rate-limited health check response."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    status: str = Field(
        default="rate_limited", description="Status of the rate-limited response"
    )
    message: str = Field(
        default="Rate limit status",
        description="Human-readable message about the rate limit status",
    )
    details: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="Additional details including retry_after and requests count",
    )
    health_check: ModelHealthResponse | None = Field(
        default=None, description="Health check result if within rate limit"
    )
    rate_limited: bool = Field(
        default=True, description="Whether the request was rate limited"
    )
    rate_limit_reset_time: datetime | None = Field(
        default=None, description="When the rate limit will reset"
    )
    remaining_requests: int | None = Field(
        default=None, description="Number of requests remaining in the current window"
    )
    error_message: str | None = Field(
        default=None, description="Error message if rate limited"
    )
