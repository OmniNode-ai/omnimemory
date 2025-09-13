"""
Health response model following ONEX standards.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelDependencyStatus(BaseModel):
    """Status of a system dependency."""

    name: str = Field(
        description="Name of the dependency"
    )
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Health status of the dependency"
    )
    latency_ms: float = Field(
        description="Response latency in milliseconds"
    )
    last_check: datetime = Field(
        description="When the dependency was last checked"
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if unhealthy"
    )


class ModelResourceMetrics(BaseModel):
    """System resource utilization metrics."""

    cpu_usage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="CPU usage percentage"
    )
    memory_usage_mb: float = Field(
        description="Memory usage in megabytes"
    )
    memory_usage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Memory usage percentage"
    )
    disk_usage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Disk usage percentage"
    )
    network_throughput_mbps: float = Field(
        description="Network throughput in megabits per second"
    )


class ModelHealthResponse(BaseModel):
    """Health check response following ONEX standards."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall system health status"
    )
    latency_ms: float = Field(
        description="Health check response time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was performed"
    )
    resource_usage: ModelResourceMetrics = Field(
        description="Current resource utilization"
    )
    dependencies: list[ModelDependencyStatus] = Field(
        default_factory=list,
        description="Status of system dependencies"
    )
    uptime_seconds: int = Field(
        description="System uptime in seconds"
    )
    version: str = Field(
        description="System version information"
    )
    environment: str = Field(
        description="Deployment environment"
    )