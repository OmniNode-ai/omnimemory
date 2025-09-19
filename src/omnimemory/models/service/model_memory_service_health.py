"""
Memory service health model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums import EnumHealthStatus, EnumTrendDirection


class ModelMemoryServiceHealth(BaseModel):
    """Memory service health information following ONEX standards."""

    # Service identification
    service_id: UUID = Field(
        description="Unique identifier for the memory service",
    )
    service_name: str = Field(
        description="Human-readable name for the memory service",
    )

    # Health status
    status: EnumHealthStatus = Field(
        description="Current status of the memory service",
    )
    is_healthy: bool = Field(
        description="Whether the memory service is considered healthy",
    )

    # Uptime information
    uptime_seconds: int = Field(
        description="Memory service uptime in seconds",
    )
    last_restart_at: datetime | None = Field(
        default=None,
        description="When the memory service was last restarted",
    )

    # Performance metrics
    response_time_ms: float = Field(
        description="Average response time in milliseconds",
    )
    requests_per_second: float = Field(
        description="Current requests per second",
    )
    error_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Error rate as a percentage",
    )

    # Resource utilization
    cpu_usage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Current CPU usage percentage",
    )
    memory_usage_mb: float = Field(
        description="Current memory usage in megabytes",
    )
    memory_usage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Memory usage as percentage of allocated",
    )

    # Connection information
    active_connections: int = Field(
        description="Number of active connections",
    )
    max_connections: int = Field(
        description="Maximum allowed connections",
    )
    connection_pool_utilization: float = Field(
        ge=0.0,
        le=1.0,
        description="Connection pool utilization percentage",
    )

    # Memory-specific metrics
    memory_cache_hit_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Memory cache hit rate percentage",
    )
    memory_operations_per_second: float = Field(
        description="Memory operations processed per second",
    )
    memory_storage_utilization: float = Field(
        ge=0.0,
        le=1.0,
        description="Memory storage utilization percentage",
    )

    # Dependency health
    dependencies_healthy: bool = Field(
        description="Whether all memory service dependencies are healthy",
    )
    unhealthy_dependencies: list[str] = Field(
        default_factory=list,
        description="List of unhealthy memory service dependencies",
    )

    # Error information
    recent_errors: list[str] = Field(
        default_factory=list,
        description="List of recent error messages",
    )
    critical_errors: int = Field(
        default=0,
        description="Number of critical errors in the last hour",
    )

    # Health check information
    last_health_check: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was performed",
    )
    health_check_duration_ms: float = Field(
        description="Duration of the health check in milliseconds",
    )

    # Memory service-specific metrics
    custom_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Memory service-specific health metrics",
    )

    # Alerts and warnings
    active_alerts: list[str] = Field(
        default_factory=list,
        description="List of active alerts",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of current warnings",
    )

    # Trend information
    health_trend: EnumTrendDirection = Field(
        default=EnumTrendDirection.STABLE,
        description="Health trend (improving, stable, degrading)",
    )
    performance_trend: EnumTrendDirection = Field(
        default=EnumTrendDirection.STABLE,
        description="Performance trend (improving, stable, degrading)",
    )
