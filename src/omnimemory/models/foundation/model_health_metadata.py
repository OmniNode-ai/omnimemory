"""
ONEX-compliant typed models for health check metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in health management, ensuring type safety and validation.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.foundation.model_semver import ModelSemVer


class HealthCheckMetadata(BaseModel):
    """Strongly typed metadata for health check operations."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    connection_url: str | None = Field(
        default=None, description="Connection URL for dependency checks"
    )

    database_version: ModelSemVer | None = Field(
        default=None,
        description="Semantic version information for database dependencies",
    )

    pool_stats: dict[str, int] | None = Field(
        default=None, description="Connection pool statistics"
    )

    request_count: int = Field(default=0, description="Number of requests processed")

    error_count: int = Field(default=0, description="Number of errors encountered")

    last_success_timestamp: datetime | None = Field(
        default=None, description="Timestamp of last successful check"
    )

    circuit_breaker_state: str | None = Field(
        default=None, description="Current circuit breaker state"
    )

    performance_metrics: dict[str, float] | None = Field(
        default=None, description="Performance metrics (latency, throughput)"
    )


class AggregateHealthMetadata(BaseModel):
    """Strongly typed metadata for aggregate health status."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    total_dependencies: int = Field(description="Total number of dependencies checked")

    healthy_dependencies: int = Field(description="Number of healthy dependencies")

    degraded_dependencies: int = Field(description="Number of degraded dependencies")

    unhealthy_dependencies: int = Field(description="Number of unhealthy dependencies")

    critical_failures: list[str] = Field(
        default_factory=list,
        description="Names of critical dependencies that are failing",
    )

    overall_health_score: float = Field(
        ge=0.0, le=1.0, description="Calculated overall health score (0.0-1.0)"
    )

    last_update_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this aggregate was last calculated",
    )

    trends: dict[str, list[float]] | None = Field(
        default=None, description="Historical trend data for key metrics"
    )


class ConfigurationChangeMetadata(BaseModel):
    """Strongly typed metadata for configuration changes."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    changed_keys: list[str] = Field(
        description="List of configuration keys that were modified"
    )

    change_source: str = Field(description="Source of the configuration change")

    validation_results: dict[str, bool] = Field(
        description="Validation results for each changed configuration"
    )

    requires_restart: bool = Field(
        default=False, description="Whether changes require service restart"
    )

    backup_created: bool = Field(
        default=False, description="Whether configuration backup was created"
    )

    rollback_available: bool = Field(
        default=False, description="Whether rollback is available for this change"
    )
