"""
ONEX-compliant typed models for health check metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in health management, ensuring type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthCheckMetadata(BaseModel):
    """Strongly typed metadata for health check operations."""

    connection_url: Optional[str] = Field(
        default=None, description="Connection URL for dependency checks"
    )

    database_version: Optional[str] = Field(
        default=None, description="Version information for database dependencies"
    )

    pool_stats: Optional[Dict[str, int]] = Field(
        default=None, description="Connection pool statistics"
    )

    request_count: int = Field(default=0, description="Number of requests processed")

    error_count: int = Field(default=0, description="Number of errors encountered")

    last_success_timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp of last successful check"
    )

    circuit_breaker_state: Optional[str] = Field(
        default=None, description="Current circuit breaker state"
    )

    performance_metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Performance metrics (latency, throughput)"
    )


class AggregateHealthMetadata(BaseModel):
    """Strongly typed metadata for aggregate health status."""

    total_dependencies: int = Field(description="Total number of dependencies checked")

    healthy_dependencies: int = Field(description="Number of healthy dependencies")

    degraded_dependencies: int = Field(description="Number of degraded dependencies")

    unhealthy_dependencies: int = Field(description="Number of unhealthy dependencies")

    critical_failures: List[str] = Field(
        default_factory=list,
        description="Names of critical dependencies that are failing",
    )

    overall_health_score: float = Field(
        description="Calculated overall health score (0.0-1.0)"
    )

    last_update_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this aggregate was last calculated",
    )

    trends: Optional[Dict[str, List[float]]] = Field(
        default=None, description="Historical trend data for key metrics"
    )


class ConfigurationChangeMetadata(BaseModel):
    """Strongly typed metadata for configuration changes."""

    changed_keys: List[str] = Field(
        description="List of configuration keys that were modified"
    )

    change_source: str = Field(description="Source of the configuration change")

    validation_results: Dict[str, bool] = Field(
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
