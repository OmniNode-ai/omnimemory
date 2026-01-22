"""
Health check Pydantic models for OmniMemory ONEX architecture.

This module contains models for health check configuration and results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.model_health_metadata import HealthCheckMetadata
from .model_resource_manager import ModelCircuitBreakerConfig

if TYPE_CHECKING:
    from ..foundation.model_health_response import ModelDependencyStatus

__all__ = [
    "DependencyType",
    "HealthStatus",
    "ModelHealthCheckConfig",
    "ModelHealthCheckDetails",
    "ModelHealthCheckResult",
    "ModelResourceHealthCheck",
    "ModelSystemHealth",
    # Backward compatibility aliases
    "HealthCheckConfig",
    "HealthCheckDetails",
    "HealthCheckResult",
    "ResourceHealthCheck",
    "SystemHealth",
]


class HealthStatus(Enum):
    """Enhanced health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"


class DependencyType(Enum):
    """Types of system dependencies."""

    DATABASE = "database"
    CACHE = "cache"
    VECTOR_DB = "vector_db"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"


class ModelHealthCheckConfig(BaseModel):
    """Configuration for individual health checks."""

    name: str = Field(description="Dependency name")
    dependency_type: DependencyType = Field(description="Type of dependency")
    timeout: float = Field(default=5.0, description="Health check timeout in seconds")
    critical: bool = Field(
        default=True, description="Whether failure affects overall health"
    )
    circuit_breaker_config: ModelCircuitBreakerConfig | None = Field(default=None)
    metadata: HealthCheckMetadata = Field(default_factory=HealthCheckMetadata)


class ModelHealthCheckResult(BaseModel):
    """Result of an individual health check."""

    config: ModelHealthCheckConfig = Field(description="Health check configuration")
    status: HealthStatus = Field(description="Health status")
    latency_ms: float = Field(description="Check latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = Field(default=None)
    metadata: HealthCheckMetadata = Field(default_factory=HealthCheckMetadata)

    def to_dependency_status(self) -> ModelDependencyStatus:
        """Convert to ModelDependencyStatus for API response."""
        # Import here to avoid circular imports
        from ..foundation.model_health_response import ModelDependencyStatus

        # Map HealthStatus to the expected Literal type
        status_map: dict[HealthStatus, Literal["healthy", "degraded", "unhealthy"]] = {
            HealthStatus.HEALTHY: "healthy",
            HealthStatus.DEGRADED: "degraded",
            HealthStatus.UNHEALTHY: "unhealthy",
            HealthStatus.UNKNOWN: "unhealthy",
            HealthStatus.TIMEOUT: "unhealthy",
            HealthStatus.RATE_LIMITED: "degraded",
            HealthStatus.CIRCUIT_OPEN: "degraded",
        }
        mapped_status = status_map.get(self.status, "unhealthy")
        return ModelDependencyStatus(
            name=self.config.name,
            status=mapped_status,
            latency_ms=self.latency_ms,
            last_check=self.timestamp,
            error_message=self.error_message,
        )


class ModelHealthCheckDetails(BaseModel):
    """Strongly typed health check details with rate-limit and circuit tracking."""

    message: str | None = Field(
        default=None, description="Human-readable status message"
    )
    error: str | None = Field(default=None, description="Error message if unhealthy")
    version: str | None = Field(default=None, description="Service version")
    connection_url: str | None = Field(default=None, description="Connection URL")
    last_check: str | None = Field(default=None, description="Last check timestamp")
    latency_ms: float | None = Field(
        default=None, description="Latency in milliseconds"
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


class ModelResourceHealthCheck(BaseModel):
    """Result of a resource health check."""

    model_config = ConfigDict(use_enum_values=False)

    status: HealthStatus = Field(description="Health status of the resource")
    response_time: float = Field(default=0.0, description="Response time in seconds")
    details: ModelHealthCheckDetails = Field(
        default_factory=ModelHealthCheckDetails, description="Additional details"
    )
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracking"
    )


class ModelSystemHealth(BaseModel):
    """Overall system health status."""

    model_config = ConfigDict(use_enum_values=False)

    overall_status: HealthStatus = Field(description="Overall system health status")
    resource_statuses: dict[str, ModelResourceHealthCheck] = Field(
        default_factory=dict, description="Health status of individual resources"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of health check",
    )


# Backward compatibility aliases
HealthCheckConfig = ModelHealthCheckConfig
HealthCheckResult = ModelHealthCheckResult
HealthCheckDetails = ModelHealthCheckDetails
ResourceHealthCheck = ModelResourceHealthCheck
SystemHealth = ModelSystemHealth
