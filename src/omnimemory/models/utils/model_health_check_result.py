# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Health check result model for OmniMemory ONEX architecture.

This module contains the ModelHealthCheckResult class.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.model_health_metadata import HealthCheckMetadata
from .model_health_status import HealthStatus

if TYPE_CHECKING:
    from ..foundation.model_health_response import ModelDependencyStatus
    from .model_health_check_config import ModelHealthCheckConfig

__all__ = [
    "ModelHealthCheckResult",
]


class ModelHealthCheckResult(BaseModel):
    """Result of an individual health check."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    config: ModelHealthCheckConfig = Field(description="Health check configuration")
    status: HealthStatus = Field(description="Health status")
    latency_ms: float = Field(
        ge=0,
        description="Check latency in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the health check was performed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the health check failed",
    )
    metadata: HealthCheckMetadata = Field(
        default_factory=HealthCheckMetadata,
        description="Additional metadata for the health check result",
    )

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
