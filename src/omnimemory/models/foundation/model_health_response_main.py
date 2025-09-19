"""
Main health response model following ONEX standards.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .model_dependency_status import ModelDependencyStatus
from .model_resource_metrics import ModelResourceMetrics


class ModelHealthResponse(BaseModel):
    """Health check response following ONEX standards."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall system health status"
    )
    latency_ms: float = Field(description="Health check response time in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was performed",
    )
    resource_usage: ModelResourceMetrics = Field(
        description="Current resource utilization"
    )
    dependencies: list[ModelDependencyStatus] = Field(
        default_factory=list, description="Status of system dependencies"
    )
    uptime_seconds: int = Field(description="System uptime in seconds")
    version: str = Field(description="System version information")
    environment: str = Field(description="Deployment environment")
