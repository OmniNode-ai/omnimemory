"""
System health model for OmniMemory ONEX architecture.

This module contains the ModelSystemHealth class.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

# Runtime imports required for Pydantic schema building (not TYPE_CHECKING)
from .model_health_status import HealthStatus  # noqa: TC001
from .model_resource_health_check import ModelResourceHealthCheck  # noqa: TC001

__all__ = [
    "ModelSystemHealth",
]


class ModelSystemHealth(BaseModel):
    """Overall system health status."""

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    overall_status: HealthStatus = Field(description="Overall system health status")
    resource_statuses: dict[str, ModelResourceHealthCheck] = Field(
        default_factory=dict, description="Health status of individual resources"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of health check",
    )
