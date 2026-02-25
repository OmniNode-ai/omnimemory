# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Resource health check model for OmniMemory ONEX architecture.

This module contains the ModelResourceHealthCheck class.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .model_health_check_details import ModelHealthCheckDetails

# Runtime import required for Pydantic schema building (not TYPE_CHECKING)
from .model_health_status import HealthStatus  # noqa: TC001

__all__ = [
    "ModelResourceHealthCheck",
]


class ModelResourceHealthCheck(BaseModel):
    """Result of a resource health check."""

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    status: HealthStatus = Field(description="Health status of the resource")
    response_time: float = Field(default=0.0, description="Response time in seconds")
    details: ModelHealthCheckDetails = Field(
        default_factory=ModelHealthCheckDetails, description="Additional details"
    )
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracking"
    )
