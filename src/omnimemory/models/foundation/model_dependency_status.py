"""
Dependency status model following ONEX standards.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelDependencyStatus(BaseModel):
    """Status of a system dependency."""

    name: str = Field(description="Name of the dependency")
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Health status of the dependency"
    )
    latency_ms: float = Field(description="Response latency in milliseconds")
    last_check: datetime = Field(description="When the dependency was last checked")
    error_message: str | None = Field(
        default=None, description="Error message if unhealthy"
    )
