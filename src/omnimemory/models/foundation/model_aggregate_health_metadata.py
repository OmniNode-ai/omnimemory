"""
ONEX-compliant typed model for aggregate health metadata.

This module provides strongly typed replacement for Dict[str, Any] patterns
in health management, ensuring type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelAggregateHealthMetadata(BaseModel):
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
