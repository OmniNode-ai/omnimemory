"""
Detailed resource utilization metrics model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelResourceMetricsDetailed(BaseModel):
    """Detailed resource utilization metrics."""

    memory_allocated_mb: float = Field(description="Memory allocated in megabytes")
    memory_used_mb: float = Field(description="Memory currently used in megabytes")
    cache_hit_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Cache hit rate percentage"
    )
    cache_size_mb: float = Field(description="Cache size in megabytes")
    database_connections_active: int = Field(
        description="Number of active database connections"
    )
    database_connections_idle: int = Field(
        description="Number of idle database connections"
    )
