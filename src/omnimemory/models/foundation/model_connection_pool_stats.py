"""
Connection pool statistics model following ONEX standards.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ModelConnectionPoolStats(BaseModel):
    """Strongly typed connection pool statistics."""

    pool_name: str = Field(description="Name of the connection pool")

    total_connections: int = Field(description="Total number of connections in pool")

    active_connections: int = Field(
        description="Number of currently active connections"
    )

    idle_connections: int = Field(description="Number of idle connections")

    max_connections: int = Field(description="Maximum allowed connections")

    pool_exhaustions: int = Field(
        default=0, description="Number of times the pool was exhausted"
    )

    average_wait_time_ms: Optional[float] = Field(
        default=None, description="Average wait time for connection acquisition"
    )

    longest_wait_time_ms: Optional[float] = Field(
        default=None, description="Longest wait time for connection acquisition"
    )

    total_connections_created: int = Field(
        default=0, description="Total connections created since startup"
    )

    total_connections_destroyed: int = Field(
        default=0, description="Total connections destroyed since startup"
    )

    health_check_failures: int = Field(
        default=0, description="Number of connection health check failures"
    )
