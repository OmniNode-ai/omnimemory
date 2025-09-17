"""
ONEX-compliant typed models for connection pool metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in connection pooling, ensuring type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelConnectionMetadata(BaseModel):
    """Strongly typed metadata for connection objects."""

    connection_id: str = Field(description="Unique identifier for this connection")

    created_at: datetime = Field(
        default_factory=datetime.now, description="When this connection was created"
    )

    last_used_at: Optional[datetime] = Field(
        default=None, description="When this connection was last used"
    )

    usage_count: int = Field(
        default=0, description="Number of times this connection has been used"
    )

    connection_string: Optional[str] = Field(
        default=None, description="Connection string (sanitized)"
    )

    database_name: Optional[str] = Field(
        default=None, description="Name of the database"
    )

    server_version: Optional[str] = Field(
        default=None, description="Server version information"
    )

    is_healthy: bool = Field(
        default=True, description="Whether the connection is healthy"
    )

    last_health_check: Optional[datetime] = Field(
        default=None, description="When the connection was last health checked"
    )

    error_count: int = Field(
        default=0, description="Number of errors encountered with this connection"
    )

    last_error: Optional[str] = Field(
        default=None, description="Last error message (sanitized)"
    )


class ConnectionPoolStats(BaseModel):
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


class SemaphoreMetrics(BaseModel):
    """Strongly typed semaphore performance metrics."""

    name: str = Field(description="Name of the semaphore")

    max_value: int = Field(description="Maximum value of the semaphore")

    current_value: int = Field(description="Current value of the semaphore")

    waiting_count: int = Field(description="Number of tasks waiting for the semaphore")

    total_acquisitions: int = Field(
        default=0, description="Total number of semaphore acquisitions"
    )

    total_releases: int = Field(
        default=0, description="Total number of semaphore releases"
    )

    average_hold_time_ms: Optional[float] = Field(
        default=None, description="Average time semaphore is held"
    )

    max_hold_time_ms: Optional[float] = Field(
        default=None, description="Maximum time semaphore was held"
    )

    acquisition_timeouts: int = Field(
        default=0, description="Number of acquisition timeouts"
    )

    fairness_violations: int = Field(
        default=0, description="Number of fairness violations detected"
    )
