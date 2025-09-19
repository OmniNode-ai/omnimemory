"""
ONEX-compliant typed model for health check metadata.

This module provides strongly typed replacement for Dict[str, Any] patterns
in health management, ensuring type safety and validation.
"""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ModelHealthCheckMetadata(BaseModel):
    """Strongly typed metadata for health check operations."""

    connection_url: Optional[str] = Field(
        default=None, description="Connection URL for dependency checks"
    )

    database_version: Optional[str] = Field(
        default=None, description="Version information for database dependencies"
    )

    pool_stats: Optional[Dict[str, int]] = Field(
        default=None, description="Connection pool statistics"
    )

    request_count: int = Field(default=0, description="Number of requests processed")

    error_count: int = Field(default=0, description="Number of errors encountered")

    last_success_timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp of last successful check"
    )

    circuit_breaker_state: Optional[str] = Field(
        default=None, description="Current circuit breaker state"
    )

    performance_metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Performance metrics (latency, throughput)"
    )
