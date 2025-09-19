"""
ONEX-compliant typed model for resource usage metadata.

This module provides strongly typed replacement for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ModelResourceUsageMetadata(BaseModel):
    """Strongly typed resource usage metrics."""

    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during operation (0-100%)",
    )

    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Memory usage in megabytes (positive values only)",
    )

    disk_io_bytes: Optional[int] = Field(default=None, description="Disk I/O in bytes")

    network_io_bytes: Optional[int] = Field(
        default=None, description="Network I/O in bytes"
    )

    operation_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Duration of operation in milliseconds (for performance monitoring)",
    )

    database_queries: Optional[int] = Field(
        default=None, description="Number of database queries performed"
    )

    cache_hits: Optional[int] = Field(default=None, description="Number of cache hits")

    cache_misses: Optional[int] = Field(
        default=None, description="Number of cache misses"
    )
