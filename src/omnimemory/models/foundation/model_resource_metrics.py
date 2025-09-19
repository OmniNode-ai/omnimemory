"""
Resource metrics model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelResourceMetrics(BaseModel):
    """System resource utilization metrics."""

    cpu_usage_percent: float = Field(
        ge=0.0, le=100.0, description="CPU usage percentage"
    )
    memory_usage_mb: float = Field(description="Memory usage in megabytes")
    memory_usage_percent: float = Field(
        ge=0.0, le=100.0, description="Memory usage percentage"
    )
    disk_usage_percent: float = Field(
        ge=0.0, le=100.0, description="Disk usage percentage"
    )
    network_throughput_mbps: float = Field(
        description="Network throughput in megabits per second"
    )
