"""
Performance metrics model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for operations."""

    average_latency_ms: float = Field(
        description="Average operation latency in milliseconds"
    )
    p95_latency_ms: float = Field(description="95th percentile latency in milliseconds")
    p99_latency_ms: float = Field(description="99th percentile latency in milliseconds")
    throughput_ops_per_second: float = Field(
        description="Operations per second throughput"
    )
    error_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Error rate as percentage"
    )
    success_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Success rate as percentage"
    )
