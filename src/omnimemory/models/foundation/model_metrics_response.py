# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Metrics response model following ONEX standards.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ModelOperationCounts(BaseModel):
    """Count of operations by type."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    storage_operations: int = Field(
        default=0, description="Number of storage operations"
    )
    retrieval_operations: int = Field(
        default=0, description="Number of retrieval operations"
    )
    query_operations: int = Field(default=0, description="Number of query operations")
    consolidation_operations: int = Field(
        default=0, description="Number of consolidation operations"
    )
    failed_operations: int = Field(default=0, description="Number of failed operations")


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for operations."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    average_latency_ms: float = Field(
        ge=0.0, description="Average operation latency in milliseconds"
    )
    p95_latency_ms: float = Field(
        ge=0.0, description="95th percentile latency in milliseconds"
    )
    p99_latency_ms: float = Field(
        ge=0.0, description="99th percentile latency in milliseconds"
    )
    throughput_ops_per_second: float = Field(
        ge=0.0, description="Operations per second throughput"
    )
    error_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Error rate as percentage"
    )
    success_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Success rate as percentage"
    )


class ModelResourceMetricsDetailed(BaseModel):
    """Detailed resource utilization metrics."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    memory_allocated_mb: float = Field(
        ge=0.0, description="Memory allocated in megabytes"
    )
    memory_used_mb: float = Field(
        ge=0.0, description="Memory currently used in megabytes"
    )
    cache_hit_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Cache hit rate percentage"
    )
    cache_size_mb: float = Field(ge=0.0, description="Cache size in megabytes")
    database_connections_active: int = Field(
        ge=0, description="Number of active database connections"
    )
    database_connections_idle: int = Field(
        ge=0, description="Number of idle database connections"
    )


class ModelMetricsResponse(BaseModel):
    """Comprehensive metrics response following ONEX standards."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When metrics were collected",
    )
    collection_duration_ms: float = Field(
        ge=0.0, description="Time taken to collect metrics in milliseconds"
    )
    operation_counts: ModelOperationCounts = Field(
        description="Count of operations by type"
    )
    performance_metrics: ModelPerformanceMetrics = Field(
        description="Performance statistics"
    )
    resource_metrics: ModelResourceMetricsDetailed = Field(
        description="Detailed resource utilization"
    )
    custom_metrics: dict[str, float] = Field(
        default_factory=dict, description="Custom application-specific metrics"
    )
    alerts: list[str] = Field(
        default_factory=list, description="Active performance alerts"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Performance improvement recommendations"
    )
