"""
Main metrics response model following ONEX standards.
"""

from datetime import datetime
from typing import Dict

from pydantic import BaseModel, Field

from .model_operation_counts import ModelOperationCounts
from .model_performance_metrics import ModelPerformanceMetrics
from .model_resource_metrics_detailed import ModelResourceMetricsDetailed


class ModelMetricsResponse(BaseModel):
    """Comprehensive metrics response following ONEX standards."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When metrics were collected"
    )
    collection_duration_ms: float = Field(
        description="Time taken to collect metrics in milliseconds"
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
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Custom application-specific metrics"
    )
    alerts: list[str] = Field(
        default_factory=list, description="Active performance alerts"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Performance improvement recommendations"
    )
