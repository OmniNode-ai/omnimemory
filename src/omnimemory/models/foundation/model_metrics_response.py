"""
Metrics response model following ONEX standards.
"""

from .model_metrics_response_main import ModelMetricsResponse

# Import and re-export models for backwards compatibility
from .model_operation_counts import ModelOperationCounts
from .model_performance_metrics import ModelPerformanceMetrics
from .model_resource_metrics_detailed import ModelResourceMetricsDetailed

__all__ = [
    "ModelOperationCounts",
    "ModelPerformanceMetrics",
    "ModelResourceMetricsDetailed",
    "ModelMetricsResponse",
]
