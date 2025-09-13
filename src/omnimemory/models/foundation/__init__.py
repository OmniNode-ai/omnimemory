"""
Foundation domain models for OmniMemory following ONEX standards.

This module provides foundation models for base implementations,
error handling, migration progress tracking, and system-level operations.
"""

from .enum_error_code import EnumErrorCode
from .enum_error_severity import EnumErrorSeverity
from .model_error_details import ModelErrorDetails
from .model_system_health import ModelSystemHealth
from .model_health_response import ModelHealthResponse, ModelDependencyStatus, ModelResourceMetrics
from .model_metrics_response import ModelMetricsResponse, ModelOperationCounts, ModelPerformanceMetrics, ModelResourceMetricsDetailed
from .model_configuration import ModelSystemConfiguration, ModelDatabaseConfig, ModelCacheConfig, ModelPerformanceConfig, ModelObservabilityConfig
from .model_migration_progress import (
    MigrationStatus,
    MigrationPriority,
    FileProcessingStatus,
    BatchProcessingMetrics,
    FileProcessingInfo,
    MigrationProgressMetrics,
    MigrationProgressTracker,
)

__all__ = [
    "EnumErrorCode",
    "EnumErrorSeverity",
    "ModelErrorDetails",
    "ModelSystemHealth",
    "ModelHealthResponse",
    "ModelDependencyStatus",
    "ModelResourceMetrics",
    "ModelMetricsResponse",
    "ModelOperationCounts",
    "ModelPerformanceMetrics",
    "ModelResourceMetricsDetailed",
    "ModelSystemConfiguration",
    "ModelDatabaseConfig",
    "ModelCacheConfig",
    "ModelPerformanceConfig",
    "ModelObservabilityConfig",

    # Migration progress tracking
    "MigrationStatus",
    "MigrationPriority",
    "FileProcessingStatus",
    "BatchProcessingMetrics",
    "FileProcessingInfo",
    "MigrationProgressMetrics",
    "MigrationProgressTracker",
]