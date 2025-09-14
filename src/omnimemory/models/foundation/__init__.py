"""
Foundation domain models for OmniMemory following ONEX standards.

This module provides foundation models for base implementations,
error handling, migration progress tracking, and system-level operations.
"""

from ...enums.enum_error_code import EnumErrorCode
from ...enums.enum_severity import EnumSeverity
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
from .model_typed_collections import (
    ModelStringList,
    ModelOptionalStringList,
    ModelKeyValuePair,
    ModelMetadata,
    ModelStructuredField,
    ModelStructuredData,
    ModelConfigurationOption,
    ModelConfiguration,
    ModelEventData,
    ModelEventCollection,
    ModelResultItem,
    ModelResultCollection,
    convert_dict_to_metadata,
    convert_list_to_string_list,
    convert_list_of_dicts_to_structured_data,
)
from .model_semver import ModelSemVer
from .model_success_metrics import ModelSuccessRate, ModelConfidenceScore, ModelQualityMetrics
from .model_notes import ModelNote, ModelNotesCollection
from .model_memory_data import (
    ModelMemoryDataValue,
    ModelMemoryDataContent,
    ModelMemoryRequestData,
    ModelMemoryResponseData,
)

__all__ = [
    "EnumErrorCode",
    "EnumSeverity",
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

    # Typed collections replacing generic types
    "ModelStringList",
    "ModelOptionalStringList",
    "ModelKeyValuePair",
    "ModelMetadata",
    "ModelStructuredField",
    "ModelStructuredData",
    "ModelConfigurationOption",
    "ModelConfiguration",
    "ModelEventData",
    "ModelEventCollection",
    "ModelResultItem",
    "ModelResultCollection",
    "convert_dict_to_metadata",
    "convert_list_to_string_list",
    "convert_list_of_dicts_to_structured_data",

    # New foundation models
    "ModelSemVer",
    "ModelSuccessRate",
    "ModelConfidenceScore",
    "ModelQualityMetrics",
    "ModelNote",
    "ModelNotesCollection",
    "ModelMemoryDataValue",
    "ModelMemoryDataContent",
    "ModelMemoryRequestData",
    "ModelMemoryResponseData",
]