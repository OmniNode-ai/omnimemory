"""
Foundation domain models for OmniMemory following ONEX standards.

This module provides foundation models for base implementations,
error handling, migration progress tracking, and system-level operations.
"""

from ...enums.enum_error_code import EnumErrorCode
from ...enums.enum_severity import EnumSeverity
from .model_audit_metadata import (
    ModelAuditEventDetails,
    ModelPerformanceAuditDetails,
    ModelResourceUsageMetadata,
    ModelSecurityAuditDetails,
)
from .model_configuration import (
    ModelCacheConfig,
    ModelDatabaseConfig,
    ModelObservabilityConfig,
    ModelPerformanceConfig,
    ModelSystemConfiguration,
)
from .model_connection_metadata import (
    ModelConnectionMetadata,
    ModelConnectionPoolStats,
    ModelSemaphoreMetrics,
)
from .model_error_details import ModelErrorDetails

# New metadata models for replacing Dict[str, Any]
from .model_health_metadata import (
    ModelAggregateHealthMetadata,
    ModelConfigurationChangeMetadata,
    ModelHealthCheckMetadata,
)
from .model_health_response import (
    ModelDependencyStatus,
    ModelHealthResponse,
    ModelResourceMetrics,
)
from .model_memory_data import (
    ModelMemoryDataContent,
    ModelMemoryDataValue,
    ModelMemoryRequestData,
    ModelMemoryResponseData,
)
from .model_metrics_response import (
    ModelMetricsResponse,
    ModelOperationCounts,
    ModelPerformanceMetrics,
    ModelResourceMetricsDetailed,
)
from .model_migration_progress import (
    FileProcessingStatus,
    MigrationPriority,
    MigrationStatus,
    ModelBatchProcessingMetrics,
    ModelFileProcessingInfo,
    ModelMigrationProgressMetrics,
    ModelMigrationProgressTracker,
)
from .model_notes import ModelNote, ModelNotesCollection
from .model_progress_summary import ModelProgressSummaryResponse
from .model_semver import ModelSemVer
from .model_success_metrics import (
    ModelConfidenceScore,
    ModelQualityMetrics,
    ModelSuccessRate,
)
from .model_system_health import ModelSystemHealth
from .model_typed_collections import (
    ModelConfiguration,
    ModelConfigurationOption,
    ModelEventCollection,
    ModelEventData,
    ModelKeyValuePair,
    ModelMetadata,
    ModelOptionalStringList,
    ModelResultCollection,
    ModelResultItem,
    ModelStringList,
    ModelStructuredData,
    ModelStructuredField,
    convert_dict_to_metadata,
    convert_list_of_dicts_to_structured_data,
    convert_list_to_string_list,
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
    "ModelBatchProcessingMetrics",
    "ModelFileProcessingInfo",
    "ModelMigrationProgressMetrics",
    "ModelMigrationProgressTracker",
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
    # New typed metadata models
    "ModelHealthCheckMetadata",
    "ModelAggregateHealthMetadata",
    "ModelConfigurationChangeMetadata",
    "ModelAuditEventDetails",
    "ModelResourceUsageMetadata",
    "ModelSecurityAuditDetails",
    "ModelPerformanceAuditDetails",
    "ModelConnectionMetadata",
    "ModelConnectionPoolStats",
    "ModelSemaphoreMetrics",
    "ModelProgressSummaryResponse",
]
