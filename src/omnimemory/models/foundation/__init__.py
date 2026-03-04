# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Foundation domain models for OmniMemory following ONEX standards.

error handling, migration progress tracking, and system-level operations.
"""

from ...enums.enum_error_code import EnumOmniMemoryErrorCode
from ...enums.enum_migration_status import (
    EnumFileProcessingStatus,
    EnumMigrationPriority,
    EnumMigrationStatus,
)
from ...enums.enum_severity import EnumSeverity
from .model_audit_metadata import (
    AuditEventDetails,
    PerformanceAuditDetails,
    ResourceUsageMetadata,
    SecurityAuditDetails,
)
from .model_configuration import (
    ModelCacheConfig,
    ModelDatabaseConfig,
    ModelObservabilityConfig,
    ModelPerformanceConfig,
    ModelSystemConfiguration,
)
from .model_connection_metadata import (
    ConnectionMetadata,
    ConnectionPoolStats,
    SemaphoreMetrics,
)
from .model_error_details import ModelErrorDetails

# New metadata models for replacing Dict[str, Any]
from .model_health_metadata import (
    AggregateHealthMetadata,
    ConfigurationChangeMetadata,
    HealthCheckMetadata,
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
    BatchProcessingMetrics,
    FileProcessingInfo,
    MigrationProgressMetrics,
    MigrationProgressTracker,
)
from .model_notes import ModelNote, ModelNotesCollection
from .model_progress_summary import (
    ModelProgressPerformanceMetrics,
    ProgressSummaryResponse,
)
from .model_semver import ModelSemVer
from .model_success_metrics import (
    ModelConfidenceInterval,
    ModelConfidenceScore,
    ModelQualityMetrics,
    ModelSuccessRate,
)
from .model_system_health import ModelSystemHealth
from .model_tags import ModelTag, ModelTagCollection, normalize_tag_name
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
    "EnumOmniMemoryErrorCode",
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
    "EnumMigrationStatus",
    "EnumMigrationPriority",
    "EnumFileProcessingStatus",
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
    "ModelConfidenceInterval",
    "ModelConfidenceScore",
    "ModelQualityMetrics",
    "ModelNote",
    "ModelNotesCollection",
    "ModelMemoryDataValue",
    "ModelMemoryDataContent",
    "ModelMemoryRequestData",
    "ModelMemoryResponseData",
    # New typed metadata models
    "HealthCheckMetadata",
    "AggregateHealthMetadata",
    "ConfigurationChangeMetadata",
    "AuditEventDetails",
    "ResourceUsageMetadata",
    "SecurityAuditDetails",
    "PerformanceAuditDetails",
    "ConnectionMetadata",
    "ConnectionPoolStats",
    "SemaphoreMetrics",
    "ModelProgressPerformanceMetrics",
    "ProgressSummaryResponse",
    # Tag normalization support
    "ModelTag",
    "ModelTagCollection",
    "normalize_tag_name",
]
