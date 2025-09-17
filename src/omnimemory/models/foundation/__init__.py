"""
Foundation domain models for OmniMemory following ONEX standards.

ONEX Compliance: One model per file, zero backwards compatibility.
"""

from .model_audit_metadata import (
    AuditEventDetails,
    ModelResourceUsageMetadata,
    PerformanceAuditDetails,
    SecurityAuditDetails,
)
from .model_confidence_score import ModelConfidenceScore
from .model_configuration import (
    ModelCacheConfig,
    ModelDatabaseConfig,
    ModelObservabilityConfig,
    ModelPerformanceConfig,
    ModelSystemConfiguration,
)
from .model_connection_metadata import (
    ConnectionPoolStats,
    ModelConnectionMetadata,
    SemaphoreMetrics,
)

# Individual model imports (ONEX compliant - one model per file)
from .model_error_details import ModelErrorDetails
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
from .model_key_value_pair import ModelKeyValuePair
from .model_memory_data import (
    ModelMemoryDataContent,
    ModelMemoryDataValue,
    ModelMemoryRequestData,
    ModelMemoryResponseData,
)
from .model_metrics_response import ModelMetricsResponse
from .model_migration_progress import (
    BatchProcessingMetrics,
    FileProcessingInfo,
    FileProcessingStatus,
    MigrationPriority,
    MigrationProgressMetrics,
    MigrationProgressTracker,
    MigrationStatus,
)
from .model_notes import ModelNote, ModelNotesCollection
from .model_operation_counts import ModelOperationCounts
from .model_optional_string_list import ModelOptionalStringList
from .model_performance_metrics import ModelPerformanceMetrics
from .model_progress_summary import ProgressSummaryResponse
from .model_quality_metrics import ModelQualityMetrics
from .model_resource_metrics_detailed import ModelResourceMetricsDetailed
from .model_string_list import ModelStringList
from .model_success_rate import ModelSuccessRate
from .model_system_health import ModelSystemHealth
from .model_tag import ModelTag
from .model_tag_collection import ModelTagCollection

__all__ = [
    "ModelErrorDetails",
    "ModelSystemHealth",
    "ModelHealthResponse",
    "ModelDependencyStatus",
    "ModelResourceMetrics",
    "ModelSystemConfiguration",
    "ModelDatabaseConfig",
    "ModelCacheConfig",
    "ModelPerformanceConfig",
    "ModelObservabilityConfig",
    "MigrationStatus",
    "MigrationPriority",
    "FileProcessingStatus",
    "BatchProcessingMetrics",
    "FileProcessingInfo",
    "MigrationProgressMetrics",
    "MigrationProgressTracker",
    "ModelOperationCounts",
    "ModelPerformanceMetrics",
    "ModelResourceMetricsDetailed",
    "ModelMetricsResponse",
    "ModelSuccessRate",
    "ModelConfidenceScore",
    "ModelQualityMetrics",
    "ModelTag",
    "ModelTagCollection",
    "ModelStringList",
    "ModelOptionalStringList",
    "ModelKeyValuePair",
    "ModelNote",
    "ModelNotesCollection",
    "ModelMemoryDataValue",
    "ModelMemoryDataContent",
    "ModelMemoryRequestData",
    "ModelMemoryResponseData",
    "ModelHealthCheckMetadata",
    "ModelAggregateHealthMetadata",
    "ModelConfigurationChangeMetadata",
    "AuditEventDetails",
    "ModelResourceUsageMetadata",
    "SecurityAuditDetails",
    "PerformanceAuditDetails",
    "ModelConnectionMetadata",
    "ConnectionPoolStats",
    "SemaphoreMetrics",
    "ProgressSummaryResponse",
]
