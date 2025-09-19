"""
ONEX Enum Package - OmniMemory Foundation Architecture

Enums are organized into functional domains following omnibase_core patterns:
- core/: Core operation and node type enums
- foundation/: Base foundation enums (severity, priority, trust, etc.)
- memory/: Memory storage and operation enums
- intelligence/: Intelligence processing and analysis enums
- service/: Service configuration and orchestration enums

This __init__.py maintains compatibility by re-exporting
all enums at the package level following ONEX standards.
"""

# Import all enums from domains
from .core import EnumNodeType, EnumOperationStatus
from .foundation import (
    EnumCalculationMethod,
    EnumDataType,
    EnumDecayFunction,
    EnumErrorCode,
    EnumHealthStatus,
    EnumMeasurementBasis,
    EnumMeasurementType,
    EnumPriorityLevel,
    EnumQualityGrade,
    EnumResourceType,
    EnumSeverity,
    EnumTrustLevel,
    FileProcessingStatus,
    MigrationPriority,
    MigrationStatus,
    OmniMemoryErrorCode,
    PriorityLevel,
)
from .intelligence import EnumIntelligenceOperationType
from .memory import (
    EnumCompressionLevel,
    EnumEncodingFormat,
    EnumMemoryItemType,
    EnumMemoryOperationType,
    EnumMemoryStorageType,
    EnumMigrationStrategy,
    EnumRetentionPolicy,
    EnumStorageBackend,
)
from .service import (
    EnumCacheEvictionPolicy,
    EnumCircuitBreakerState,
    EnumDiscoveryMethod,
    EnumEnvironment,
    EnumProtocol,
    EnumServiceType,
    EnumTrendDirection,
)

# Re-export all enums at package level
__all__ = [
    # Core enums
    "EnumOperationStatus",
    "EnumNodeType",
    # Foundation enums
    "EnumCalculationMethod",
    "EnumDataType",
    "EnumErrorCode",
    "OmniMemoryErrorCode",
    "EnumHealthStatus",
    "EnumMeasurementBasis",
    "EnumMeasurementType",
    "EnumPriorityLevel",
    "PriorityLevel",
    "EnumQualityGrade",
    "EnumResourceType",
    "EnumSeverity",
    "EnumTrustLevel",
    "EnumDecayFunction",
    "MigrationStatus",
    "MigrationPriority",
    "FileProcessingStatus",
    # Memory enums
    "EnumCompressionLevel",
    "EnumEncodingFormat",
    "EnumMemoryItemType",
    "EnumMemoryOperationType",
    "EnumMemoryStorageType",
    "EnumMigrationStrategy",
    "EnumRetentionPolicy",
    "EnumStorageBackend",
    # Intelligence enums
    "EnumIntelligenceOperationType",
    # Service enums
    "EnumProtocol",
    "EnumCircuitBreakerState",
    "EnumDiscoveryMethod",
    "EnumEnvironment",
    "EnumServiceType",
    "EnumTrendDirection",
    "EnumCacheEvictionPolicy",
]

# Compatibility aliases for commonly used enums
NodeType = EnumNodeType
