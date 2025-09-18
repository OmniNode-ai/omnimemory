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
    EnumDataType,
    EnumDecayFunction,
    EnumErrorCode,
    EnumHealthStatus,
    EnumPriorityLevel,
    EnumSeverity,
    EnumTrustLevel,
    FileProcessingStatus,
    MigrationPriority,
    MigrationStatus,
    OmniMemoryErrorCode,
    PriorityLevel,
)
from .intelligence import EnumIntelligenceOperationType
from .memory import EnumMemoryOperationType, EnumMemoryStorageType
from .service import (
    EnumCircuitBreakerState,
    EnumDiscoveryMethod,
    EnumEnvironment,
    EnumProtocol,
    EnumServiceType,
)

# Re-export all enums at package level
__all__ = [
    # Core enums
    "EnumOperationStatus",
    "EnumNodeType",
    # Foundation enums
    "EnumDataType",
    "EnumErrorCode",
    "OmniMemoryErrorCode",
    "EnumHealthStatus",
    "EnumPriorityLevel",
    "PriorityLevel",
    "EnumSeverity",
    "EnumTrustLevel",
    "EnumDecayFunction",
    "MigrationStatus",
    "MigrationPriority",
    "FileProcessingStatus",
    # Memory enums
    "EnumMemoryOperationType",
    "EnumMemoryStorageType",
    # Intelligence enums
    "EnumIntelligenceOperationType",
    # Service enums
    "EnumProtocol",
    "EnumCircuitBreakerState",
    "EnumDiscoveryMethod",
    "EnumEnvironment",
    "EnumServiceType",
]

# Compatibility aliases for commonly used enums
NodeType = EnumNodeType
