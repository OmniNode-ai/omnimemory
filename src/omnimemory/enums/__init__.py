"""
ONEX-compliant enums for omnimemory system.

All enums are centralized here for better maintainability and ONEX compliance.
"""

from .enum_error_code import OmniMemoryErrorCode
# Keep backward compatibility during migration
EnumErrorCode = OmniMemoryErrorCode
from .enum_intelligence_operation_type import EnumIntelligenceOperationType
from .enum_memory_operation_type import EnumMemoryOperationType
from .enum_memory_storage_type import EnumMemoryStorageType
from .enum_migration_status import MigrationStatus, MigrationPriority, FileProcessingStatus
from .enum_trust_level import EnumTrustLevel, EnumDecayFunction
from .enum_priority_level import EnumPriorityLevel

__all__ = [
    "OmniMemoryErrorCode",
    "EnumErrorCode",  # Backward compatibility alias
    "EnumIntelligenceOperationType",
    "EnumMemoryOperationType",
    "EnumMemoryStorageType",
    "MigrationStatus",
    "MigrationPriority",
    "FileProcessingStatus",
    "EnumTrustLevel",
    "EnumDecayFunction",
    "EnumPriorityLevel",
]