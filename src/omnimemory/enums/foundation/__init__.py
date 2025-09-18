"""
Foundation domain enums for OmniMemory following ONEX standards.

This module provides foundation enums for base implementations,
error handling, severity levels, and system-level operations.
"""

from .enum_data_type import EnumDataType
from .enum_error_code import EnumErrorCode, OmniMemoryErrorCode
from .enum_health_status import EnumHealthStatus
from .enum_migration_status import (
    FileProcessingStatus,
    MigrationPriority,
    MigrationStatus,
)
from .enum_priority_level import EnumPriorityLevel, PriorityLevel
from .enum_severity import EnumSeverity
from .enum_trust_level import EnumDecayFunction, EnumTrustLevel

__all__ = [
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
]
