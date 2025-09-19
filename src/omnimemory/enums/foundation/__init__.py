"""
Foundation domain enums for OmniMemory following ONEX standards.

This module provides foundation enums for base implementations,
error handling, severity levels, and system-level operations.
"""

from .enum_calculation_method import EnumCalculationMethod
from .enum_data_type import EnumDataType
from .enum_error_code import EnumErrorCode, OmniMemoryErrorCode
from .enum_health_status import EnumHealthStatus
from .enum_measurement_basis import EnumMeasurementBasis
from .enum_measurement_type import EnumMeasurementType
from .enum_migration_status import (
    FileProcessingStatus,
    MigrationPriority,
    MigrationStatus,
)
from .enum_priority_level import EnumPriorityLevel, PriorityLevel
from .enum_quality_grade import EnumQualityGrade
from .enum_resource_type import EnumResourceType
from .enum_severity import EnumSeverity
from .enum_trust_level import EnumDecayFunction, EnumTrustLevel

__all__ = [
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
    "EnumSeverity",
    "EnumTrustLevel",
    "EnumDecayFunction",
    "MigrationStatus",
    "MigrationPriority",
    "FileProcessingStatus",
    "EnumResourceType",
]
