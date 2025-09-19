"""
Operation type enumeration for ONEX Foundation Architecture.

Defines standardized operation types for system-wide consistency.
"""

from enum import Enum


class EnumOperationType(str, Enum):
    """Standardized operation types for system operations."""

    # Memory operations
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_SEARCH = "memory_search"

    # Semantic operations
    SEMANTIC_SEARCH = "semantic_search"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    SEMANTIC_INDEXING = "semantic_indexing"

    # Intelligence operations
    PATTERN_RECOGNITION = "pattern_recognition"
    INTELLIGENCE_ANALYSIS = "intelligence_analysis"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"

    # System operations
    HEALTH_CHECK = "health_check"
    CONFIGURATION_UPDATE = "configuration_update"
    MIGRATION = "migration"
    BACKUP = "backup"
    RESTORE = "restore"

    # Cache operations
    CACHE_SET = "cache_set"
    CACHE_GET = "cache_get"
    CACHE_DELETE = "cache_delete"
    CACHE_CLEAR = "cache_clear"

    # Validation operations
    VALIDATION = "validation"
    SCHEMA_VALIDATION = "schema_validation"
    DATA_VALIDATION = "data_validation"

    # Monitoring operations
    METRICS_COLLECTION = "metrics_collection"
    PERFORMANCE_MONITORING = "performance_monitoring"
    ERROR_TRACKING = "error_tracking"
