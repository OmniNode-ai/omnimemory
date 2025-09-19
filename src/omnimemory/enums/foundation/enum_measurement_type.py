"""
Measurement type enumeration for success metrics.
"""

from enum import Enum


class EnumMeasurementType(str, Enum):
    """Types of operations that can be measured for success metrics."""

    MEMORY_STORAGE = "memory_storage"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_DELETE = "memory_delete"
    MEMORY_UPDATE = "memory_update"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    DATABASE_QUERY = "database_query"
    DATABASE_WRITE = "database_write"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    VALIDATION = "validation"
    SANITIZATION = "sanitization"
    INDEXING = "indexing"
    SEARCH = "search"
    MIGRATION = "migration"
    BACKUP = "backup"
    RESTORE = "restore"
    HEALTH_CHECK = "health_check"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
