"""
Storage backend enum for OmniMemory following ONEX standards.
"""

from enum import Enum


class EnumStorageBackend(str, Enum):
    """Storage backend types for memory operations."""

    MEMORY_CACHE = "memory_cache"
    POSTGRESQL = "postgresql"
    PINECONE = "pinecone"
    SUPABASE = "supabase"
    REDIS = "redis"
    FILE_SYSTEM = "file_system"
    S3 = "s3"
    VECTOR_DB = "vector_db"
