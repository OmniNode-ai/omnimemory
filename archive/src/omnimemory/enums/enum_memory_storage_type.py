"""
Enum for memory storage types following ONEX standards.
"""

from enum import Enum


class EnumMemoryStorageType(str, Enum):
    """Types of memory storage in the ONEX memory system."""

    VECTOR_DATABASE = "vector_database"
    RELATIONAL_DATABASE = "relational_database"
    DOCUMENT_STORE = "document_store"
    KEY_VALUE_STORE = "key_value_store"
    GRAPH_DATABASE = "graph_database"
    TIME_SERIES_DATABASE = "time_series_database"
    CACHE = "cache"
    FILE_SYSTEM = "file_system"
