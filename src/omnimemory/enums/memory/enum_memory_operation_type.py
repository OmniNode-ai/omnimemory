"""
Memory operation type enumeration.
"""

from enum import Enum


class EnumMemoryOperationType(Enum):
    """Memory operation types."""

    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    BULK_STORE = "bulk_store"
    BULK_RETRIEVE = "bulk_retrieve"
