"""
Enum for memory operation types following ONEX standards.
"""

from enum import Enum


class EnumMemoryOperationType(str, Enum):
    """Types of operations in the ONEX memory system."""

    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    ANALYZE = "analyze"
    CONSOLIDATE = "consolidate"
    OPTIMIZE = "optimize"
    HEALTH_CHECK = "health_check"
    SYNC = "sync"