"""
Memory domain enums for OmniMemory following ONEX standards.

This module provides memory-specific enums for storage, retrieval,
and memory management operations.
"""

from .enum_memory_operation_type import EnumMemoryOperationType
from .enum_memory_storage_type import EnumMemoryStorageType

__all__ = [
    "EnumMemoryOperationType",
    "EnumMemoryStorageType",
]
