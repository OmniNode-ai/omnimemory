"""
Enum for memory operation types following ONEX standards.
"""

from enum import Enum


class EnumMemoryOperationType(str, Enum):
    """
    Types of operations in the ONEX memory system.

    Defines all possible operations that can be performed on memory data:
    - STORE: Store new memory data in the system
    - RETRIEVE: Fetch existing memory data by key or query
    - UPDATE: Modify existing memory data
    - DELETE: Remove memory data from the system
    - SEARCH: Perform semantic or structured search
    - ANALYZE: Analyze memory patterns and relationships
    - CONSOLIDATE: Merge or consolidate related memories
    - OPTIMIZE: Optimize memory storage and retrieval performance
    - HEALTH_CHECK: Check system health and availability
    - SYNC: Synchronize memory data across nodes or systems
    """

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