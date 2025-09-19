"""
Memory storage type enumeration.
"""

from enum import Enum


class EnumMemoryStorageType(Enum):
    """Memory storage types."""

    PERSISTENT = "persistent"
    TEMPORARY = "temporary"
    CACHED = "cached"
    VECTOR = "vector"
    GRAPH = "graph"
    DISTRIBUTED = "distributed"
