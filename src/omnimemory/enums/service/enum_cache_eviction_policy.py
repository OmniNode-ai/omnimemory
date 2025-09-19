"""
Cache eviction policy enum for ONEX standards.
"""

from enum import Enum


class EnumCacheEvictionPolicy(str, Enum):
    """Cache eviction policies for memory management."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    RANDOM = "random"
    TTL = "ttl"
    NONE = "none"
