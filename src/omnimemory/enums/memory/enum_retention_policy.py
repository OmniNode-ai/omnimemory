"""
Retention policy enum for OmniMemory following ONEX standards.
"""

from enum import Enum


class EnumRetentionPolicy(str, Enum):
    """Memory retention policies for lifecycle management."""

    PERMANENT = "permanent"
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    SIZE_BASED = "size_based"
    CONDITIONAL = "conditional"
    MANUAL = "manual"
