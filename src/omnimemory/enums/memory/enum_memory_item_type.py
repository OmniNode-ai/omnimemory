"""
Memory item type enum for OmniMemory following ONEX standards.
"""

from enum import Enum


class EnumMemoryItemType(str, Enum):
    """Types of memory items in the OmniMemory system."""

    DOCUMENT = "document"
    EMBEDDING = "embedding"
    TEMPORAL = "temporal"
    PERSISTENT = "persistent"
    VECTOR = "vector"
    STRUCTURED = "structured"
    INTELLIGENCE = "intelligence"
    PATTERN = "pattern"
    METADATA = "metadata"
    CACHE = "cache"
    INDEX = "index"
    RELATIONSHIP = "relationship"
