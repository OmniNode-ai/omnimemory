"""
Compression level enum for OmniMemory following ONEX standards.
"""

from enum import Enum


class EnumCompressionLevel(str, Enum):
    """Compression levels for memory storage optimization."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"
