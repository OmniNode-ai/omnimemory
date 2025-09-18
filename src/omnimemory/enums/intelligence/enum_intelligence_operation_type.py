"""
Intelligence operation type enumeration.
"""

from enum import Enum


class EnumIntelligenceOperationType(Enum):
    """Intelligence operation types."""

    ANALYZE = "analyze"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    PREDICT = "predict"
    OPTIMIZE = "optimize"
