"""
Priority level enumeration.
"""

from enum import IntEnum


class EnumPriorityLevel(IntEnum):
    """Priority levels."""

    LOWEST = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6


# Alias for compatibility
PriorityLevel = EnumPriorityLevel
