"""
Severity level enumeration.
"""

from enum import IntEnum


class EnumSeverity(IntEnum):
    """Severity levels."""

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5
    FATAL = 6
