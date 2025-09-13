"""
Enum for error severity levels following ONEX standards.
"""

from enum import Enum


class EnumErrorSeverity(str, Enum):
    """Severity levels for errors in the ONEX memory system."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"