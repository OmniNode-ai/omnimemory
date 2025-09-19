"""
Trend direction enum for ONEX standards.
"""

from enum import Enum


class EnumTrendDirection(str, Enum):
    """Trend directions for health and performance metrics."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"
    CRITICAL = "critical"
