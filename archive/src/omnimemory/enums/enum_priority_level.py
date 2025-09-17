"""
Priority level enumerations for ONEX compliance.

This module contains priority level enum types following ONEX standards.
"""

from enum import Enum


class EnumPriorityLevel(str, Enum):
    """Priority levels for ONEX operations."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

    def get_numeric_value(self) -> float:
        """Get numeric value for priority calculations."""
        priority_values = {
            "critical": 100.0,
            "high": 75.0,
            "normal": 50.0,
            "low": 25.0,
            "background": 10.0,
        }
        return priority_values.get(self.value, 50.0)

    def is_high_priority(self) -> bool:
        """Check if this is high priority."""
        return self in (EnumPriorityLevel.CRITICAL, EnumPriorityLevel.HIGH)

    def requires_immediate_action(self) -> bool:
        """Check if this requires immediate action."""
        return self == EnumPriorityLevel.CRITICAL
