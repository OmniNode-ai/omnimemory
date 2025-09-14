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