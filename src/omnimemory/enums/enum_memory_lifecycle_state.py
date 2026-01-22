"""
Enum for memory lifecycle states following ONEX standards.

Note: PROMOTED is not a state - it's a transition that returns to ACTIVE.
"""

from enum import Enum


class EnumMemoryLifecycleState(str, Enum):
    """Memory lifecycle states for the state machine."""

    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DELETED = "deleted"
