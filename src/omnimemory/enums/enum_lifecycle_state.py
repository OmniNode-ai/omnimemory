"""
Lifecycle state enumeration following ONEX standards.
"""

from enum import Enum


class EnumLifecycleState(str, Enum):
    """
    Memory lifecycle states for the lifecycle orchestrator.

    Represents the current lifecycle state of a memory entity:
    - ACTIVE: Memory is live and accessible for reads/writes
    - EXPIRED: Memory TTL has passed, pending archive transition
    - ARCHIVED: Memory moved to cold storage, read-only access
    - DELETED: Memory permanently removed (soft delete marker for audit trail)
    """

    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DELETED = "deleted"
