# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Lifecycle state enumeration following ONEX standards.
"""

from enum import Enum


class EnumLifecycleState(str, Enum):
    """
    Memory lifecycle states for the lifecycle orchestrator.

    Represents the current lifecycle state of a memory entity:
    - ACTIVE: Memory is live and accessible for reads/writes
    - STALE: Memory is outdated but still accessible (soft TTL exceeded)
    - EXPIRED: Memory TTL has passed, pending archive transition
    - ARCHIVED: Memory moved to cold storage, read-only access
    - DELETED: Memory permanently removed (soft delete marker for audit trail)

    Lifecycle transitions:
        ACTIVE -> STALE -> EXPIRED -> ARCHIVED -> DELETED

    STALE is an intermediate state between ACTIVE and EXPIRED that indicates
    the memory's soft TTL has been exceeded but it remains accessible. This
    allows for graceful degradation where consumers can be notified of staleness
    before hard expiration occurs.
    """

    ACTIVE = "active"
    STALE = "stale"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DELETED = "deleted"
