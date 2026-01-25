# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator Validators.

Validation logic for lifecycle state transitions.

Validators (to be implemented):
    - ValidatorLifecycleTransition: Validates state transition rules

Validation Rules:
    State transitions must follow the lifecycle state machine:
    - ACTIVE -> STALE (access timeout)
    - ACTIVE -> EXPIRED (explicit expiration)
    - ACTIVE -> ARCHIVED (explicit archival)
    - STALE -> EXPIRED (TTL expiration)
    - STALE -> ARCHIVED (explicit archival)
    - EXPIRED -> ARCHIVED (post-expiration archival)
    - ARCHIVED -> ACTIVE (restore command)

Invalid Transitions:
    - DELETED -> any state (terminal)
    - ARCHIVED -> EXPIRED (must restore first)
    - any state -> ACTIVE (except restore from ARCHIVED)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1453.

Ticket: OMN-1453
"""

# TODO(OMN-1453): Add validator imports as implemented:
#   ValidatorLifecycleTransition

__all__: list[str] = []
