# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator Validators.

Validation logic for lifecycle state transitions.

Validators (to be implemented):
    - ValidatorLifecycleTransition: Validates state transition rules

Validation Rules:
    State transitions must follow the lifecycle state machine:
    - ACTIVE -> EXPIRED (TTL expiration)
    - ACTIVE -> ARCHIVED (explicit archival)
    - EXPIRED -> ARCHIVED (post-expiration archival)
    - ARCHIVED -> ACTIVE (restore command)
    - Any -> DELETED (terminal state)

Invalid Transitions:
    - DELETED -> any state (terminal, no recovery)
    - ARCHIVED -> EXPIRED (must restore first)
    - EXPIRED -> ACTIVE (must archive then restore)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1453.

Ticket: OMN-1453
"""

# TODO(OMN-1453): Add validator imports as implemented:
#   ValidatorLifecycleTransition

__all__: list[str] = []
