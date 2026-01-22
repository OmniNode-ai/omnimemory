# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator - ONEX Node (Core 8 Foundation).

Full lifecycle management: store -> analyze -> consolidate -> archive.

This orchestrator manages the memory lifecycle state machine:
    ACTIVE -> EXPIRED -> ARCHIVED -> DELETED

Any state (except DELETED) can be PROMOTED back to ACTIVE.

Components:
    Adapters:
        - AdapterRuntimeTickMemory: Tick-based lifecycle processing
        - ModelMemoryLifecycleTickResult: Result model for tick processing

    Validators:
        - validate_transition: Check if a state transition is valid
        - apply_transition: Validate and compute transition results
        - VALID_TRANSITIONS: Dict of valid state transitions
        - ModelLifecycleTransitionResult: Result model for transitions
"""

from omnimemory.nodes.memory_lifecycle_orchestrator.adapters import (
    AdapterRuntimeTickMemory,
    ModelMemoryLifecycleTickResult,
)
from omnimemory.nodes.memory_lifecycle_orchestrator.validators import (
    VALID_TRANSITIONS,
    ModelLifecycleTransitionResult,
    apply_transition,
    validate_transition,
)

__all__ = [
    # Adapters
    "AdapterRuntimeTickMemory",
    "ModelMemoryLifecycleTickResult",
    # Validators
    "VALID_TRANSITIONS",
    "ModelLifecycleTransitionResult",
    "apply_transition",
    "validate_transition",
]
