# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator - ONEX Node (Core 8 Foundation).

Manages memory lifecycle transitions: ACTIVE -> EXPIRED -> ARCHIVED -> DELETED.
Handles TTL expiration via RuntimeTick events and optimistic locking
for concurrent safety.

Node Type: ORCHESTRATOR
- Workflow coordination for memory lifecycle state transitions
- TTL expiration evaluation triggered by RuntimeTick events
- Explicit archival, expiration, and restoration commands
- Access tracking for TTL extension and pattern analysis

Time Injection:
    The orchestrator receives deterministic timestamps from RuntimeTick events
    via the `now` parameter. All timeout evaluation uses injected time rather
    than system clock, enabling deterministic testing and consistent behavior
    across distributed deployments.

Lifecycle States:
    - ACTIVE: Memory is available for retrieval and actively used
    - EXPIRED: Memory has exceeded TTL, pending cleanup
    - ARCHIVED: Memory has been moved to cold storage
    - DELETED: Memory has been permanently removed (terminal state)

ONEX 4.0 Declarative Pattern:
    This node follows the fully declarative ONEX pattern:
    - contract.yaml defines the node type, inputs, outputs, and dependencies
    - Business logic lives in handlers (handler_memory_tick, handler_archive_memory, etc.)
    - No node.py class needed - the contract IS the node definition

Handlers::

    from omnimemory.nodes.memory_lifecycle_orchestrator import (
        HandlerMemoryTick,
        HandlerMemoryArchive,
        HandlerMemoryExpire,
        HandlerRestoreMemory,
        HandlerMemoryAccessed,
    )

Models::

    from omnimemory.nodes.memory_lifecycle_orchestrator import (
        ModelLifecycleOrchestratorInput,
        ModelLifecycleOrchestratorOutput,
        ModelArchiveMemoryCommand,
        ModelExpireMemoryCommand,
        ModelRestoreMemoryCommand,
    )

.. versionadded:: 0.1.0
    Initial implementation for OMN-1453.

Ticket: OMN-1453
"""

# Implemented handler imports
from .handlers import (
    HandlerMemoryExpire,
    HandlerMemoryTick,
    ModelExpireMemoryCommand,
    ModelMemoryCurrentState,
    ModelMemoryExpireResult,
    ModelMemoryTickResult,
)

# TODO(OMN-1453): Add handler imports as implemented:
#   HandlerMemoryArchive, HandlerRestoreMemory, HandlerMemoryAccessed

# TODO(OMN-1453): Add model imports as implemented:
#   ModelLifecycleOrchestratorInput, ModelLifecycleOrchestratorOutput,
#   ModelArchiveMemoryCommand, ModelRestoreMemoryCommand

__all__: list[str] = [
    # Implemented handlers
    "HandlerMemoryTick",
    "HandlerMemoryExpire",
    "ModelMemoryTickResult",
    "ModelExpireMemoryCommand",
    "ModelMemoryExpireResult",
    "ModelMemoryCurrentState",
]
