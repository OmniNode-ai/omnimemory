# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator Handlers.

This module exports handlers for the memory lifecycle orchestrator node.

Handlers:
    HandlerMemoryTick: Processes RuntimeTick events to evaluate memory TTL
        expirations and archive candidates. Emits MemoryExpired and
        ArchiveInitiated events for memories past their deadlines.

    HandlerMemoryExpire: Performs state transition ACTIVE -> EXPIRED with
        optimistic locking. Uses revision-based concurrency control to
        handle concurrent access safely.

    HandlerMemoryArchive: Archives EXPIRED memories to cold storage with
        gzip compression and atomic writes. Uses optimistic locking to
        prevent double-archive race conditions.

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration
    - OMN-1524: Atomic write primitive (pending)
"""

from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_archive import (
    HandlerMemoryArchive,
    ModelArchiveMemoryCommand,
    ModelArchiveRecord,
    ModelMemoryArchiveResult,
    ProtocolOrphanedArchiveTracker,
)
from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
    HandlerMemoryExpire,
    ModelExpireMemoryCommand,
    ModelMemoryCurrentState,
    ModelMemoryExpireResult,
)
from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_tick import (
    HandlerMemoryTick,
    ModelMemoryTickResult,
)

__all__ = [
    # Tick handler
    "HandlerMemoryTick",
    "ModelMemoryTickResult",
    # Expire handler
    "HandlerMemoryExpire",
    "ModelExpireMemoryCommand",
    "ModelMemoryCurrentState",
    "ModelMemoryExpireResult",
    # Archive handler
    "HandlerMemoryArchive",
    "ModelArchiveMemoryCommand",
    "ModelArchiveRecord",
    "ModelMemoryArchiveResult",
    "ProtocolOrphanedArchiveTracker",
]
