# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator Adapters.

Storage adapters for memory lifecycle operations including runtime tick
processing and memory deactivation.

Adapters:
    AdapterRuntimeTickMemory: Tick-based lifecycle detection adapter.
        Wraps HandlerMemoryTick to provide TTL expiration and archive
        candidate detection via runtime tick events.

    AdapterPostgresDeactivateMemory: Memory expiration operations adapter.
        Wraps HandlerMemoryExpire to provide ACTIVE -> EXPIRED state
        transitions with optimistic locking against PostgreSQL.

Adapter Pattern:
    Adapters wrap external dependencies (storage backends, databases) and
    provide a consistent interface for handlers. All adapters implement
    protocol-based interfaces for dependency injection and testing.

Related Tickets:
    - OMN-1603: Add adapter implementations for memory lifecycle orchestrator
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration

.. versionadded:: 0.1.0
    Initial implementation for OMN-1603.
"""

from omnimemory.nodes.node_memory_lifecycle_orchestrator.adapters.adapter_postgres_deactivate_memory import (
    AdapterPostgresDeactivateMemory,
    ModelDeactivateAdapterHealth,
    ModelDeactivateAdapterMetadata,
)
from omnimemory.nodes.node_memory_lifecycle_orchestrator.adapters.adapter_runtime_tick_memory import (
    AdapterRuntimeTickMemory,
    ModelRuntimeTickAdapterHealth,
    ModelRuntimeTickAdapterMetadata,
)

__all__: list[str] = [
    "AdapterRuntimeTickMemory",
    "ModelRuntimeTickAdapterHealth",
    "ModelRuntimeTickAdapterMetadata",
    "AdapterPostgresDeactivateMemory",
    "ModelDeactivateAdapterHealth",
    "ModelDeactivateAdapterMetadata",
]
