# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Handler Adapters.

This package contains adapter layers that wrap omnibase_infra handlers
to provide memory-specific interfaces. Adapters translate between
memory domain concepts and underlying infrastructure operations.

Available Adapters:
    - AdapterGraphMemory: Wraps HandlerGraph for relationship-based memory queries

Example::

    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        AdapterGraphMemoryConfig,
    )

    config = AdapterGraphMemoryConfig(max_depth=3)
    adapter = AdapterGraphMemory(config)
    await adapter.initialize(connection_uri="bolt://localhost:7687")

    related = await adapter.find_related("memory_123", depth=2)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from omnimemory.handlers.adapters.adapter_graph_memory import (
    AdapterGraphMemory,
    AdapterGraphMemoryConfig,
    ModelConnectionsResult,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
)

__all__ = [
    "AdapterGraphMemory",
    "AdapterGraphMemoryConfig",
    "ModelConnectionsResult",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
]
