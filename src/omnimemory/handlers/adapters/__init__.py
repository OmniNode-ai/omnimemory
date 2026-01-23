# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Handler Adapters.

This package contains adapter layers that wrap omnibase_infra handlers
to provide memory-specific interfaces. Adapters translate between
memory domain concepts and underlying infrastructure operations.

Available Adapters:
    - AdapterGraphMemory: Wraps HandlerGraph for relationship-based memory queries
    - AdapterValkey: Valkey/Redis adapter for subscription caching

Example::

    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        AdapterGraphMemoryConfig,
        AdapterValkey,
        AdapterValkeyConfig,
    )

    # Graph adapter for memory relationships
    config = AdapterGraphMemoryConfig(max_depth=3)
    adapter = AdapterGraphMemory(config)
    await adapter.initialize(connection_uri="bolt://localhost:7687")
    related = await adapter.find_related("memory_123", depth=2)

    # Valkey adapter for caching
    valkey_config = AdapterValkeyConfig(host="localhost", port=6379)
    valkey = AdapterValkey(valkey_config)
    await valkey.initialize()
    await valkey.set_key("key", "value")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.

.. versionadded:: 0.1.0
    AdapterValkey added for OMN-1393.
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
from omnimemory.handlers.adapters.adapter_valkey import (
    AdapterValkey,
    AdapterValkeyConfig,
    ModelValkeyHealth,
)

__all__ = [
    # Graph Memory Adapter
    "AdapterGraphMemory",
    "AdapterGraphMemoryConfig",
    "ModelConnectionsResult",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    # Valkey Adapter
    "AdapterValkey",
    "AdapterValkeyConfig",
    "ModelValkeyHealth",
]
