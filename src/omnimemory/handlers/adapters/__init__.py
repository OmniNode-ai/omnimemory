# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Handler Adapters.

This package contains adapter layers that wrap omnibase_infra handlers
to provide memory-specific interfaces. Adapters translate between
memory domain concepts and underlying infrastructure operations.

Available Adapters:
    - AdapterGraphMemory: Wraps HandlerGraph for relationship-based memory queries
    - AdapterIntentGraph: Wraps HandlerGraph for intent classification storage

Example::

    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        AdapterGraphMemoryConfig,
        AdapterIntentGraph,
        ModelAdapterIntentGraphConfig,
    )

    # Memory graph adapter
    config = AdapterGraphMemoryConfig(max_depth=3)
    adapter = AdapterGraphMemory(config)
    await adapter.initialize(connection_uri="bolt://localhost:7687")
    related = await adapter.find_related("memory_123", depth=2)

    # Intent graph adapter
    intent_config = ModelAdapterIntentGraphConfig(timeout_seconds=30.0)
    intent_adapter = AdapterIntentGraph(intent_config)
    await intent_adapter.initialize(connection_uri="bolt://localhost:7687")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389 (AdapterGraphMemory).

.. versionadded:: 0.1.0
    Added AdapterIntentGraph for OMN-1457.
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
from omnimemory.handlers.adapters.adapter_intent_graph import (
    AdapterIntentGraph,
    IntentCypherTemplates,
)
from omnimemory.handlers.adapters.models import (
    ModelAdapterIntentGraphConfig,
    ModelIntentClassificationOutput,
    ModelIntentGraphHealth,
    ModelIntentQueryResult,
    ModelIntentRecord,
    ModelIntentStorageResult,
)

__all__ = [
    "AdapterGraphMemory",
    "AdapterGraphMemoryConfig",
    "AdapterIntentGraph",
    "ModelAdapterIntentGraphConfig",
    "IntentCypherTemplates",
    "ModelConnectionsResult",
    "ModelGraphMemoryHealth",
    "ModelIntentClassificationOutput",
    "ModelIntentGraphHealth",
    "ModelIntentQueryResult",
    "ModelIntentRecord",
    "ModelIntentStorageResult",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
]
