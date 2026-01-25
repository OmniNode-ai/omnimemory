# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent Query Effect - ONEX Node.

This node provides event-driven intent query operations via Kafka,
supporting multiple query types:

- **Distribution Query**: Get intent distribution across categories
- **Session Query**: Get intents for a specific session
- **Recent Query**: Get recent intents across all sessions

The node follows the ONEX EFFECT pattern, consuming events from Kafka,
querying Memgraph for intent data, and publishing response events.

Example::

    import asyncio
    from omnimemory.nodes.intent_query_effect import HandlerIntentQuery
    from omnimemory.handlers.adapters import (
        AdapterIntentGraph,
        ModelAdapterIntentGraphConfig,
    )

    async def main():
        # Initialize adapter first
        adapter_config = ModelAdapterIntentGraphConfig()
        adapter = AdapterIntentGraph(adapter_config)
        await adapter.initialize(connection_uri="bolt://localhost:7687")

        # Initialize handler with adapter
        handler = HandlerIntentQuery()
        await handler.initialize(adapter)

        # Handler processes Kafka events automatically
        # Or can be called directly for testing

        await handler.shutdown()
        await adapter.shutdown()

    asyncio.run(main())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
from omnimemory.nodes.intent_query_effect.models import ModelHandlerIntentQueryConfig
from omnimemory.nodes.intent_query_effect.registry import RegistryIntentQueryEffect
from omnimemory.nodes.intent_query_effect.utils import (
    map_intent_records,
    map_to_intent_payload,
)

__all__ = [
    "HandlerIntentQuery",
    "ModelHandlerIntentQueryConfig",
    "RegistryIntentQueryEffect",
    "map_intent_records",
    "map_to_intent_payload",
]
