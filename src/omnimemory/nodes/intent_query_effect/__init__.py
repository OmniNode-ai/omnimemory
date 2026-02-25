# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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

This node uses the container-driven pattern where the handler owns
and manages the adapter lifecycle internally.

Example::

    import asyncio
    import os
    from omnibase_core.container import ModelONEXContainer
    from omnimemory.nodes.intent_query_effect import HandlerIntentQuery

    async def main():
        # Create container and handler
        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        # Initialize handler (creates and owns adapter internally)
        await handler.initialize(
            connection_uri=os.getenv("MEMGRAPH_URI", "bolt://localhost:7687"),
            auth=(os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", "")),
        )

        # Handler processes Kafka events automatically
        # Or can be called directly for testing

        # Shutdown releases all resources including adapter
        await handler.shutdown()

    asyncio.run(main())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.

.. versionchanged:: 0.2.0
    Refactored to container-driven pattern for OMN-1577.
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
