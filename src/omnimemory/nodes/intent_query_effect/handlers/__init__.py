# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent Query Effect handlers.

This package contains IHandler implementations for the intent_query_effect node.

Handlers:
    - HandlerIntentQuery: Main handler for processing intent query events

Example::

    from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
    from omnimemory.handlers.adapters import AdapterIntentGraph

    # Initialize adapter
    adapter = AdapterIntentGraph(config)
    await adapter.initialize(connection_uri="bolt://localhost:7687")

    # Initialize handler
    handler = HandlerIntentQuery()
    await handler.initialize(adapter)

    # Execute query
    response = await handler.execute(request_event)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from omnimemory.nodes.intent_query_effect.handlers.handler_intent_query import (
    HandlerIntentQuery,
)

__all__ = ["HandlerIntentQuery"]
