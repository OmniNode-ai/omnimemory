# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Effect - ONEX Node (Core 8 Foundation).

This node provides semantic, temporal, and contextual search operations
for memory retrieval across multiple backends:

- **Semantic Search**: Vector similarity search via Qdrant
- **Full-Text Search**: SQL-based text search via PostgreSQL
- **Graph Traversal**: Relationship-based retrieval via Neo4j/Memgraph

The node follows the ONEX EFFECT pattern, performing I/O operations
against external storage systems with proper error handling and
circuit breaker patterns.

Example::

    import asyncio
    from omnimemory.nodes.memory_retrieval_effect import (
        HandlerMemoryRetrieval,
        ModelHandlerMemoryRetrievalConfig,
        ModelMemoryRetrievalRequest,
    )

    async def main():
        # Initialize handler
        handler = HandlerMemoryRetrieval()
        await handler.initialize()

        # Semantic search
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="user authentication decisions",
            limit=10,
            similarity_threshold=0.7,
        )
        response = await handler.execute(request)

        for result in response.results:
            print(f"[{result.score:.2f}] {result.snapshot.subject}")

        await handler.shutdown()

    asyncio.run(main())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.

Status: Implementation complete with mock adapters.
"""

from .handler_memory_retrieval import (
    HandlerMemoryRetrieval,
    ModelHandlerMemoryRetrievalConfig,
)
from .models import (
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
)

__all__: list[str] = [
    # Main handler
    "HandlerMemoryRetrieval",
    "ModelHandlerMemoryRetrievalConfig",
    # Models
    "ModelMemoryRetrievalRequest",
    "ModelMemoryRetrievalResponse",
    "ModelSearchResult",
]
