# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Effect - ONEX Node (Core 8 Foundation).

for memory retrieval across multiple backends:

- **Semantic Search**: Vector similarity search via Qdrant
- **Full-Text Search**: SQL-based text search via PostgreSQL
- **Graph Traversal**: Relationship-based retrieval via Neo4j/Memgraph

The node follows the ONEX EFFECT pattern, performing I/O operations
against external storage systems with proper error handling and
circuit breaker patterns.

Example::

    import asyncio
    from omnimemory.nodes.node_memory_retrieval_effect import (
        HandlerMemoryRetrieval,
        ModelMemoryRetrievalRequest,
    )

    async def main():
        # Stores come from settings (container.registry_memory_retrieval)
        handler = HandlerMemoryRetrieval(vector_store=vector_store)

        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="user authentication decisions",
            limit=10,
            similarity_threshold=0.7,
        )
        response = await handler.execute(request)
        # status "refused" names the store when it is missing or unreachable
        for result in response.results:
            print(f"[{result.score:.2f}] {result.snapshot.subject}")

        await handler.shutdown()

    asyncio.run(main())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.

Status: stores are injected; an absent or unreachable store is refused by name.
"""

from .handlers import HandlerMemoryRetrieval
from .models import (
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
    ModelStoreRefusal,
)

__all__: list[str] = [
    # Main handler
    "HandlerMemoryRetrieval",
    # Models
    "ModelMemoryRetrievalRequest",
    "ModelMemoryRetrievalResponse",
    "ModelSearchResult",
    "ModelStoreRefusal",
]
