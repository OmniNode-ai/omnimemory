# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Handler - Routes requests to appropriate backend handlers.

routing requests to the appropriate handler based on the operation type:
- search: Routes to Qdrant handler for semantic similarity search
- search_text: Routes to Database handler for full-text search
- search_graph: Routes to Graph handler for relationship traversal

The handler abstracts the underlying handlers, providing a unified interface
for memory retrieval operations.

Example::

    import asyncio
    from omnimemory.nodes.node_memory_retrieval_effect.handlers import (
        HandlerMemoryRetrieval,
    )
    from omnimemory.nodes.node_memory_retrieval_effect.models import (
        ModelHandlerMemoryRetrievalConfig,
        ModelMemoryRetrievalRequest,
    )

    async def example():
        config = ModelHandlerMemoryRetrievalConfig()
        handler = HandlerMemoryRetrieval(config)
        await handler.initialize()

        # Semantic search
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="authentication decision",
        )
        response = await handler.execute(request)

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from collections.abc import Sequence

from omnibase_core.models.omnimemory import (
    ModelMemorySnapshot,
)

from ..models import (
    ModelHandlerMemoryRetrievalConfig,
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
)
from .handler_db_mock import HandlerDbMock
from .handler_fusion import (
    MIN_LEXICAL_TS_RANK,
    MIN_VECTOR_COSINE,
    fuse_rrf,
    select_relevant,
)
from .handler_graph_mock import HandlerGraphMock
from .handler_qdrant_mock import HandlerQdrantMock

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerMemoryRetrieval",
]


class HandlerMemoryRetrieval:
    """Main handler for memory retrieval operations.

    operations, routing requests to the appropriate backend handler based
    on the operation type.

    Supported operations:
        - search: Semantic similarity search (Qdrant)
        - search_text: Full-text search (PostgreSQL)
        - search_graph: Relationship traversal (Graph DB)

    The handler manages handler lifecycle and ensures consistent error
    handling across all backends.

    Attributes:
        config: The handler configuration.

    Example::

        handler = HandlerMemoryRetrieval(ModelHandlerMemoryRetrievalConfig())
        await handler.initialize()

        # Seed test data (for mock handlers)
        handler.seed_snapshots([snapshot1, snapshot2])

        # Execute search
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="user authentication",
        )
        response = await handler.execute(request)
        for result in response.results:
            print(f"{result.snapshot.snapshot_id}: {result.score:.2f}")
    """

    def __init__(self, config: ModelHandlerMemoryRetrievalConfig | None = None) -> None:
        """Initialize the handler with configuration.

        Args:
            config: The handler configuration. If None, defaults are used.
        """
        self._config = config or ModelHandlerMemoryRetrievalConfig()
        self._qdrant_handler: HandlerQdrantMock | None = None
        self._db_handler: HandlerDbMock | None = None
        self._graph_handler: HandlerGraphMock | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelHandlerMemoryRetrievalConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the handler and all sub-handlers.

        Thread-safe: Uses asyncio.Lock to prevent concurrent initialization.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            if self._config.use_stub_handlers:
                # Initialize stub (mock) handlers
                self._qdrant_handler = HandlerQdrantMock(
                    self._config.qdrant_mock_config
                )
                self._db_handler = HandlerDbMock(self._config.db_config)
                self._graph_handler = HandlerGraphMock(self._config.graph_config)

                await asyncio.gather(
                    self._qdrant_handler.initialize(),
                    self._db_handler.initialize(),
                    self._graph_handler.initialize(),
                )

                logger.info("Memory retrieval handler initialized with stub handlers")
            else:
                # stub-ok: OMN-4475 — HandlerQdrant production wiring deferred to Task 5
                raise NotImplementedError(
                    "Production handlers not yet implemented. Set use_stub_handlers=True"
                )

            self._initialized = True

    def seed_snapshots(
        self,
        snapshots: Sequence[ModelMemorySnapshot],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Seed all handlers with test snapshots.

        This is primarily for testing with mock handlers. Each handler
        receives the same set of snapshots.

        Args:
            snapshots: List of snapshots to seed.
            embeddings: Optional pre-computed embeddings for Qdrant handler.

        Raises:
            RuntimeError: If the handler is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        if self._qdrant_handler:
            self._qdrant_handler.seed_snapshots(snapshots, embeddings)
        if self._db_handler:
            self._db_handler.seed_snapshots(snapshots)
        if self._graph_handler:
            self._graph_handler.seed_snapshots(snapshots)

        logger.debug("Seeded %d snapshots into all handlers", len(snapshots))

    def add_graph_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add a relationship to the graph handler.

        Args:
            source_id: The source snapshot ID.
            target_id: The target snapshot ID.
            relationship_type: The type of relationship.
            weight: Optional weight (0.0-1.0).

        Raises:
            RuntimeError: If handler is not initialized or graph unavailable.
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")
        if not self._graph_handler:
            raise RuntimeError("Graph handler not available")

        self._graph_handler.add_relationship(
            source_id, target_id, relationship_type, weight
        )

    def clear(self) -> None:
        """Clear all data from all handlers.

        Raises:
            RuntimeError: If the handler is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        if self._qdrant_handler:
            self._qdrant_handler.clear()
        if self._db_handler:
            self._db_handler.clear()
        if self._graph_handler:
            self._graph_handler.clear()

    async def handle(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Canonical definition-B dispatch entrypoint (CLAUDE.md rule 7a).

        This is the entrypoint the runtime auto-wiring binds — it resolves
        ``handle_async`` then ``handle`` — and the name the omnimarket
        dispatch-entrypoint gate (OMN-14617) looks for on every handler
        declared in ``handler_routing.handlers[]``.

        Added, never renamed: ``execute`` remains the implementation and keeps
        its existing call sites (the mock sub-handlers, this module, and
        ``node_memory_retrieval_effect/__init__.py``).

        Args:
            request: The retrieval request.

        Returns:
            Response with search results or error information.
        """
        return await self.execute(request)

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Execute a memory retrieval operation.

        Routes the request to the appropriate handler based on operation type.

        Args:
            request: The retrieval request.

        Returns:
            Response with search results or error information.
        """
        if not self._initialized:
            await self.initialize()

        try:
            match request.operation:
                case "search":
                    if not self._qdrant_handler:
                        return ModelMemoryRetrievalResponse(
                            status="error",
                            error_message=(
                                f"{self.__class__.__name__}: Qdrant handler not "
                                f"available for operation '{request.operation}'"
                            ),
                        )
                    return await self._qdrant_handler.execute(request)

                case "search_text":
                    if not self._db_handler:
                        return ModelMemoryRetrievalResponse(
                            status="error",
                            error_message=(
                                f"{self.__class__.__name__}: Database handler not "
                                f"available for operation '{request.operation}'"
                            ),
                        )
                    return await self._db_handler.execute(request)

                case "search_graph":
                    if not self._graph_handler:
                        return ModelMemoryRetrievalResponse(
                            status="error",
                            error_message=(
                                f"{self.__class__.__name__}: Graph handler not "
                                f"available for operation '{request.operation}'"
                            ),
                        )
                    return await self._graph_handler.execute(request)

                case "search_hybrid":
                    if not self._qdrant_handler or not self._db_handler:
                        return ModelMemoryRetrievalResponse(
                            status="error",
                            error_message=(
                                f"{self.__class__.__name__}: Hybrid search needs "
                                f"both the Qdrant and Database handlers for "
                                f"operation '{request.operation}'"
                            ),
                        )
                    return await self._execute_hybrid(request)

                case _:
                    assert_never(request.operation)

        except Exception as e:
            logger.exception(
                "Error executing retrieval operation %s",
                request.operation,
            )
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: Retrieval failed: {e} "
                    f"for operation '{request.operation}'"
                ),
            )

    async def _execute_hybrid(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Fan out to the semantic and full-text legs, then fuse their rankings.

        The two legs are dispatched concurrently — they share no state and
        neither depends on the other's result. Each is sent as the operation it
        already implements, so this adds no new behaviour to either sub-handler.

        Ranking is a two-step process, and the steps are deliberately distinct:
        each leg is first filtered against its own relevance floor, then the
        survivors are fused by rank. Fusing first would let a leg that matched
        nothing useful contribute its best-of-a-bad-set at full rank-1 weight
        and displace a leg that answered correctly. See ``handler_fusion``.

        Args:
            request: The originating ``search_hybrid`` request.

        Returns:
            One response carrying the fused, deduplicated ranking, truncated to
            the request's limit. If either leg errors, that error is returned
            rather than a half-fused ranking presented as a whole answer.
        """
        if self._qdrant_handler is None or self._db_handler is None:
            raise RuntimeError("Hybrid search requires both Qdrant and DB handlers")

        vector_response, lexical_response = await asyncio.gather(
            self._qdrant_handler.execute(
                request.model_copy(update={"operation": "search"})
            ),
            self._db_handler.execute(
                request.model_copy(update={"operation": "search_text"})
            ),
        )

        for leg_response in (vector_response, lexical_response):
            if leg_response.status == "error":
                return leg_response

        by_id = {
            str(result.snapshot.snapshot_id): result
            for result in (*lexical_response.results, *vector_response.results)
        }

        fused_ids = fuse_rrf(
            select_relevant(
                [
                    (str(r.snapshot.snapshot_id), r.score)
                    for r in vector_response.results
                ],
                MIN_VECTOR_COSINE,
            ),
            select_relevant(
                [
                    (str(r.snapshot.snapshot_id), r.score)
                    for r in lexical_response.results
                ],
                MIN_LEXICAL_TS_RANK,
            ),
        )

        results = [by_id[doc_id] for doc_id in fused_ids][: request.limit]

        logger.debug(
            "Hybrid search fused %d vector and %d lexical results into %d",
            len(vector_response.results),
            len(lexical_response.results),
            len(results),
        )

        return ModelMemoryRetrievalResponse(
            status="success" if results else "no_results",
            results=results,
            total_count=len(results),
            query_embedding_used=vector_response.query_embedding_used,
        )

    async def shutdown(self) -> None:
        """Shutdown the handler and all sub-handlers."""
        if not self._initialized:
            return

        shutdown_tasks = []
        if self._qdrant_handler:
            shutdown_tasks.append(self._qdrant_handler.shutdown())
        if self._db_handler:
            shutdown_tasks.append(self._db_handler.shutdown())
        if self._graph_handler:
            shutdown_tasks.append(self._graph_handler.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks)

        self._qdrant_handler = None
        self._db_handler = None
        self._graph_handler = None
        self._initialized = False

        logger.info("Memory retrieval handler shutdown complete")
