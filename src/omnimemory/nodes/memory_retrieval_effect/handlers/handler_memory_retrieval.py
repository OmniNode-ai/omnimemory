# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Handler - Routes requests to appropriate backend handlers.

This module provides the main handler for the memory_retrieval_effect node,
routing requests to the appropriate handler based on the operation type:
- search: Routes to Qdrant handler for semantic similarity search
- search_text: Routes to Database handler for full-text search
- search_graph: Routes to Graph handler for relationship traversal

The handler abstracts the underlying handlers, providing a unified interface
for memory retrieval operations.

Example::

    import asyncio
    from omnimemory.nodes.memory_retrieval_effect.handlers import (
        HandlerMemoryRetrieval,
    )
    from omnimemory.nodes.memory_retrieval_effect.models import (
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
    ModelMemorySnapshot,  # noqa: TC002 - Pydantic needs runtime access
)

from ..models import (
    ModelHandlerMemoryRetrievalConfig,
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
)
from .handler_db_mock import HandlerDbMock
from .handler_graph_mock import HandlerGraphMock
from .handler_qdrant_mock import HandlerQdrantMock

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerMemoryRetrieval",
]


class HandlerMemoryRetrieval:
    """Main handler for memory retrieval operations.

    This handler provides a unified interface for all memory retrieval
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

            if self._config.use_mock_handlers:
                # Initialize mock handlers
                self._qdrant_handler = HandlerQdrantMock(self._config.qdrant_config)
                self._db_handler = HandlerDbMock(self._config.db_config)
                self._graph_handler = HandlerGraphMock(self._config.graph_config)

                await asyncio.gather(
                    self._qdrant_handler.initialize(),
                    self._db_handler.initialize(),
                    self._graph_handler.initialize(),
                )

                logger.info("Memory retrieval handler initialized with mock handlers")
            else:
                # TODO: Initialize real handlers from omnibase_infra
                raise NotImplementedError(
                    "Real handlers not yet implemented. Set use_mock_handlers=True"
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
