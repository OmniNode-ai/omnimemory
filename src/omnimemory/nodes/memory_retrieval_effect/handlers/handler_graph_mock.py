# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Mock Graph Handler for relationship traversal operations.

This module provides a mock handler that simulates `HandlerGraph` behavior
for graph-based memory traversal. It allows development and testing of the
memory_retrieval_effect node without requiring a running Neo4j/Memgraph instance.

The mock maintains an in-memory graph structure of snapshot relationships
and supports breadth-first traversal with depth limits and relationship
type filtering.

Example::

    import asyncio
    from omnimemory.nodes.memory_retrieval_effect.handlers import (
        HandlerGraphMock,
        ModelHandlerGraphMockConfig,
    )

    async def example():
        config = ModelHandlerGraphMockConfig()
        handler = HandlerGraphMock(config)
        await handler.initialize()

        # Seed with test data and relationships
        handler.seed_snapshots([snapshot1, snapshot2, snapshot3])
        handler.add_relationship("snap1", "snap2", "related_to")
        handler.add_relationship("snap2", "snap3", "caused_by")

        # Traverse
        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id="snap1",
            traversal_depth=2,
        )
        response = await handler.execute(request)

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omnibase_core.models.omnimemory import (
    ModelMemorySnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

from ..models import (
    ModelHandlerGraphMockConfig,
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerGraphMock",
    "ModelHandlerGraphMockConfig",
    "HandlerGraphRelationship",
]


@dataclass
class HandlerGraphRelationship:
    """Represents a relationship between two snapshots.

    Attributes:
        source_id: The source snapshot ID.
        target_id: The target snapshot ID.
        relationship_type: The type of relationship (e.g., "related_to", "caused_by").
        weight: Optional weight/strength of the relationship (0.0-1.0).
    """

    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0


class HandlerGraphMock:
    """Mock handler that simulates HandlerGraph for relationship traversal.

    This handler provides a development-friendly interface for testing
    graph traversal functionality without requiring a real graph database.
    It maintains an in-memory graph structure and supports BFS traversal.

    The handler can be seeded with test data and relationships for
    reproducible testing.

    Attributes:
        config: The handler configuration.

    Example::

        async def example():
            handler = HandlerGraphMock(ModelHandlerGraphMockConfig())
            await handler.initialize()

            # Seed test data
            handler.seed_snapshots([snapshot1, snapshot2])
            handler.add_relationship("snap1", "snap2", "related_to")

            # Execute traversal
            request = ModelMemoryRetrievalRequest(
                operation="search_graph",
                snapshot_id="snap1",
                traversal_depth=3,
            )
            response = await handler.execute(request)
    """

    def __init__(self, config: ModelHandlerGraphMockConfig) -> None:
        """Initialize the mock handler with configuration.

        Args:
            config: The handler configuration.
        """
        self._config = config
        self._snapshots: dict[str, ModelMemorySnapshot] = {}
        self._relationships: list[HandlerGraphRelationship] = []
        self._adjacency: dict[
            str, list[tuple[str, str, float]]
        ] = {}  # id -> [(target, type, weight)]
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelHandlerGraphMockConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    @property
    def snapshot_count(self) -> int:
        """Get the number of stored snapshots."""
        return len(self._snapshots)

    @property
    def relationship_count(self) -> int:
        """Get the number of relationships."""
        return len(self._relationships)

    async def initialize(self) -> None:
        """Initialize the mock handler.

        Thread-safe: Uses asyncio.Lock to prevent concurrent initialization.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Mock Graph handler initialized")
            self._initialized = True

    def seed_snapshots(self, snapshots: Sequence[ModelMemorySnapshot]) -> None:
        """Seed the mock store with test snapshots.

        Args:
            snapshots: List of snapshots to add to the mock store.

        Note:
            Snapshots with invalid or empty IDs are skipped with a warning.
        """
        valid_count = 0
        for snapshot in snapshots:
            # Validate snapshot ID is non-empty
            if not snapshot.snapshot_id or not str(snapshot.snapshot_id).strip():
                logger.warning(
                    "Skipping snapshot with invalid/empty ID: %r",
                    snapshot.snapshot_id,
                )
                continue

            snapshot_id = str(snapshot.snapshot_id)
            self._snapshots[snapshot_id] = snapshot
            if snapshot_id not in self._adjacency:
                self._adjacency[snapshot_id] = []
            valid_count += 1

        logger.debug("Seeded %d snapshots into mock graph store", valid_count)

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add a relationship between two snapshots.

        Args:
            source_id: The source snapshot ID.
            target_id: The target snapshot ID.
            relationship_type: The type of relationship.
            weight: Optional weight (0.0-1.0). Defaults to 1.0.
        """
        relationship = HandlerGraphRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
        )
        self._relationships.append(relationship)

        # Update adjacency list
        if source_id not in self._adjacency:
            self._adjacency[source_id] = []
        self._adjacency[source_id].append((target_id, relationship_type, weight))

        # If bidirectional, add reverse edge
        if self._config.bidirectional:
            if target_id not in self._adjacency:
                self._adjacency[target_id] = []
            self._adjacency[target_id].append((source_id, relationship_type, weight))

        logger.debug(
            "Added relationship: %s -[%s]-> %s (weight=%.2f)",
            source_id,
            relationship_type,
            target_id,
            weight,
        )

    def clear(self) -> None:
        """Clear all snapshots and relationships from the mock store."""
        self._snapshots.clear()
        self._relationships.clear()
        self._adjacency.clear()

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Execute a graph traversal operation.

        Args:
            request: The retrieval request (must have operation="search_graph").

        Returns:
            Response with traversal results ordered by path distance.

        Raises:
            ValueError: If operation is not "search_graph".
        """
        if not self._initialized:
            await self.initialize()

        if request.operation != "search_graph":
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: Only supports 'search_graph', "
                    f"got '{request.operation}'"
                ),
            )

        if request.snapshot_id is None:
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: snapshot_id is required "
                    f"for operation '{request.operation}'"
                ),
            )

        # Validate start node exists
        if request.snapshot_id not in self._snapshots:
            return ModelMemoryRetrievalResponse(
                status="no_results",
                results=[],
                total_count=0,
                error_message=(
                    f"{self.__class__.__name__}: Start snapshot "
                    f"'{request.snapshot_id}' not found "
                    f"for operation '{request.operation}'"
                ),
            )

        # Simulate latency if configured
        if self._config.simulate_latency_ms > 0:
            await asyncio.sleep(self._config.simulate_latency_ms / 1000)

        # Perform BFS traversal
        max_depth = min(request.traversal_depth, self._config.max_traversal_depth)
        allowed_types = (
            set(request.relationship_types) if request.relationship_types else None
        )

        results = self._traverse_bfs(
            start_id=request.snapshot_id,
            max_depth=max_depth,
            allowed_types=allowed_types,
            limit=request.limit,
        )

        if not results:
            return ModelMemoryRetrievalResponse(
                status="no_results",
                results=[],
                total_count=0,
            )

        return ModelMemoryRetrievalResponse(
            status="success",
            results=results,
            total_count=len(results),
        )

    def _traverse_bfs(
        self,
        start_id: str,
        max_depth: int,
        allowed_types: set[str] | None,
        limit: int,
    ) -> list[ModelSearchResult]:
        """Perform breadth-first traversal from a start node.

        Args:
            start_id: The starting snapshot ID.
            max_depth: Maximum traversal depth.
            allowed_types: Set of allowed relationship types (None = all).
            limit: Maximum number of results.

        Returns:
            List of search results with path information.
        """
        results: list[ModelSearchResult] = []
        visited: set[str] = {start_id}

        # Queue entries are tuples of node_id, depth, path, cumulative_weight
        queue: deque[tuple[str, int, list[str], float]] = deque()
        queue.append((start_id, 0, [start_id], 1.0))

        while queue and len(results) < limit:
            current_id, depth, path, cum_weight = queue.popleft()

            # Add to results if not the start node
            if current_id != start_id and current_id in self._snapshots:
                # Score based on path weight and inverse depth
                score = cum_weight * (1.0 / (depth + 1))
                results.append(
                    ModelSearchResult(
                        snapshot=self._snapshots[current_id],
                        score=min(1.0, score),
                        path=path.copy(),
                    )
                )

            # Don't explore beyond max depth
            if depth >= max_depth:
                continue

            # Explore neighbors
            for target_id, rel_type, weight in self._adjacency.get(current_id, []):
                # Skip if already visited
                if target_id in visited:
                    continue

                # Filter by relationship type
                if allowed_types and rel_type not in allowed_types:
                    continue

                visited.add(target_id)
                new_path = path + [target_id]
                new_weight = cum_weight * weight
                queue.append((target_id, depth + 1, new_path, new_weight))

        # Sort by score (highest first)
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        if self._initialized:
            self._snapshots.clear()
            self._relationships.clear()
            self._adjacency.clear()
            self._initialized = False
            logger.debug("Mock Graph handler shutdown complete")
