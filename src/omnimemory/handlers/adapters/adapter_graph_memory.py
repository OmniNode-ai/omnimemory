# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph Handler Adapter for relationship-based memory queries.

This module provides an adapter that wraps `HandlerGraph` from omnibase_infra
to support memory-specific graph operations. It enables "memories related to X"
queries via graph traversal, translating between memory domain concepts and
graph database operations.

The adapter transforms memory operations into graph operations:
    - find_related(memory_id) -> traverse(start_node, depth)
    - get_connections(memory_id) -> execute_query() with edge retrieval

Example::

    import asyncio
    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        AdapterGraphMemoryConfig,
    )

    async def example():
        config = AdapterGraphMemoryConfig(max_depth=5)
        adapter = AdapterGraphMemory(config)
        await adapter.initialize(
            connection_uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )

        # Find memories related to a specific memory
        related = await adapter.find_related("memory_abc123", depth=2)
        for memory in related.memories:
            print(f"Related: {memory.memory_id} (score={memory.score:.2f})")

        # Get direct connections
        connections = await adapter.get_connections("memory_abc123")
        for conn in connections:
            print(f"{conn.source_id} --[{conn.relationship_type}]--> {conn.target_id}")

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from collections.abc import Mapping
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

# omnibase_infra is a dev dependency - make imports conditional
_OMNIBASE_INFRA_AVAILABLE = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

try:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.models.graph import (
        ModelGraphDatabaseNode,
        ModelGraphRelationship,
        ModelGraphTraversalFilters,
        ModelGraphTraversalResult,
    )
    from omnibase_infra.errors import InfraConnectionError
    from omnibase_infra.handlers.handler_graph import HandlerGraph

    _OMNIBASE_INFRA_AVAILABLE = True
except ImportError as e:
    _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

    # Provide stub types for type checking and to allow module to load
    class InfraConnectionError(Exception):  # type: ignore[no-redef]
        """Stub for InfraConnectionError when omnibase_infra is not installed."""

        pass

    class HandlerGraph:  # type: ignore[no-redef]
        """Stub for HandlerGraph when omnibase_infra is not installed."""

        def __init__(self, container: object) -> None:
            raise ImportError(
                f"omnibase_infra is required for AdapterGraphMemory. "
                f"Install it with: poetry install --with dev. "
                f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
            )

    class ModelONEXContainer:  # type: ignore[no-redef]
        """Stub for ModelONEXContainer."""

        pass

    class ModelGraphTraversalResult:  # type: ignore[no-redef]
        """Stub for ModelGraphTraversalResult."""

        pass

    class ModelGraphTraversalFilters:  # type: ignore[no-redef]
        """Stub for ModelGraphTraversalFilters."""

        pass

    class ModelGraphDatabaseNode:  # type: ignore[no-redef]
        """Stub for ModelGraphDatabaseNode."""

        pass

    class ModelGraphRelationship:  # type: ignore[no-redef]
        """Stub for ModelGraphRelationship."""

        pass


logger = logging.getLogger(__name__)

__all__ = [
    "AdapterGraphMemory",
    "AdapterGraphMemoryConfig",
    "ModelConnectionsResult",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
]


# =============================================================================
# Cypher Query Templates
# =============================================================================
# All templates use parameterized queries to prevent injection attacks.
# See docs/handler_reuse_matrix.md Security section for guidelines.


class CypherTemplates:
    """Parameterized Cypher query templates for memory graph operations.

    Direction Behavior:
        - GET_CONNECTIONS: Bidirectional by default (matches both incoming and outgoing)
        - GET_CONNECTIONS_BY_TYPE: Outgoing-only (for directed relationship queries)
        - GET_CONNECTIONS_BY_TYPE_BIDIRECTIONAL: Bidirectional with type filtering

        The `get_connections()` method accepts a `bidirectional` parameter to select
        the appropriate template when filtering by relationship type.

    Security:
        All queries use parameters ($param) instead of string interpolation.
        NEVER construct queries by concatenating user input.
    """

    # Find direct edges (relationships) for a memory node
    GET_CONNECTIONS = """
    MATCH (m:Memory {memory_id: $memory_id})-[r]-(n:Memory)
    RETURN
        m.memory_id AS source_id,
        n.memory_id AS target_id,
        type(r) AS relationship_type,
        r.weight AS weight,
        r.created_at AS created_at,
        startNode(r) = m AS is_outgoing
    LIMIT $limit
    """

    # Find connections filtered by relationship type (outgoing only)
    GET_CONNECTIONS_BY_TYPE = """
    MATCH (m:Memory {memory_id: $memory_id})-[r]->(n:Memory)
    WHERE type(r) IN $relationship_types
    RETURN
        m.memory_id AS source_id,
        n.memory_id AS target_id,
        type(r) AS relationship_type,
        r.weight AS weight,
        r.created_at AS created_at,
        true AS is_outgoing
    LIMIT $limit
    """

    # Find connections filtered by relationship type (bidirectional)
    GET_CONNECTIONS_BY_TYPE_BIDIRECTIONAL = """
    MATCH (m:Memory {memory_id: $memory_id})-[r]-(n:Memory)
    WHERE type(r) IN $relationship_types
    RETURN
        m.memory_id AS source_id,
        n.memory_id AS target_id,
        type(r) AS relationship_type,
        r.weight AS weight,
        r.created_at AS created_at,
        startNode(r) = m AS is_outgoing
    LIMIT $limit
    """

    # Count connections for a memory
    COUNT_CONNECTIONS = """
    MATCH (m:Memory {memory_id: $memory_id})-[r]-()
    RETURN count(r) AS connection_count
    """

    # Check if a memory node exists
    NODE_EXISTS = """
    MATCH (m:Memory {memory_id: $memory_id})
    RETURN m.memory_id AS memory_id, elementId(m) AS element_id
    LIMIT 1
    """


# =============================================================================
# Models
# =============================================================================


class ModelMemoryConnection(BaseModel):
    """Represents a connection (relationship) between two memories.

    Attributes:
        source_id: The source memory ID.
        target_id: The target memory ID.
        relationship_type: The type of relationship (e.g., "related_to", "caused_by").
        weight: Strength of the connection (0.0-1.0). Defaults to 1.0.
        is_outgoing: True if this is an outgoing edge from source, False if incoming.
        created_at: ISO timestamp when the connection was created.
    """

    source_id: str = Field(
        ...,
        description="Source memory ID",
    )
    target_id: str = Field(
        ...,
        description="Target memory ID",
    )
    relationship_type: str = Field(
        ...,
        description="Type of relationship between memories",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strength of connection (0.0-1.0)",
    )
    is_outgoing: bool = Field(
        default=True,
        description="True if outgoing from source, False if incoming",
    )
    created_at: str | None = Field(
        default=None,
        description="ISO timestamp when connection was created",
    )


class ModelRelatedMemory(BaseModel):
    """A memory found through relationship traversal.

    Attributes:
        memory_id: The related memory's identifier.
        score: Relevance score based on path weight and distance (0.0-1.0).
        path: Traversal path from start memory to this memory (list of memory IDs).
        depth: Number of hops from the starting memory.
        labels: Graph labels on the memory node.
        properties: Additional properties from the graph node.
    """

    memory_id: str = Field(
        ...,
        description="The related memory's identifier",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0-1.0)",
    )
    path: list[str] = Field(
        default_factory=list,
        description=(
            "Partial traversal path containing start and end memory IDs. "
            "Note: Intermediate nodes are not included; full path reconstruction "
            "would require additional queries. For complete path information, "
            "use the depth field which indicates the number of hops."
        ),
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Number of hops from starting memory",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Graph labels on the memory node",
    )
    properties: dict[str, object] = Field(
        default_factory=dict,
        description="Additional node properties",
    )


class ModelRelatedMemoryResult(BaseModel):
    """Result of a find_related operation.

    Attributes:
        status: Operation status (success, error, not_found, no_results).
        memories: List of related memories ordered by relevance score.
        total_count: Total number of related memories found.
        max_depth_reached: The maximum traversal depth that was reached.
        execution_time_ms: Time taken to execute the query in milliseconds.
        error_message: Error details if status is "error".
    """

    status: Literal["success", "error", "not_found", "no_results"] = Field(
        ...,
        description="Operation status",
    )
    memories: list[ModelRelatedMemory] = Field(
        default_factory=list,
        description="Related memories ordered by relevance",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of related memories found",
    )
    max_depth_reached: int = Field(
        default=0,
        ge=0,
        description="Maximum traversal depth reached",
    )
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if status is error",
    )


class ModelConnectionsResult(BaseModel):
    """Result of a get_connections operation.

    Attributes:
        status: Operation status (success, error, not_found, no_results).
        connections: List of connections for the memory.
        total_count: Total number of connections found.
        error_message: Error details if status is "error".
    """

    status: Literal["success", "error", "not_found", "no_results"] = Field(
        ...,
        description="Operation status",
    )
    connections: list[ModelMemoryConnection] = Field(
        default_factory=list,
        description="Connections for the memory",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of connections",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if status is error",
    )


class ModelGraphMemoryHealth(BaseModel):
    """Health status information for the graph memory adapter.

    Attributes:
        is_healthy: Overall health status.
        initialized: Whether the adapter has been initialized.
        handler_healthy: Health status from the underlying graph handler.
        error_message: Error details if unhealthy.
    """

    is_healthy: bool = Field(
        ...,
        description="Overall health status",
    )
    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )
    handler_healthy: bool | None = Field(
        default=None,
        description="Health status from underlying graph handler (None if not checked)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if unhealthy",
    )


# =============================================================================
# Configuration
# =============================================================================


class AdapterGraphMemoryConfig(BaseModel):
    """Configuration for the Graph Memory adapter.

    Attributes:
        max_depth: Maximum allowed traversal depth. Bounded to prevent
            expensive deep traversals. Defaults to 5.
        default_depth: Default traversal depth if not specified. Defaults to 2.
        default_limit: Default result limit. Defaults to 100.
        max_limit: Maximum allowed result limit. Defaults to 1000.
        bidirectional: Whether to traverse relationships in both directions.
            Defaults to True.
        memory_node_label: Graph label for memory nodes. Defaults to "Memory".
        timeout_seconds: Query timeout in seconds. Defaults to 30.0.
    """

    max_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum allowed traversal depth",
    )
    default_depth: int = Field(
        default=2,
        ge=1,
        description="Default traversal depth",
    )
    default_limit: int = Field(
        default=100,
        ge=1,
        description="Default result limit",
    )
    max_limit: int = Field(
        default=1000,
        ge=1,
        description="Maximum allowed result limit",
    )
    bidirectional: bool = Field(
        default=True,
        description="Traverse relationships in both directions",
    )
    memory_node_label: str = Field(
        default="Memory",
        description="Graph label for memory nodes",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Query timeout in seconds",
    )

    @model_validator(mode="after")
    def validate_depth_bounds(self) -> "AdapterGraphMemoryConfig":
        """Ensure default_depth does not exceed max_depth."""
        if self.default_depth > self.max_depth:
            msg = (
                f"default_depth ({self.default_depth}) "
                f"must be <= max_depth ({self.max_depth})"
            )
            raise ValueError(msg)
        return self


# =============================================================================
# Adapter
# =============================================================================


class AdapterGraphMemory:
    """Adapter that wraps HandlerGraph for memory-specific graph operations.

    This adapter provides a memory-domain interface on top of the generic
    graph handler, translating memory operations into graph queries:

    - find_related(memory_id): Uses traverse() to find connected memories
    - get_connections(memory_id): Uses execute_query() to get direct edges

    The adapter handles:
    - Memory ID to graph node ID mapping
    - Depth limiting to prevent expensive traversals
    - Score calculation based on path weight and distance
    - Cypher query parameterization for security

    Attributes:
        config: The adapter configuration.
        handler: The underlying HandlerGraph instance.

    Example::

        async def example():
            config = AdapterGraphMemoryConfig(max_depth=3)
            adapter = AdapterGraphMemory(config)
            await adapter.initialize(
                connection_uri="bolt://localhost:7687",
            )

            # Find related memories up to 2 hops away
            result = await adapter.find_related("mem_123", depth=2)
            for mem in result.memories:
                print(f"Found: {mem.memory_id} at depth {mem.depth}")

            await adapter.shutdown()
    """

    def __init__(
        self,
        config: AdapterGraphMemoryConfig,
        container: ModelONEXContainer | None = None,
    ) -> None:
        """Initialize the adapter with configuration.

        Args:
            config: The adapter configuration.
            container: Optional ONEX container for dependency injection.
                If not provided, a minimal container will be created.
        """
        self._config = config
        self._container = container
        self._handler: HandlerGraph | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> AdapterGraphMemoryConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def handler(self) -> HandlerGraph | None:
        """Get the underlying graph handler (None if not initialized)."""
        return self._handler

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter has been initialized."""
        return self._initialized

    async def initialize(
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the adapter and underlying graph handler.

        Establishes connection to the graph database and prepares
        the handler for memory queries.

        Args:
            connection_uri: Graph database URI (e.g., "bolt://localhost:7687").
            auth: Optional (username, password) tuple for authentication.
            options: Additional connection options passed to HandlerGraph.

        Raises:
            RuntimeError: If initialization fails.
            InfraConnectionError: If connection to graph database fails.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                # Create container if not provided
                if self._container is None:
                    # Import here to get the real class
                    from omnibase_core.container import ModelONEXContainer

                    self._container = ModelONEXContainer()

                # Create and initialize handler
                self._handler = HandlerGraph(self._container)

                init_options: dict[str, object] = {
                    "timeout_seconds": self._config.timeout_seconds,
                }
                if options:
                    init_options.update(options)

                await self._handler.initialize(
                    connection_uri=connection_uri,
                    auth=auth,
                    options=init_options,
                )

                self._initialized = True
                # Safely extract host info without credentials
                parsed_uri = urlparse(connection_uri)
                safe_uri = f"{parsed_uri.scheme}://{parsed_uri.hostname}"
                if parsed_uri.port:
                    safe_uri += f":{parsed_uri.port}"
                logger.info(
                    "AdapterGraphMemory initialized with connection to %s",
                    safe_uri,
                )

            except InfraConnectionError:
                raise
            except Exception as e:
                logger.error(
                    "Failed to initialize AdapterGraphMemory: %s",
                    e,
                )
                raise RuntimeError(f"Initialization failed: {e}") from e

    def _ensure_initialized(self) -> HandlerGraph:
        """Ensure adapter is initialized and return handler.

        Returns:
            The initialized HandlerGraph.

        Raises:
            RuntimeError: If adapter is not initialized.
        """
        if not self._initialized or self._handler is None:
            raise RuntimeError(
                "AdapterGraphMemory not initialized. Call initialize() first."
            )
        return self._handler

    async def find_related(
        self,
        memory_id: str,
        *,
        depth: int | None = None,
        relationship_types: list[str] | None = None,
        limit: int | None = None,
        min_score: float = 0.0,
    ) -> ModelRelatedMemoryResult:
        """Find memories related to a given memory via graph traversal.

        Performs breadth-first traversal from the starting memory node,
        following relationships up to the specified depth. Results are
        scored based on path weight and distance from the starting node.

        Args:
            memory_id: The starting memory's identifier.
            depth: Maximum traversal depth. Bounded by config.max_depth.
                Defaults to config.default_depth.
            relationship_types: Optional list of relationship types to follow.
                If None, all relationship types are followed.
            limit: Maximum number of results. Bounded by config.max_limit.
                Defaults to config.default_limit.
            min_score: Minimum score threshold (0.0-1.0). Results below
                this score are filtered out. Defaults to 0.0.

        Returns:
            ModelRelatedMemoryResult with related memories ordered by score.

        Raises:
            RuntimeError: If adapter is not initialized.
        """
        handler = self._ensure_initialized()

        # Apply bounds
        effective_depth = min(
            depth or self._config.default_depth, self._config.max_depth
        )
        effective_limit = min(
            limit or self._config.default_limit, self._config.max_limit
        )

        # Determine traversal direction
        direction = "both" if self._config.bidirectional else "outgoing"

        # Build traversal filters
        filters: ModelGraphTraversalFilters | None = None
        if self._config.memory_node_label:
            filters = ModelGraphTraversalFilters(
                node_labels=[self._config.memory_node_label],
            )

        try:
            # First, find the graph node for this memory_id
            node_result = await handler.execute_query(
                query=CypherTemplates.NODE_EXISTS,
                parameters={"memory_id": memory_id},
            )

            if not node_result.records:
                return ModelRelatedMemoryResult(
                    status="not_found",
                    error_message=f"Memory '{memory_id}' not found in graph",
                )

            element_id = node_result.records[0]["element_id"]

            # Perform traversal from the memory node
            traversal_result: ModelGraphTraversalResult = await handler.traverse(
                start_node_id=element_id,
                relationship_types=relationship_types,
                direction=direction,
                max_depth=effective_depth,
                filters=filters,
            )

            # Convert graph nodes to related memories
            memories: list[ModelRelatedMemory] = []

            for node in traversal_result.nodes:
                # Extract memory_id from properties with explicit type check
                node_memory_id = node.properties.get("memory_id")
                if not isinstance(node_memory_id, str) or node_memory_id == memory_id:
                    continue

                # Calculate relevance score based on traversal depth (edge count).
                # In path array, index 0 is start node, so node at index N is N
                # edges away. Score: 1/(depth+1) gives closer nodes higher scores.
                # E.g.: depth=1 -> 0.5, depth=2 -> 0.33, depth=3 -> 0.25
                node_paths = [p for p in traversal_result.paths if node.element_id in p]
                if node_paths:
                    shortest_path = min(node_paths, key=len)
                    # Position in path = number of edges from start (depth)
                    depth_to_node = shortest_path.index(node.element_id)
                else:
                    logger.warning(
                        "Path info unavailable for node %s, defaulting to depth=1",
                        node.element_id,
                    )
                    depth_to_node = 1

                score = 1.0 / (depth_to_node + 1)

                if score < min_score:
                    continue

                # Build path as memory IDs
                path_memory_ids = [memory_id]
                # Note: Full path reconstruction would require additional queries

                memories.append(
                    ModelRelatedMemory(
                        memory_id=str(node_memory_id),
                        score=score,
                        path=path_memory_ids + [str(node_memory_id)],
                        depth=depth_to_node,
                        labels=list(node.labels),
                        properties=dict(node.properties),
                    )
                )

            # Use heapq for O(n log k) instead of O(n log n) full sort
            memories = heapq.nlargest(effective_limit, memories, key=lambda m: m.score)

            if not memories:
                return ModelRelatedMemoryResult(
                    status="no_results",
                    memories=[],
                    total_count=0,
                    max_depth_reached=traversal_result.depth_reached,
                    execution_time_ms=traversal_result.execution_time_ms,
                )

            return ModelRelatedMemoryResult(
                status="success",
                memories=memories,
                total_count=len(memories),
                max_depth_reached=traversal_result.depth_reached,
                execution_time_ms=traversal_result.execution_time_ms,
            )

        except InfraConnectionError as e:
            logger.warning(
                "Graph traversal failed for memory %s: %s",
                memory_id,
                e,
            )
            return ModelRelatedMemoryResult(
                status="error",
                error_message=f"Graph traversal failed: {e}",
            )
        except Exception as e:
            logger.error(
                "Unexpected error finding related memories for %s: %s",
                memory_id,
                e,
            )
            return ModelRelatedMemoryResult(
                status="error",
                error_message=f"Unexpected error: {e}",
            )

    async def get_connections(
        self,
        memory_id: str,
        *,
        relationship_types: list[str] | None = None,
        limit: int | None = None,
        bidirectional: bool | None = None,
    ) -> ModelConnectionsResult:
        """Get direct connections (edges) for a memory node.

        Retrieves all relationships connected to the specified memory,
        optionally filtered by relationship type.

        Args:
            memory_id: The memory's identifier.
            relationship_types: Optional list of relationship types to include.
                If None, all types are returned.
            limit: Maximum number of connections. Defaults to config.default_limit.
            bidirectional: Whether to include both incoming and outgoing connections.
                If None, defaults to config.bidirectional. When True, returns
                connections in both directions. When False, returns only outgoing
                connections (only applies when relationship_types is specified).

        Returns:
            ModelConnectionsResult with the memory's connections.

        Raises:
            RuntimeError: If adapter is not initialized.
        """
        handler = self._ensure_initialized()

        effective_limit = min(
            limit or self._config.default_limit, self._config.max_limit
        )
        # Use config.bidirectional if not explicitly specified
        effective_bidirectional = (
            bidirectional if bidirectional is not None else self._config.bidirectional
        )

        try:
            # Choose query based on whether we're filtering by type and direction
            if relationship_types:
                if effective_bidirectional:
                    query = CypherTemplates.GET_CONNECTIONS_BY_TYPE_BIDIRECTIONAL
                else:
                    query = CypherTemplates.GET_CONNECTIONS_BY_TYPE
                parameters: dict[str, object] = {
                    "memory_id": memory_id,
                    "relationship_types": relationship_types,
                    "limit": effective_limit,
                }
            else:
                query = CypherTemplates.GET_CONNECTIONS
                parameters = {
                    "memory_id": memory_id,
                    "limit": effective_limit,
                }

            result = await handler.execute_query(
                query=query,
                parameters=parameters,
            )

            if not result.records:
                # Check if node exists
                exists_result = await handler.execute_query(
                    query=CypherTemplates.NODE_EXISTS,
                    parameters={"memory_id": memory_id},
                )
                if not exists_result.records:
                    return ModelConnectionsResult(
                        status="not_found",
                        error_message=f"Memory '{memory_id}' not found in graph",
                    )

                return ModelConnectionsResult(
                    status="no_results",
                    connections=[],
                    total_count=0,
                )

            # Convert records to connections
            connections: list[ModelMemoryConnection] = []
            for record in result.records:
                connections.append(
                    ModelMemoryConnection(
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        relationship_type=record["relationship_type"],
                        weight=(
                            record.get("weight")
                            if record.get("weight") is not None
                            else 1.0
                        ),
                        is_outgoing=record.get("is_outgoing", True),
                        created_at=record.get("created_at"),
                    )
                )

            return ModelConnectionsResult(
                status="success",
                connections=connections,
                total_count=len(connections),
            )

        except InfraConnectionError as e:
            logger.warning(
                "Failed to get connections for memory %s: %s",
                memory_id,
                e,
            )
            return ModelConnectionsResult(
                status="error",
                error_message=f"Query failed: {e}",
            )
        except Exception as e:
            logger.error(
                "Unexpected error getting connections for %s: %s",
                memory_id,
                e,
            )
            return ModelConnectionsResult(
                status="error",
                error_message=f"Unexpected error: {e}",
            )

    async def health_check(self) -> ModelGraphMemoryHealth:
        """Check if the graph connection is healthy.

        Returns:
            ModelGraphMemoryHealth with detailed health status information.
        """
        if not self._initialized or self._handler is None:
            return ModelGraphMemoryHealth(
                is_healthy=False,
                initialized=False,
                handler_healthy=None,
                error_message="Adapter not initialized",
            )

        try:
            health = await self._handler.health_check()
            handler_healthy = bool(health.healthy)
            return ModelGraphMemoryHealth(
                is_healthy=handler_healthy,
                initialized=True,
                handler_healthy=handler_healthy,
                error_message=None if handler_healthy else "Handler reports unhealthy",
            )
        except Exception as e:
            return ModelGraphMemoryHealth(
                is_healthy=False,
                initialized=True,
                handler_healthy=None,
                error_message=f"Health check failed: {e}",
            )

    async def shutdown(self) -> None:
        """Shutdown the adapter and release resources."""
        if self._initialized and self._handler is not None:
            await self._handler.shutdown()
            self._handler = None
            self._initialized = False
            logger.info("AdapterGraphMemory shutdown complete")
