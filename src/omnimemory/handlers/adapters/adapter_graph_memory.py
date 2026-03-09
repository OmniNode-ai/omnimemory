# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Graph Handler Adapter for relationship-based memory queries.

from omnibase_spi to support memory-specific graph operations. It enables
"memories related to X" queries via graph traversal, translating between
memory domain concepts and graph database operations.

The adapter transforms memory operations into graph operations:
    - find_related(memory_id) -> execute_query() with BFS traversal
    - get_connections(memory_id) -> execute_query() with edge retrieval

Example::

    import asyncio
    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        ModelGraphMemoryConfig,
    )

    async def example():
        config = ModelGraphMemoryConfig(max_depth=5)
        adapter = AdapterGraphMemory(config)
        await adapter.initialize(
            connection_uri="bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}",
            auth=("neo4j", "password"),
        )

        # Find memories related to a specific memory
        related = await adapter.find_related("memory_abc123", depth=2)
        for memory in related.memories:
            print(f"Related: {memory.memory_id} (score={memory.score:.2f})")

        # Get direct connections
        result = await adapter.get_connections("memory_abc123")
        for conn in result.connections:
            print(f"{conn.source_id} --[{conn.relationship_type}]--> {conn.target_id}")

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

import asyncio
import heapq
import importlib
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar, cast
from urllib.parse import urlparse

from omnibase_core.container import ModelONEXContainer
from omnibase_core.types.type_json import JsonType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_spi.protocols.storage.protocol_graph_database_handler import (
    ProtocolGraphDatabaseHandler,
)

from omnimemory.models.adapters import (
    ModelConnectionsResult,
    ModelGraphMemoryConfig,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
    PropertyValue,
)

logger = logging.getLogger(__name__)

# TypeVar for the generic retry helper return type
_T = TypeVar("_T")

# Transient error types that trigger automatic retry with exponential backoff.
# Permanent errors (InfraAuthenticationError, InfraProtocolError, etc.) are not
# included here because retrying them would not resolve the underlying problem.
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)


def _create_graph_handler(
    container: ModelONEXContainer,
) -> ProtocolGraphDatabaseHandler:
    """Lazily resolve and instantiate the concrete HandlerGraph at runtime.

    Uses ``importlib`` to import the concrete ``HandlerGraph`` class from
    ``omnibase_infra`` so that this module depends only on the SPI protocol
    at import time, not on the concrete infrastructure implementation.

    Args:
        container: The ONEX container for dependency injection.

    Returns:
        A concrete handler instance satisfying ProtocolGraphDatabaseHandler.

    Raises:
        RuntimeError: If the concrete class cannot be imported or instantiated.
    """
    try:
        module = importlib.import_module("omnibase_infra.handlers.handler_graph")
        handler_cls = module.HandlerGraph
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "Failed to resolve concrete HandlerGraph from omnibase_infra. "
            "Ensure omnibase_infra is installed."
        ) from e
    handler: ProtocolGraphDatabaseHandler = handler_cls(container)
    return handler


__all__ = [
    "AdapterGraphMemory",
    "ModelConnectionsResult",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    "PropertyValue",
]


# =============================================================================
# Cypher Query Templates
# =============================================================================
# All templates use parameterized queries to prevent injection attacks.
# See docs/handler_reuse_matrix.md Security section for guidelines.


class CypherTemplates:
    """Parameterized Cypher query templates for memory graph operations.

    All template methods accept a ``node_label`` parameter to allow configurable
    node labels (defaults to "Memory" in ModelGraphMemoryConfig).

    Direction Behavior:
        - get_connections: Bidirectional (matches both incoming and outgoing)
        - get_connections_by_type: Bidirectional with type filtering
        - get_connections_outgoing: Outgoing only (from source to target)
        - get_connections_by_type_outgoing: Outgoing only with type filtering

        Bidirectional templates use ``startNode(r) = m AS is_outgoing`` to dynamically
        determine edge direction. Outgoing-only templates always return ``true`` for
        is_outgoing since all edges are outgoing by definition.

    Security:
        All queries use parameters ($param) instead of string interpolation.
        The ``node_label`` parameter is safe from injection as it comes from
        config validation (string type with pydantic validation).
        NEVER construct queries by concatenating user input.
    """

    @staticmethod
    def get_connections(node_label: str) -> str:
        """Generate query to find direct edges for a memory node (bidirectional).

        Args:
            node_label: Graph label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string.
        """
        return f"""
        MATCH (m:{node_label} {{memory_id: $memory_id}})-[r]-(n:{node_label})
        RETURN
            m.memory_id AS source_id,
            n.memory_id AS target_id,
            type(r) AS relationship_type,
            r.weight AS weight,
            r.created_at AS created_at,
            startNode(r) = m AS is_outgoing
        LIMIT $limit
        """

    @staticmethod
    def get_connections_by_type(node_label: str) -> str:
        """Generate query to find connections filtered by type (bidirectional).

        Args:
            node_label: Graph label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string.
        """
        return f"""
        MATCH (m:{node_label} {{memory_id: $memory_id}})-[r]-(n:{node_label})
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

    @staticmethod
    def get_connections_outgoing(node_label: str) -> str:
        """Generate query to find outgoing edges only (from source to target).

        Args:
            node_label: Graph label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string.
        """
        return f"""
        MATCH (m:{node_label} {{memory_id: $memory_id}})-[r]->(n:{node_label})
        RETURN
            m.memory_id AS source_id,
            n.memory_id AS target_id,
            type(r) AS relationship_type,
            r.weight AS weight,
            r.created_at AS created_at,
            true AS is_outgoing
        LIMIT $limit
        """

    @staticmethod
    def get_connections_by_type_outgoing(node_label: str) -> str:
        """Generate query to find outgoing connections filtered by type.

        Args:
            node_label: Graph label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string.
        """
        return f"""
        MATCH (m:{node_label} {{memory_id: $memory_id}})-[r]->(n:{node_label})
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

    @staticmethod
    def count_connections(node_label: str) -> str:
        """Generate query to count connections for a memory.

        Args:
            node_label: Graph label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string.
        """
        return f"""
        MATCH (m:{node_label} {{memory_id: $memory_id}})-[r]-()
        RETURN count(r) AS connection_count
        """

    @staticmethod
    def node_exists(node_label: str) -> str:
        """Generate query to check if a memory node exists.

        Note: Using id(m) instead of elementId(m) for Memgraph compatibility
        (Neo4j 5.x prefers elementId() but id() still works).

        Args:
            node_label: Graph label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string.
        """
        return f"""
        MATCH (m:{node_label} {{memory_id: $memory_id}})
        RETURN m.memory_id AS memory_id, id(m) AS element_id
        LIMIT 1
        """

    # Template for index creation - uses node_label parameter
    # NOTE: Index creation is idempotent in Memgraph (creating an existing index
    # returns an error that we can safely ignore). The index dramatically improves
    # query performance for memory_id lookups.
    @staticmethod
    def create_memory_index(node_label: str) -> str:
        """Generate index creation query for memory_id property.

        Args:
            node_label: The label for memory nodes (e.g., "Memory").

        Returns:
            Cypher query string to create the index.
        """
        return f"CREATE INDEX ON :{node_label}(memory_id);"

    # Template functions for find_related queries
    # NOTE: Memgraph does NOT support parameterized depth in variable-length paths
    # (e.g., `[*1..$max_depth]` fails), so we must embed the depth value directly.
    # This is safe because depth is bounded by config validation (1-10 integer).
    @staticmethod
    def find_related_query(
        max_depth: int, node_label: str, bidirectional: bool = True
    ) -> str:
        """Generate FIND_RELATED query with embedded depth value.

        Args:
            max_depth: Maximum traversal depth (must be a bounded integer, 1-10).
            node_label: Graph label for memory nodes (e.g., "Memory").
            bidirectional: Whether to traverse in both directions.

        Returns:
            Cypher query string with depth embedded.
        """
        direction = "-" if bidirectional else "->"
        return f"""
        MATCH (start:{node_label} {{memory_id: $memory_id}})
              -[r*1..{max_depth}]{direction}(related:{node_label})
        WHERE related.memory_id <> $memory_id
        RETURN DISTINCT
            related.memory_id AS memory_id,
            labels(related) AS labels,
            properties(related) AS properties,
            size(r) AS depth
        ORDER BY depth ASC
        LIMIT $limit
        """

    @staticmethod
    def find_related_by_type_query(
        max_depth: int, node_label: str, bidirectional: bool = True
    ) -> str:
        """Generate FIND_RELATED_BY_TYPE query with embedded depth value.

        Args:
            max_depth: Maximum traversal depth (must be a bounded integer, 1-10).
            node_label: Graph label for memory nodes (e.g., "Memory").
            bidirectional: Whether to traverse in both directions.

        Returns:
            Cypher query string with depth embedded.
        """
        direction = "-" if bidirectional else "->"
        return f"""
        MATCH (start:{node_label} {{memory_id: $memory_id}})
              -[r*1..{max_depth}]{direction}(related:{node_label})
        WHERE related.memory_id <> $memory_id
          AND ALL(rel IN r WHERE type(rel) IN $relationship_types)
        RETURN DISTINCT
            related.memory_id AS memory_id,
            labels(related) AS labels,
            properties(related) AS properties,
            size(r) AS depth
        ORDER BY depth ASC
        LIMIT $limit
        """


# =============================================================================
# Adapter
# =============================================================================


class AdapterGraphMemory:
    """Adapter that wraps a ProtocolGraphDatabaseHandler for memory-specific graph operations.

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
        handler: The underlying ProtocolGraphDatabaseHandler instance.

    Example::

        async def example():
            config = ModelGraphMemoryConfig(max_depth=3)
            adapter = AdapterGraphMemory(config)
            await adapter.initialize(
                connection_uri="bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}",
            )

            # Find related memories up to 2 hops away
            result = await adapter.find_related("mem_123", depth=2)
            for mem in result.memories:
                print(f"Found: {mem.memory_id} at depth {mem.depth}")

            await adapter.shutdown()
    """

    def __init__(
        self,
        config: ModelGraphMemoryConfig,
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
        self._handler: ProtocolGraphDatabaseHandler | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelGraphMemoryConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def handler(self) -> ProtocolGraphDatabaseHandler | None:
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
        options: Mapping[str, JsonType] | None = None,
    ) -> None:
        """Initialize the adapter and underlying graph handler.

        Establishes connection to the graph database and prepares
        the handler for memory queries.

        Args:
            connection_uri: Graph database URI (e.g., "bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}").
            auth: Optional (username, password) tuple for authentication.
            options: Additional connection options passed to the graph handler.

        Raises:
            RuntimeError: If initialization fails.
            ValueError: If connection_uri is malformed.
            InfraConnectionError: If connection to graph database fails.
        """
        # Validate URI format before attempting connection
        parsed_uri = urlparse(connection_uri)
        if not parsed_uri.scheme or not parsed_uri.hostname:
            raise ValueError(
                f"Invalid connection_uri: '{connection_uri}'. "
                "Expected format: 'bolt://hostname:port' or 'bolt+s://hostname:port'"
            )
        if parsed_uri.scheme not in ("bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s"):
            logger.warning(
                "Unexpected URI scheme '%s' in connection_uri. "
                "Expected bolt, bolt+s, bolt+ssc, neo4j, or neo4j+s.",
                parsed_uri.scheme,
            )

        try:
            # Timeout covers both lock acquisition and initialization work
            # to prevent hanging if another coroutine holds the lock indefinitely
            async with asyncio.timeout(self._config.timeout_seconds):
                async with self._init_lock:
                    # Early return for already-initialized
                    if self._initialized:
                        return
                    try:
                        # Create container if not provided
                        if self._container is None:
                            # Import here to get the real class
                            from omnibase_core.container import ModelONEXContainer

                            self._container = ModelONEXContainer()

                        # Create and initialize handler via lazy resolution
                        self._handler = _create_graph_handler(self._container)

                        init_options: dict[str, JsonType] = {
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

                        # Ensure indexes exist for optimal query performance
                        if self._config.ensure_indexes:
                            try:
                                await self._handler.execute_query(
                                    query=CypherTemplates.create_memory_index(
                                        self._config.memory_node_label
                                    ),
                                    parameters={},
                                )
                                logger.info(
                                    "Ensured index on %s(memory_id)",
                                    self._config.memory_node_label,
                                )
                            except Exception as e:
                                # Index may already exist - log but don't fail
                                error_msg = str(e).lower()
                                if (
                                    "already exists" in error_msg
                                    or "duplicate" in error_msg
                                ):
                                    logger.debug(
                                        "Index already exists on %s(memory_id)",
                                        self._config.memory_node_label,
                                    )
                                else:
                                    logger.warning(
                                        "Index creation failed for %s(memory_id): %s",
                                        self._config.memory_node_label,
                                        e,
                                    )

                    except InfraConnectionError:
                        raise
                    except Exception as e:
                        logger.error(
                            "Failed to initialize AdapterGraphMemory: %s",
                            e,
                        )
                        raise RuntimeError(f"Initialization failed: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(
                f"Initialization timed out after {self._config.timeout_seconds}s. "
                "Possible causes: (1) Lock contention - another coroutine may be "
                "holding the initialization lock (e.g., a concurrent initialize() "
                "call is still in progress); (2) Database connection issue - the "
                "graph database may be slow or unresponsive. Suggestions: Check if "
                "another initialization is in progress, verify the database is "
                "reachable, or increase timeout_seconds in ModelGraphMemoryConfig."
            ) from e

    def _ensure_initialized(self) -> ProtocolGraphDatabaseHandler:
        """Ensure adapter is initialized and return handler.

        Returns:
            The initialized ProtocolGraphDatabaseHandler.

        Raises:
            RuntimeError: If adapter is not initialized.
        """
        if not self._initialized or self._handler is None:
            raise RuntimeError(
                "AdapterGraphMemory not initialized. Call initialize() first."
            )
        return self._handler

    async def _execute_with_retry(
        self,
        operation_name: str,
        fn: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Execute an async operation with exponential backoff retry.

        Retries the given callable on transient graph DB failures
        (InfraConnectionError, InfraTimeoutError, InfraUnavailableError).
        Permanent errors (authentication, protocol violations, etc.) propagate
        immediately without retrying.

        Backoff formula: ``delay = min(base * 2^attempt + jitter, max_delay)``
        where ``jitter`` is a uniform random value in ``[0, base]`` to prevent
        thundering-herd effects when multiple callers retry simultaneously.

        Args:
            operation_name: Human-readable name used in log messages (e.g.,
                ``"find_related"`` or ``"get_connections"``).
            fn: Zero-argument async callable that performs the graph operation.

        Returns:
            The return value of ``fn()`` on the first successful attempt.

        Raises:
            InfraConnectionError | InfraTimeoutError | InfraUnavailableError:
                Re-raised after all retry attempts are exhausted.
            Exception: Any non-transient exception propagates immediately.
        """
        max_retries = self._config.max_retries
        base_delay = self._config.retry_base_delay_seconds
        max_delay = self._config.retry_max_delay_seconds

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await fn()
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if attempt >= max_retries:
                    # All retries exhausted - re-raise the last transient error
                    logger.warning(
                        "Graph operation '%s' failed after %d attempt(s): %s",
                        operation_name,
                        attempt + 1,
                        exc,
                    )
                    raise
                # Calculate exponential backoff with jitter.
                # random.uniform (default PRNG) is acceptable here: asyncio is
                # single-threaded so there is no shared-state race, and retry
                # jitter does not require cryptographic quality randomness.
                delay = min(
                    base_delay * (2**attempt) + random.uniform(0.0, base_delay),
                    max_delay,
                )
                logger.warning(
                    "Graph operation '%s' attempt %d/%d failed with transient "
                    "error %s: %s. Retrying in %.2fs.",
                    operation_name,
                    attempt + 1,
                    max_retries + 1,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # This line is unreachable in practice: ModelGraphMemoryConfig enforces
        # max_retries >= 0, so the loop executes at least once (attempt 0) and
        # always assigns last_exc before raising or completing all retries.
        # The assertion satisfies mypy's exhaustiveness analysis.
        assert last_exc is not None
        raise last_exc  # pragma: no cover

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

                **Practical min_score guidance** (score = 1/(depth+1)):

                - ``min_score=0.5`` includes only depth=1 (direct neighbors)
                - ``min_score=0.3`` filters out depth > 2 (keeps depth 1-2)
                - ``min_score=0.25`` filters out depth > 3 (keeps depth 1-3)
                - ``min_score=0.2`` filters out depth > 4 (keeps depth 1-4)
                - ``min_score=0.15`` filters out depth > 5 (keeps depth 1-5)
                - ``min_score=0.1`` keeps most results up to depth 9
                - ``min_score=0.0`` keeps all results (no filtering)

        Returns:
            ModelRelatedMemoryResult with related memories ordered by score.

        Raises:
            RuntimeError: If adapter is not initialized.

        Note:
            **Score Filtering and Result Count Tradeoff**

            When ``min_score`` is set above 0.0, this method fetches extra
            candidates (controlled by ``config.score_filter_multiplier``,
            default 3.0) to account for results that will be filtered out.

            However, if ``min_score`` is high (e.g., 0.9) and most graph
            relationships have lower scores, fewer results than ``limit``
            may be returned. This is expected behavior.

            To increase the likelihood of receiving ``limit`` results when
            using high ``min_score`` values, increase
            ``score_filter_multiplier`` in the adapter configuration (up to
            10.0). Be aware that higher multipliers increase query cost.
        """
        handler = self._ensure_initialized()

        # Apply bounds - use explicit None checks to allow 0 as a valid value
        effective_depth = min(
            depth if depth is not None else self._config.default_depth,
            self._config.max_depth,
        )
        effective_limit = min(
            limit if limit is not None else self._config.default_limit,
            self._config.max_limit,
        )
        # Clamp min_score to valid range [0.0, 1.0]
        effective_min_score = max(0.0, min(min_score, 1.0))

        # Determine traversal direction (bidirectional or outgoing-only)
        is_bidirectional = self._config.bidirectional

        # Select appropriate query template based on direction and filters
        # Request more results than needed to account for min_score filtering.
        # The multiplier is configurable via score_filter_multiplier.
        query_limit = min(
            int(effective_limit * self._config.score_filter_multiplier),
            self._config.max_limit,
        )

        # Generate query with embedded depth (required for Memgraph compatibility)
        # Memgraph does NOT support parameterized depth in variable-length paths
        node_label = self._config.memory_node_label
        if relationship_types:
            traversal_query = CypherTemplates.find_related_by_type_query(
                max_depth=effective_depth,
                node_label=node_label,
                bidirectional=is_bidirectional,
            )
            traversal_params: dict[str, JsonType] = {
                "memory_id": memory_id,
                "relationship_types": cast("list[JsonType]", relationship_types),
                "limit": query_limit,
            }
        else:
            traversal_query = CypherTemplates.find_related_query(
                max_depth=effective_depth,
                node_label=node_label,
                bidirectional=is_bidirectional,
            )
            traversal_params = {
                "memory_id": memory_id,
                "limit": query_limit,
            }

        async def _do_find_related() -> ModelRelatedMemoryResult:
            start_time = time.perf_counter()

            # First, check if the memory node exists
            node_result = await handler.execute_query(
                query=CypherTemplates.node_exists(node_label),
                parameters={"memory_id": memory_id},
            )

            if not node_result.records:
                return ModelRelatedMemoryResult(
                    status="not_found",
                    error_message=f"Memory '{memory_id}' not found in graph",
                )

            # Execute the traversal query
            result = await handler.execute_query(
                query=traversal_query,
                parameters=traversal_params,
            )

            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            # Convert query results to related memories
            memories: list[ModelRelatedMemory] = []
            max_depth_reached = 0
            # Track candidates before min_score filtering for observability
            candidates_found = 0

            for record in result.records:
                node_memory_id = record.get("memory_id")
                if not isinstance(node_memory_id, str):
                    continue

                # Count valid candidates before score filtering
                candidates_found += 1

                # Get depth from query result (path length)
                raw_depth = record.get("depth")
                depth_to_node = (
                    int(raw_depth) if isinstance(raw_depth, (int, float, str)) else 1
                )
                max_depth_reached = max(max_depth_reached, depth_to_node)

                # Calculate relevance score based on traversal depth (edge count).
                # Score formula: 1/(depth+1) gives closer nodes higher scores.
                # Score range: 0.5 (depth=1) to ~0.09 (depth=10, max allowed).
                # Note: Score never reaches 1.0 since minimum traversal depth is 1.
                # Examples: depth=1 -> 0.5, depth=2 -> 0.33, depth=3 -> 0.25
                score = 1.0 / (depth_to_node + 1)

                if score < effective_min_score:
                    continue

                # Extract labels and properties from result
                labels = record.get("labels", [])
                if not isinstance(labels, list):
                    labels = []

                properties = record.get("properties", {})
                if not isinstance(properties, dict):
                    properties = {}

                # Build path as memory IDs (start and end nodes)
                path_memory_ids = [memory_id, str(node_memory_id)]

                memories.append(
                    ModelRelatedMemory(
                        memory_id=node_memory_id,  # Verified as str above
                        score=score,
                        path=path_memory_ids,
                        depth=depth_to_node,
                        labels=list(labels),
                        properties=dict(properties),
                    )
                )

            # Use heapq for O(n log k) instead of O(n log n) full sort
            memories = heapq.nlargest(effective_limit, memories, key=lambda m: m.score)

            if not memories:
                return ModelRelatedMemoryResult(
                    status="no_results",
                    memories=[],
                    total_count=0,
                    candidates_found=candidates_found,
                    max_depth_reached=max_depth_reached,
                    execution_time_ms=execution_time_ms,
                )

            return ModelRelatedMemoryResult(
                status="success",
                memories=memories,
                total_count=len(memories),
                candidates_found=candidates_found,
                max_depth_reached=max_depth_reached,
                execution_time_ms=execution_time_ms,
            )

        try:
            return await self._execute_with_retry("find_related", _do_find_related)
        except _TRANSIENT_ERRORS as e:
            logger.warning(
                "Graph traversal failed for memory %s after all retries: %s",
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

        Retrieves relationships connected to the specified memory,
        optionally filtered by relationship type and direction.

        Args:
            memory_id: The memory's identifier.
            relationship_types: Optional list of relationship types to include.
                If None, all types are returned.
            limit: Maximum number of connections. Defaults to config.default_limit.
            bidirectional: Whether to include both incoming and outgoing connections.
                If None, defaults to config.bidirectional. When True, returns
                connections in both directions (using `-[r]-` pattern). When False,
                returns only outgoing connections (using `-[r]->` pattern).

        Returns:
            ModelConnectionsResult with the memory's connections.

        Raises:
            RuntimeError: If adapter is not initialized.
        """
        handler = self._ensure_initialized()

        # Validate limit if provided - return early for non-positive values
        # (consistent with find_related() behavior for edge case handling)
        if limit is not None and limit < 1:
            return ModelConnectionsResult(
                status="no_results",
                connections=[],
                total_count=0,
            )

        # Apply bounds - default to config if None, cap at max_limit
        effective_limit = min(
            limit if limit is not None else self._config.default_limit,
            self._config.max_limit,
        )

        # Resolve bidirectional: use passed value if not None, otherwise use config.
        # This allows callers to override per-call while falling back to config default.
        # Verified correct: explicit False passes through, explicit True passes through,
        # None falls back to self._config.bidirectional.
        effective_bidirectional = (
            bidirectional if bidirectional is not None else self._config.bidirectional
        )

        # Choose query based on bidirectional flag and relationship_types
        node_label = self._config.memory_node_label
        if effective_bidirectional and relationship_types:
            # Bidirectional with type filter
            conn_query = CypherTemplates.get_connections_by_type(node_label)
            conn_params: dict[str, JsonType] = {
                "memory_id": memory_id,
                "relationship_types": cast("list[JsonType]", relationship_types),
                "limit": effective_limit,
            }
        elif effective_bidirectional:
            # Bidirectional without type filter
            conn_query = CypherTemplates.get_connections(node_label)
            conn_params = {
                "memory_id": memory_id,
                "limit": effective_limit,
            }
        elif relationship_types:
            # Outgoing-only with type filter
            conn_query = CypherTemplates.get_connections_by_type_outgoing(node_label)
            conn_params = {
                "memory_id": memory_id,
                "relationship_types": cast("list[JsonType]", relationship_types),
                "limit": effective_limit,
            }
        else:
            # Outgoing-only without type filter
            conn_query = CypherTemplates.get_connections_outgoing(node_label)
            conn_params = {
                "memory_id": memory_id,
                "limit": effective_limit,
            }

        async def _do_get_connections() -> ModelConnectionsResult:
            result = await handler.execute_query(
                query=conn_query,
                parameters=conn_params,
            )

            if not result.records:
                # Check if node exists
                exists_result = await handler.execute_query(
                    query=CypherTemplates.node_exists(node_label),
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
                # Preserve weight=0.0 (falsy but valid), default to 1.0 only for None
                raw_weight = record.get("weight")
                weight = raw_weight if raw_weight is not None else 1.0

                connections.append(
                    ModelMemoryConnection(
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        relationship_type=record["relationship_type"],
                        weight=weight,
                        is_outgoing=record.get("is_outgoing", True),
                        created_at=record.get("created_at"),
                    )
                )

            return ModelConnectionsResult(
                status="success",
                connections=connections,
                total_count=len(connections),
            )

        try:
            return await self._execute_with_retry(
                "get_connections", _do_get_connections
            )
        except _TRANSIENT_ERRORS as e:
            logger.warning(
                "Failed to get connections for memory %s after all retries: %s",
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
            # Log summary at WARNING, full traceback at DEBUG to reduce noise
            logger.warning(
                "Health check failed with %s: %s",
                type(e).__name__,
                e,
            )
            logger.debug(
                "Health check exception traceback",
                exc_info=True,
            )
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
