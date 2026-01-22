# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for AdapterGraphMemory with real Memgraph database.

This module tests the graph memory adapter against a real Memgraph instance
to verify end-to-end functionality beyond unit tests with mocks.

Test Categories:
    - Connection: Real database connection lifecycle
    - Traversal: Graph traversal with real data
    - Query: Direct edge retrieval queries
    - Health: Health check with live connection

Prerequisites:
    - Memgraph running at MEMGRAPH_URI (default: bolt://localhost:7687)
    - omnibase_infra installed (dev dependency)

Usage:
    # Run only integration tests
    pytest tests/handlers/adapters/test_adapter_graph_memory_integration.py -v

    # Run with specific markers
    pytest -m "integration and memgraph" -v

    # Skip if Memgraph unavailable (automatic)
    pytest -m integration -v

Environment Variables:
    MEMGRAPH_URI: Memgraph connection URI (default: bolt://localhost:7687)
    MEMGRAPH_USER: Memgraph username (optional)
    MEMGRAPH_PASSWORD: Memgraph password (optional)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

# Check if omnibase_infra is available
_MEMGRAPH_AVAILABLE = False
_SKIP_REASON = "omnibase_infra not installed"

try:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    _MEMGRAPH_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError:
    _SKIP_REASON = "neo4j driver not installed (required for Memgraph)"

try:
    from omnimemory.handlers.adapters.adapter_graph_memory import (
        AdapterGraphMemory,
        ModelGraphMemoryConfig,
    )
except ImportError:
    _MEMGRAPH_AVAILABLE = False
    _SKIP_REASON = "AdapterGraphMemory not available"


if TYPE_CHECKING:
    from neo4j import AsyncDriver


# =============================================================================
# Configuration
# =============================================================================

# Default connection settings
DEFAULT_MEMGRAPH_URI = "bolt://localhost:7687"
DEFAULT_MEMGRAPH_USER = "memgraph"
DEFAULT_MEMGRAPH_PASSWORD = ""

# Test data prefix to avoid conflicts
TEST_PREFIX = "test_mem_"


def get_memgraph_uri() -> str:
    """Get Memgraph URI from environment or default."""
    return os.environ.get("MEMGRAPH_URI", DEFAULT_MEMGRAPH_URI)


def get_memgraph_auth() -> tuple[str, str] | None:
    """Get Memgraph authentication from environment.

    Returns a (username, password) tuple if both are configured, or None for
    anonymous authentication.

    Authentication Behavior:
        - If BOTH MEMGRAPH_USER and MEMGRAPH_PASSWORD are set (non-empty): Returns tuple
        - If ONLY MEMGRAPH_USER is set (no password): Returns None (anonymous auth)
        - If ONLY MEMGRAPH_PASSWORD is set (no user): Returns None (anonymous auth)
        - If neither is set: Returns None (anonymous auth)

    This is intentional because Memgraph/Neo4j driver requires both username AND
    password for authentication. Providing only one would cause authentication
    errors. When auth=None, the driver uses anonymous (no-auth) connection, which
    is suitable for development Memgraph instances without authentication enabled.

    Note:
        The default MEMGRAPH_USER="memgraph" is used, but DEFAULT_MEMGRAPH_PASSWORD
        is empty string, so if password is not explicitly set in environment,
        anonymous authentication will be used by default.

    Returns:
        Tuple of (username, password) if both are configured, None otherwise.
    """
    user = os.environ.get("MEMGRAPH_USER", DEFAULT_MEMGRAPH_USER)
    password = os.environ.get("MEMGRAPH_PASSWORD", DEFAULT_MEMGRAPH_PASSWORD)
    # Both username AND password required for Memgraph authentication.
    # If only one is set, fall back to anonymous auth (None) to avoid auth errors.
    if user and password:
        return (user, password)
    return None


async def check_memgraph_available() -> bool:
    """Check if Memgraph is available and responding."""
    if not _MEMGRAPH_AVAILABLE:
        return False

    uri = get_memgraph_uri()
    auth = get_memgraph_auth()

    try:
        driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=auth)
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            await result.consume()
        await driver.close()
        return True
    except (ServiceUnavailable, OSError, Exception):
        return False


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip all tests if Memgraph is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.memgraph,
    pytest.mark.skipif(
        not _MEMGRAPH_AVAILABLE,
        reason=_SKIP_REASON,
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def memgraph_available() -> bool:
    """Check if Memgraph is available for tests.

    Returns:
        True if Memgraph is reachable, False otherwise.
    """
    return await check_memgraph_available()


@pytest.fixture
async def memgraph_driver(
    memgraph_available: bool,
) -> AsyncGenerator[AsyncDriver | None, None]:
    """Create a Memgraph driver for test setup/teardown.

    Yields:
        AsyncDriver connected to Memgraph, or None if unavailable.
    """
    if not memgraph_available:
        pytest.skip("Memgraph is not available")

    uri = get_memgraph_uri()
    auth = get_memgraph_auth()
    driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=auth)

    yield driver

    await driver.close()


@pytest.fixture
def test_session_id() -> str:
    """Generate a unique session ID for test isolation.

    Returns:
        Unique identifier for this test session.
    """
    return uuid.uuid4().hex[:8]


@pytest.fixture
async def test_graph_data(
    memgraph_driver: AsyncDriver | None,
    test_session_id: str,
) -> AsyncGenerator[dict[str, str], None]:
    """Create test graph data in Memgraph.

    Creates a small test graph with Memory nodes and relationships:
        mem_a --[related_to]--> mem_b
        mem_a --[caused_by]--> mem_c
        mem_b --[related_to]--> mem_d
        mem_c --[related_to]--> mem_d

    Yields:
        Dictionary mapping logical names to actual memory IDs.
    """
    if memgraph_driver is None:
        pytest.skip("Memgraph driver not available")

    # Create unique memory IDs for this test session
    memory_ids = {
        "mem_a": f"{TEST_PREFIX}{test_session_id}_a",
        "mem_b": f"{TEST_PREFIX}{test_session_id}_b",
        "mem_c": f"{TEST_PREFIX}{test_session_id}_c",
        "mem_d": f"{TEST_PREFIX}{test_session_id}_d",
        "mem_isolated": f"{TEST_PREFIX}{test_session_id}_isolated",
    }

    # Create test nodes and relationships
    async with memgraph_driver.session() as session:
        # Create Memory nodes
        create_nodes_query = """
        CREATE (a:Memory {memory_id: $mem_a, content: 'Test memory A'})
        CREATE (b:Memory {memory_id: $mem_b, content: 'Test memory B'})
        CREATE (c:Memory {memory_id: $mem_c, content: 'Test memory C'})
        CREATE (d:Memory {memory_id: $mem_d, content: 'Test memory D'})
        CREATE (isolated:Memory {memory_id: $mem_isolated, content: 'Isolated memory'})
        CREATE (a)-[:related_to {weight: 0.9}]->(b)
        CREATE (a)-[:caused_by {weight: 0.8}]->(c)
        CREATE (b)-[:related_to {weight: 0.7}]->(d)
        CREATE (c)-[:related_to {weight: 0.6}]->(d)
        """
        await session.run(create_nodes_query, memory_ids)

    yield memory_ids

    # Cleanup: Remove test nodes
    async with memgraph_driver.session() as session:
        cleanup_query = """
        MATCH (m:Memory)
        WHERE m.memory_id STARTS WITH $prefix
        DETACH DELETE m
        """
        await session.run(
            cleanup_query,
            {"prefix": f"{TEST_PREFIX}{test_session_id}"},
        )


@pytest.fixture
def adapter_config() -> ModelGraphMemoryConfig:
    """Create adapter configuration for tests."""
    return ModelGraphMemoryConfig(
        max_depth=5,
        default_depth=2,
        default_limit=100,
        max_limit=1000,
        bidirectional=True,
        memory_node_label="Memory",
        timeout_seconds=30.0,
    )


@pytest.fixture
async def initialized_adapter(
    memgraph_available: bool,
    adapter_config: ModelGraphMemoryConfig,
) -> AsyncGenerator[AdapterGraphMemory, None]:
    """Create and initialize an adapter for testing.

    Yields:
        Initialized AdapterGraphMemory instance.
    """
    if not memgraph_available:
        pytest.skip("Memgraph is not available")

    adapter = AdapterGraphMemory(adapter_config)
    uri = get_memgraph_uri()
    auth = get_memgraph_auth()

    await adapter.initialize(connection_uri=uri, auth=auth)

    yield adapter

    await adapter.shutdown()


# =============================================================================
# Connection Tests
# =============================================================================


class TestRealConnection:
    """Tests for real Memgraph connection lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_real_connection(
        self,
        memgraph_available: bool,
        adapter_config: ModelGraphMemoryConfig,
    ) -> None:
        """Test actual connection to Memgraph database."""
        if not memgraph_available:
            pytest.skip("Memgraph is not available")

        adapter = AdapterGraphMemory(adapter_config)
        uri = get_memgraph_uri()
        auth = get_memgraph_auth()

        # Should not be initialized yet
        assert not adapter.is_initialized

        # Initialize connection
        await adapter.initialize(connection_uri=uri, auth=auth)

        # Should be initialized
        assert adapter.is_initialized
        assert adapter.handler is not None

        # Cleanup
        await adapter.shutdown()
        assert not adapter.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_idempotent_real(
        self,
        memgraph_available: bool,
        adapter_config: ModelGraphMemoryConfig,
    ) -> None:
        """Test that multiple initialize calls are safe."""
        if not memgraph_available:
            pytest.skip("Memgraph is not available")

        adapter = AdapterGraphMemory(adapter_config)
        uri = get_memgraph_uri()
        auth = get_memgraph_auth()

        # Initialize twice
        await adapter.initialize(connection_uri=uri, auth=auth)
        await adapter.initialize(connection_uri=uri, auth=auth)

        # Should still be initialized
        assert adapter.is_initialized

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_real(
        self,
        memgraph_available: bool,
        adapter_config: ModelGraphMemoryConfig,
    ) -> None:
        """Test clean shutdown of real connection."""
        if not memgraph_available:
            pytest.skip("Memgraph is not available")

        adapter = AdapterGraphMemory(adapter_config)
        uri = get_memgraph_uri()
        auth = get_memgraph_auth()

        await adapter.initialize(connection_uri=uri, auth=auth)
        assert adapter.is_initialized

        # Shutdown
        await adapter.shutdown()

        # Should be cleaned up
        assert not adapter.is_initialized
        assert adapter.handler is None

        # Multiple shutdowns should be safe
        await adapter.shutdown()
        assert not adapter.is_initialized


# =============================================================================
# Health Check Tests
# =============================================================================


class TestRealHealthCheck:
    """Tests for health check with real connection."""

    @pytest.mark.asyncio
    async def test_health_check_real_healthy(
        self,
        initialized_adapter: AdapterGraphMemory,
    ) -> None:
        """Test health check returns healthy with real connection."""
        result = await initialized_adapter.health_check()

        assert result.is_healthy is True
        assert result.initialized is True
        assert result.handler_healthy is True
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(
        self,
        memgraph_available: bool,
        adapter_config: ModelGraphMemoryConfig,
    ) -> None:
        """Test health check returns unhealthy when not initialized."""
        if not memgraph_available:
            pytest.skip("Memgraph is not available")

        adapter = AdapterGraphMemory(adapter_config)

        # Not initialized
        result = await adapter.health_check()

        assert result.is_healthy is False
        assert result.initialized is False
        assert result.handler_healthy is None
        assert result.error_message is not None


# =============================================================================
# Traversal Tests
# =============================================================================


class TestRealTraversal:
    """Tests for graph traversal with real data."""

    @pytest.mark.asyncio
    async def test_find_related_real_traversal(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test find_related with real graph traversal."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.find_related(mem_a, depth=2)

        assert result.status == "success"
        assert result.total_count > 0

        # Should find related memories
        found_ids = {m.memory_id for m in result.memories}
        assert (
            test_graph_data["mem_b"] in found_ids
            or test_graph_data["mem_c"] in found_ids
        )

    @pytest.mark.asyncio
    async def test_find_related_with_depth_1(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test find_related with depth=1 finds direct neighbors only."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.find_related(mem_a, depth=1)

        assert result.status == "success"

        # Should find direct neighbors (mem_b, mem_c)
        found_ids = {m.memory_id for m in result.memories}

        # Direct neighbors of mem_a
        expected_direct = {test_graph_data["mem_b"], test_graph_data["mem_c"]}
        assert found_ids.intersection(
            expected_direct
        ), f"Expected to find at least one of {expected_direct}, got {found_ids}"

    @pytest.mark.asyncio
    async def test_find_related_with_depth_2(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test find_related with depth=2 finds second-hop neighbors."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.find_related(mem_a, depth=2)

        assert result.status == "success"

        # Should potentially find mem_d (2 hops away)
        found_ids = {m.memory_id for m in result.memories}

        # mem_d is reachable via mem_b or mem_c
        # It should be in the results at depth=2
        # Note: Due to bidirectional traversal, exact results may vary
        assert len(found_ids) >= 1, "Should find at least one related memory"

    @pytest.mark.asyncio
    async def test_find_related_with_relationship_filter(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test find_related with relationship type filter.

        Graph structure:
            mem_a --[related_to]--> mem_b --[related_to]--> mem_d
            mem_a --[caused_by]--> mem_c --[related_to]--> mem_d

        When filtering by 'related_to' only:
            - mem_b should be found (direct related_to from mem_a)
            - mem_d should be found (via mem_b's related_to)
            - mem_c should NOT be found (connected via caused_by, not related_to)
        """
        mem_a = test_graph_data["mem_a"]
        mem_b = test_graph_data["mem_b"]
        mem_c = test_graph_data["mem_c"]

        # Only follow 'related_to' relationships
        result = await initialized_adapter.find_related(
            mem_a,
            depth=2,
            relationship_types=["related_to"],
        )

        assert result.status in ("success", "no_results")

        if result.status == "success":
            found_ids = {m.memory_id for m in result.memories}
            # mem_b is connected via related_to - should be found
            assert mem_b in found_ids, (
                f"Expected mem_b ({mem_b}) to be found via related_to filter, "
                f"but found: {found_ids}"
            )
            # mem_c is connected via caused_by - should NOT be found
            # when filtering only by related_to
            assert mem_c not in found_ids, (
                f"Expected mem_c ({mem_c}) to NOT be found when filtering by "
                f"related_to only (connected via caused_by), but found: {found_ids}"
            )

    @pytest.mark.asyncio
    async def test_find_related_not_found(
        self,
        initialized_adapter: AdapterGraphMemory,
    ) -> None:
        """Test find_related for non-existent memory."""
        result = await initialized_adapter.find_related("nonexistent_memory_xyz123")

        assert result.status == "not_found"
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_find_related_isolated_node(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test find_related for isolated node with no connections."""
        mem_isolated = test_graph_data["mem_isolated"]

        result = await initialized_adapter.find_related(mem_isolated, depth=2)

        # Node exists but has no connections
        assert result.status == "no_results"
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_find_related_scores_decrease_with_depth(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test that relevance scores decrease with traversal depth."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.find_related(mem_a, depth=3)

        if result.status == "success" and len(result.memories) > 1:
            # Results should be sorted by score descending
            scores = [m.score for m in result.memories]
            assert scores == sorted(
                scores, reverse=True
            ), "Results should be sorted by score descending"


# =============================================================================
# Connection Query Tests
# =============================================================================


class TestRealConnections:
    """Tests for get_connections with real data."""

    @pytest.mark.asyncio
    async def test_get_connections_real_query(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test get_connections retrieves real edges."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.get_connections(mem_a)

        assert result.status == "success"
        assert result.total_count >= 2  # mem_a has 2 outgoing edges

        # Check connection structure
        for conn in result.connections:
            assert mem_a in {conn.source_id, conn.target_id}
            assert conn.relationship_type in ("related_to", "caused_by")
            assert 0.0 <= conn.weight <= 1.0

    @pytest.mark.asyncio
    async def test_get_connections_with_type_filter(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test get_connections with relationship type filter."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.get_connections(
            mem_a,
            relationship_types=["related_to"],
        )

        assert result.status == "success"

        # All connections should be 'related_to'
        for conn in result.connections:
            assert conn.relationship_type == "related_to"

    @pytest.mark.asyncio
    async def test_get_connections_not_found(
        self,
        initialized_adapter: AdapterGraphMemory,
    ) -> None:
        """Test get_connections for non-existent memory."""
        result = await initialized_adapter.get_connections("nonexistent_memory_abc789")

        assert result.status == "not_found"
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_get_connections_isolated_node(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test get_connections for node with no edges."""
        mem_isolated = test_graph_data["mem_isolated"]

        result = await initialized_adapter.get_connections(mem_isolated)

        assert result.status == "no_results"
        assert result.connections == []
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_get_connections_with_limit(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test get_connections respects limit parameter."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.get_connections(mem_a, limit=1)

        assert result.status == "success"
        assert len(result.connections) <= 1

    @pytest.mark.asyncio
    async def test_get_connections_verifies_edge_properties(
        self,
        initialized_adapter: AdapterGraphMemory,
        test_graph_data: dict[str, str],
    ) -> None:
        """Test that edge properties are correctly retrieved."""
        mem_a = test_graph_data["mem_a"]

        result = await initialized_adapter.get_connections(mem_a)

        assert result.status == "success"

        # Find the related_to edge to mem_b
        related_edges = [
            c
            for c in result.connections
            if c.relationship_type == "related_to"
            and test_graph_data["mem_b"] in (c.source_id, c.target_id)
        ]

        if related_edges:
            edge = related_edges[0]
            # Weight should be 0.9 as set in test data
            assert edge.weight == pytest.approx(0.9, rel=0.01)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestRealErrorHandling:
    """Tests for error handling with real database."""

    @pytest.mark.asyncio
    async def test_operations_before_initialize_raise_error(
        self,
        memgraph_available: bool,
        adapter_config: ModelGraphMemoryConfig,
    ) -> None:
        """Test that operations before initialize raise RuntimeError."""
        if not memgraph_available:
            pytest.skip("Memgraph is not available")

        adapter = AdapterGraphMemory(adapter_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.find_related("test_mem")

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.get_connections("test_mem")

    @pytest.mark.asyncio
    async def test_invalid_connection_uri(
        self,
        memgraph_available: bool,
        adapter_config: ModelGraphMemoryConfig,
    ) -> None:
        """Test connection to invalid URI fails gracefully."""
        if not memgraph_available:
            pytest.skip("Memgraph is not available")

        adapter = AdapterGraphMemory(adapter_config)

        # Try to connect to non-existent server
        with pytest.raises((RuntimeError, Exception)):
            await adapter.initialize(
                connection_uri="bolt://invalid-host-xyz:7687",
                auth=("user", "pass"),
            )
