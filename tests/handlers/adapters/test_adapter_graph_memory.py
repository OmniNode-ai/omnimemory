# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for AdapterGraphMemory.

This module tests the graph memory adapter that wraps HandlerGraph
for memory-specific graph operations.

Test Categories:
    - Configuration: Config validation and defaults
    - Models: Pydantic model validation
    - find_related: Graph traversal to find related memories
    - get_connections: Direct edge retrieval
    - Error Handling: Failure scenarios
    - Lifecycle: Initialize and shutdown

Usage:
    pytest tests/handlers/adapters/test_adapter_graph_memory.py -v
    pytest tests/handlers/adapters/ -v -k "find_related"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimemory.handlers.adapters.adapter_graph_memory import (
    AdapterGraphMemory,
    AdapterGraphMemoryConfig,
    CypherTemplates,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock as MagicMockType  # noqa: F401


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> AdapterGraphMemoryConfig:
    """Create a default adapter configuration."""
    return AdapterGraphMemoryConfig(
        max_depth=5,
        default_depth=2,
        default_limit=100,
        max_limit=1000,
        bidirectional=True,
    )


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create a mock HandlerGraph.

    Returns:
        MagicMock configured with async methods matching HandlerGraph interface:
            - initialize: AsyncMock for handler initialization
            - shutdown: AsyncMock for handler shutdown
            - execute_query: AsyncMock for Cypher query execution
            - traverse: AsyncMock for graph traversal operations
            - health_check: AsyncMock for health status checks
    """
    handler: MagicMock = MagicMock()
    handler.initialize = AsyncMock()
    handler.shutdown = AsyncMock()
    handler.execute_query = AsyncMock()
    handler.traverse = AsyncMock()
    handler.health_check = AsyncMock()
    return handler


@pytest.fixture
def adapter_with_mock(
    config: AdapterGraphMemoryConfig,
    mock_handler: MagicMock,
) -> AdapterGraphMemory:
    """Create an adapter with a mock handler injected.

    Args:
        config: AdapterGraphMemoryConfig fixture with test configuration.
        mock_handler: MagicMock fixture configured as HandlerGraph.

    Returns:
        AdapterGraphMemory instance with mock handler injected and
        initialization state set to True for immediate use in tests.
    """
    adapter: AdapterGraphMemory = AdapterGraphMemory(config)
    adapter._handler = mock_handler
    adapter._initialized = True
    return adapter


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAdapterGraphMemoryConfig:
    """Tests for AdapterGraphMemoryConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AdapterGraphMemoryConfig()

        assert config.max_depth == 5
        assert config.default_depth == 2
        assert config.default_limit == 100
        assert config.max_limit == 1000
        assert config.bidirectional is True
        assert config.memory_node_label == "Memory"
        assert config.timeout_seconds == 30.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AdapterGraphMemoryConfig(
            max_depth=3,
            default_depth=1,
            default_limit=50,
            bidirectional=False,
            memory_node_label="MemoryNode",
            timeout_seconds=60.0,
        )

        assert config.max_depth == 3
        assert config.default_depth == 1
        assert config.default_limit == 50
        assert config.bidirectional is False
        assert config.memory_node_label == "MemoryNode"
        assert config.timeout_seconds == 60.0

    def test_max_depth_bounds(self) -> None:
        """Test max_depth has valid bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            AdapterGraphMemoryConfig(max_depth=0)

        # Too high
        with pytest.raises(ValidationError):
            AdapterGraphMemoryConfig(max_depth=11)

    def test_timeout_must_be_positive(self) -> None:
        """Test timeout must be positive."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AdapterGraphMemoryConfig(timeout_seconds=0)


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for Pydantic model validation."""

    def test_memory_connection_model(self) -> None:
        """Test ModelMemoryConnection creation."""
        conn = ModelMemoryConnection(
            source_id="mem_1",
            target_id="mem_2",
            relationship_type="related_to",
            weight=0.8,
            is_outgoing=True,
        )

        assert conn.source_id == "mem_1"
        assert conn.target_id == "mem_2"
        assert conn.relationship_type == "related_to"
        assert conn.weight == 0.8
        assert conn.is_outgoing is True

    def test_memory_connection_defaults(self) -> None:
        """Test ModelMemoryConnection default values."""
        conn = ModelMemoryConnection(
            source_id="mem_1",
            target_id="mem_2",
            relationship_type="related",
        )

        assert conn.weight == 1.0
        assert conn.is_outgoing is True
        assert conn.created_at is None

    def test_related_memory_model(self) -> None:
        """Test ModelRelatedMemory creation."""
        memory = ModelRelatedMemory(
            memory_id="mem_123",
            score=0.95,
            path=["mem_start", "mem_123"],
            depth=1,
            labels=["Memory"],
            properties={"key": "value"},
        )

        assert memory.memory_id == "mem_123"
        assert memory.score == 0.95
        assert memory.path == ["mem_start", "mem_123"]
        assert memory.depth == 1
        assert memory.labels == ["Memory"]
        assert memory.properties == {"key": "value"}

    def test_related_memory_result_success(self) -> None:
        """Test ModelRelatedMemoryResult success case."""
        result = ModelRelatedMemoryResult(
            status="success",
            memories=[
                ModelRelatedMemory(memory_id="mem_1", score=0.9),
                ModelRelatedMemory(memory_id="mem_2", score=0.8),
            ],
            total_count=2,
            max_depth_reached=2,
            execution_time_ms=50.0,
        )

        assert result.status == "success"
        assert len(result.memories) == 2
        assert result.total_count == 2
        assert result.error_message is None

    def test_related_memory_result_error(self) -> None:
        """Test ModelRelatedMemoryResult error case."""
        result = ModelRelatedMemoryResult(
            status="error",
            error_message="Connection failed",
        )

        assert result.status == "error"
        assert result.memories == []
        assert result.error_message == "Connection failed"


# =============================================================================
# Cypher Templates Tests
# =============================================================================


class TestCypherTemplates:
    """Tests for Cypher query templates."""

    def test_templates_use_parameters(self) -> None:
        """Verify all templates use parameterized queries (no string interpolation)."""
        templates = [
            CypherTemplates.GET_CONNECTIONS,
            CypherTemplates.GET_CONNECTIONS_BY_TYPE,
            CypherTemplates.COUNT_CONNECTIONS,
            CypherTemplates.NODE_EXISTS,
        ]

        for template in templates:
            # Templates should use $param syntax for parameters
            assert "$" in template, f"Template missing parameter: {template[:50]}..."
            # Templates should not have Python f-string or .format() placeholders
            # Note: Cypher uses {key: $value} for property matching, which is safe
            # We check for patterns like {0}, {name} (without $) that indicate
            # Python string formatting
            import re

            # Match Python format patterns but not Cypher property patterns
            unsafe_patterns = [
                r"\{[0-9]+\}",  # {0}, {1} positional args
                r"\{[a-zA-Z_][a-zA-Z0-9_]*\}",  # {name} w/o colon
            ]
            for pattern in unsafe_patterns:
                matches = re.findall(pattern, template)
                # Filter out Cypher property patterns (followed by colon)
                actual_unsafe = [m for m in matches if f"{m[1:-1]}:" not in template]
                assert not actual_unsafe, (
                    f"Template has unsafe format pattern {actual_unsafe}: "
                    f"{template[:50]}..."
                )

    def test_get_connections_template(self) -> None:
        """Test GET_CONNECTIONS template structure."""
        template = CypherTemplates.GET_CONNECTIONS
        assert "$memory_id" in template
        assert "$limit" in template
        assert "MATCH" in template
        assert "RETURN" in template

    def test_node_exists_template(self) -> None:
        """Test NODE_EXISTS template structure."""
        template = CypherTemplates.NODE_EXISTS
        assert "$memory_id" in template
        assert "LIMIT 1" in template


# =============================================================================
# find_related Tests
# =============================================================================


class TestFindRelated:
    """Tests for find_related method."""

    @pytest.mark.asyncio
    async def test_find_related_success(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test successful find_related operation."""
        # Mock the node exists check
        mock_handler.execute_query.return_value = MagicMock(
            records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]
        )

        # Mock the traverse result
        mock_node = MagicMock()
        mock_node.element_id = "4:abc:456"
        mock_node.labels = ["Memory"]
        mock_node.properties = {"memory_id": "mem_related"}

        mock_handler.traverse.return_value = MagicMock(
            nodes=[mock_node],
            relationships=[],
            paths=[["4:abc:123", "4:abc:456"]],
            depth_reached=1,
            execution_time_ms=25.0,
        )

        result = await adapter_with_mock.find_related("mem_start", depth=2)

        assert result.status == "success"
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_related"

    @pytest.mark.asyncio
    async def test_find_related_not_found(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related when start memory doesn't exist."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        result = await adapter_with_mock.find_related("nonexistent_mem")

        assert result.status == "not_found"
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_find_related_no_results(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related when no related memories exist."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[{"memory_id": "mem_isolated", "element_id": "4:abc:123"}]
        )
        mock_handler.traverse.return_value = MagicMock(
            nodes=[],
            relationships=[],
            paths=[],
            depth_reached=0,
            execution_time_ms=10.0,
        )

        result = await adapter_with_mock.find_related("mem_isolated")

        assert result.status == "no_results"
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_find_related_respects_depth_limit(
        self,
        config: AdapterGraphMemoryConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test that find_related respects max_depth configuration."""
        config.max_depth = 3
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        mock_handler.execute_query.return_value = MagicMock(
            records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]
        )
        mock_handler.traverse.return_value = MagicMock(
            nodes=[], relationships=[], paths=[], depth_reached=0, execution_time_ms=5.0
        )

        # Request depth=10, should be capped to max_depth=3
        await adapter.find_related("mem_start", depth=10)

        # Verify traverse was called with bounded depth
        mock_handler.traverse.assert_called_once()
        call_kwargs = mock_handler.traverse.call_args[1]
        assert call_kwargs["max_depth"] == 3

    @pytest.mark.asyncio
    async def test_find_related_with_relationship_filter(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related with relationship type filter."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]
        )
        mock_handler.traverse.return_value = MagicMock(
            nodes=[], relationships=[], paths=[], depth_reached=0, execution_time_ms=5.0
        )

        await adapter_with_mock.find_related(
            "mem_start",
            relationship_types=["related_to", "caused_by"],
        )

        call_kwargs = mock_handler.traverse.call_args[1]
        assert call_kwargs["relationship_types"] == ["related_to", "caused_by"]

    @pytest.mark.asyncio
    async def test_find_related_not_initialized(
        self,
        config: AdapterGraphMemoryConfig,
    ) -> None:
        """Test find_related raises error when not initialized."""
        adapter = AdapterGraphMemory(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.find_related("mem_123")


# =============================================================================
# get_connections Tests
# =============================================================================


class TestGetConnections:
    """Tests for get_connections method."""

    @pytest.mark.asyncio
    async def test_get_connections_success(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test successful get_connections operation."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "source_id": "mem_1",
                    "target_id": "mem_2",
                    "relationship_type": "related_to",
                    "weight": 0.9,
                    "is_outgoing": True,
                    "created_at": None,
                },
                {
                    "source_id": "mem_1",
                    "target_id": "mem_3",
                    "relationship_type": "caused_by",
                    "weight": 0.7,
                    "is_outgoing": True,
                    "created_at": None,
                },
            ]
        )

        result = await adapter_with_mock.get_connections("mem_1")

        assert result.status == "success"
        assert len(result.connections) == 2
        assert result.connections[0].target_id == "mem_2"
        assert result.connections[1].target_id == "mem_3"

    @pytest.mark.asyncio
    async def test_get_connections_not_found(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections when memory doesn't exist."""
        # First call returns empty (no connections)
        # Second call (node exists check) also returns empty
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[]),  # GET_CONNECTIONS returns empty
            MagicMock(records=[]),  # NODE_EXISTS returns empty
        ]

        result = await adapter_with_mock.get_connections("nonexistent")

        assert result.status == "not_found"

    @pytest.mark.asyncio
    async def test_get_connections_no_results(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections when node exists but has no connections."""
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[]),  # GET_CONNECTIONS returns empty
            MagicMock(
                records=[{"memory_id": "mem_isolated", "element_id": "4:abc:123"}]
            ),
        ]

        result = await adapter_with_mock.get_connections("mem_isolated")

        assert result.status == "no_results"
        assert result.connections == []

    @pytest.mark.asyncio
    async def test_get_connections_with_type_filter(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections with relationship type filter."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "source_id": "mem_1",
                    "target_id": "mem_2",
                    "relationship_type": "related_to",
                    "weight": 1.0,
                    "is_outgoing": True,
                    "created_at": None,
                }
            ]
        )

        result = await adapter_with_mock.get_connections(
            "mem_1",
            relationship_types=["related_to"],
        )

        assert result.status == "success"
        # Verify the filtered query was used
        call_args = mock_handler.execute_query.call_args
        assert "relationship_types" in call_args[1]["parameters"]


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for adapter initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        config: AdapterGraphMemoryConfig,
    ) -> None:
        """Test successful initialization."""
        adapter = AdapterGraphMemory(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                    auth=("neo4j", "password"),
                )

            assert adapter.is_initialized
            mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self,
        config: AdapterGraphMemoryConfig,
    ) -> None:
        """Test that initialize is idempotent."""
        adapter = AdapterGraphMemory(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                await adapter.initialize("bolt://localhost:7687")
                await adapter.initialize("bolt://localhost:7687")

            # Should only create handler once
            assert MockHandler.call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test shutdown releases resources."""
        assert adapter_with_mock.is_initialized

        await adapter_with_mock.shutdown()

        assert not adapter_with_mock.is_initialized
        mock_handler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check returns healthy status when handler is healthy."""
        mock_handler.health_check.return_value = MagicMock(healthy=True)

        result = await adapter_with_mock.health_check()

        assert result.is_healthy is True
        assert result.initialized is True
        assert result.handler_healthy is True
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check returns unhealthy status when handler is unhealthy."""
        mock_handler.health_check.return_value = MagicMock(healthy=False)

        result = await adapter_with_mock.health_check()

        assert result.is_healthy is False
        assert result.initialized is True
        assert result.handler_healthy is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(
        self,
        config: AdapterGraphMemoryConfig,
    ) -> None:
        """Test health check returns unhealthy status when not initialized."""
        adapter = AdapterGraphMemory(config)

        result = await adapter.health_check()

        assert result.is_healthy is False
        assert result.initialized is False
        assert result.handler_healthy is None
        assert result.error_message is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_find_related_handles_connection_error(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related handles connection errors gracefully."""
        from omnimemory.handlers.adapters.adapter_graph_memory import (
            InfraConnectionError,
        )

        mock_handler.execute_query.side_effect = InfraConnectionError("Connection lost")

        result = await adapter_with_mock.find_related("mem_123")

        assert result.status == "error"
        assert "failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_get_connections_handles_connection_error(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections handles connection errors gracefully."""
        from omnimemory.handlers.adapters.adapter_graph_memory import (
            InfraConnectionError,
        )

        mock_handler.execute_query.side_effect = InfraConnectionError("Query timeout")

        result = await adapter_with_mock.get_connections("mem_123")

        assert result.status == "error"
        assert "failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_handles_unexpected_exception(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test adapter handles unexpected exceptions."""
        mock_handler.execute_query.side_effect = RuntimeError("Unexpected error")

        result = await adapter_with_mock.find_related("mem_123")

        assert result.status == "error"
        assert "unexpected" in result.error_message.lower()
