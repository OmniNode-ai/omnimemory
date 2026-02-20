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

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests in this module if omnibase_infra is not installed
pytest.importorskip(
    "omnibase_infra", reason="omnibase_infra required for adapter tests"
)

from omnimemory.handlers.adapters.adapter_graph_memory import (
    AdapterGraphMemory,
    CypherTemplates,
    ModelGraphMemoryConfig,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> ModelGraphMemoryConfig:
    """Create a default adapter configuration."""
    return ModelGraphMemoryConfig(
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
    config: ModelGraphMemoryConfig,
    mock_handler: MagicMock,
) -> AdapterGraphMemory:
    """Create an adapter with a mock handler injected.

    Args:
        config: ModelGraphMemoryConfig fixture with test configuration.
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


class TestModelGraphMemoryConfig:
    """Tests for ModelGraphMemoryConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelGraphMemoryConfig()

        assert config.max_depth == 5
        assert config.default_depth == 2
        assert config.default_limit == 100
        assert config.max_limit == 1000
        assert config.bidirectional is True
        assert config.memory_node_label == "Memory"
        assert config.timeout_seconds == 30.0
        assert config.score_filter_multiplier == 3.0
        assert config.ensure_indexes is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ModelGraphMemoryConfig(
            max_depth=3,
            default_depth=1,
            default_limit=50,
            bidirectional=False,
            memory_node_label="MemoryNode",
            timeout_seconds=60.0,
            score_filter_multiplier=5.0,
            ensure_indexes=False,
        )

        assert config.max_depth == 3
        assert config.default_depth == 1
        assert config.default_limit == 50
        assert config.bidirectional is False
        assert config.memory_node_label == "MemoryNode"
        assert config.timeout_seconds == 60.0
        assert config.score_filter_multiplier == 5.0
        assert config.ensure_indexes is False

    def test_max_depth_bounds(self) -> None:
        """Test max_depth has valid bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(max_depth=0)

        # Too high
        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(max_depth=11)

    def test_timeout_must_be_positive(self) -> None:
        """Test timeout must be positive."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(timeout_seconds=0)

    def test_score_filter_multiplier_bounds(self) -> None:
        """Test score_filter_multiplier has valid bounds (1.0 to 10.0)."""
        from pydantic import ValidationError

        # Valid at bounds
        config_min = ModelGraphMemoryConfig(score_filter_multiplier=1.0)
        assert config_min.score_filter_multiplier == 1.0

        config_max = ModelGraphMemoryConfig(score_filter_multiplier=10.0)
        assert config_max.score_filter_multiplier == 10.0

        # Too low
        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(score_filter_multiplier=0.5)

        # Too high
        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(score_filter_multiplier=11.0)

    def test_default_depth_exceeds_max_depth_raises(self) -> None:
        """Test that default_depth > max_depth raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="default_depth"):
            ModelGraphMemoryConfig(max_depth=3, default_depth=5)

    def test_default_limit_exceeds_max_limit_raises(self) -> None:
        """Test that default_limit > max_limit raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="default_limit"):
            ModelGraphMemoryConfig(max_limit=500, default_limit=1000)


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
            candidates_found=5,  # 5 candidates, 2 passed min_score filter
            max_depth_reached=2,
            execution_time_ms=50.0,
        )

        assert result.status == "success"
        assert len(result.memories) == 2
        assert result.total_count == 2
        assert result.candidates_found == 5
        assert result.error_message is None

    def test_related_memory_result_candidates_vs_total(self) -> None:
        """Test that candidates_found can differ from total_count after filtering."""
        result = ModelRelatedMemoryResult(
            status="success",
            memories=[ModelRelatedMemory(memory_id="mem_1", score=0.5)],
            total_count=1,
            candidates_found=10,  # 10 found, only 1 passed min_score
        )

        assert result.total_count == 1
        assert result.candidates_found == 10
        # Difference shows 9 were filtered out
        assert result.candidates_found - result.total_count == 9

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
        # All templates now require node_label parameter
        node_label = "Memory"
        templates = [
            CypherTemplates.get_connections(node_label),
            CypherTemplates.get_connections_by_type(node_label),
            CypherTemplates.count_connections(node_label),
            CypherTemplates.node_exists(node_label),
        ]

        for template in templates:
            # Templates should use $param syntax for parameters
            assert "$" in template, f"Template missing parameter: {template[:50]}..."
            # Templates should not have Python f-string or .format() placeholders
            # Note: Cypher uses {key: $value} for property matching, which is safe
            # We check for patterns like {0}, {name} (without $) that indicate
            # Python string formatting

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
        """Test get_connections template structure."""
        template = CypherTemplates.get_connections("Memory")
        assert "$memory_id" in template
        assert "$limit" in template
        assert "MATCH" in template
        assert "RETURN" in template

    def test_node_exists_template(self) -> None:
        """Test node_exists template structure."""
        template = CypherTemplates.node_exists("Memory")
        assert "$memory_id" in template
        assert "LIMIT 1" in template

    def test_bidirectional_templates_use_undirected_pattern(self) -> None:
        """Verify bidirectional templates use -[r]- undirected pattern."""
        node_label = "Memory"
        bidirectional_templates = [
            CypherTemplates.get_connections(node_label),
            CypherTemplates.get_connections_by_type(node_label),
        ]

        for template in bidirectional_templates:
            normalized = template.replace(" ", "").replace("\n", "")
            # Should have undirected pattern -[r]- (not -[r]->)
            assert (
                "-[r]-" in normalized
            ), f"Bidirectional template missing -[r]- pattern: {template[:80]}..."
            # Should NOT have directed outgoing pattern
            assert (
                "-[r]->" not in normalized
            ), f"Bidirectional template has directed pattern -[r]->: {template[:80]}..."

    def test_outgoing_templates_use_directed_pattern(self) -> None:
        """Verify outgoing templates use -[r]-> directed pattern."""
        node_label = "Memory"
        outgoing_templates = [
            CypherTemplates.get_connections_outgoing(node_label),
            CypherTemplates.get_connections_by_type_outgoing(node_label),
        ]

        for template in outgoing_templates:
            normalized = template.replace(" ", "").replace("\n", "")
            # Should have directed outgoing pattern -[r]->
            assert (
                "-[r]->" in normalized
            ), f"Outgoing template missing -[r]-> pattern: {template[:80]}..."

    def test_outgoing_templates_have_required_parameters(self) -> None:
        """Verify outgoing templates have required parameters."""
        node_label = "Memory"
        # get_connections_outgoing should have memory_id and limit
        template = CypherTemplates.get_connections_outgoing(node_label)
        assert "$memory_id" in template
        assert "$limit" in template

        # get_connections_by_type_outgoing should also have relationship_types
        template_by_type = CypherTemplates.get_connections_by_type_outgoing(node_label)
        assert "$memory_id" in template_by_type
        assert "$limit" in template_by_type
        assert "$relationship_types" in template_by_type

    def test_create_memory_index_template(self) -> None:
        """Test create_memory_index generates correct Cypher query."""
        # Default label
        query = CypherTemplates.create_memory_index("Memory")
        assert query == "CREATE INDEX ON :Memory(memory_id);"

        # Custom label
        query_custom = CypherTemplates.create_memory_index("CustomNode")
        assert query_custom == "CREATE INDEX ON :CustomNode(memory_id);"

    def test_templates_use_configured_node_label(self) -> None:
        """Verify templates correctly use the configured node_label."""
        custom_label = "CustomMemory"

        # Test all templates use the custom label
        templates_and_checks = [
            (CypherTemplates.get_connections(custom_label), ":CustomMemory"),
            (CypherTemplates.get_connections_by_type(custom_label), ":CustomMemory"),
            (CypherTemplates.get_connections_outgoing(custom_label), ":CustomMemory"),
            (
                CypherTemplates.get_connections_by_type_outgoing(custom_label),
                ":CustomMemory",
            ),
            (CypherTemplates.count_connections(custom_label), ":CustomMemory"),
            (CypherTemplates.node_exists(custom_label), ":CustomMemory"),
            (
                CypherTemplates.find_related_query(3, custom_label),
                ":CustomMemory",
            ),
            (
                CypherTemplates.find_related_by_type_query(3, custom_label),
                ":CustomMemory",
            ),
        ]

        for template, expected_label in templates_and_checks:
            assert expected_label in template, (
                f"Template should contain {expected_label}, "
                f"got: {template[:100]}..."
            )
            # Ensure default "Memory" label is NOT present
            assert ":Memory " not in template, (
                f"Template should NOT contain ':Memory ' when custom label is used, "
                f"got: {template[:100]}..."
            )
            assert ":Memory{" not in template, (
                f"Template should NOT contain ':Memory{{' when custom label is used, "
                f"got: {template[:100]}..."
            )


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
        # Mock execute_query with side_effect for multiple calls:
        # 1. NODE_EXISTS check
        # 2. FIND_RELATED query
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]),
            MagicMock(
                records=[
                    {
                        "memory_id": "mem_related",
                        "labels": ["Memory"],
                        "properties": {"memory_id": "mem_related", "content": "test"},
                        "depth": 1,
                    }
                ]
            ),
        ]

        result = await adapter_with_mock.find_related("mem_start", depth=2)

        assert result.status == "success"
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_related"
        assert result.memories[0].depth == 1
        assert result.memories[0].score == 0.5  # 1/(1+1)

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
        # Mock execute_query with side_effect for multiple calls:
        # 1. NODE_EXISTS check (node exists)
        # 2. FIND_RELATED query (no results)
        mock_handler.execute_query.side_effect = [
            MagicMock(
                records=[{"memory_id": "mem_isolated", "element_id": "4:abc:123"}]
            ),
            MagicMock(records=[]),  # No related memories
        ]

        result = await adapter_with_mock.find_related("mem_isolated")

        assert result.status == "no_results"
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_find_related_respects_depth_limit(
        self,
        config: ModelGraphMemoryConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test that find_related respects max_depth configuration."""
        config = config.model_copy(update={"max_depth": 3})
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        # Mock execute_query with side_effect for multiple calls:
        # 1. NODE_EXISTS check
        # 2. FIND_RELATED query
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]),
            MagicMock(records=[]),  # No related memories
        ]

        # Request depth=10, should be capped to max_depth=3
        await adapter.find_related("mem_start", depth=10)

        # Verify execute_query was called twice (node exists + find_related)
        assert mock_handler.execute_query.call_count == 2

        # Check second call (FIND_RELATED) has bounded max_depth embedded in query
        # Note: max_depth embedded in query string (not param) for Memgraph compat
        find_related_call = mock_handler.execute_query.call_args_list[1]
        query = find_related_call[1]["query"]
        # The query should contain "[r*1..3]" (bounded depth of 3)
        assert "*1..3]" in query, f"Expected depth 3 in query, got: {query}"

    @pytest.mark.asyncio
    async def test_find_related_with_relationship_filter(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related with relationship type filter."""
        # Mock execute_query with side_effect for multiple calls:
        # 1. NODE_EXISTS check
        # 2. FIND_RELATED_BY_TYPE query
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]),
            MagicMock(records=[]),  # No related memories
        ]

        await adapter_with_mock.find_related(
            "mem_start",
            relationship_types=["related_to", "caused_by"],
        )

        # Verify execute_query was called twice (node exists + find_related_by_type)
        assert mock_handler.execute_query.call_count == 2

        # Check the second call has the relationship_types parameter
        find_related_call = mock_handler.execute_query.call_args_list[1]
        parameters = find_related_call[1]["parameters"]
        assert parameters["relationship_types"] == ["related_to", "caused_by"]

    @pytest.mark.asyncio
    async def test_find_related_uses_score_filter_multiplier(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that find_related uses the configurable score_filter_multiplier.

        The multiplier determines how many extra candidates are fetched
        to account for min_score filtering. This test verifies:
        1. Default multiplier (3.0) results in query_limit = limit * 3
        2. Custom multiplier changes the query_limit accordingly
        """
        # Create config with custom multiplier
        config = ModelGraphMemoryConfig(
            default_limit=100,
            max_limit=1000,
            score_filter_multiplier=5.0,  # Custom multiplier
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        # Mock execute_query with side_effect for multiple calls
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]),
            MagicMock(records=[]),  # No related memories
        ]

        # Call find_related with default limit (100)
        await adapter.find_related("mem_start", limit=100)

        # Verify execute_query was called twice
        assert mock_handler.execute_query.call_count == 2

        # Check second call (FIND_RELATED) has query_limit = 100 * 5.0 = 500
        find_related_call = mock_handler.execute_query.call_args_list[1]
        parameters = find_related_call[1]["parameters"]
        assert parameters["limit"] == 500  # 100 * 5.0 multiplier

    @pytest.mark.asyncio
    async def test_find_related_multiplier_respects_max_limit(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that query_limit is capped by max_limit even with high multiplier."""
        config = ModelGraphMemoryConfig(
            default_limit=100,
            max_limit=200,  # Low max_limit
            score_filter_multiplier=5.0,  # Would give 500, but capped at 200
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        mock_handler.execute_query.side_effect = [
            MagicMock(records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]),
            MagicMock(records=[]),
        ]

        await adapter.find_related("mem_start", limit=100)

        # query_limit should be min(100 * 5.0, 200) = 200 (capped by max_limit)
        find_related_call = mock_handler.execute_query.call_args_list[1]
        parameters = find_related_call[1]["parameters"]
        assert parameters["limit"] == 200

    @pytest.mark.asyncio
    async def test_find_related_tracks_candidates_found(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test that find_related tracks candidates_found before min_score filtering.

        This verifies the candidates_found field shows how many results were found
        before min_score filtering was applied, helping users understand data loss.
        """
        # Mock 5 results at various depths (1-5), with scores ranging from 0.5 to 0.17
        mock_handler.execute_query.side_effect = [
            MagicMock(records=[{"memory_id": "mem_start", "element_id": "4:abc:123"}]),
            MagicMock(
                records=[
                    {"memory_id": "mem_1", "labels": [], "properties": {}, "depth": 1},
                    {"memory_id": "mem_2", "labels": [], "properties": {}, "depth": 2},
                    {"memory_id": "mem_3", "labels": [], "properties": {}, "depth": 3},
                    {"memory_id": "mem_4", "labels": [], "properties": {}, "depth": 4},
                    {"memory_id": "mem_5", "labels": [], "properties": {}, "depth": 5},
                ]
            ),
        ]

        # Use min_score=0.3, which filters out depth >= 3 (score < 0.3)
        # depth=1: score=0.5, depth=2: score=0.33, depth=3: score=0.25, etc.
        result = await adapter_with_mock.find_related("mem_start", min_score=0.3)

        assert result.status == "success"
        # Only depth 1 and 2 pass the min_score filter
        assert result.total_count == 2
        # But we found 5 candidates total before filtering
        assert result.candidates_found == 5
        # Verify the returned memories are the high-scoring ones
        assert all(m.score >= 0.3 for m in result.memories)

    @pytest.mark.asyncio
    async def test_find_related_not_initialized(
        self,
        config: ModelGraphMemoryConfig,
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

    @pytest.mark.asyncio
    async def test_get_connections_bidirectional_with_type_filter(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections uses bidirectional template with type filter."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "source_id": "mem_1",
                    "target_id": "mem_2",
                    "relationship_type": "related_to",
                    "weight": 1.0,
                    "is_outgoing": True,
                    "created_at": None,
                },
                {
                    "source_id": "mem_1",
                    "target_id": "mem_3",
                    "relationship_type": "related_to",
                    "weight": 0.8,
                    "is_outgoing": False,
                    "created_at": None,
                },
            ]
        )

        result = await adapter_with_mock.get_connections(
            "mem_1",
            relationship_types=["related_to"],
            bidirectional=True,
        )

        assert result.status == "success"
        assert len(result.connections) == 2
        # Verify the bidirectional template was used (contains -[r]- not -[r]->)
        call_args = mock_handler.execute_query.call_args
        query = call_args[1]["query"]
        assert "-[r]-" in query.replace(" ", "").replace("\n", "")
        assert "-[r]->" not in query.replace(" ", "").replace("\n", "")

    @pytest.mark.asyncio
    async def test_get_connections_outgoing_only(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections with bidirectional=False uses outgoing template."""
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
            ]
        )

        result = await adapter_with_mock.get_connections(
            "mem_1",
            bidirectional=False,
        )

        assert result.status == "success"
        assert len(result.connections) == 1
        # Verify the outgoing template was used (contains -[r]-> not -[r]-)
        call_args = mock_handler.execute_query.call_args
        query = call_args[1]["query"]
        normalized_query = query.replace(" ", "").replace("\n", "")
        assert (
            "-[r]->" in normalized_query
        ), f"Expected outgoing pattern -[r]-> in query: {query}"

    @pytest.mark.asyncio
    async def test_get_connections_outgoing_with_type_filter(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections with bidirectional=False and type filter."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "source_id": "mem_1",
                    "target_id": "mem_2",
                    "relationship_type": "related_to",
                    "weight": 1.0,
                    "is_outgoing": True,
                    "created_at": None,
                },
            ]
        )

        result = await adapter_with_mock.get_connections(
            "mem_1",
            relationship_types=["related_to"],
            bidirectional=False,
        )

        assert result.status == "success"
        # Verify the outgoing template with type filter was used
        call_args = mock_handler.execute_query.call_args
        query = call_args[1]["query"]
        normalized_query = query.replace(" ", "").replace("\n", "")
        assert (
            "-[r]->" in normalized_query
        ), f"Expected outgoing pattern -[r]-> in query: {query}"
        # Verify type filter parameters are passed
        assert "relationship_types" in call_args[1]["parameters"]

    @pytest.mark.asyncio
    async def test_get_connections_uses_config_bidirectional_default(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections uses config.bidirectional when param not specified."""
        # Create config with bidirectional=False
        config = ModelGraphMemoryConfig(
            max_depth=5,
            default_depth=2,
            default_limit=100,
            max_limit=1000,
            bidirectional=False,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

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
            ]
        )

        # Call get_connections WITHOUT bidirectional param (should use config default)
        result = await adapter.get_connections("mem_1")

        assert result.status == "success"
        # Verify the outgoing template was used (respecting config.bidirectional=False)
        call_args = mock_handler.execute_query.call_args
        query = call_args[1]["query"]
        normalized_query = query.replace(" ", "").replace("\n", "")
        assert (
            "-[r]->" in normalized_query
        ), f"Expected outgoing pattern -[r]-> when config.bidirectional=False: {query}"

    @pytest.mark.asyncio
    async def test_get_connections_param_overrides_config_default(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections bidirectional param overrides config default."""
        # Create config with bidirectional=False
        config = ModelGraphMemoryConfig(
            max_depth=5,
            default_depth=2,
            default_limit=100,
            max_limit=1000,
            bidirectional=False,  # Config says outgoing only
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

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
            ]
        )

        # Call with explicit bidirectional=True (should override config)
        result = await adapter.get_connections("mem_1", bidirectional=True)

        assert result.status == "success"
        # Verify the bidirectional template was used (param override)
        call_args = mock_handler.execute_query.call_args
        query = call_args[1]["query"]
        normalized_query = query.replace(" ", "").replace("\n", "")
        # Bidirectional uses -[r]- without arrow
        assert (
            "-[r]-(" in normalized_query or "-[r]-(n" in normalized_query
        ), f"Expected bidirectional pattern -[r]- when param=True: {query}"

    @pytest.mark.asyncio
    async def test_get_connections_weight_zero_preserved(
        self,
        adapter_with_mock: AdapterGraphMemory,
        mock_handler: MagicMock,
    ) -> None:
        """Test that weight=0.0 is preserved, not defaulted to 1.0.

        This verifies the walrus operator correctly handles the edge case where
        weight=0.0 (falsy but valid) vs weight=None (should default to 1.0).
        """
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "source_id": "mem_1",
                    "target_id": "mem_2",
                    "relationship_type": "weak_link",
                    "weight": 0.0,  # Explicit zero weight - should be preserved
                    "is_outgoing": True,
                    "created_at": None,
                },
                {
                    "source_id": "mem_1",
                    "target_id": "mem_3",
                    "relationship_type": "related",
                    "weight": None,  # None should default to 1.0
                    "is_outgoing": True,
                    "created_at": None,
                },
            ]
        )

        result = await adapter_with_mock.get_connections("mem_1")

        assert result.status == "success"
        assert len(result.connections) == 2
        # weight=0.0 should be preserved (not converted to 1.0)
        assert result.connections[0].weight == 0.0
        assert result.connections[0].relationship_type == "weak_link"
        # weight=None should default to 1.0
        assert result.connections[1].weight == 1.0
        assert result.connections[1].relationship_type == "related"


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for adapter initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        config: ModelGraphMemoryConfig,
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
        config: ModelGraphMemoryConfig,
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
        config: ModelGraphMemoryConfig,
    ) -> None:
        """Test health check returns unhealthy status when not initialized."""
        adapter = AdapterGraphMemory(config)

        result = await adapter.health_check()

        assert result.is_healthy is False
        assert result.initialized is False
        assert result.handler_healthy is None
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_initialize_timeout_on_lock_acquisition(self) -> None:
        """Test that initialization times out if lock cannot be acquired.

        This verifies the timeout handling for the initialization lock, which
        prevents indefinite hangs if another initialization is in progress or
        the database is unresponsive. The asyncio.timeout now wraps both lock
        acquisition and initialization work.
        """
        # Use a short timeout (0.5s is more reliable than 0.1s in CI)
        config = ModelGraphMemoryConfig(timeout_seconds=0.5)
        adapter = AdapterGraphMemory(config)

        # Acquire the lock to simulate another initialization in progress
        async with adapter._init_lock:
            # Try to initialize while lock is held - should timeout
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.initialize(connection_uri="bolt://localhost:7687")

            # Verify the error message mentions timeout
            assert "timed out" in str(exc_info.value).lower()
            assert "0.5" in str(exc_info.value)

        # Adapter should not be initialized after timeout
        assert not adapter.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_creates_index_by_default(
        self,
        config: ModelGraphMemoryConfig,
    ) -> None:
        """Test that index is created during initialization by default."""
        adapter = AdapterGraphMemory(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                    auth=("neo4j", "password"),
                )

            assert adapter.is_initialized
            # Verify execute_query was called with index creation query
            mock_instance.execute_query.assert_called_once()
            call_args = mock_instance.execute_query.call_args
            query = call_args[1]["query"]
            assert "CREATE INDEX ON :Memory(memory_id)" in query

    @pytest.mark.asyncio
    async def test_initialize_skips_index_when_disabled(self) -> None:
        """Test that index creation is skipped when ensure_indexes=False."""
        config = ModelGraphMemoryConfig(ensure_indexes=False)
        adapter = AdapterGraphMemory(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            assert adapter.is_initialized
            # Verify execute_query was NOT called (no index creation)
            mock_instance.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_handles_index_already_exists(
        self,
        config: ModelGraphMemoryConfig,
    ) -> None:
        """Test that index creation error (e.g., already exists) doesn't fail init."""
        adapter = AdapterGraphMemory(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            # Simulate index already exists error
            mock_instance.execute_query = AsyncMock(
                side_effect=Exception("Index already exists")
            )
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                # Should NOT raise - index errors are caught and logged
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            # Adapter should still be initialized despite index error
            assert adapter.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_uses_custom_node_label_for_index(self) -> None:
        """Test that index creation uses the configured memory_node_label."""
        config = ModelGraphMemoryConfig(memory_node_label="CustomMemory")
        adapter = AdapterGraphMemory(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            # Verify the custom label was used in index creation
            call_args = mock_instance.execute_query.call_args
            query = call_args[1]["query"]
            assert "CREATE INDEX ON :CustomMemory(memory_id)" in query

    @pytest.mark.asyncio
    async def test_initialize_validates_uri_format(
        self,
        config: ModelGraphMemoryConfig,
    ) -> None:
        """Test that initialize validates the connection_uri format."""
        adapter = AdapterGraphMemory(config)

        # Missing scheme and hostname
        with pytest.raises(ValueError, match="Invalid connection_uri"):
            await adapter.initialize(connection_uri="invalid-uri")

        # Missing hostname
        with pytest.raises(ValueError, match="Invalid connection_uri"):
            await adapter.initialize(connection_uri="bolt://")

        # Empty string
        with pytest.raises(ValueError, match="Invalid connection_uri"):
            await adapter.initialize(connection_uri="")

    @pytest.mark.asyncio
    async def test_initialize_accepts_valid_uri_schemes(
        self,
        config: ModelGraphMemoryConfig,
    ) -> None:
        """Test that initialize accepts all valid bolt URI schemes."""
        adapter = AdapterGraphMemory(config)

        valid_uris = [
            "bolt://localhost:7687",
            "bolt+s://localhost:7687",
            "bolt+ssc://localhost:7687",
            "neo4j://localhost:7687",
            "neo4j+s://localhost:7687",
        ]

        with patch(
            "omnimemory.handlers.adapters.adapter_graph_memory.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.ModelONEXContainer"
            ):
                for uri in valid_uris:
                    # Reset adapter state for each URI
                    adapter._initialized = False
                    adapter._handler = None

                    # Should not raise ValueError for valid schemes
                    await adapter.initialize(connection_uri=uri)
                    assert adapter.is_initialized


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


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for exponential backoff retry behavior on transient errors."""

    # -------------------------------------------------------------------------
    # Configuration tests
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_retry_config_defaults(self) -> None:
        """Test default retry configuration values."""
        config = ModelGraphMemoryConfig()

        assert config.max_retries == 3
        assert config.retry_base_delay_seconds == 1.0
        assert config.retry_max_delay_seconds == 30.0

    @pytest.mark.unit
    def test_retry_config_custom(self) -> None:
        """Test custom retry configuration is accepted."""
        config = ModelGraphMemoryConfig(
            max_retries=5,
            retry_base_delay_seconds=0.5,
            retry_max_delay_seconds=60.0,
        )

        assert config.max_retries == 5
        assert config.retry_base_delay_seconds == 0.5
        assert config.retry_max_delay_seconds == 60.0

    @pytest.mark.unit
    def test_retry_config_zero_max_retries(self) -> None:
        """Test that max_retries=0 is valid (disables retries)."""
        config = ModelGraphMemoryConfig(max_retries=0)
        assert config.max_retries == 0

    @pytest.mark.unit
    def test_retry_config_max_retries_upper_bound(self) -> None:
        """Test that max_retries is bounded at 10."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(max_retries=11)

    @pytest.mark.unit
    def test_retry_config_base_delay_must_be_positive(self) -> None:
        """Test that retry_base_delay_seconds must be > 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(retry_base_delay_seconds=0.0)

    @pytest.mark.unit
    def test_retry_config_max_delay_must_be_positive(self) -> None:
        """Test that retry_max_delay_seconds must be > 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelGraphMemoryConfig(retry_max_delay_seconds=0.0)

    # -------------------------------------------------------------------------
    # _execute_with_retry unit tests
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_succeeds_on_first_attempt(
        self,
        adapter_with_mock: AdapterGraphMemory,
    ) -> None:
        """Test _execute_with_retry returns immediately on first success."""
        call_count = 0

        async def succeeding_op() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await adapter_with_mock._execute_with_retry("test_op", succeeding_op)

        assert result == "ok"
        assert call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_on_transient_error_then_succeeds(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that transient errors are retried and eventual success is returned."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=3,
            retry_base_delay_seconds=0.01,  # Fast retries for test
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def flaky_op() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise InfraConnectionError("Transient connection failure")
            return "success_after_retry"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._execute_with_retry("test_op", flaky_op)

        assert result == "success_after_retry"
        assert call_count == 3  # Failed twice, succeeded on third attempt

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausts_all_retries(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that after max_retries exhaustion, the last error is re-raised."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def always_failing_op() -> str:
            nonlocal call_count
            call_count += 1
            raise InfraConnectionError("Persistent connection failure")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(
                InfraConnectionError, match="Persistent connection failure"
            ):
                await adapter._execute_with_retry("test_op", always_failing_op)

        # Should have been called max_retries + 1 times (original + retries)
        assert call_count == 3  # 1 original + 2 retries

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_no_retry_on_zero_max_retries(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that max_retries=0 disables retry (fails immediately)."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(max_retries=0)
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def always_failing_op() -> str:
            nonlocal call_count
            call_count += 1
            raise InfraConnectionError("Connection refused")

        with pytest.raises(InfraConnectionError):
            await adapter._execute_with_retry("test_op", always_failing_op)

        assert call_count == 1  # Only the original attempt, no retries

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_does_not_retry_permanent_errors(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that non-transient errors propagate immediately without retrying."""
        config = ModelGraphMemoryConfig(
            max_retries=3,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def permanent_error_op() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid Cypher query syntax")

        with pytest.raises(ValueError, match="Invalid Cypher query syntax"):
            await adapter._execute_with_retry("test_op", permanent_error_op)

        # Permanent error must not retry - only 1 call
        assert call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_infra_timeout_error(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that InfraTimeoutError is treated as transient and retried."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraTimeoutError, ModelTimeoutErrorContext

        config = ModelGraphMemoryConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0
        timeout_context = ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.GRAPH,
            operation="execute_query",
            timeout_seconds=5.0,
        )

        async def timeout_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise InfraTimeoutError("Query timed out", context=timeout_context)
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._execute_with_retry("test_op", timeout_then_success)

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_infra_unavailable_error(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that InfraUnavailableError is treated as transient and retried."""
        from omnibase_infra.errors import InfraUnavailableError

        config = ModelGraphMemoryConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def unavailable_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise InfraUnavailableError("Service temporarily unavailable")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._execute_with_retry(
                "test_op", unavailable_then_success
            )

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_uses_exponential_backoff(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that sleep durations increase exponentially between retries."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=3,
            retry_base_delay_seconds=1.0,
            retry_max_delay_seconds=100.0,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        async def always_failing_op() -> str:
            raise InfraConnectionError("Connection refused")

        sleep_calls: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(InfraConnectionError):
                await adapter._execute_with_retry("test_op", always_failing_op)

        # Should have 3 sleep calls (for 3 retries, not the final failure)
        assert len(sleep_calls) == 3

        # Each delay should be >= base * 2^attempt (before jitter is added)
        # We verify that later delays are >= earlier delays (exponential growth trend)
        # With jitter, exact values vary but the trend must hold:
        # Attempt 0: base * 2^0 + jitter ~ 1.0 to 2.0
        # Attempt 1: base * 2^1 + jitter ~ 2.0 to 3.0
        # Attempt 2: base * 2^2 + jitter ~ 4.0 to 5.0
        assert sleep_calls[0] >= 1.0, f"First retry delay too short: {sleep_calls[0]}"
        assert sleep_calls[1] >= 2.0, f"Second retry delay too short: {sleep_calls[1]}"
        assert sleep_calls[2] >= 4.0, f"Third retry delay too short: {sleep_calls[2]}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retry_respects_max_delay_cap(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that retry delay is capped at retry_max_delay_seconds."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=5,
            retry_base_delay_seconds=10.0,
            retry_max_delay_seconds=15.0,  # Low cap
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        async def always_failing_op() -> str:
            raise InfraConnectionError("Connection refused")

        sleep_calls: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(InfraConnectionError):
                await adapter._execute_with_retry("test_op", always_failing_op)

        # All delays must be <= max_delay_seconds
        for delay in sleep_calls:
            assert delay <= 15.0, f"Delay {delay} exceeds max_delay_seconds=15.0"

    # -------------------------------------------------------------------------
    # Integration with find_related and get_connections
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_related_retries_on_transient_error(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related retries transient errors and returns success result."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def flaky_execute_query(**kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise InfraConnectionError("Transient failure")
            # Return successful results on 3rd attempt
            if call_count == 3:
                # node_exists call
                return MagicMock(
                    records=[{"memory_id": "mem_start", "element_id": "1"}]
                )
            # find_related call
            return MagicMock(records=[])

        mock_handler.execute_query = AsyncMock(side_effect=flaky_execute_query)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.find_related("mem_start")

        # Should succeed after retries
        assert result.status in ("success", "no_results", "not_found")
        # Should have retried (at least 3 calls since first 2 raise)
        assert call_count >= 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_related_returns_error_after_all_retries_exhausted(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test find_related returns error status when all retries are exhausted."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=1,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        mock_handler.execute_query = AsyncMock(
            side_effect=InfraConnectionError("Persistent failure")
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.find_related("mem_123")

        assert result.status == "error"
        assert "failed" in result.error_message.lower()
        # Should have retried: 1 original + 1 retry = 2 total calls
        assert mock_handler.execute_query.call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_connections_retries_on_transient_error(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections retries transient errors and returns success result."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        call_count = 0

        async def flaky_execute_query(**kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise InfraConnectionError("Transient failure")
            return MagicMock(
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

        mock_handler.execute_query = AsyncMock(side_effect=flaky_execute_query)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.get_connections("mem_1")

        assert result.status == "success"
        assert len(result.connections) == 1
        assert call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_connections_returns_error_after_all_retries_exhausted(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_connections returns error status when all retries are exhausted."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=1,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        mock_handler.execute_query = AsyncMock(
            side_effect=InfraConnectionError("Persistent failure")
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.get_connections("mem_123")

        assert result.status == "error"
        assert "failed" in result.error_message.lower()
        # Should have retried: 1 original + 1 retry = 2 total calls
        assert mock_handler.execute_query.call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_logs_warning_on_each_attempt(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test that retry attempts are logged at WARNING level for observability."""
        from omnibase_infra.errors import InfraConnectionError

        config = ModelGraphMemoryConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.1,
        )
        adapter = AdapterGraphMemory(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        async def always_failing_op() -> str:
            raise InfraConnectionError("Connection refused")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.logger"
            ) as mock_logger:
                with pytest.raises(InfraConnectionError):
                    await adapter._execute_with_retry("test_op", always_failing_op)

                # Should have warning calls for retry attempts and final failure
                assert mock_logger.warning.call_count >= 1
