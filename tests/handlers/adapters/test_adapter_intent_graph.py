# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for AdapterIntentGraph.

This module tests the intent graph adapter that wraps HandlerGraph
for intent classification storage and retrieval operations.

Test Categories:
    - Configuration: Config validation and defaults
    - Models: Pydantic model validation
    - Cypher Templates: Query generation and parameter handling
    - Lifecycle: Initialize and shutdown
    - store_intent: Intent storage operations
    - get_session_intents: Intent query operations
    - get_intent_distribution: Analytics queries
    - health_check: Health monitoring
    - Error Handling: Failure scenarios

Usage:
    pytest tests/handlers/adapters/test_adapter_intent_graph.py -v
    pytest tests/handlers/adapters/ -v -k "intent"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1457.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

# Test UUIDs for consistent test data
TEST_INTENT_ID_1 = UUID("11111111-1111-1111-1111-111111111111")
TEST_INTENT_ID_2 = UUID("22222222-2222-2222-2222-222222222222")
TEST_INTENT_ID_EXISTING = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
TEST_CORRELATION_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
TEST_CORRELATION_ID_2 = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

# Test datetimes for consistent test data
TEST_CREATED_AT_1 = datetime(2025, 1, 22, 10, 30, 0, tzinfo=UTC)
TEST_CREATED_AT_2 = datetime(2025, 1, 22, 10, 25, 0, tzinfo=UTC)

# Skip all tests in this module if omnibase_infra is not installed
pytest.importorskip(
    "omnibase_infra", reason="omnibase_infra required for adapter tests"
)

from omnibase_core.enums.intelligence import EnumIntentCategory

from omnimemory.handlers.adapters import (
    AdapterIntentGraph,
    IntentCypherTemplates,
    ModelAdapterIntentGraphConfig,
    ModelIntentClassificationOutput,
    ModelIntentGraphHealth,
    ModelIntentQueryResult,
    ModelIntentRecord,
    ModelIntentStorageResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> ModelAdapterIntentGraphConfig:
    """Create a default adapter configuration."""
    return ModelAdapterIntentGraphConfig(
        timeout_seconds=30.0,
        session_node_label="Session",
        intent_node_label="Intent",
        relationship_type="HAD_INTENT",
        max_intents_per_session=100,
        default_confidence_threshold=0.0,
    )


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create a mock HandlerGraph.

    Returns:
        MagicMock configured with async methods matching HandlerGraph interface:
            - initialize: AsyncMock for handler initialization
            - shutdown: AsyncMock for handler shutdown
            - execute_query: AsyncMock for Cypher query execution
            - health_check: AsyncMock for health status checks
    """
    handler: MagicMock = MagicMock()
    handler.initialize = AsyncMock()
    handler.shutdown = AsyncMock()
    handler.execute_query = AsyncMock()
    handler.health_check = AsyncMock()
    return handler


@pytest.fixture
def adapter_with_mock(
    config: ModelAdapterIntentGraphConfig,
    mock_handler: MagicMock,
) -> AdapterIntentGraph:
    """Create an adapter with a mock handler injected.

    Args:
        config: ModelAdapterIntentGraphConfig fixture with test configuration.
        mock_handler: MagicMock fixture configured as HandlerGraph.

    Returns:
        AdapterIntentGraph instance with mock handler injected and
        initialization state set to True for immediate use in tests.
    """
    adapter: AdapterIntentGraph = AdapterIntentGraph(config)
    adapter._handler = mock_handler
    adapter._initialized = True
    return adapter


@pytest.fixture
def sample_intent_classification() -> ModelIntentClassificationOutput:
    """Create a sample intent classification for testing."""
    return ModelIntentClassificationOutput(
        intent_category=EnumIntentCategory.DEBUGGING,
        confidence=0.92,
        keywords=["error", "traceback", "fix"],
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestModelAdapterIntentGraphConfig:
    """Tests for ModelAdapterIntentGraphConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelAdapterIntentGraphConfig()

        assert config.timeout_seconds == 30.0
        assert config.session_node_label == "Session"
        assert config.intent_node_label == "Intent"
        assert config.relationship_type == "HAD_INTENT"
        assert config.max_intents_per_session == 100
        assert config.default_confidence_threshold == 0.0
        assert config.auto_create_indexes is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ModelAdapterIntentGraphConfig(
            timeout_seconds=60.0,
            session_node_label="CustomSession",
            intent_node_label="CustomIntent",
            relationship_type="EXPRESSED_INTENT",
            max_intents_per_session=50,
            default_confidence_threshold=0.5,
        )

        assert config.timeout_seconds == 60.0
        assert config.session_node_label == "CustomSession"
        assert config.intent_node_label == "CustomIntent"
        assert config.relationship_type == "EXPRESSED_INTENT"
        assert config.max_intents_per_session == 50
        assert config.default_confidence_threshold == 0.5

    def test_timeout_seconds_bounds(self) -> None:
        """Test timeout_seconds has valid bounds (0.1 to 300.0)."""
        from pydantic import ValidationError

        # Valid at bounds
        config_min = ModelAdapterIntentGraphConfig(timeout_seconds=0.1)
        assert config_min.timeout_seconds == 0.1

        config_max = ModelAdapterIntentGraphConfig(timeout_seconds=300.0)
        assert config_max.timeout_seconds == 300.0

        # Too low
        with pytest.raises(ValidationError):
            ModelAdapterIntentGraphConfig(timeout_seconds=0.05)

        # Too high
        with pytest.raises(ValidationError):
            ModelAdapterIntentGraphConfig(timeout_seconds=400.0)

    def test_max_intents_per_session_bounds(self) -> None:
        """Test max_intents_per_session has valid bounds (1 to 1000)."""
        from pydantic import ValidationError

        # Valid at bounds
        config_min = ModelAdapterIntentGraphConfig(max_intents_per_session=1)
        assert config_min.max_intents_per_session == 1

        config_max = ModelAdapterIntentGraphConfig(max_intents_per_session=1000)
        assert config_max.max_intents_per_session == 1000

        # Too low
        with pytest.raises(ValidationError):
            ModelAdapterIntentGraphConfig(max_intents_per_session=0)

        # Too high
        with pytest.raises(ValidationError):
            ModelAdapterIntentGraphConfig(max_intents_per_session=1001)

    def test_default_confidence_threshold_bounds(self) -> None:
        """Test default_confidence_threshold has valid bounds (0.0 to 1.0)."""
        from pydantic import ValidationError

        # Valid at bounds
        config_min = ModelAdapterIntentGraphConfig(default_confidence_threshold=0.0)
        assert config_min.default_confidence_threshold == 0.0

        config_max = ModelAdapterIntentGraphConfig(default_confidence_threshold=1.0)
        assert config_max.default_confidence_threshold == 1.0

        # Too low
        with pytest.raises(ValidationError):
            ModelAdapterIntentGraphConfig(default_confidence_threshold=-0.1)

        # Too high
        with pytest.raises(ValidationError):
            ModelAdapterIntentGraphConfig(default_confidence_threshold=1.1)

    def test_session_node_label_validation(self) -> None:
        """Test session_node_label must be a valid Cypher identifier."""
        from pydantic import ValidationError

        # Valid labels
        config_underscore = ModelAdapterIntentGraphConfig(session_node_label="_Session")
        assert config_underscore.session_node_label == "_Session"

        config_alphanum = ModelAdapterIntentGraphConfig(session_node_label="Session123")
        assert config_alphanum.session_node_label == "Session123"

        # Invalid labels
        with pytest.raises(ValidationError, match="valid Cypher identifier"):
            ModelAdapterIntentGraphConfig(session_node_label="123Invalid")

        with pytest.raises(ValidationError, match="valid Cypher identifier"):
            ModelAdapterIntentGraphConfig(session_node_label="has-dash")

        with pytest.raises(ValidationError, match="valid Cypher identifier"):
            ModelAdapterIntentGraphConfig(session_node_label="has space")

    def test_intent_node_label_validation(self) -> None:
        """Test intent_node_label must be a valid Cypher identifier."""
        from pydantic import ValidationError

        # Valid labels
        config_underscore = ModelAdapterIntentGraphConfig(intent_node_label="_Intent")
        assert config_underscore.intent_node_label == "_Intent"

        # Invalid labels
        with pytest.raises(ValidationError, match="valid Cypher identifier"):
            ModelAdapterIntentGraphConfig(intent_node_label="has.dot")

    def test_relationship_type_validation(self) -> None:
        """Test relationship_type must be a valid Cypher identifier."""
        from pydantic import ValidationError

        # Valid relationship types
        config_valid = ModelAdapterIntentGraphConfig(
            relationship_type="EXPRESSED_INTENT"
        )
        assert config_valid.relationship_type == "EXPRESSED_INTENT"

        config_underscore = ModelAdapterIntentGraphConfig(
            relationship_type="_HAS_INTENT"
        )
        assert config_underscore.relationship_type == "_HAS_INTENT"

        # Invalid relationship types
        with pytest.raises(ValidationError, match="valid Cypher identifier"):
            ModelAdapterIntentGraphConfig(relationship_type="has-dash")

    def test_auto_create_indexes_option(self) -> None:
        """Test auto_create_indexes configuration option."""
        # Default is True
        config_default = ModelAdapterIntentGraphConfig()
        assert config_default.auto_create_indexes is True

        # Can be set to False
        config_disabled = ModelAdapterIntentGraphConfig(auto_create_indexes=False)
        assert config_disabled.auto_create_indexes is False

        # Can be set to True explicitly
        config_enabled = ModelAdapterIntentGraphConfig(auto_create_indexes=True)
        assert config_enabled.auto_create_indexes is True

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected (ConfigDict extra='forbid')."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="extra"):
            ModelAdapterIntentGraphConfig(
                timeout_seconds=30.0,
                unknown_field="should fail",  # type: ignore[call-arg]
            )


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for Pydantic model validation."""

    def test_intent_classification_output_required_fields(self) -> None:
        """Test ModelIntentClassificationOutput requires intent_category and confidence."""
        from pydantic import ValidationError

        # Verify that omitting required fields raises ValidationError
        with pytest.raises(ValidationError, match="intent_category"):
            ModelIntentClassificationOutput(confidence=0.5)

        with pytest.raises(ValidationError, match="confidence"):
            ModelIntentClassificationOutput(intent_category="debugging")

        # Verify construction with only required fields succeeds
        classification = ModelIntentClassificationOutput(
            intent_category="unknown",
            confidence=0.0,
        )

        assert classification.intent_category == "unknown"
        assert classification.confidence == 0.0
        assert classification.keywords == []
        assert classification.raw_text is None
        assert classification.metadata == {}

    def test_intent_classification_output_all_fields(self) -> None:
        """Test ModelIntentClassificationOutput with all fields."""
        classification = ModelIntentClassificationOutput(
            intent_category=EnumIntentCategory.CODE_GENERATION,
            confidence=0.95,
            keywords=["python", "function", "async"],
            raw_text="Write a python async function",
            metadata={"model_version": "1.0"},
        )

        assert (
            classification.intent_category == EnumIntentCategory.CODE_GENERATION.value
        )
        assert classification.confidence == 0.95
        assert classification.keywords == ["python", "function", "async"]
        assert classification.raw_text == "Write a python async function"
        assert classification.metadata == {"model_version": "1.0"}

    def test_intent_classification_output_confidence_bounds(self) -> None:
        """Test ModelIntentClassificationOutput confidence must be 0.0-1.0."""
        from pydantic import ValidationError

        # Valid bounds
        ModelIntentClassificationOutput(intent_category="debugging", confidence=0.0)
        ModelIntentClassificationOutput(intent_category="debugging", confidence=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            ModelIntentClassificationOutput(
                intent_category="debugging", confidence=-0.1
            )

        with pytest.raises(ValidationError):
            ModelIntentClassificationOutput(intent_category="debugging", confidence=1.1)

    def test_intent_storage_result_success(self) -> None:
        """Test ModelIntentStorageResult success case."""
        result = ModelIntentStorageResult(
            status="success",
            intent_id=TEST_INTENT_ID_1,
            session_id="session_123",
            created=True,
        )

        assert result.status == "success"
        assert result.intent_id == TEST_INTENT_ID_1
        assert result.created is True
        assert result.error_message is None

    def test_intent_storage_result_error(self) -> None:
        """Test ModelIntentStorageResult error case."""
        result = ModelIntentStorageResult(
            status="error",
            session_id="session_123",
            error_message="Connection timeout",
        )

        assert result.status == "error"
        assert result.intent_id is None
        assert result.created is False
        assert result.error_message == "Connection timeout"

    def test_intent_storage_result_merged(self) -> None:
        """Test ModelIntentStorageResult when intent was merged (not created)."""
        result = ModelIntentStorageResult(
            status="success",
            intent_id=TEST_INTENT_ID_EXISTING,
            session_id="session_123",
            created=False,  # Merged with existing
        )

        assert result.status == "success"
        assert result.created is False

    def test_intent_record_model(self) -> None:
        """Test ModelIntentRecord creation."""
        record = ModelIntentRecord(
            intent_id=TEST_INTENT_ID_1,
            session_ref="session_123",
            intent_category=EnumIntentCategory.DEBUGGING.value,
            confidence=0.92,
            keywords=["error", "fix"],
            created_at_utc=TEST_CREATED_AT_1,
            correlation_id=TEST_CORRELATION_ID,
        )

        assert record.intent_id == TEST_INTENT_ID_1
        assert record.session_ref == "session_123"
        assert record.intent_category == EnumIntentCategory.DEBUGGING.value
        assert record.confidence == 0.92
        assert record.keywords == ["error", "fix"]
        assert record.created_at_utc == TEST_CREATED_AT_1
        assert record.correlation_id == TEST_CORRELATION_ID

    def test_intent_record_defaults(self) -> None:
        """Test ModelIntentRecord default values."""
        record = ModelIntentRecord(
            intent_id=TEST_INTENT_ID_1,
            session_ref="session_123",
            intent_category=EnumIntentCategory.UNKNOWN.value,
            confidence=0.5,
            created_at_utc=TEST_CREATED_AT_1,
        )

        assert record.keywords == []
        assert record.correlation_id is None

    def test_intent_query_result_success(self) -> None:
        """Test ModelIntentQueryResult success case with results."""
        intents = [
            ModelIntentRecord(
                intent_id=TEST_INTENT_ID_1,
                session_ref="session_123",
                intent_category=EnumIntentCategory.DEBUGGING.value,
                confidence=0.9,
                created_at_utc=TEST_CREATED_AT_1,
            ),
            ModelIntentRecord(
                intent_id=TEST_INTENT_ID_2,
                session_ref="session_123",
                intent_category=EnumIntentCategory.CODE_GENERATION.value,
                confidence=0.85,
                created_at_utc=TEST_CREATED_AT_2,
            ),
        ]

        result = ModelIntentQueryResult(
            status="success",
            intents=intents,
        )

        assert result.status == "success"
        assert len(result.intents) == 2
        assert result.error_message is None

    def test_intent_query_result_no_results(self) -> None:
        """Test ModelIntentQueryResult when no intents found."""
        result = ModelIntentQueryResult(
            status="no_results",
            intents=[],
        )

        assert result.status == "no_results"
        assert result.intents == []

    def test_intent_query_result_error(self) -> None:
        """Test ModelIntentQueryResult error case."""
        result = ModelIntentQueryResult(
            status="error",
            error_message="Query timeout",
        )

        assert result.status == "error"
        assert result.intents == []
        assert result.error_message == "Query timeout"

    def test_intent_query_result_status_values(self) -> None:
        """Test ModelIntentQueryResult with success/failure states."""
        # Success case
        result_success = ModelIntentQueryResult(status="success")
        assert result_success.status == "success"

        # Error case
        result_error = ModelIntentQueryResult(status="error", error_message="Failed")
        assert result_error.status == "error"

    def test_intent_graph_health_healthy(self) -> None:
        """Test ModelIntentGraphHealth when healthy."""
        health = ModelIntentGraphHealth(
            is_healthy=True,
            initialized=True,
            handler_healthy=True,
            session_count=100,
            intent_count=500,
            last_check_timestamp="2025-01-22T10:30:00Z",
        )

        assert health.is_healthy is True
        assert health.initialized is True
        assert health.handler_healthy is True
        assert health.error_message is None
        assert health.session_count == 100
        assert health.intent_count == 500

    def test_intent_graph_health_not_initialized(self) -> None:
        """Test ModelIntentGraphHealth when not initialized."""
        health = ModelIntentGraphHealth(
            is_healthy=False,
            initialized=False,
            handler_healthy=None,
            error_message="Adapter not initialized",
        )

        assert health.is_healthy is False
        assert health.initialized is False
        assert health.handler_healthy is None
        assert health.error_message == "Adapter not initialized"
        assert health.session_count is None
        assert health.intent_count is None

    def test_intent_graph_health_unhealthy_handler(self) -> None:
        """Test ModelIntentGraphHealth when handler is unhealthy."""
        health = ModelIntentGraphHealth(
            is_healthy=False,
            initialized=True,
            handler_healthy=False,
            error_message="Handler reports unhealthy",
        )

        assert health.is_healthy is False
        assert health.initialized is True
        assert health.handler_healthy is False


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


class TestProtocolConformance:
    """Tests that AdapterIntentGraph conforms to ProtocolIntentGraph."""

    def test_isinstance_check(self, config: ModelAdapterIntentGraphConfig) -> None:
        """Test that AdapterIntentGraph is an instance of ProtocolIntentGraph."""
        from omnibase_spi.protocols import ProtocolIntentGraph

        adapter = AdapterIntentGraph(config)
        assert isinstance(adapter, ProtocolIntentGraph)

    def test_inherits_from_protocol(self) -> None:
        """Test that AdapterIntentGraph explicitly inherits from ProtocolIntentGraph."""
        from omnibase_spi.protocols import ProtocolIntentGraph

        assert issubclass(AdapterIntentGraph, ProtocolIntentGraph)


# =============================================================================
# Cypher Templates Tests
# =============================================================================


class TestIntentCypherTemplates:
    """Tests for IntentCypherTemplates."""

    def test_templates_use_parameters(self) -> None:
        """Verify all templates use parameterized queries (no string interpolation)."""
        session_label = "Session"
        intent_label = "Intent"
        rel_type = "HAD_INTENT"

        templates = [
            IntentCypherTemplates.store_intent_query(
                session_label, intent_label, rel_type
            ),
            IntentCypherTemplates.get_session_intents_query(
                session_label, intent_label, rel_type
            ),
            IntentCypherTemplates.get_intent_distribution_query(intent_label),
            IntentCypherTemplates.count_all_query(session_label, intent_label),
        ]

        for template in templates:
            # Templates should use $param syntax for parameters
            assert (
                "$" in template or "count" in template.lower()
            ), f"Template missing parameter: {template[:50]}..."
            # Templates should not have Python f-string or .format() placeholders
            unsafe_patterns = [
                r"\{[0-9]+\}",  # {0}, {1} positional args
            ]
            for pattern in unsafe_patterns:
                matches = re.findall(pattern, template)
                assert not matches, (
                    f"Template has unsafe format pattern {matches}: "
                    f"{template[:50]}..."
                )

    def test_store_intent_query_structure(self) -> None:
        """Test store_intent_query template structure."""
        template = IntentCypherTemplates.store_intent_query(
            "Session", "Intent", "HAD_INTENT"
        )

        # Required parameters
        assert "$session_id" in template
        assert "$intent_category" in template
        assert "$confidence" in template
        assert "$keywords" in template
        assert "$correlation_id" in template
        assert "$intent_id" in template
        assert "$created_at_utc" in template
        assert "$timestamp_utc" in template

        # Required Cypher keywords
        assert "MERGE" in template
        assert "ON CREATE SET" in template
        assert "ON MATCH SET" in template
        assert "RETURN" in template

    def test_store_intent_query_uses_configured_labels(self) -> None:
        """Test that store_intent_query uses the configured labels."""
        template = IntentCypherTemplates.store_intent_query(
            "CustomSession", "CustomIntent", "EXPRESSED"
        )

        assert ":CustomSession" in template
        assert ":CustomIntent" in template
        assert ":EXPRESSED" in template

        # Ensure defaults are not present
        assert ":Session" not in template
        assert ":Intent" not in template
        assert ":HAD_INTENT" not in template

    def test_get_session_intents_query_structure(self) -> None:
        """Test get_session_intents_query template structure."""
        template = IntentCypherTemplates.get_session_intents_query(
            "Session", "Intent", "HAD_INTENT"
        )

        # Required parameters
        assert "$session_id" in template
        assert "$min_confidence" in template
        assert "$limit" in template

        # Required Cypher keywords
        assert "MATCH" in template
        assert "WHERE" in template
        assert "RETURN" in template
        assert "ORDER BY" in template
        assert "LIMIT" in template

    def test_get_intent_distribution_query_structure(self) -> None:
        """Test get_intent_distribution_query template structure."""
        template = IntentCypherTemplates.get_intent_distribution_query("Intent")

        # Required parameters
        assert "$since_utc" in template

        # Required Cypher keywords
        assert "MATCH" in template
        assert "WHERE" in template
        assert "RETURN" in template
        assert "count" in template.lower()
        assert "ORDER BY" in template

    def test_create_indexes_queries_returns_list(self) -> None:
        """Test create_indexes_queries returns list of index creation queries."""
        queries = IntentCypherTemplates.create_indexes_queries(
            "Session", "Intent", "HAD_INTENT"
        )

        assert isinstance(queries, list)
        # 4 node property indexes + 1 edge property index
        assert len(queries) == 5

        # Check for expected node property indexes
        assert any("session_id" in q for q in queries)
        assert any("intent_id" in q for q in queries)
        assert any("intent_category" in q for q in queries)
        assert any("created_at_utc" in q for q in queries)

        # Check for edge property index on relationship timestamp
        assert any("timestamp_utc" in q and "EDGE INDEX" in q for q in queries)

        # Verify IF NOT EXISTS syntax is used for idempotent index creation
        for query in queries:
            assert "IF NOT EXISTS" in query, f"Query missing IF NOT EXISTS: {query}"

    def test_create_indexes_queries_uses_configured_labels(self) -> None:
        """Test create_indexes_queries uses the configured labels."""
        queries = IntentCypherTemplates.create_indexes_queries(
            "CustomSession", "CustomIntent", "CUSTOM_HAD_INTENT"
        )

        # Should use custom labels for node indexes
        assert any(":CustomSession" in q for q in queries)
        assert any(":CustomIntent" in q for q in queries)

        # Should use custom relationship type for edge index
        assert any(":CUSTOM_HAD_INTENT" in q for q in queries)

        # Should NOT use defaults
        assert not any(":Session" in q and "Custom" not in q for q in queries)
        assert not any(":Intent" in q and "Custom" not in q for q in queries)
        assert not any(":HAD_INTENT" in q and "CUSTOM" not in q for q in queries)

    def test_count_all_query_structure(self) -> None:
        """Test count_all_query generates valid Cypher."""
        template = IntentCypherTemplates.count_all_query("Session", "Intent")

        # Should use OPTIONAL MATCH for both node types
        assert "OPTIONAL MATCH" in template
        assert ":Session" in template
        assert ":Intent" in template
        # Should return both counts
        assert "session_count" in template
        assert "intent_count" in template


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for adapter initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test successful initialization."""
        adapter = AdapterIntentGraph(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                    auth=("neo4j", "password"),
                )

            assert adapter.is_initialized
            mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test that initialize is idempotent."""
        adapter = AdapterIntentGraph(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                await adapter.initialize("bolt://localhost:7687")
                await adapter.initialize("bolt://localhost:7687")

            # Should only create handler once
            assert MockHandler.call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test shutdown releases resources."""
        assert adapter_with_mock.is_initialized

        await adapter_with_mock.shutdown()

        assert not adapter_with_mock.is_initialized
        mock_handler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test shutdown is idempotent (safe to call multiple times)."""
        await adapter_with_mock.shutdown()
        await adapter_with_mock.shutdown()

        # Should only call handler shutdown once
        mock_handler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_raises_when_not_initialized(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test _ensure_initialized raises when not initialized."""
        adapter = AdapterIntentGraph(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            adapter._ensure_initialized()

    @pytest.mark.asyncio
    async def test_initialize_validates_uri_format(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test that initialize validates the connection_uri format."""
        adapter = AdapterIntentGraph(config)

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
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test that initialize accepts all valid bolt URI schemes."""
        adapter = AdapterIntentGraph(config)

        valid_uris = [
            "bolt://localhost:7687",
            "bolt+s://localhost:7687",
            "bolt+ssc://localhost:7687",
            "neo4j://localhost:7687",
            "neo4j+s://localhost:7687",
        ]

        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                for uri in valid_uris:
                    # Reset adapter state for each URI
                    adapter._initialized = False
                    adapter._handler = None

                    # Should not raise ValueError for valid schemes
                    await adapter.initialize(connection_uri=uri)
                    assert adapter.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_timeout_on_lock_acquisition(self) -> None:
        """Test that initialization times out if lock cannot be acquired."""
        # Use a short timeout
        config = ModelAdapterIntentGraphConfig(timeout_seconds=0.5)
        adapter = AdapterIntentGraph(config)

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
    async def test_initialize_creates_indexes(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test that indexes are created during initialization."""
        adapter = AdapterIntentGraph(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            assert adapter.is_initialized
            # Should have 5 index creation calls (4 node + 1 edge property index)
            assert mock_instance.execute_query.call_count == 5

    @pytest.mark.asyncio
    async def test_initialize_handles_index_creation_errors(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test that index creation errors don't fail initialization.

        Index creation errors are non-fatal since indexes improve performance
        but are not required for correctness. This test verifies the adapter
        gracefully handles index creation failures.
        """
        adapter = AdapterIntentGraph(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            # Simulate index creation error (e.g., permission denied, syntax error)
            mock_instance.execute_query = AsyncMock(
                side_effect=Exception("Index creation failed: permission denied")
            )
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                # Should NOT raise - index errors are caught and logged
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            # Adapter should still be initialized despite index error
            assert adapter.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_skips_indexes_when_disabled(self) -> None:
        """Test that index creation is skipped when auto_create_indexes=False."""
        config = ModelAdapterIntentGraphConfig(auto_create_indexes=False)
        adapter = AdapterIntentGraph(config)

        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute_query = AsyncMock()
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                await adapter.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            assert adapter.is_initialized
            # execute_query should NOT have been called for index creation
            mock_instance.execute_query.assert_not_called()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test that async context manager returns the adapter instance."""
        adapter = AdapterIntentGraph(config)

        async with adapter as ctx:
            assert ctx is adapter

    @pytest.mark.asyncio
    async def test_context_manager_calls_shutdown_on_exit(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test that shutdown is called automatically on context exit."""
        assert adapter_with_mock.is_initialized

        async with adapter_with_mock:
            # Do some work
            pass

        # shutdown should have been called
        mock_handler.shutdown.assert_called_once()
        assert not adapter_with_mock.is_initialized

    @pytest.mark.asyncio
    async def test_context_manager_calls_shutdown_on_exception(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test that shutdown is called even when exception occurs inside context."""
        assert adapter_with_mock.is_initialized

        with pytest.raises(ValueError, match="test error"):
            async with adapter_with_mock:
                raise ValueError("test error")

        # shutdown should still have been called despite the exception
        mock_handler.shutdown.assert_called_once()
        assert not adapter_with_mock.is_initialized

    @pytest.mark.asyncio
    async def test_context_manager_full_lifecycle(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test full lifecycle with context manager: create, initialize, use, shutdown."""
        with patch(
            "omnimemory.handlers.adapters.adapter_intent_graph.HandlerGraph"
        ) as MockHandler:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.shutdown = AsyncMock()
            mock_instance.execute_query = AsyncMock(
                return_value=MagicMock(
                    records=[{"intent_id": str(TEST_INTENT_ID_1), "was_created": True}]
                )
            )
            MockHandler.return_value = mock_instance

            with patch("omnibase_core.container.ModelONEXContainer"):
                async with AdapterIntentGraph(config) as adapter:
                    await adapter.initialize(
                        connection_uri="bolt://localhost:7687",
                    )
                    assert adapter.is_initialized

                    # Store an intent
                    result = await adapter.store_intent(
                        session_id="session_123",
                        intent_data=ModelIntentClassificationOutput(
                            intent_category=EnumIntentCategory.DEBUGGING,
                            confidence=0.9,
                        ),
                        correlation_id=str(TEST_CORRELATION_ID),
                    )
                    assert result.success is True

                # After context exit, shutdown should be called
                mock_instance.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_shutdown_idempotent(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test that context manager shutdown is safe if shutdown was already called."""
        async with adapter_with_mock:
            # Manually call shutdown inside the context
            await adapter_with_mock.shutdown()
            mock_handler.shutdown.assert_called_once()

        # Context manager exit should not fail even though shutdown was already called
        # shutdown should still only have been called once (idempotent behavior)
        mock_handler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_without_initialize(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test context manager works even if initialize was never called."""
        adapter = AdapterIntentGraph(config)

        # Should not raise, even without initialization
        async with adapter:
            assert not adapter.is_initialized

        # shutdown on uninitialized adapter is a no-op
        assert not adapter.is_initialized


# =============================================================================
# store_intent Tests
# =============================================================================


class TestStoreIntent:
    """Tests for store_intent method."""

    @pytest.mark.asyncio
    async def test_store_intent_success(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
        sample_intent_classification: ModelIntentClassificationOutput,
    ) -> None:
        """Test successful intent storage."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "intent_id": str(TEST_INTENT_ID_1),
                    "was_created": True,
                }
            ]
        )

        result = await adapter_with_mock.store_intent(
            session_id="session_xyz789",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        assert result.success is True
        assert result.intent_id == TEST_INTENT_ID_1
        assert result.created is True
        assert result.error_message is None

        # Verify query was called with correct parameters
        call_args = mock_handler.execute_query.call_args
        params = call_args[1]["parameters"]
        assert params["session_id"] == "session_xyz789"
        assert params["intent_category"] == "debugging"
        assert params["confidence"] == 0.92
        assert params["keywords"] == ["error", "traceback", "fix"]
        assert params["correlation_id"] == str(TEST_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_store_intent_merged_existing(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
        sample_intent_classification: ModelIntentClassificationOutput,
    ) -> None:
        """Test storing intent that merges with existing."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "intent_id": str(TEST_INTENT_ID_EXISTING),
                    "was_created": False,  # Merged, not created
                }
            ]
        )

        result = await adapter_with_mock.store_intent(
            session_id="session_xyz789",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        assert result.success is True
        assert result.created is False
        assert result.intent_id == TEST_INTENT_ID_EXISTING

    @pytest.mark.asyncio
    async def test_store_intent_not_initialized(
        self,
        config: ModelAdapterIntentGraphConfig,
        sample_intent_classification: ModelIntentClassificationOutput,
    ) -> None:
        """Test store_intent returns error when not initialized."""
        adapter = AdapterIntentGraph(config)

        result = await adapter.store_intent(
            session_id="session_123",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        assert result.success is False
        assert result.error_message is not None
        assert "not initialized" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_store_intent_handles_connection_error(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
        sample_intent_classification: ModelIntentClassificationOutput,
    ) -> None:
        """Test store_intent handles connection errors gracefully."""
        mock_handler.execute_query.side_effect = Exception("Connection lost")

        result = await adapter_with_mock.store_intent(
            session_id="session_123",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        assert result.success is False
        assert result.error_message is not None
        assert "failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_store_intent_with_user_context(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
        sample_intent_classification: ModelIntentClassificationOutput,
    ) -> None:
        """Test store_intent passes through correlation_id correctly."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[{"intent_id": str(TEST_INTENT_ID_1), "was_created": True}]
        )

        await adapter_with_mock.store_intent(
            session_id="session_123",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        call_args = mock_handler.execute_query.call_args
        params = call_args[1]["parameters"]
        assert params["correlation_id"] == str(TEST_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_store_intent_rejects_empty_session_id(
        self,
        adapter_with_mock: AdapterIntentGraph,
        sample_intent_classification: ModelIntentClassificationOutput,
    ) -> None:
        """Test store_intent returns error for empty or whitespace-only session_id."""
        # Test empty string
        result = await adapter_with_mock.store_intent(
            session_id="",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        assert result.success is False
        assert result.error_message is not None
        assert "session_id cannot be empty" in result.error_message

        # Test whitespace-only string
        result_whitespace = await adapter_with_mock.store_intent(
            session_id="   ",
            intent_data=sample_intent_classification,
            correlation_id=str(TEST_CORRELATION_ID_2),
        )

        assert result_whitespace.success is False
        assert result_whitespace.error_message is not None
        assert "session_id cannot be empty" in result_whitespace.error_message


# =============================================================================
# get_session_intents Tests
# =============================================================================


class TestGetSessionIntents:
    """Tests for get_session_intents method."""

    @pytest.mark.asyncio
    async def test_get_session_intents_success(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test successful intent query."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "intent_id": str(TEST_INTENT_ID_1),
                    "intent_category": "debugging",
                    "confidence": 0.92,
                    "keywords": ["error", "fix"],
                    "created_at_utc": TEST_CREATED_AT_1.isoformat(),
                    "correlation_id": str(TEST_CORRELATION_ID),
                },
                {
                    "intent_id": str(TEST_INTENT_ID_2),
                    "intent_category": "unknown",  # Use valid enum value
                    "confidence": 0.85,
                    "keywords": ["why", "how"],
                    "created_at_utc": TEST_CREATED_AT_2.isoformat(),
                    "correlation_id": str(TEST_CORRELATION_ID_2),
                },
            ]
        )

        result = await adapter_with_mock.get_session_intents(
            session_id="session_123",
        )

        assert result.success is True
        assert len(result.intents) == 2

        # Verify first intent (core ModelIntentRecord uses EnumIntentCategory, not str)
        assert result.intents[0].intent_id == TEST_INTENT_ID_1
        assert result.intents[0].intent_category == EnumIntentCategory.DEBUGGING
        assert result.intents[0].confidence == 0.92
        assert result.intents[0].keywords == ["error", "fix"]

    @pytest.mark.asyncio
    async def test_get_session_intents_no_results(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_session_intents when no intents found."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        result = await adapter_with_mock.get_session_intents(
            session_id="session_empty",
        )

        assert result.success is True
        assert result.intents == []

    @pytest.mark.asyncio
    async def test_get_session_intents_with_min_confidence(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_session_intents with min_confidence filter."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        await adapter_with_mock.get_session_intents(
            session_id="session_123",
            min_confidence=0.8,
        )

        call_args = mock_handler.execute_query.call_args
        params = call_args[1]["parameters"]
        assert params["min_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_get_session_intents_with_limit(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_session_intents with custom limit."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        await adapter_with_mock.get_session_intents(
            session_id="session_123",
            limit=10,
        )

        call_args = mock_handler.execute_query.call_args
        params = call_args[1]["parameters"]
        assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_session_intents_uses_config_defaults(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_session_intents uses config defaults when params not specified."""
        config = ModelAdapterIntentGraphConfig(
            default_confidence_threshold=0.5,
            max_intents_per_session=50,
        )
        adapter = AdapterIntentGraph(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        mock_handler.execute_query.return_value = MagicMock(records=[])

        await adapter.get_session_intents(session_id="session_123")

        call_args = mock_handler.execute_query.call_args
        params = call_args[1]["parameters"]
        assert params["min_confidence"] == 0.5
        assert params["limit"] == 50

    @pytest.mark.asyncio
    async def test_get_session_intents_not_initialized(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test get_session_intents returns error when not initialized."""
        adapter = AdapterIntentGraph(config)

        result = await adapter.get_session_intents(session_id="session_123")

        assert result.success is False
        assert result.error_message is not None
        assert "not initialized" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_get_session_intents_handles_error(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_session_intents handles errors gracefully."""
        mock_handler.execute_query.side_effect = Exception("Query failed")

        result = await adapter_with_mock.get_session_intents(
            session_id="session_123",
        )

        assert result.success is False
        assert result.error_message is not None
        assert "failed" in result.error_message.lower()


# =============================================================================
# get_intent_distribution Tests
# =============================================================================


class TestGetIntentDistribution:
    """Tests for get_intent_distribution method."""

    @pytest.mark.asyncio
    async def test_get_intent_distribution_success(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test successful intent distribution query."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {"category": "debugging", "count": 150},
                {"category": "code_generation", "count": 89},
                {"category": "explanation", "count": 45},
            ]
        )

        result = await adapter_with_mock.get_intent_distribution(time_range_hours=24)

        assert result.status == "success"
        assert result.distribution == {
            "debugging": 150,
            "code_generation": 89,
            "explanation": 45,
        }
        assert result.total_intents == 284
        assert result.time_range_hours == 24
        assert result.execution_time_ms > 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_get_intent_distribution_empty(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_intent_distribution with no results."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        result = await adapter_with_mock.get_intent_distribution()

        assert result.status == "success"
        assert result.distribution == {}
        assert result.total_intents == 0

    @pytest.mark.asyncio
    async def test_get_intent_distribution_custom_time_range(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_intent_distribution with custom time range."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        await adapter_with_mock.get_intent_distribution(time_range_hours=48)

        call_args = mock_handler.execute_query.call_args
        params = call_args[1]["parameters"]
        assert "since_utc" in params

    @pytest.mark.asyncio
    async def test_get_intent_distribution_not_initialized(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test get_intent_distribution returns error when not initialized."""
        adapter = AdapterIntentGraph(config)

        result = await adapter.get_intent_distribution()

        assert result.status == "error"
        assert "not initialized" in result.error_message.lower()
        assert result.distribution == {}

    @pytest.mark.asyncio
    async def test_get_intent_distribution_handles_error(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_intent_distribution returns error on query failure."""
        mock_handler.execute_query.side_effect = Exception("Query failed")

        result = await adapter_with_mock.get_intent_distribution()

        assert result.status == "error"
        assert "failed" in result.error_message.lower()
        assert result.distribution == {}

    @pytest.mark.asyncio
    async def test_get_intent_distribution_filters_invalid_records(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test that invalid records are filtered out."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {"category": "debugging", "count": 150},
                {"category": None, "count": 50},  # Invalid category
                {"category": "explanation", "count": "invalid"},  # Invalid count
                {"category": "valid", "count": 10},
            ]
        )

        result = await adapter_with_mock.get_intent_distribution()

        assert result.status == "success"
        # Only valid records should be included
        assert result.distribution == {
            "debugging": 150,
            "valid": 10,
        }
        assert result.total_intents == 160

    @pytest.mark.asyncio
    async def test_get_intent_distribution_zero_time_range(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_intent_distribution handles zero time_range_hours gracefully."""
        mock_handler.execute_query.return_value = MagicMock(records=[])

        # Call with zero time range - should be handled gracefully
        result = await adapter_with_mock.get_intent_distribution(time_range_hours=0)

        # Should succeed, not error
        assert result.status == "success"
        # Time range should be clamped to at least 1
        assert result.time_range_hours >= 1


# =============================================================================
# health_check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check returns True when handler is healthy."""
        mock_handler.health_check.return_value = MagicMock(healthy=True)

        result = await adapter_with_mock.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(
        self,
        config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Test health check returns False when not initialized."""
        adapter = AdapterIntentGraph(config)

        result = await adapter.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_handler_unhealthy(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check returns False when handler reports unhealthy."""
        mock_handler.health_check.return_value = MagicMock(healthy=False)

        result = await adapter_with_mock.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_count_query_failure(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check returns True even if handler raises on health_check."""
        mock_handler.health_check.side_effect = Exception("Health check failed")

        result = await adapter_with_mock.health_check()

        # Should return False when exception occurs
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check handles timeout gracefully."""
        import asyncio

        config = ModelAdapterIntentGraphConfig(timeout_seconds=0.1)
        adapter = AdapterIntentGraph(config)
        adapter._handler = mock_handler
        adapter._initialized = True

        # Simulate slow health check
        async def slow_health_check() -> MagicMock:
            await asyncio.sleep(1.0)
            return MagicMock(healthy=True)

        mock_handler.health_check = slow_health_check

        result = await adapter.health_check()

        assert result is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_store_intent_handles_unexpected_exception(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test adapter handles unexpected exceptions."""
        mock_handler.execute_query.side_effect = RuntimeError("Unexpected error")

        classification = ModelIntentClassificationOutput(
            intent_category=EnumIntentCategory.UNKNOWN,
            confidence=0.5,
        )

        result = await adapter_with_mock.store_intent(
            session_id="session_123",
            intent_data=classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )

        assert result.success is False
        assert result.error_message is not None
        assert "failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_get_session_intents_handles_malformed_records(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test get_session_intents handles malformed records gracefully."""
        mock_handler.execute_query.return_value = MagicMock(
            records=[
                {
                    "intent_id": str(TEST_INTENT_ID_1),  # Valid UUID
                    "intent_category": "debugging",
                    "confidence": 0.9,
                    "keywords": ["test"],
                    "created_at_utc": TEST_CREATED_AT_1.isoformat(),
                },
                {
                    # Missing intent_id - should be skipped
                    "intent_category": "debugging",
                    "confidence": 0.8,
                },
                {
                    "intent_id": None,  # None intent_id - should be skipped
                    "intent_category": "unknown",
                    "confidence": 0.7,
                },
                {
                    "intent_id": "not-a-valid-uuid",  # Invalid UUID - should be skipped
                    "intent_category": "unknown",
                    "confidence": 0.6,
                    "created_at_utc": TEST_CREATED_AT_1.isoformat(),
                },
            ]
        )

        result = await adapter_with_mock.get_session_intents(session_id="session_123")

        assert result.success is True
        # Only the valid record with proper UUID should be included
        assert len(result.intents) == 1
        assert result.intents[0].intent_id == TEST_INTENT_ID_1

    @pytest.mark.asyncio
    async def test_operations_after_shutdown(
        self,
        adapter_with_mock: AdapterIntentGraph,
        mock_handler: MagicMock,
    ) -> None:
        """Test operations return errors after shutdown."""
        await adapter_with_mock.shutdown()

        # All operations should return error status
        classification = ModelIntentClassificationOutput(
            intent_category=EnumIntentCategory.UNKNOWN,
            confidence=0.5,
        )

        store_result = await adapter_with_mock.store_intent(
            session_id="session_123",
            intent_data=classification,
            correlation_id=str(TEST_CORRELATION_ID),
        )
        assert store_result.success is False

        query_result = await adapter_with_mock.get_session_intents(
            session_id="session_123",
        )
        assert query_result.success is False

        distribution = await adapter_with_mock.get_intent_distribution()
        assert distribution.status == "error"
        assert distribution.distribution == {}

        health = await adapter_with_mock.health_check()
        assert health is False
