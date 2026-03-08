# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerIntentQuery with real Memgraph.

These tests require a running Memgraph instance. They will be skipped
if Memgraph is not available.

Test Categories:
    - Distribution: Query intent distribution by category
    - Session: Query intents for a specific session
    - Recent: Query recent intents across all sessions
    - Data round-trip: Create, query, and verify intent data
    - Filtering: Test confidence and other filter criteria
    - Error handling: Test error conditions and edge cases

Prerequisites:
    - Memgraph running at MEMGRAPH_URI (default: bolt://localhost:7687)
    - omnibase_infra installed (dev dependency)

Usage:
    # Run only integration tests
    pytest tests/nodes/node_intent_query_effect/test_handler_intent_query_integration.py -v

    # Run with specific markers
    pytest -m "integration and memgraph" -v

    # Skip if Memgraph unavailable (automatic)
    pytest -m integration -v

Environment Variables:
    MEMGRAPH_URI: Memgraph connection URI (default: bolt://localhost:7687)
    MEMGRAPH_USER: Memgraph username (optional)
    MEMGRAPH_PASSWORD: Memgraph password (optional)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Environment configuration
DEFAULT_MEMGRAPH_URI = "bolt://localhost:7687"
DEFAULT_MEMGRAPH_USER = "memgraph"
DEFAULT_MEMGRAPH_PASSWORD = ""


def get_memgraph_uri() -> str:
    """Get Memgraph URI from environment."""
    return os.environ.get("MEMGRAPH_URI", DEFAULT_MEMGRAPH_URI)


def get_memgraph_auth() -> tuple[str, str] | None:
    """Get Memgraph auth from environment.

    Returns a (username, password) tuple if both are configured, or None for
    anonymous authentication.
    """
    user = os.environ.get("MEMGRAPH_USER", DEFAULT_MEMGRAPH_USER)
    password = os.environ.get("MEMGRAPH_PASSWORD", DEFAULT_MEMGRAPH_PASSWORD)
    if user and password:
        return (user, password)
    return None


# =============================================================================
# Availability Check
# =============================================================================

_MEMGRAPH_AVAILABLE = False
_SKIP_REASON = "Memgraph not available"

try:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    from omnibase_core.container import ModelONEXContainer

    from omnimemory.handlers.adapters import AdapterIntentGraph
    from omnimemory.handlers.adapters.models import ModelAdapterIntentGraphConfig
    from omnimemory.nodes.node_intent_query_effect.handlers import HandlerIntentQuery
    from omnimemory.nodes.node_intent_query_effect.models import (
        ModelHandlerIntentQueryConfig,
    )

    _MEMGRAPH_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as e:
    _SKIP_REASON = f"Required dependencies not installed: {e}"


async def check_memgraph_available() -> bool:
    """Check if Memgraph is reachable."""
    if not _MEMGRAPH_AVAILABLE:
        return False

    try:
        uri = get_memgraph_uri()
        auth = get_memgraph_auth()
        driver = AsyncGraphDatabase.driver(uri, auth=auth)
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            await result.consume()
        await driver.close()
        return True
    except (ServiceUnavailable, OSError, Exception):
        return False


# Check availability at module load time
try:
    import asyncio

    _loop = asyncio.new_event_loop()
    _MEMGRAPH_AVAILABLE = _loop.run_until_complete(check_memgraph_available())
    _loop.close()
    if not _MEMGRAPH_AVAILABLE:
        _SKIP_REASON = "Memgraph is not available or not responding"
except Exception as e:
    _MEMGRAPH_AVAILABLE = False
    _SKIP_REASON = f"Failed to check Memgraph availability: {e}"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.memgraph,
    pytest.mark.skipif(not _MEMGRAPH_AVAILABLE, reason=_SKIP_REASON),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_session_id() -> str:
    """Generate unique session ID for test isolation."""
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def memgraph_available() -> bool:
    """Check if Memgraph is available for tests."""
    return await check_memgraph_available()


@pytest.fixture
def adapter_config() -> ModelAdapterIntentGraphConfig:
    """Create adapter configuration for tests."""
    return ModelAdapterIntentGraphConfig(
        timeout_seconds=30.0,
        auto_create_indexes=False,  # Skip index creation in tests
    )


@pytest.fixture
def handler_config() -> ModelHandlerIntentQueryConfig:
    """Create handler configuration for tests."""
    return ModelHandlerIntentQueryConfig(
        timeout_seconds=30.0,
        default_time_range_hours=24,
        default_limit=100,
    )


@pytest.fixture
async def initialized_adapter(
    memgraph_available: bool,
    adapter_config: ModelAdapterIntentGraphConfig,
) -> AsyncGenerator[AdapterIntentGraph, None]:
    """Create and initialize adapter for tests."""
    if not memgraph_available:
        pytest.skip("Memgraph is not available")

    adapter = AdapterIntentGraph(adapter_config)
    await adapter.initialize(get_memgraph_uri(), get_memgraph_auth())

    yield adapter

    await adapter.shutdown()


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create a container for handler tests."""
    return ModelONEXContainer()


@pytest.fixture
async def initialized_handler(
    memgraph_available: bool,
    container: ModelONEXContainer,
    handler_config: ModelHandlerIntentQueryConfig,
    adapter_config: ModelAdapterIntentGraphConfig,
) -> AsyncGenerator[HandlerIntentQuery, None]:
    """Create and initialize handler for tests.

    The handler now owns the adapter lifecycle (container-driven pattern).
    """
    if not memgraph_available:
        pytest.skip("Memgraph is not available")

    handler = HandlerIntentQuery(container)
    await handler.initialize(
        connection_uri=get_memgraph_uri(),
        auth=get_memgraph_auth(),
        config=handler_config,
        adapter_config=adapter_config,
    )

    yield handler

    await handler.shutdown()


@pytest.fixture
async def test_intents_in_db(
    initialized_adapter: AdapterIntentGraph,
    test_session_id: str,
) -> list[dict[str, Any]]:
    """Create test intents in database for query testing.

    This fixture stores test intents with varying categories and confidence
    levels to enable comprehensive query testing. The intents are linked to
    the unique test_session_id to ensure test isolation.

    Returns:
        List of dictionaries containing stored intent metadata:
        - intent_id: UUID of the stored intent
        - session_id: Session the intent belongs to
        - category: Intent category string
        - confidence: Confidence score (0.0-1.0)
        - keywords: List of keywords for this intent

    Note:
        No cleanup is needed since each test run uses a unique session_id,
        providing natural test isolation without explicit deletion.
    """
    from uuid import uuid4

    from omnibase_core.enums.intelligence import EnumIntentCategory
    from omnibase_core.models.intelligence import ModelIntentClassificationOutput

    # Create test intents with varying categories and confidence levels
    test_intents: list[dict[str, Any]] = []
    test_data: list[tuple[EnumIntentCategory, float, list[str]]] = [
        (EnumIntentCategory.DEBUGGING, 0.80, ["error", "traceback"]),
        (EnumIntentCategory.CODE_GENERATION, 0.85, ["add", "implement"]),
        (EnumIntentCategory.DOCUMENTATION, 0.90, ["docs", "explain"]),
    ]

    for category, confidence, keywords in test_data:
        intent_data = ModelIntentClassificationOutput(
            success=True,
            intent_category=category,
            confidence=confidence,
            keywords=keywords,
        )
        correlation_id = uuid4()

        result = await initialized_adapter.store_intent(
            session_id=test_session_id,
            intent_data=intent_data,
            correlation_id=str(correlation_id),
        )

        if result.success and result.intent_id is not None:
            test_intents.append(
                {
                    "intent_id": result.intent_id,
                    "session_id": test_session_id,
                    "category": category.value,
                    "confidence": confidence,
                    "keywords": keywords,
                    "correlation_id": correlation_id,
                }
            )

    return test_intents


# =============================================================================
# Integration Tests
# =============================================================================


class TestHandlerIntentQueryIntegration:
    """Integration tests for HandlerIntentQuery with real Memgraph."""

    @pytest.mark.asyncio
    async def test_distribution_query_empty(
        self,
        initialized_handler: HandlerIntentQuery,
    ) -> None:
        """Test distribution query returns valid response even with no data."""
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=1,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.query_id == request.query_id
        assert response.query_type == "distribution"
        assert response.status in ("success", "no_results")
        assert response.correlation_id == request.correlation_id
        assert response.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_session_query_not_found(
        self,
        initialized_handler: HandlerIntentQuery,
        test_session_id: str,
    ) -> None:
        """Test session query for non-existent session."""
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        request = ModelIntentQueryRequestedEvent.create_session_query(
            session_ref=test_session_id,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.query_id == request.query_id
        assert response.query_type == "session"
        assert response.status in ("success", "no_results", "not_found")
        assert response.intents == []
        assert response.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_recent_query_empty(
        self,
        initialized_handler: HandlerIntentQuery,
    ) -> None:
        """Test recent query returns valid response."""
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        request = ModelIntentQueryRequestedEvent.create_recent_query(
            time_range_hours=1,
            limit=10,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.query_id == request.query_id
        assert response.query_type == "recent"
        assert response.status in ("success", "no_results")
        assert response.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_session_query_missing_ref_returns_error(
        self,
        initialized_handler: HandlerIntentQuery,
    ) -> None:
        """Test session query without session_ref returns error response.

        Note: The omnibase_core model allows session_ref=None at creation time.
        Validation happens at the handler level, which returns an error response.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        # Create request with missing session_ref for a session query
        request = ModelIntentQueryRequestedEvent(
            query_type="session",
            session_ref=None,  # Missing!
            requester_name="test",
        )

        # Handler-level validation returns error response
        response = await initialized_handler.execute(request)

        assert response.status == "error"
        assert "session_ref is required" in (response.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(
        self,
        initialized_handler: HandlerIntentQuery,
    ) -> None:
        """Test correlation_id is echoed in response."""
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        correlation_id = uuid.uuid4()
        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=1,
            correlation_id=correlation_id,
        )

        response = await initialized_handler.execute(request)

        assert response.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_execution_time_is_positive(
        self,
        initialized_handler: HandlerIntentQuery,
    ) -> None:
        """Test execution time is tracked and positive."""
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=24,
        )

        response = await initialized_handler.execute(request)

        assert response.execution_time_ms is not None
        assert response.execution_time_ms > 0

    # =========================================================================
    # Data Round-Trip Tests (Create -> Query -> Verify)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_session_query_returns_stored_intents(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
        test_session_id: str,
    ) -> None:
        """Test session query returns intents that were stored.

        This test validates the full round-trip: intents are stored via the
        adapter, then queried via the handler, and results are verified to
        match the stored data.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        # Skip if no intents were stored (indicates storage failure)
        if not test_intents_in_db:
            pytest.skip("No test intents were stored - storage may have failed")

        request = ModelIntentQueryRequestedEvent.create_session_query(
            session_ref=test_session_id,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "success", f"Query failed: {response.error_message}"
        assert len(response.intents) == len(test_intents_in_db)

        # Verify all expected categories are present
        response_categories = {i.intent_category for i in response.intents}
        expected_categories = {i["category"] for i in test_intents_in_db}
        assert response_categories == expected_categories

        # Verify confidence values match
        response_confidences = {
            i.intent_category: i.confidence for i in response.intents
        }
        for stored_intent in test_intents_in_db:
            category = stored_intent["category"]
            expected_confidence = stored_intent["confidence"]
            assert category in response_confidences
            assert abs(response_confidences[category] - expected_confidence) < 0.001

    @pytest.mark.asyncio
    async def test_recent_query_returns_stored_intents(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
    ) -> None:
        """Test recent query returns recently stored intents.

        Since test intents are created just before this test runs, they
        should appear in a recent query with a short time range.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        if not test_intents_in_db:
            pytest.skip("No test intents were stored - storage may have failed")

        request = ModelIntentQueryRequestedEvent.create_recent_query(
            time_range_hours=1,  # Just created, should be within 1 hour
            limit=100,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "success", f"Query failed: {response.error_message}"
        # Should have at least our test intents (might have more from other tests)
        assert len(response.intents) >= len(test_intents_in_db)

        # Verify our test categories appear in the results
        response_categories = {i.intent_category for i in response.intents}
        expected_categories = {i["category"] for i in test_intents_in_db}
        assert expected_categories.issubset(response_categories)

    @pytest.mark.asyncio
    async def test_distribution_query_includes_stored_categories(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
    ) -> None:
        """Test distribution query includes stored intent categories.

        The distribution query aggregates intent counts by category. Our
        test intents should appear in the distribution with correct counts.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        if not test_intents_in_db:
            pytest.skip("No test intents were stored - storage may have failed")

        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=1,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "success", f"Query failed: {response.error_message}"
        assert response.distribution is not None

        # Our test categories should appear in distribution
        for test_intent in test_intents_in_db:
            category = test_intent["category"]
            assert category in response.distribution, (
                f"Expected category '{category}' not found in distribution. "
                f"Available categories: {list(response.distribution.keys())}"
            )
            # Each category should have at least 1 intent
            assert response.distribution[category] >= 1

    @pytest.mark.asyncio
    async def test_session_query_with_confidence_filter(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
        test_session_id: str,
    ) -> None:
        """Test session query filters by min_confidence.

        Test data has confidence values: 0.80, 0.85, 0.90
        Filtering with min_confidence=0.84 should exclude the first intent.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        if not test_intents_in_db:
            pytest.skip("No test intents were stored - storage may have failed")

        # Filter for confidence > 0.84, should exclude 0.80
        min_confidence = 0.84
        request = ModelIntentQueryRequestedEvent.create_session_query(
            session_ref=test_session_id,
            min_confidence=min_confidence,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        # Should have filtered out the low-confidence intent
        assert response.status in ("success", "no_results")

        if response.status == "success":
            # All returned intents should have confidence >= min_confidence
            for intent in response.intents:
                assert intent.confidence >= min_confidence, (
                    f"Intent with confidence {intent.confidence} should have "
                    f"been filtered out (min_confidence={min_confidence})"
                )

            # Count how many test intents should pass the filter
            expected_count = sum(
                1 for i in test_intents_in_db if i["confidence"] >= min_confidence
            )
            assert len(response.intents) == expected_count

    @pytest.mark.asyncio
    async def test_session_query_returns_intent_ids(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
        test_session_id: str,
    ) -> None:
        """Test session query returns valid intent IDs.

        Verify that the returned intent records have valid UUIDs that match
        the IDs returned during storage.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        if not test_intents_in_db:
            pytest.skip("No test intents were stored - storage may have failed")

        request = ModelIntentQueryRequestedEvent.create_session_query(
            session_ref=test_session_id,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "success"

        # All returned intents should have valid UUIDs
        for intent in response.intents:
            assert intent.intent_id is not None
            # UUID should be valid (not raise on str conversion)
            assert str(intent.intent_id)

        # The returned intent_ids should match what was stored
        response_ids = {intent.intent_id for intent in response.intents}
        stored_ids = {i["intent_id"] for i in test_intents_in_db}
        assert response_ids == stored_ids

    @pytest.mark.asyncio
    async def test_recent_query_with_limit(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
    ) -> None:
        """Test recent query respects limit parameter.

        When limit is less than the number of stored intents, only
        that many should be returned.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        if len(test_intents_in_db) < 2:
            pytest.skip("Need at least 2 intents for limit test")

        # Request fewer intents than we stored
        limit = 2
        request = ModelIntentQueryRequestedEvent.create_recent_query(
            time_range_hours=1,
            limit=limit,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "success"
        # Should not exceed the requested limit
        assert len(response.intents) <= limit

    @pytest.mark.asyncio
    async def test_intent_keywords_preserved(
        self,
        initialized_handler: HandlerIntentQuery,
        test_intents_in_db: list[dict[str, Any]],
        test_session_id: str,
    ) -> None:
        """Test that intent keywords are preserved through storage and retrieval.

        Keywords are an important part of intent classification and should
        be accurately stored and returned.
        """
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        if not test_intents_in_db:
            pytest.skip("No test intents were stored - storage may have failed")

        request = ModelIntentQueryRequestedEvent.create_session_query(
            session_ref=test_session_id,
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "success"

        # Build a map of category -> keywords from stored data
        stored_keywords: dict[str, list[str]] = {
            i["category"]: i["keywords"] for i in test_intents_in_db
        }

        # Verify keywords match for each returned intent
        for intent in response.intents:
            expected_keywords = stored_keywords.get(intent.intent_category)
            assert expected_keywords is not None
            assert set(intent.keywords) == set(expected_keywords), (
                f"Keywords mismatch for category '{intent.intent_category}': "
                f"expected {expected_keywords}, got {intent.keywords}"
            )


# =============================================================================
# Unit Tests (No Memgraph Required)
# =============================================================================


class TestHandlerIntentQueryUnit:
    """Unit tests that don't require Memgraph."""

    @pytest.mark.asyncio
    async def test_handler_not_initialized_returns_error(self) -> None:
        """Test execute before initialize returns error."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )
        from omnimemory.nodes.node_intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)
        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=1,
        )

        response = await handler.execute(request)

        assert response.status == "error"
        assert "not initialized" in (response.error_message or "").lower()

    def test_handler_config_none_before_initialize(self) -> None:
        """Test handler config is None before initialization."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        # Config is None before initialize
        assert handler.config is None

    def test_handler_has_container(self) -> None:
        """Test handler has container reference."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        assert handler.container is container

    def test_handler_not_initialized_by_default(self) -> None:
        """Test handler is not initialized by default."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        assert not handler.is_initialized

    @pytest.mark.asyncio
    async def test_handler_shutdown_idempotent(self) -> None:
        """Test shutdown can be called multiple times safely."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        # Should not raise
        await handler.shutdown()
        await handler.shutdown()

        assert not handler.is_initialized

    @pytest.mark.asyncio
    async def test_handler_describe_metadata(self) -> None:
        """Test describe returns handler metadata."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        metadata = await handler.describe()

        assert metadata.name == "HandlerIntentQuery"
        assert metadata.node_type == "EFFECT"
        assert "distribution" in metadata.supported_query_types
        assert "session" in metadata.supported_query_types
        assert "recent" in metadata.supported_query_types
        assert metadata.initialized is False

    @pytest.mark.asyncio
    async def test_handler_health_check_not_initialized(self) -> None:
        """Test health_check returns unhealthy when not initialized."""
        pytest.importorskip("omnimemory.nodes.node_intent_query_effect.handlers")
        from omnibase_core.container import ModelONEXContainer

        from omnimemory.nodes.node_intent_query_effect.handlers import (
            HandlerIntentQuery,
        )

        container = ModelONEXContainer()
        handler = HandlerIntentQuery(container)

        health = await handler.health_check()

        assert health.healthy is False
        assert health.initialized is False
        assert "not initialized" in (health.error_message or "").lower()


# Remove integration markers for unit tests
TestHandlerIntentQueryUnit.pytestmark = []  # type: ignore[attr-defined]
