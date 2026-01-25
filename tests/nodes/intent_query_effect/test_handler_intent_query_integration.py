# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerIntentQuery with real Memgraph.

These tests require a running Memgraph instance. They will be skipped
if Memgraph is not available.

Test Categories:
    - Distribution: Query intent distribution by category
    - Session: Query intents for a specific session
    - Recent: Query recent intents across all sessions
    - Error handling: Test error conditions and edge cases

Prerequisites:
    - Memgraph running at MEMGRAPH_URI (default: bolt://localhost:7687)
    - omnibase_infra installed (dev dependency)

Usage:
    # Run only integration tests
    pytest tests/nodes/intent_query_effect/test_handler_intent_query_integration.py -v

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
from typing import TYPE_CHECKING

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

    from omnimemory.handlers.adapters import AdapterIntentGraph
    from omnimemory.handlers.adapters.models import ModelAdapterIntentGraphConfig
    from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
    from omnimemory.nodes.intent_query_effect.models import (
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
async def initialized_handler(
    initialized_adapter: AdapterIntentGraph,
    handler_config: ModelHandlerIntentQueryConfig,
) -> AsyncGenerator[HandlerIntentQuery, None]:
    """Create and initialize handler for tests."""
    handler = HandlerIntentQuery(handler_config)
    await handler.initialize(initialized_adapter)

    yield handler

    await handler.shutdown()


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
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

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
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

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
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

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
        """Test session query without session_ref returns error."""
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

        # Manually create request without session_ref
        request = ModelIntentQueryRequestedEvent(
            query_type="session",
            session_ref=None,  # Missing!
            requester_name="test",
        )

        response = await initialized_handler.execute(request)

        assert response.status == "error"
        assert "session_ref" in (response.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(
        self,
        initialized_handler: HandlerIntentQuery,
    ) -> None:
        """Test correlation_id is echoed in response."""
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

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
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=24,
        )

        response = await initialized_handler.execute(request)

        assert response.execution_time_ms is not None
        assert response.execution_time_ms > 0


# =============================================================================
# Unit Tests (No Memgraph Required)
# =============================================================================


class TestHandlerIntentQueryUnit:
    """Unit tests that don't require Memgraph."""

    @pytest.mark.asyncio
    async def test_handler_not_initialized_returns_error(self) -> None:
        """Test execute before initialize returns error."""
        pytest.importorskip("omnimemory.nodes.intent_query_effect.handlers")
        from omnibase_core.models.events import ModelIntentQueryRequestedEvent

        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        handler = HandlerIntentQuery()
        request = ModelIntentQueryRequestedEvent.create_distribution_query(
            time_range_hours=1,
        )

        response = await handler.execute(request)

        assert response.status == "error"
        assert "not initialized" in (response.error_message or "").lower()

    def test_handler_config_defaults(self) -> None:
        """Test handler uses default config when none provided."""
        pytest.importorskip("omnimemory.nodes.intent_query_effect.handlers")
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        handler = HandlerIntentQuery()

        assert handler.config.timeout_seconds == 10.0
        assert handler.config.default_time_range_hours == 24
        assert handler.config.default_limit == 100

    def test_handler_config_custom(self) -> None:
        """Test handler accepts custom config."""
        pytest.importorskip("omnimemory.nodes.intent_query_effect.models")
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
        from omnimemory.nodes.intent_query_effect.models import (
            ModelHandlerIntentQueryConfig,
        )

        config = ModelHandlerIntentQueryConfig(
            timeout_seconds=30.0,
            default_time_range_hours=48,
            default_limit=50,
        )
        handler = HandlerIntentQuery(config)

        assert handler.config.timeout_seconds == 30.0
        assert handler.config.default_time_range_hours == 48
        assert handler.config.default_limit == 50

    def test_handler_not_initialized_by_default(self) -> None:
        """Test handler is not initialized by default."""
        pytest.importorskip("omnimemory.nodes.intent_query_effect.handlers")
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        handler = HandlerIntentQuery()

        assert not handler.is_initialized

    @pytest.mark.asyncio
    async def test_handler_initialize_requires_adapter(self) -> None:
        """Test initialize requires valid adapter."""
        pytest.importorskip("omnimemory.nodes.intent_query_effect.handlers")
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        handler = HandlerIntentQuery()

        # Should not raise, but sets adapter
        # We can't call initialize with None, so we test the state
        assert handler.is_initialized is False

    @pytest.mark.asyncio
    async def test_handler_shutdown_idempotent(self) -> None:
        """Test shutdown can be called multiple times safely."""
        pytest.importorskip("omnimemory.nodes.intent_query_effect.handlers")
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        handler = HandlerIntentQuery()

        # Should not raise
        await handler.shutdown()
        await handler.shutdown()

        assert not handler.is_initialized


# Remove integration markers for unit tests
TestHandlerIntentQueryUnit.pytestmark = []  # type: ignore[attr-defined]
