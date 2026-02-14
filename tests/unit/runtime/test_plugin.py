# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for PluginMemory lifecycle and protocol compliance.

Validates:
    - PluginMemory satisfies ProtocolDomainPlugin (structural typing)
    - plugin_id and display_name properties return expected values
    - should_activate() checks OMNIMEMORY_ENABLED env var
    - initialize() returns success result
    - wire_handlers() verifies handler importability
    - wire_dispatchers() creates dispatch engine and stores introspection state
    - shutdown() cleans up resources, introspection state, and publishes shutdown
    - Concurrent shutdown is guarded by _shutdown_in_progress flag

Related:
    - OMN-2216: Phase 5 -- Runtime plugin PluginMemory
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnimemory.runtime.introspection import IntrospectionResult
from omnimemory.runtime.plugin import PluginMemory

from .conftest import StubConfig, StubEventBus

# =============================================================================
# Helpers
# =============================================================================


def _make_config(
    event_bus: object | None = None,
    correlation_id: object | None = None,
) -> object:
    """Create a minimal ModelDomainPluginConfig-compatible object."""
    bus = event_bus if event_bus is not None else StubEventBus()
    cid = correlation_id if correlation_id is not None else uuid4()
    return StubConfig(event_bus=bus, correlation_id=cid)


# =============================================================================
# Fixtures: Introspection mocking
# =============================================================================


@pytest.fixture
def mock_introspection(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Mock publish_memory_introspection to avoid heartbeat task leaks.

    Returns an IntrospectionResult with registered node names but no
    proxies (so no background heartbeat tasks are created). The mock
    is applied at the source module so that the local import inside
    wire_dispatchers picks it up.

    Returns:
        A list that records each call's keyword arguments for assertions.
    """
    calls: list[dict[str, Any]] = []

    async def _mock_publish(
        event_bus: object,
        *,
        correlation_id: UUID | None = None,
        enable_heartbeat: bool = True,
        heartbeat_interval_seconds: float = 30.0,
    ) -> IntrospectionResult:
        calls.append(
            {
                "event_bus": event_bus,
                "correlation_id": correlation_id,
                "enable_heartbeat": enable_heartbeat,
                "heartbeat_interval_seconds": heartbeat_interval_seconds,
            }
        )
        return IntrospectionResult(
            registered_nodes=["node_1", "node_2"],
            proxies=[],
        )

    monkeypatch.setattr(
        "omnimemory.runtime.introspection.publish_memory_introspection",
        _mock_publish,
    )
    return calls


@pytest.fixture
def mock_shutdown_introspection(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Mock publish_memory_shutdown to track shutdown calls.

    Returns:
        A list that records each call's keyword arguments for assertions.
    """
    calls: list[dict[str, Any]] = []

    async def _mock_shutdown(
        event_bus: object,
        *,
        proxies: list[object] | None = None,
        correlation_id: UUID | None = None,
    ) -> None:
        calls.append(
            {
                "event_bus": event_bus,
                "proxies": proxies,
                "correlation_id": correlation_id,
            }
        )

    monkeypatch.setattr(
        "omnimemory.runtime.introspection.publish_memory_shutdown",
        _mock_shutdown,
    )
    return calls


# =============================================================================
# Tests: Protocol compliance
# =============================================================================


class TestPluginProtocol:
    """Verify PluginMemory satisfies ProtocolDomainPlugin."""

    def test_satisfies_protocol(self) -> None:
        """PluginMemory should be recognized as ProtocolDomainPlugin."""
        from omnibase_infra.runtime.protocol_domain_plugin import (
            ProtocolDomainPlugin,
        )

        plugin = PluginMemory()
        assert isinstance(plugin, ProtocolDomainPlugin)

    def test_plugin_id(self) -> None:
        """plugin_id should return 'memory'."""
        plugin = PluginMemory()
        assert plugin.plugin_id == "memory"

    def test_display_name(self) -> None:
        """display_name should return 'Memory'."""
        plugin = PluginMemory()
        assert plugin.display_name == "Memory"


# =============================================================================
# Tests: should_activate
# =============================================================================


class TestPluginShouldActivate:
    """Validate should_activate checks OMNIMEMORY_ENABLED."""

    def test_inactive_without_env(self) -> None:
        """should_activate returns False when env var is not set."""
        plugin = PluginMemory()
        config = _make_config()
        with patch.dict("os.environ", {}, clear=True):
            assert plugin.should_activate(config) is False  # type: ignore[arg-type]

    def test_active_with_env(self) -> None:
        """should_activate returns True when OMNIMEMORY_ENABLED is set."""
        plugin = PluginMemory()
        config = _make_config()
        with patch.dict("os.environ", {"OMNIMEMORY_ENABLED": "true"}):
            assert plugin.should_activate(config) is True  # type: ignore[arg-type]

    def test_active_with_any_value(self) -> None:
        """Any truthy value for OMNIMEMORY_ENABLED activates the plugin."""
        plugin = PluginMemory()
        config = _make_config()
        with patch.dict("os.environ", {"OMNIMEMORY_ENABLED": "1"}):
            assert plugin.should_activate(config) is True  # type: ignore[arg-type]


# =============================================================================
# Tests: initialize
# =============================================================================


class TestPluginInitialize:
    """Validate initialize() returns success."""

    @pytest.mark.asyncio
    async def test_initialize_succeeds(self) -> None:
        """initialize should return a success result."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.initialize(config)  # type: ignore[arg-type]

        assert result.success
        assert result.plugin_id == "memory"
        assert result.duration_seconds >= 0.0


# =============================================================================
# Tests: wire_handlers
# =============================================================================


class TestPluginWireHandlers:
    """Validate wire_handlers() verifies handler importability."""

    @pytest.mark.asyncio
    async def test_wire_handlers_succeeds(self) -> None:
        """wire_handlers should return success with registered services."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.wire_handlers(config)  # type: ignore[arg-type]

        assert result.success
        assert len(result.services_registered) > 0
        assert "HandlerIntentEventConsumer" in result.services_registered
        assert "HandlerIntentQuery" in result.services_registered
        assert "HandlerSubscription" in result.services_registered

    @pytest.mark.asyncio
    async def test_wire_handlers_stores_services(self) -> None:
        """wire_handlers should store registered services in plugin state."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_handlers(config)  # type: ignore[arg-type]

        assert len(plugin._services_registered) > 0


# =============================================================================
# Tests: wire_dispatchers
# =============================================================================


class TestPluginWireDispatchers:
    """Validate wire_dispatchers() creates the dispatch engine and introspection state."""

    @pytest.mark.asyncio
    async def test_wire_dispatchers_creates_engine(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """wire_dispatchers should create and store a dispatch engine."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert result.success, f"wire_dispatchers failed: {result.error_message}"
        assert plugin._dispatch_engine is not None
        assert plugin._dispatch_engine.is_frozen

    @pytest.mark.asyncio
    async def test_wire_dispatchers_engine_has_six_routes(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """Engine should have exactly 6 routes (2 handler + 1 retrieval + 3 lifecycle)."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert plugin._dispatch_engine is not None
        assert plugin._dispatch_engine.route_count == 6

    @pytest.mark.asyncio
    async def test_wire_dispatchers_engine_has_four_handlers(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """Engine should have exactly 4 handlers."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert plugin._dispatch_engine is not None
        assert plugin._dispatch_engine.handler_count == 4

    @pytest.mark.asyncio
    async def test_wire_dispatchers_returns_resources_created(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """Result should list dispatch_engine and node_introspection in resources_created."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert "dispatch_engine" in result.resources_created
        assert "node_introspection" in result.resources_created

    @pytest.mark.asyncio
    async def test_wire_dispatchers_stores_event_bus(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """wire_dispatchers should store the event_bus reference for shutdown."""
        event_bus = StubEventBus()
        plugin = PluginMemory()
        config = _make_config(event_bus=event_bus)

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert plugin._event_bus is event_bus

    @pytest.mark.asyncio
    async def test_wire_dispatchers_stores_introspection_nodes(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """wire_dispatchers should populate _introspection_nodes from the result."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert plugin._introspection_nodes == ["node_1", "node_2"]

    @pytest.mark.asyncio
    async def test_wire_dispatchers_error_cleans_introspection_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If introspection publishing fails, state should be cleaned up.

        Simulates publish_memory_introspection raising an exception after
        the dispatch engine and event_bus have already been captured. The
        except block in wire_dispatchers should reset all state fields
        (event_bus, introspection_nodes, introspection_proxies,
        dispatch_engine) and reset the single-call guard.
        """

        async def _boom_publish(
            event_bus: object, **kwargs: Any
        ) -> IntrospectionResult:
            raise RuntimeError("simulated introspection failure")

        monkeypatch.setattr(
            "omnimemory.runtime.introspection.publish_memory_introspection",
            _boom_publish,
        )

        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert not result.success
        assert "simulated introspection failure" in (result.error_message or "")
        assert plugin._event_bus is None
        assert plugin._introspection_nodes == []
        assert plugin._introspection_proxies == []
        assert plugin._dispatch_engine is None


# =============================================================================
# Tests: start_consumers
# =============================================================================


class TestPluginStartConsumers:
    """Validate start_consumers subscribes to topics."""

    @pytest.mark.asyncio
    async def test_returns_skipped_without_engine(self) -> None:
        """Without wire_dispatchers, start_consumers should return skipped."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.start_consumers(config)  # type: ignore[arg-type]

        assert result.success
        assert "skipped" in result.message.lower()

    @pytest.mark.asyncio
    async def test_subscribes_to_all_topics(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """After wire_dispatchers, all topics should be subscribed."""
        from omnimemory.runtime.plugin import MEMORY_SUBSCRIBE_TOPICS

        event_bus = StubEventBus()
        plugin = PluginMemory()
        config = _make_config(event_bus=event_bus)

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]
        result = await plugin.start_consumers(config)  # type: ignore[arg-type]

        assert result.success
        assert len(event_bus.subscriptions) == len(MEMORY_SUBSCRIBE_TOPICS)

    @pytest.mark.asyncio
    async def test_no_subscriptions_without_engine(self) -> None:
        """Without wire_dispatchers, no topics should be subscribed."""
        event_bus = StubEventBus()
        plugin = PluginMemory()
        config = _make_config(event_bus=event_bus)

        await plugin.start_consumers(config)  # type: ignore[arg-type]

        assert len(event_bus.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_all_topics_use_dispatch_callback(
        self, mock_introspection: list[dict[str, Any]]
    ) -> None:
        """All subscribed topics should use dispatch callback (not noop)."""

        event_bus = StubEventBus()
        plugin = PluginMemory()
        config = _make_config(event_bus=event_bus)

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]
        await plugin.start_consumers(config)  # type: ignore[arg-type]

        for sub in event_bus.subscriptions:
            handler = sub["on_message"]
            assert handler is not None, f"Topic {sub['topic']} has no handler"
            assert callable(handler), f"Topic {sub['topic']} handler not callable"


# =============================================================================
# Tests: shutdown
# =============================================================================


class TestPluginShutdown:
    """Validate shutdown() cleans up resources and introspection state."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_engine(
        self,
        mock_introspection: list[dict[str, Any]],
        mock_shutdown_introspection: list[dict[str, Any]],
    ) -> None:
        """After shutdown, _dispatch_engine and introspection state should be None/empty."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]
        assert plugin._dispatch_engine is not None
        assert plugin._event_bus is not None
        assert len(plugin._introspection_nodes) > 0

        await plugin.shutdown(config)  # type: ignore[arg-type]
        assert plugin._dispatch_engine is None
        assert plugin._event_bus is None
        assert plugin._introspection_nodes == []
        assert plugin._introspection_proxies == []

    @pytest.mark.asyncio
    async def test_shutdown_clears_services(self) -> None:
        """After shutdown, _services_registered should be empty."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_handlers(config)  # type: ignore[arg-type]
        assert len(plugin._services_registered) > 0

        await plugin.shutdown(config)  # type: ignore[arg-type]
        assert len(plugin._services_registered) == 0

    @pytest.mark.asyncio
    async def test_shutdown_returns_success(self) -> None:
        """Shutdown should return success result."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.shutdown(config)  # type: ignore[arg-type]

        assert result.success
        assert result.plugin_id == "memory"

    @pytest.mark.asyncio
    async def test_concurrent_shutdown_skipped(self) -> None:
        """Second concurrent shutdown call should be skipped."""
        plugin = PluginMemory()
        config = _make_config()

        # Simulate concurrent shutdown by setting the flag
        plugin._shutdown_in_progress = True

        result = await plugin.shutdown(config)  # type: ignore[arg-type]

        assert result.success
        assert "skipped" in result.message.lower()

    @pytest.mark.asyncio
    async def test_shutdown_publishes_shutdown_introspection(
        self,
        mock_introspection: list[dict[str, Any]],
        mock_shutdown_introspection: list[dict[str, Any]],
    ) -> None:
        """Shutdown should call publish_memory_shutdown when event_bus is set."""
        event_bus = StubEventBus()
        plugin = PluginMemory()
        config = _make_config(event_bus=event_bus)

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]
        assert plugin._event_bus is event_bus

        await plugin.shutdown(config)  # type: ignore[arg-type]

        assert len(mock_shutdown_introspection) == 1
        call = mock_shutdown_introspection[0]
        assert call["event_bus"] is event_bus

    @pytest.mark.asyncio
    async def test_shutdown_skips_introspection_without_event_bus(
        self,
        mock_shutdown_introspection: list[dict[str, Any]],
    ) -> None:
        """Shutdown should not call publish_memory_shutdown when event_bus is None."""
        plugin = PluginMemory()
        config = _make_config()

        # Do not call wire_dispatchers, so _event_bus remains None
        assert plugin._event_bus is None

        await plugin.shutdown(config)  # type: ignore[arg-type]

        assert len(mock_shutdown_introspection) == 0


# =============================================================================
# Tests: get_status_line
# =============================================================================


class TestPluginStatusLine:
    """Validate get_status_line() output."""

    def test_disabled_without_env(self) -> None:
        """Status should be 'disabled' when env var is not set."""
        plugin = PluginMemory()
        with patch.dict("os.environ", {}, clear=True):
            assert plugin.get_status_line() == "disabled"

    def test_enabled_with_env(self) -> None:
        """Status should indicate 'enabled' with topic count."""
        plugin = PluginMemory()
        with patch.dict("os.environ", {"OMNIMEMORY_ENABLED": "true"}):
            status = plugin.get_status_line()
            assert status.startswith("enabled")
            assert "topics" in status
