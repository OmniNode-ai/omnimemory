# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for PluginMemory lifecycle and protocol compliance.

Validates:
    - PluginMemory satisfies ProtocolDomainPlugin (structural typing)
    - plugin_id and display_name properties return expected values
    - should_activate() checks OMNIMEMORY_ENABLED env var
    - initialize() returns success result
    - wire_handlers() verifies handler importability
    - shutdown() cleans up resources and clears state
    - Concurrent shutdown is guarded by _shutdown_in_progress flag

Related:
    - OMN-2216: Phase 5 -- Runtime plugin PluginMemory
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

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
    """Validate wire_dispatchers() creates the dispatch engine."""

    @pytest.mark.asyncio
    async def test_wire_dispatchers_creates_engine(self) -> None:
        """wire_dispatchers should create and store a dispatch engine."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert result.success, f"wire_dispatchers failed: {result.error_message}"
        assert plugin._dispatch_engine is not None
        assert plugin._dispatch_engine.is_frozen

    @pytest.mark.asyncio
    async def test_wire_dispatchers_engine_has_six_routes(self) -> None:
        """Engine should have exactly 6 routes (2 handler + 1 retrieval + 3 lifecycle)."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert plugin._dispatch_engine is not None
        assert plugin._dispatch_engine.route_count == 6

    @pytest.mark.asyncio
    async def test_wire_dispatchers_engine_has_four_handlers(self) -> None:
        """Engine should have exactly 4 handlers."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert plugin._dispatch_engine is not None
        assert plugin._dispatch_engine.handler_count == 4

    @pytest.mark.asyncio
    async def test_wire_dispatchers_returns_resources_created(self) -> None:
        """Result should list dispatch_engine in resources_created."""
        plugin = PluginMemory()
        config = _make_config()

        result = await plugin.wire_dispatchers(config)  # type: ignore[arg-type]

        assert "dispatch_engine" in result.resources_created


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
    async def test_subscribes_to_all_topics(self) -> None:
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
    async def test_all_topics_use_dispatch_callback(self) -> None:
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
    """Validate shutdown() cleans up resources."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_engine(self) -> None:
        """After shutdown, _dispatch_engine should be None."""
        plugin = PluginMemory()
        config = _make_config()

        await plugin.wire_dispatchers(config)  # type: ignore[arg-type]
        assert plugin._dispatch_engine is not None

        await plugin.shutdown(config)  # type: ignore[arg-type]
        assert plugin._dispatch_engine is None

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
