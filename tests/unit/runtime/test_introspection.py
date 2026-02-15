# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for memory node introspection registration.

Validates:
    - MEMORY_NODES structure: count, types, uniqueness, determinism
    - _NodeDescriptor: uuid5 generation, version defaults
    - MemoryNodeIntrospectionProxy: creation and name property
    - publish_memory_introspection: single-call guard, no-op with None bus
    - publish_memory_shutdown: proxy cleanup and guard reset
    - reset_introspection_guard: allows re-invocation after reset

Related:
    - OMN-2216: Phase 5 -- Wire memory nodes into registration + introspection
"""

from __future__ import annotations

from uuid import NAMESPACE_DNS, uuid5

import pytest
from omnibase_core.enums import EnumNodeKind

from omnimemory.runtime.introspection import (
    MEMORY_NODES,
    IntrospectionResult,
    MemoryNodeIntrospectionProxy,
    _NodeDescriptor,
    publish_memory_introspection,
    publish_memory_shutdown,
    reset_introspection_guard,
)

from .conftest import StubEventBus

# =============================================================================
# Helpers
# =============================================================================


class MockEventBus:
    """Event bus mock that records publish calls for assertion.

    Provides both ``publish`` and ``subscribe`` to satisfy the
    ProtocolEventBus interface used by MixinNodeIntrospection.
    """

    def __init__(self) -> None:
        self.published: list[dict[str, object]] = []
        self.subscriptions: list[dict[str, object]] = []

    async def publish(
        self,
        topic: str = "",
        value: bytes = b"",
        key: bytes | None = None,
    ) -> None:
        self.published.append({"topic": topic, "value": value, "key": key})

    async def subscribe(
        self,
        topic: str = "",
        group_id: str = "",
        on_message: object = None,
        **kwargs: object,
    ) -> object:
        self.subscriptions.append(
            {"topic": topic, "group_id": group_id, "on_message": on_message}
        )

        async def _unsub() -> None:
            pass

        return _unsub


# =============================================================================
# Tests: MEMORY_NODES structure
# =============================================================================


class TestMemoryNodesStructure:
    """Validate the MEMORY_NODES tuple contents and invariants."""

    @pytest.mark.unit
    def test_has_exactly_nine_nodes(self) -> None:
        """MEMORY_NODES should contain exactly 9 node descriptors."""
        assert len(MEMORY_NODES) == 9

    @pytest.mark.unit
    def test_contains_two_orchestrators(self) -> None:
        """MEMORY_NODES should contain exactly 2 orchestrator nodes."""
        orchestrators = [
            n for n in MEMORY_NODES if n.node_type == EnumNodeKind.ORCHESTRATOR
        ]
        assert len(orchestrators) == 2

    @pytest.mark.unit
    def test_contains_two_compute_nodes(self) -> None:
        """MEMORY_NODES should contain exactly 2 compute nodes."""
        compute_nodes = [n for n in MEMORY_NODES if n.node_type == EnumNodeKind.COMPUTE]
        assert len(compute_nodes) == 2

    @pytest.mark.unit
    def test_contains_five_effect_nodes(self) -> None:
        """MEMORY_NODES should contain exactly 5 effect nodes."""
        effect_nodes = [n for n in MEMORY_NODES if n.node_type == EnumNodeKind.EFFECT]
        assert len(effect_nodes) == 5

    @pytest.mark.unit
    def test_all_names_are_unique(self) -> None:
        """All node names in MEMORY_NODES should be unique."""
        names = [n.name for n in MEMORY_NODES]
        assert len(names) == len(
            set(names)
        ), f"Duplicate names found: {[n for n in names if names.count(n) > 1]}"

    @pytest.mark.unit
    def test_all_node_ids_are_unique(self) -> None:
        """All node_ids in MEMORY_NODES should be unique (deterministic UUID5)."""
        node_ids = [n.node_id for n in MEMORY_NODES]
        assert len(node_ids) == len(set(node_ids)), "Duplicate node_ids found"

    @pytest.mark.unit
    def test_node_id_determinism(self) -> None:
        """Same node name should always produce the same node_id."""
        for descriptor in MEMORY_NODES:
            first_call = descriptor.node_id
            second_call = descriptor.node_id
            assert (
                first_call == second_call
            ), f"Non-deterministic node_id for {descriptor.name}"


# =============================================================================
# Tests: _NodeDescriptor
# =============================================================================


class TestNodeDescriptor:
    """Validate _NodeDescriptor initialization and properties."""

    @pytest.mark.unit
    def test_node_id_uses_uuid5_with_omnimemory_prefix(self) -> None:
        """node_id should be uuid5(NAMESPACE_DNS, 'omnimemory.<name>')."""
        descriptor = _NodeDescriptor("test_node", EnumNodeKind.EFFECT)
        expected = uuid5(NAMESPACE_DNS, "omnimemory.test_node")
        assert descriptor.node_id == expected

    @pytest.mark.unit
    def test_version_defaults_to_1_0_0(self) -> None:
        """Version should default to '1.0.0' when not specified."""
        descriptor = _NodeDescriptor("some_node", EnumNodeKind.COMPUTE)
        assert descriptor.version == "1.0.0"

    @pytest.mark.unit
    def test_version_can_be_overridden(self) -> None:
        """Version can be explicitly set to a custom value."""
        descriptor = _NodeDescriptor(
            "versioned_node", EnumNodeKind.ORCHESTRATOR, version="2.3.1"
        )
        assert descriptor.version == "2.3.1"

    @pytest.mark.unit
    def test_name_and_node_type_stored(self) -> None:
        """Name and node_type should be stored as provided."""
        descriptor = _NodeDescriptor("my_effect", EnumNodeKind.EFFECT)
        assert descriptor.name == "my_effect"
        assert descriptor.node_type == EnumNodeKind.EFFECT

    @pytest.mark.unit
    def test_different_names_produce_different_node_ids(self) -> None:
        """Distinct node names should produce distinct node_ids."""
        desc_a = _NodeDescriptor("node_a", EnumNodeKind.EFFECT)
        desc_b = _NodeDescriptor("node_b", EnumNodeKind.EFFECT)
        assert desc_a.node_id != desc_b.node_id


# =============================================================================
# Tests: MemoryNodeIntrospectionProxy
# =============================================================================


class TestMemoryNodeIntrospectionProxy:
    """Validate proxy creation and name property."""

    @pytest.mark.unit
    def test_can_create_with_none_event_bus(self) -> None:
        """Proxy should be creatable with event_bus=None."""
        descriptor = _NodeDescriptor("proxy_test", EnumNodeKind.EFFECT)
        proxy = MemoryNodeIntrospectionProxy(descriptor=descriptor, event_bus=None)
        assert proxy is not None

    @pytest.mark.unit
    def test_name_returns_descriptor_name(self) -> None:
        """name property should return the descriptor's name."""
        descriptor = _NodeDescriptor("my_proxy_node", EnumNodeKind.COMPUTE)
        proxy = MemoryNodeIntrospectionProxy(descriptor=descriptor, event_bus=None)
        assert proxy.name == "my_proxy_node"

    @pytest.mark.unit
    def test_can_create_with_event_bus(self) -> None:
        """Proxy should be creatable with a real event bus stub."""
        descriptor = _NodeDescriptor("bus_proxy", EnumNodeKind.EFFECT)
        bus = StubEventBus()
        proxy = MemoryNodeIntrospectionProxy(
            descriptor=descriptor,
            event_bus=bus,  # type: ignore[arg-type]
        )
        assert proxy.name == "bus_proxy"


# =============================================================================
# Tests: publish_memory_introspection
# =============================================================================


class TestPublishMemoryIntrospection:
    """Validate the publish_memory_introspection function behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_none_bus_returns_empty_result(self) -> None:
        """With event_bus=None, should return an empty IntrospectionResult."""
        result = await publish_memory_introspection(event_bus=None)
        assert isinstance(result, IntrospectionResult)
        assert result.registered_nodes == []
        assert result.proxies == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_none_bus_does_not_set_guard(self) -> None:
        """With event_bus=None, the guard should NOT be set.

        A subsequent call with a real bus must still succeed.
        """
        await publish_memory_introspection(event_bus=None)
        # If guard were set, this would raise RuntimeError
        await publish_memory_introspection(event_bus=None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_none_bus_multiple_calls_no_error(self) -> None:
        """Calling with event_bus=None multiple times should never raise."""
        for _ in range(5):
            result = await publish_memory_introspection(event_bus=None)
            assert result.registered_nodes == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_mock_bus_returns_registered_nodes(self) -> None:
        """With a mock event bus, should return IntrospectionResult with nodes."""
        bus = MockEventBus()
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert isinstance(result, IntrospectionResult)
        assert len(result.registered_nodes) == len(MEMORY_NODES)
        for descriptor in MEMORY_NODES:
            assert descriptor.name in result.registered_nodes

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_mock_bus_publishes_events(self) -> None:
        """With a mock event bus, introspection events should be published."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        # At least one publish call per node
        assert len(bus.published) >= len(MEMORY_NODES)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_second_call_raises_runtime_error(self) -> None:
        """Calling twice with a real event bus should raise RuntimeError."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        with pytest.raises(RuntimeError, match="already been called"):
            await publish_memory_introspection(
                event_bus=bus,  # type: ignore[arg-type]
                enable_heartbeat=False,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_after_reset_can_call_again(self) -> None:
        """After reset_introspection_guard, publish should succeed again."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        await reset_introspection_guard()
        # Should not raise
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result.registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_none_bus_then_real_bus_succeeds(self) -> None:
        """Calling with None bus first, then real bus, should succeed."""
        await publish_memory_introspection(event_bus=None)
        bus = MockEventBus()
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result.registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_heartbeat_disabled_returns_no_proxies(self) -> None:
        """With enable_heartbeat=False, proxies list should be empty."""
        bus = MockEventBus()
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert result.proxies == []


# =============================================================================
# Tests: publish_memory_shutdown
# =============================================================================


class TestPublishMemoryShutdown:
    """Validate the publish_memory_shutdown function behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stops_introspection_tasks_on_proxies(self) -> None:
        """shutdown should call stop_introspection_tasks on provided proxies."""
        stopped: list[str] = []

        class FakeProxy:
            def __init__(self, name: str) -> None:
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            async def stop_introspection_tasks(self) -> None:
                stopped.append(self._name)

        proxies = [FakeProxy("proxy_a"), FakeProxy("proxy_b")]
        await publish_memory_shutdown(
            event_bus=None,
            proxies=proxies,  # type: ignore[arg-type]
        )
        assert "proxy_a" in stopped
        assert "proxy_b" in stopped

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resets_guard_after_shutdown(self) -> None:
        """After shutdown, the guard should be reset."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        # Guard is now set; calling again would raise
        await publish_memory_shutdown(
            event_bus=bus,  # type: ignore[arg-type]
        )
        # Guard should be reset; calling again should succeed
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result.registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_none_bus_stops_tasks_and_resets_guard(self) -> None:
        """With event_bus=None, shutdown stops tasks and resets guard."""
        bus = MockEventBus()
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        # Guard is set
        await publish_memory_shutdown(
            event_bus=None,
            proxies=result.proxies,
        )
        # Guard should be reset even with None bus
        result2 = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result2.registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_real_bus_publishes_shutdown_events(self) -> None:
        """With a real bus, shutdown should publish SHUTDOWN events."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        published_before = len(bus.published)
        await publish_memory_shutdown(
            event_bus=bus,  # type: ignore[arg-type]
        )
        # Should have published additional shutdown events
        assert len(bus.published) > published_before

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_with_no_proxies(self) -> None:
        """Shutdown with proxies=None should not raise."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        # Should not raise
        await publish_memory_shutdown(
            event_bus=bus,  # type: ignore[arg-type]
            proxies=None,
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_tolerates_proxy_stop_error(self) -> None:
        """Shutdown should continue even if a proxy's stop_introspection_tasks fails."""

        class FailingProxy:
            @property
            def name(self) -> str:
                return "failing_proxy"

            async def stop_introspection_tasks(self) -> None:
                raise RuntimeError("stop failed")

        # Should not raise despite the proxy error
        await publish_memory_shutdown(
            event_bus=None,
            proxies=[FailingProxy()],  # type: ignore[arg-type]
        )


# =============================================================================
# Tests: reset_introspection_guard
# =============================================================================


class TestResetIntrospectionGuard:
    """Validate the reset_introspection_guard function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_allows_republish_after_reset(self) -> None:
        """After reset, publish_memory_introspection can be called again."""
        bus = MockEventBus()
        await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        await reset_introspection_guard()
        # Should not raise
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result.registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_resets_are_safe(self) -> None:
        """Calling reset_introspection_guard multiple times should not raise."""
        for _ in range(10):
            await reset_introspection_guard()


# =============================================================================
# Tests: IntrospectionResult
# =============================================================================


class TestIntrospectionResult:
    """Validate IntrospectionResult dataclass."""

    @pytest.mark.unit
    def test_default_empty_lists(self) -> None:
        """Default IntrospectionResult should have empty lists."""
        result = IntrospectionResult()
        assert result.registered_nodes == []
        assert result.proxies == []

    @pytest.mark.unit
    def test_fields_are_mutable(self) -> None:
        """registered_nodes and proxies should be appendable."""
        result = IntrospectionResult()
        result.registered_nodes.append("test_node")
        assert "test_node" in result.registered_nodes


# =============================================================================
# Tests: Introspection Lifecycle Integration
# =============================================================================


class TestIntrospectionLifecycle:
    """Integration-level tests for the full introspection lifecycle.

    Validates end-to-end flows including publish -> shutdown -> re-publish,
    concurrent publish guards (TOCTOU fix), and failure-retry behavior.
    """

    @pytest.mark.unit
    async def test_full_lifecycle(self) -> None:
        """Publish -> verify nodes -> shutdown -> cleanup -> re-publish succeeds.

        Exercises the full plugin lifecycle: initial introspection publish
        registers all nodes, shutdown cleans up proxies and resets the guard,
        and a subsequent publish succeeds (proving the guard was reset by
        shutdown).
        """
        bus = MockEventBus()

        # Phase 1: Publish introspection
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result.registered_nodes) == len(MEMORY_NODES)
        for descriptor in MEMORY_NODES:
            assert descriptor.name in result.registered_nodes
        assert len(bus.published) >= len(MEMORY_NODES)

        # Phase 2: Shutdown -- resets guard and publishes SHUTDOWN events
        published_before_shutdown = len(bus.published)
        await publish_memory_shutdown(
            event_bus=bus,  # type: ignore[arg-type]
            proxies=result.proxies,
        )
        # Shutdown should have published additional events
        assert len(bus.published) > published_before_shutdown

        # Phase 3: Re-publish succeeds (guard was reset by shutdown)
        result2 = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result2.registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    async def test_concurrent_publish_only_one_succeeds(self) -> None:
        """Two concurrent publishes: exactly one succeeds, the other raises.

        Validates the TOCTOU fix: the guard is set atomically inside the
        asyncio.Lock, so even with concurrent coroutines, only one can
        proceed past the guard check.
        """
        import asyncio

        bus = MockEventBus()

        results = await asyncio.gather(
            publish_memory_introspection(
                event_bus=bus,  # type: ignore[arg-type]
                enable_heartbeat=False,
            ),
            publish_memory_introspection(
                event_bus=bus,  # type: ignore[arg-type]
                enable_heartbeat=False,
            ),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, IntrospectionResult)]
        failures = [r for r in results if isinstance(r, RuntimeError)]

        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert (
            len(failures) == 1
        ), f"Expected exactly 1 RuntimeError, got {len(failures)}"
        assert "already been called" in str(failures[0])

        # The successful result should have all nodes registered
        assert len(successes[0].registered_nodes) == len(MEMORY_NODES)

    @pytest.mark.unit
    async def test_publish_failure_allows_retry(self) -> None:
        """When publish fails mid-execution, the guard is reset for retry.

        Validates the try/except guard-reset path: if the publish loop
        raises an unrecoverable exception, ``_introspection_published``
        is reset to ``False`` so a subsequent call can succeed instead of
        being permanently blocked.
        """
        from unittest.mock import patch

        class _FailingIterable:
            """Iterable that raises on iteration to trigger outer except."""

            def __iter__(self) -> None:  # type: ignore[override]
                raise RuntimeError("simulated iteration failure")

        bus = MockEventBus()

        # Patch MEMORY_NODES to an iterable that raises, triggering the
        # outer except block which resets the guard.
        with patch(
            "omnimemory.runtime.introspection.MEMORY_NODES",
            _FailingIterable(),
        ):
            with pytest.raises(RuntimeError, match="simulated iteration failure"):
                await publish_memory_introspection(
                    event_bus=bus,  # type: ignore[arg-type]
                    enable_heartbeat=False,
                )

        # Guard should have been reset by the except block.
        # A retry with the real MEMORY_NODES should succeed.
        result = await publish_memory_introspection(
            event_bus=bus,  # type: ignore[arg-type]
            enable_heartbeat=False,
        )
        assert len(result.registered_nodes) == len(MEMORY_NODES)
