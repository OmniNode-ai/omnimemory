# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for runtime protocol adapters.

Validates:
    - ProtocolEventBusPublish conformance (runtime_checkable isinstance checks)
    - ProtocolEventBusHealthCheck conformance
    - ProtocolEventBusLifecycle conformance
    - AdapterKafkaPublisher serialization and delegation
    - create_default_event_bus factory return type

Related:
    - OMN-2214: Phase 3 -- ARCH-002 compliance, abstract Kafka from handlers
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimemory.runtime.adapters import (
    AdapterKafkaPublisher,
    ProtocolEventBusHealthCheck,
    ProtocolEventBusLifecycle,
    ProtocolEventBusPublish,
    create_default_event_bus,
)

# =============================================================================
# Helpers: Stub classes for protocol conformance testing
# =============================================================================


class _StubPublishOnly:
    """Stub that implements only the publish protocol."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        pass


class _StubHealthCheckOnly:
    """Stub that implements only the health_check protocol."""

    async def health_check(self) -> dict[str, object]:
        return {"healthy": True}


class _StubLifecycleOnly:
    """Stub that implements only the lifecycle protocol."""

    async def initialize(self, config: dict[str, object]) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _StubFullBus:
    """Stub implementing all three protocols (publish, health, lifecycle)."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        pass

    async def health_check(self) -> dict[str, object]:
        return {"healthy": True}

    async def initialize(self, config: dict[str, object]) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _StubNotABus:
    """Stub that does NOT implement any event bus protocol."""

    async def do_something(self) -> None:
        pass


# =============================================================================
# Tests: ProtocolEventBusPublish conformance
# =============================================================================


class TestProtocolEventBusPublishConformance:
    """Validate ProtocolEventBusPublish runtime_checkable isinstance checks."""

    def test_stub_publish_satisfies_protocol(self) -> None:
        """Class with matching publish() signature satisfies the protocol."""
        bus = _StubPublishOnly()
        assert isinstance(bus, ProtocolEventBusPublish)

    def test_full_bus_satisfies_publish_protocol(self) -> None:
        """Class with all protocols still satisfies publish protocol."""
        bus = _StubFullBus()
        assert isinstance(bus, ProtocolEventBusPublish)

    def test_non_bus_does_not_satisfy_publish_protocol(self) -> None:
        """Class without publish() does not satisfy the protocol."""
        obj = _StubNotABus()
        assert not isinstance(obj, ProtocolEventBusPublish)

    def test_health_only_does_not_satisfy_publish_protocol(self) -> None:
        """Class with only health_check() does not satisfy publish protocol."""
        obj = _StubHealthCheckOnly()
        assert not isinstance(obj, ProtocolEventBusPublish)

    @pytest.mark.asyncio
    async def test_publish_accepts_positional_args(self) -> None:
        """Protocol publish() must accept positional arguments (not keyword-only).

        This verifies the fix for the signature mismatch where the protocol
        used keyword-only args (``*,``) but EventBusKafka uses positional args.
        """
        bus = _StubPublishOnly()
        # This should not raise - positional args must be accepted
        await bus.publish("topic", None, b"value")


# =============================================================================
# Tests: ProtocolEventBusHealthCheck conformance
# =============================================================================


class TestProtocolEventBusHealthCheckConformance:
    """Validate ProtocolEventBusHealthCheck runtime_checkable isinstance checks."""

    def test_stub_health_satisfies_protocol(self) -> None:
        """Class with matching health_check() satisfies the protocol."""
        bus = _StubHealthCheckOnly()
        assert isinstance(bus, ProtocolEventBusHealthCheck)

    def test_full_bus_satisfies_health_protocol(self) -> None:
        """Class with all protocols still satisfies health check protocol."""
        bus = _StubFullBus()
        assert isinstance(bus, ProtocolEventBusHealthCheck)

    def test_publish_only_does_not_satisfy_health_protocol(self) -> None:
        """Class with only publish() does not satisfy health check protocol."""
        obj = _StubPublishOnly()
        assert not isinstance(obj, ProtocolEventBusHealthCheck)

    def test_non_bus_does_not_satisfy_health_protocol(self) -> None:
        """Class without health_check() does not satisfy the protocol."""
        obj = _StubNotABus()
        assert not isinstance(obj, ProtocolEventBusHealthCheck)


# =============================================================================
# Tests: ProtocolEventBusLifecycle conformance
# =============================================================================


class TestProtocolEventBusLifecycleConformance:
    """Validate ProtocolEventBusLifecycle runtime_checkable isinstance checks."""

    def test_stub_lifecycle_satisfies_protocol(self) -> None:
        """Class with matching initialize()/shutdown() satisfies the protocol."""
        bus = _StubLifecycleOnly()
        assert isinstance(bus, ProtocolEventBusLifecycle)

    def test_full_bus_satisfies_lifecycle_protocol(self) -> None:
        """Class with all protocols still satisfies lifecycle protocol."""
        bus = _StubFullBus()
        assert isinstance(bus, ProtocolEventBusLifecycle)

    def test_publish_only_does_not_satisfy_lifecycle_protocol(self) -> None:
        """Class with only publish() does not satisfy lifecycle protocol."""
        obj = _StubPublishOnly()
        assert not isinstance(obj, ProtocolEventBusLifecycle)

    def test_non_bus_does_not_satisfy_lifecycle_protocol(self) -> None:
        """Class without initialize()/shutdown() does not satisfy the protocol."""
        obj = _StubNotABus()
        assert not isinstance(obj, ProtocolEventBusLifecycle)


# =============================================================================
# Tests: AdapterKafkaPublisher delegation
# =============================================================================


class TestAdapterKafkaPublisherDelegation:
    """Validate that AdapterKafkaPublisher delegates to the underlying bus."""

    @pytest.fixture
    def mock_bus(self) -> AsyncMock:
        """Create a mock event bus with publish method."""
        bus = AsyncMock(spec=_StubPublishOnly)
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def publisher(self, mock_bus: AsyncMock) -> AdapterKafkaPublisher:
        """Create an AdapterKafkaPublisher wrapping the mock bus."""
        return AdapterKafkaPublisher(mock_bus)

    @pytest.mark.asyncio
    async def test_publish_delegates_to_bus(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must call the underlying bus publish method."""
        await publisher.publish("test.topic", "my-key", {"data": "value"})
        mock_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_passes_correct_topic(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must pass the topic unchanged to the bus."""
        await publisher.publish("my.topic.name", "key", {"x": 1})
        call_kwargs = mock_bus.publish.call_args.kwargs
        assert call_kwargs["topic"] == "my.topic.name"

    @pytest.mark.asyncio
    async def test_publish_encodes_key_to_bytes(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must encode string key to UTF-8 bytes."""
        await publisher.publish("topic", "my-key", {"x": 1})
        call_kwargs = mock_bus.publish.call_args.kwargs
        assert call_kwargs["key"] == b"my-key"

    @pytest.mark.asyncio
    async def test_publish_none_key_passes_none(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must pass None key as None (not b'None')."""
        await publisher.publish("topic", None, {"x": 1})
        call_kwargs = mock_bus.publish.call_args.kwargs
        assert call_kwargs["key"] is None

    @pytest.mark.asyncio
    async def test_publish_serializes_dict_value_to_json_bytes(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must serialize dict value to compact JSON bytes."""
        import json

        payload = {"action": "created", "entity_id": "abc-123"}
        await publisher.publish("topic", "key", payload)
        call_kwargs = mock_bus.publish.call_args.kwargs
        value_bytes: bytes = call_kwargs["value"]

        # Must be bytes
        assert isinstance(value_bytes, bytes)

        # Must be valid JSON
        decoded = json.loads(value_bytes)
        assert decoded == payload

    @pytest.mark.asyncio
    async def test_publish_uses_compact_json_separators(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must use compact separators (no whitespace)."""
        await publisher.publish("topic", "key", {"a": 1, "b": 2})
        call_kwargs = mock_bus.publish.call_args.kwargs
        value_str = call_kwargs["value"].decode("utf-8")
        # Compact JSON has no spaces after : or ,
        assert " " not in value_str

    @pytest.mark.asyncio
    async def test_publish_empty_string_key_encodes_to_empty_bytes(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must encode empty string key as b'' (not None)."""
        await publisher.publish("topic", "", {"x": 1})
        call_kwargs = mock_bus.publish.call_args.kwargs
        assert call_kwargs["key"] == b""

    @pytest.mark.asyncio
    async def test_publish_propagates_bus_exception(
        self,
        publisher: AdapterKafkaPublisher,
        mock_bus: AsyncMock,
    ) -> None:
        """publish() must propagate exceptions from the underlying bus."""
        mock_bus.publish.side_effect = ConnectionError("bus unavailable")
        with pytest.raises(ConnectionError, match="bus unavailable"):
            await publisher.publish("topic", "key", {"x": 1})


# =============================================================================
# Tests: create_default_event_bus factory
# =============================================================================


class TestCreateDefaultEventBus:
    """Validate create_default_event_bus factory behavior.

    These tests mock the omnibase_infra imports since the actual Kafka
    infrastructure is not available in unit tests. The factory uses
    local imports so patches target the source modules.
    """

    @pytest.mark.asyncio
    async def test_factory_returns_protocol_conforming_object(self) -> None:
        """create_default_event_bus must return ProtocolEventBusPublish."""
        mock_config_cls = MagicMock()
        mock_bus_instance = AsyncMock(spec=_StubFullBus)
        mock_bus_instance.start = AsyncMock()
        mock_bus_instance.publish = AsyncMock()
        mock_bus_cls = MagicMock(return_value=mock_bus_instance)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.EventBusKafka",
                mock_bus_cls,
            ),
            patch(
                "omnibase_infra.event_bus.models.config.ModelKafkaEventBusConfig",
                mock_config_cls,
            ),
        ):
            result = await create_default_event_bus(
                bootstrap_servers="localhost:9092",
            )

        # Result must not be None
        assert result is not None

    @pytest.mark.asyncio
    async def test_factory_calls_start_on_bus(self) -> None:
        """create_default_event_bus must call start() on the bus."""
        mock_config_cls = MagicMock()
        mock_bus_instance = AsyncMock(spec=_StubFullBus)
        mock_bus_instance.start = AsyncMock()
        mock_bus_cls = MagicMock(return_value=mock_bus_instance)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.EventBusKafka",
                mock_bus_cls,
            ),
            patch(
                "omnibase_infra.event_bus.models.config.ModelKafkaEventBusConfig",
                mock_config_cls,
            ),
        ):
            await create_default_event_bus(
                bootstrap_servers="localhost:9092",
            )

        mock_bus_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_factory_raises_runtime_error_on_start_failure(self) -> None:
        """create_default_event_bus must raise RuntimeError if start() fails."""
        mock_config_cls = MagicMock()
        mock_bus_instance = AsyncMock(spec=_StubFullBus)
        mock_bus_instance.start = AsyncMock(side_effect=ConnectionError("broker down"))
        mock_bus_cls = MagicMock(return_value=mock_bus_instance)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.EventBusKafka",
                mock_bus_cls,
            ),
            patch(
                "omnibase_infra.event_bus.models.config.ModelKafkaEventBusConfig",
                mock_config_cls,
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to initialize event bus"):
                await create_default_event_bus(
                    bootstrap_servers="localhost:9092",
                )
