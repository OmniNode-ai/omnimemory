# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol adapters for omnimemory handler dependencies.

Bridges available infrastructure (event bus, handlers) to the protocol
interfaces expected by omnimemory domain handlers.

Adapters:
    - AdapterKafkaPublisher: event bus -> ProtocolEventBusPublish
      Wraps a ProtocolEventBusPublish-conforming event bus into the
      higher-level publish interface used by HandlerSubscription.

Design:
    Each adapter is a thin explicit boundary that prevents accidental coupling
    to infrastructure-specific methods. Protocol conformance is verified in
    tests, not at import time (avoids import-time landmines with optional deps).

    ARCH-002 Compliance:
        No handler or node file may import directly from aiokafka,
        omnibase_infra.event_bus, or other transport modules. All event bus
        interaction goes through the ProtocolEventBusPublish protocol defined
        here. Concrete wiring happens in the runtime layer only.

Related:
    - OMN-2214: Phase 3 -- ARCH-002 compliance, abstract Kafka from handlers
    - omniintelligence runtime/adapters.py (reference implementation)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definitions
# =============================================================================


@runtime_checkable
class ProtocolEventBusPublish(Protocol):
    """Minimal protocol for event bus publish capability.

    Matches the publish signature used by EventBusKafka and EventBusInmemory.
    Handlers depend on this protocol, never on concrete implementations.

    Example::

        class MyPublisher:
            def __init__(self, bus: ProtocolEventBusPublish) -> None:
                self._bus = bus

            async def emit(self, topic: str, data: bytes) -> None:
                await self._bus.publish(topic=topic, key=None, value=data)
    """

    async def publish(self, *, topic: str, key: bytes | None, value: bytes) -> None:
        """Publish raw bytes to a topic.

        Args:
            topic: Target topic name.
            key: Optional message key for partitioning.
            value: Message payload as bytes.
        """
        ...


@runtime_checkable
class ProtocolEventBusHealthCheck(Protocol):
    """Minimal protocol for event bus health check capability.

    Matches the health_check signature used by EventBusKafka.
    """

    async def health_check(self) -> dict[str, object]:
        """Return health status.

        Returns:
            Dict with at minimum a "healthy" key (bool).
        """
        ...


@runtime_checkable
class ProtocolEventBusLifecycle(Protocol):
    """Minimal protocol for event bus lifecycle (initialize/shutdown).

    Matches the initialize/shutdown signatures used by EventBusKafka.
    """

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the event bus with configuration.

        Args:
            config: Configuration dictionary.
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus."""
        ...


# =============================================================================
# AdapterKafkaPublisher
# =============================================================================


class AdapterKafkaPublisher:
    """Wraps a ProtocolEventBusPublish event bus for handler-level publishing.

    Normalizes the handler's publish interface (topic, key: str, value: dict)
    to the event bus protocol interface (topic, key: bytes, value: bytes).

    Serialization:
        - key: UTF-8 encoded bytes
        - value: compact JSON bytes (no whitespace, non-ASCII preserved)

    Example::

        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        bus = EventBusKafka(config=config)
        await bus.start()

        publisher = AdapterKafkaPublisher(bus)
        await publisher.publish("my.topic", "key", {"data": "value"})
    """

    __slots__ = ("_event_bus",)

    def __init__(self, event_bus: ProtocolEventBusPublish) -> None:
        self._event_bus = event_bus

    async def publish(
        self,
        topic: str,
        key: str,
        value: Mapping[str, object],
        headers: dict[str, str] | None = None,
    ) -> None:
        """Publish event to topic via event bus protocol.

        Args:
            topic: Target topic name.
            key: Message key (for partitioning).
            value: Event payload dict (serialized to JSON bytes).
            headers: Optional headers (logged but not passed to raw publish;
                extend ProtocolEventBusPublish if header support is required).
        """
        value_bytes = json.dumps(
            value, separators=(",", ":"), ensure_ascii=False, default=str
        ).encode("utf-8")
        key_bytes = key.encode("utf-8") if key else None

        if headers:
            logger.debug(
                "Publishing to %s with headers: %s",
                topic,
                list(headers.keys()),
            )

        await self._event_bus.publish(topic=topic, key=key_bytes, value=value_bytes)


# =============================================================================
# Factory Functions
# =============================================================================


async def create_default_event_bus(
    bootstrap_servers: str = "localhost:9092",
) -> ProtocolEventBusPublish:
    """Create and initialize a default event bus instance.

    This factory centralizes the concrete transport construction so that
    handler and node files never import from omnibase_infra.event_bus
    directly (ARCH-002 compliance).

    Args:
        bootstrap_servers: Event bus bootstrap servers (comma-separated).

    Returns:
        An initialized event bus conforming to ProtocolEventBusPublish.
        The returned object also satisfies ProtocolEventBusLifecycle and
        ProtocolEventBusHealthCheck.

    Raises:
        RuntimeError: If the event bus cannot be initialized.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

    event_bus = EventBusKafka()
    try:
        await event_bus.initialize({"bootstrap_servers": bootstrap_servers})
    except Exception as e:
        raise RuntimeError(f"Failed to initialize event bus: {e}") from e
    # EventBusKafka satisfies ProtocolEventBusPublish (has async publish method)
    # but mypy cannot verify cross-package structural subtyping.
    return cast(ProtocolEventBusPublish, event_bus)


__all__ = [
    "AdapterKafkaPublisher",
    "ProtocolEventBusHealthCheck",
    "ProtocolEventBusLifecycle",
    "ProtocolEventBusPublish",
    "create_default_event_bus",
]
