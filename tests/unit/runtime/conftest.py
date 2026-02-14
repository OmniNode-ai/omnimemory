# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared test fixtures for runtime unit tests.

Provides stub implementations of ModelDomainPluginConfig and related
types used across plugin and wiring tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class StubContainer:
    """Minimal container stub for plugin config."""

    service_registry: object = None


class StubEventBus:
    """Event bus stub that tracks subscriptions."""

    def __init__(self) -> None:
        self.subscriptions: list[dict[str, object]] = []

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

    async def publish(
        self,
        topic: str = "",
        key: bytes | None = None,
        value: bytes = b"",
    ) -> None:
        pass


@dataclass
class StubConfig:
    """Minimal ModelDomainPluginConfig-compatible stub."""

    container: object = field(default_factory=StubContainer)
    event_bus: object = field(default_factory=lambda: StubEventBus())
    correlation_id: object = field(default_factory=uuid4)
    input_topic: str = "test.input"
    output_topic: str = "test.output"
    consumer_group: str = "test-consumer"
    dispatch_engine: object = None
    node_identity: object = None
    kafka_bootstrap_servers: str | None = None
