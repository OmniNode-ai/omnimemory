# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Shared test fixtures for runtime unit tests.

Provides stub implementations of ModelDomainPluginConfig and related
types used across plugin and wiring tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from omnimemory.runtime.message_type_registration import _reset_for_testing


@pytest.fixture(autouse=True)
def _reset_observability_globals() -> Iterator[None]:
    """Reset module-level observability state before each test.

    Without this reset the globals (_registry_ready, _registered_count,
    _registration_failure_count) persist across tests.  Under parallel
    execution (pytest-xdist) or reordered collection this causes flaky
    assertions that depend on "clean slate" state.
    """
    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture(autouse=True)
async def _reset_introspection_guard() -> AsyncIterator[None]:
    """Reset the introspection single-call guard before and after each test.

    ``publish_memory_introspection()`` sets a module-level
    ``_introspection_published`` guard that prevents it from being called
    twice in the same process.  Without resetting this guard, the first
    test that triggers ``wire_dispatchers()`` (which calls
    ``publish_memory_introspection``) permanently blocks all subsequent
    tests from calling it again, causing ``RuntimeError``.

    Resetting both before **and** after ensures clean state regardless of
    test ordering or whether a previous test failed mid-execution.
    """
    from omnimemory.runtime.introspection import reset_introspection_guard

    await reset_introspection_guard()
    yield
    await reset_introspection_guard()


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
