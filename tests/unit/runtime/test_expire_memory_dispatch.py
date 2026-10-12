# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""An expire-memory command reaches HandlerMemoryExpire and its outcome is published.

Every test drives the runtime's own entry point: a raw bus message goes through
``create_dispatch_callback`` into the ``MessageDispatchEngine`` built by
``create_memory_dispatch_engine``, exactly as ``PluginMemory.start_consumers``
wires it. The database is a fake asyncpg pool; ``HandlerMemoryExpire`` is the
real handler.

Contract under test:
    - ACTIVE entity -> exactly one ``memory-expired.v1`` carrying the entity id
      and no ``lifecycle-transition-failed.v1``
    - Entity that is not ACTIVE, a revision conflict, a malformed command or a
      missing handler -> one ``lifecycle-transition-failed.v1`` and no
      ``memory-expired.v1``
    - A command is never dropped silently: either an outcome event is published
      or the message is nacked.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from omnibase_core.container import ModelONEXContainer

from omnimemory.enums import EnumLifecycleState
from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
    HandlerMemoryExpire,
)
from omnimemory.runtime.dispatch_handlers import (
    DISPATCH_ALIAS_EXPIRE_MEMORY,
    create_dispatch_callback,
    create_memory_dispatch_engine,
)
from omnimemory.topics import EnumMemoryEventTopic

if TYPE_CHECKING:
    from asyncpg import Pool

MEMORY_EXPIRED_TOPIC = "onex.evt.omnimemory.memory-expired.v1"
TRANSITION_FAILED_TOPIC = "onex.evt.omnimemory.lifecycle-transition-failed.v1"

SYNTHETIC_REASON = "synthetic-lifecycle-proof"


@dataclass
class _BusMessage:
    """Raw bus message as the event bus hands it to the subscribed callback."""

    value: bytes
    acked: bool = False
    nacked: bool = False

    async def ack(self) -> None:
        self.acked = True

    async def nack(self) -> None:
        self.nacked = True


@dataclass
class _Published:
    """Collects everything the dispatch handler publishes to the bus."""

    events: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    async def __call__(self, topic: str, value: dict[str, object]) -> None:
        self.events.append((topic, value))

    def on(self, topic: str) -> list[dict[str, object]]:
        return [value for t, value in self.events if t == topic]


class _FakePool:
    """asyncpg pool double: ``fetchrow`` replays the scripted rows in order."""

    def __init__(self, rows: list[dict[str, object] | None]) -> None:
        self.conn = MagicMock()
        self.conn.fetchrow = AsyncMock(side_effect=rows)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[MagicMock]:
        yield self.conn


async def _expire_handler(rows: list[dict[str, object] | None]) -> HandlerMemoryExpire:
    handler = HandlerMemoryExpire(MagicMock(spec=ModelONEXContainer))
    await handler.initialize(db_pool=cast("Pool", _FakePool(rows)))
    return handler


def _state_row(
    memory_id: UUID, state: EnumLifecycleState, revision: int
) -> dict[str, object]:
    return {
        "id": memory_id,
        "lifecycle_state": state.value,
        "lifecycle_revision": revision,
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


def _command_message(memory_id: UUID, **extra: object) -> _BusMessage:
    body: dict[str, object] = {
        "memory_id": str(memory_id),
        "expected_revision": 1,
        "reason": SYNTHETIC_REASON,
        **extra,
    }
    return _BusMessage(value=json.dumps(body).encode("utf-8"))


async def _deliver(
    message: _BusMessage,
    *,
    published: _Published,
    expire_handler: HandlerMemoryExpire | None,
) -> None:
    engine = create_memory_dispatch_engine(
        intent_consumer=MagicMock(),
        intent_query_handler=MagicMock(),
        publish_callback=published,
        expire_handler=expire_handler,
    )
    callback = create_dispatch_callback(
        engine=engine,
        dispatch_topic=DISPATCH_ALIAS_EXPIRE_MEMORY,
    )
    await callback(message)


@pytest.mark.unit
class TestExpireMemoryOutcomeTopics:
    """The outcome topics are the contract's, not strings invented in a handler."""

    def test_enum_carries_the_contract_topics(self) -> None:
        assert EnumMemoryEventTopic.MEMORY_EXPIRED.value == MEMORY_EXPIRED_TOPIC
        assert (
            EnumMemoryEventTopic.LIFECYCLE_TRANSITION_FAILED.value
            == TRANSITION_FAILED_TOPIC
        )


@pytest.mark.unit
class TestExpireActiveMemory:
    """AC2: an ACTIVE entity yields exactly one memory-expired event."""

    @pytest.mark.asyncio
    async def test_active_entity_yields_one_memory_expired_with_its_id(self) -> None:
        memory_id = uuid4()
        published = _Published()
        handler = await _expire_handler([{"lifecycle_revision": 2}])
        message = _command_message(memory_id)

        await _deliver(message, published=published, expire_handler=handler)

        expired = published.on(MEMORY_EXPIRED_TOPIC)
        assert len(expired) == 1
        assert expired[0]["entity_id"] == str(memory_id)
        assert expired[0]["memory_id"] == str(memory_id)
        assert expired[0]["lifecycle_revision"] == 2
        assert published.on(TRANSITION_FAILED_TOPIC) == []
        assert message.acked
        assert not message.nacked

    @pytest.mark.asyncio
    async def test_command_envelope_keys_do_not_reject_the_command(self) -> None:
        """`onex kafka produce --envelope` flattens its keys into the payload."""
        memory_id = uuid4()
        published = _Published()
        handler = await _expire_handler([{"lifecycle_revision": 2}])
        message = _command_message(
            memory_id,
            correlation_id=str(uuid4()),
            requested_by="onex-cli",
            timestamp="2026-10-12T11:55:37+00:00",
        )

        await _deliver(message, published=published, expire_handler=handler)

        assert len(published.on(MEMORY_EXPIRED_TOPIC)) == 1
        assert published.on(TRANSITION_FAILED_TOPIC) == []


@pytest.mark.unit
class TestExpireRefusals:
    """AC3: a refused expire yields lifecycle-transition-failed and no memory-expired."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state",
        [
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.STALE,
            EnumLifecycleState.ARCHIVED,
            EnumLifecycleState.DELETED,
        ],
    )
    async def test_entity_not_active_yields_transition_failed_only(
        self, state: EnumLifecycleState
    ) -> None:
        memory_id = uuid4()
        published = _Published()
        # UPDATE ... WHERE lifecycle_state='active' matches no row; the diagnostic
        # read then shows the entity at the expected revision in a non-ACTIVE state.
        handler = await _expire_handler([None, _state_row(memory_id, state, 1)])
        message = _command_message(memory_id)

        await _deliver(message, published=published, expire_handler=handler)

        assert published.on(MEMORY_EXPIRED_TOPIC) == []
        failed = published.on(TRANSITION_FAILED_TOPIC)
        assert len(failed) == 1
        assert failed[0]["memory_id"] == str(memory_id)
        assert failed[0]["operation"] == "expire"
        assert failed[0]["previous_state"] == state.value
        assert failed[0]["conflict"] is False
        assert message.acked

    @pytest.mark.asyncio
    async def test_unknown_entity_yields_transition_failed_only(self) -> None:
        memory_id = uuid4()
        published = _Published()
        handler = await _expire_handler([None, None])

        await _deliver(
            _command_message(memory_id), published=published, expire_handler=handler
        )

        assert published.on(MEMORY_EXPIRED_TOPIC) == []
        failed = published.on(TRANSITION_FAILED_TOPIC)
        assert len(failed) == 1
        assert failed[0]["reason"] == "Memory not found"

    @pytest.mark.asyncio
    async def test_revision_conflict_yields_transition_failed_with_conflict(
        self,
    ) -> None:
        memory_id = uuid4()
        published = _Published()
        handler = await _expire_handler(
            [None, _state_row(memory_id, EnumLifecycleState.ACTIVE, 7)]
        )

        await _deliver(
            _command_message(memory_id), published=published, expire_handler=handler
        )

        assert published.on(MEMORY_EXPIRED_TOPIC) == []
        failed = published.on(TRANSITION_FAILED_TOPIC)
        assert len(failed) == 1
        assert failed[0]["conflict"] is True

    @pytest.mark.asyncio
    async def test_no_expire_handler_yields_transition_failed_not_silence(
        self,
    ) -> None:
        memory_id = uuid4()
        published = _Published()
        message = _command_message(memory_id)

        await _deliver(message, published=published, expire_handler=None)

        assert published.on(MEMORY_EXPIRED_TOPIC) == []
        failed = published.on(TRANSITION_FAILED_TOPIC)
        assert len(failed) == 1
        assert failed[0]["memory_id"] == str(memory_id)
        assert "not configured" in cast("str", failed[0]["reason"])
        assert message.acked

    @pytest.mark.asyncio
    async def test_malformed_command_yields_transition_failed_without_entity(
        self,
    ) -> None:
        published = _Published()
        handler = await _expire_handler([])
        message = _BusMessage(value=json.dumps({"reason": "no memory id"}).encode())

        await _deliver(message, published=published, expire_handler=handler)

        assert published.on(MEMORY_EXPIRED_TOPIC) == []
        failed = published.on(TRANSITION_FAILED_TOPIC)
        assert len(failed) == 1
        assert failed[0]["memory_id"] is None
        assert message.acked


@pytest.mark.unit
class TestExpireIsNeverDroppedSilently:
    """A lost outcome event nacks the message instead of acking a silent drop."""

    @pytest.mark.asyncio
    async def test_publish_failure_nacks_the_command(self) -> None:
        memory_id = uuid4()
        handler = await _expire_handler([{"lifecycle_revision": 2}])
        publish = AsyncMock(side_effect=RuntimeError("broker down"))
        engine = create_memory_dispatch_engine(
            intent_consumer=MagicMock(),
            intent_query_handler=MagicMock(),
            publish_callback=publish,
            expire_handler=handler,
        )
        callback = create_dispatch_callback(
            engine=engine, dispatch_topic=DISPATCH_ALIAS_EXPIRE_MEMORY
        )
        message = _command_message(memory_id)

        await callback(message)

        assert message.nacked
        assert not message.acked

    @pytest.mark.asyncio
    async def test_no_publisher_nacks_the_command(self) -> None:
        memory_id = uuid4()
        handler = await _expire_handler([{"lifecycle_revision": 2}])
        engine = create_memory_dispatch_engine(
            intent_consumer=MagicMock(),
            intent_query_handler=MagicMock(),
            expire_handler=handler,
        )
        callback = create_dispatch_callback(
            engine=engine, dispatch_topic=DISPATCH_ALIAS_EXPIRE_MEMORY
        )
        message = _command_message(memory_id)

        await callback(message)

        assert message.nacked
        assert not message.acked


@pytest.mark.unit
class TestLifecycleBridgeDoesNotTreatExpireAsShutdown:
    """The tick/archive bridge once mapped an expire-memory command to shutdown."""

    @pytest.mark.asyncio
    async def test_expire_command_does_not_stop_the_lifecycle_subsystem(self) -> None:
        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
        from omnibase_core.models.orchestrator.model_orchestrator_context import (
            ModelOrchestratorContext,
        )

        from omnimemory.runtime.dispatch_handlers import (
            _create_lifecycle_bridge_handler,
        )
        from omnimemory.runtime.handler_lifecycle import HandlerMemoryLifecycle

        lifecycle = HandlerMemoryLifecycle()
        await lifecycle.handle_startup()
        bridge = _create_lifecycle_bridge_handler(lifecycle=lifecycle)
        correlation_id = uuid4()
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload={"command": "expire-memory"},
            correlation_id=correlation_id,
        )
        context = ModelOrchestratorContext(
            correlation_id=correlation_id, envelope_id=uuid4()
        )

        await bridge(envelope, context)

        assert lifecycle.is_started()
