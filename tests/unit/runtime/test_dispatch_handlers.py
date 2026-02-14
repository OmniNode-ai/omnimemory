# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for OmniMemory dispatch bridge handlers.

Validates:
    - Dispatch engine factory creates a frozen engine with correct routes/handlers
    - Bridge handler for intent-classified events delegates to consumer
    - Bridge handler for intent-query-requested events delegates to query handler
    - Lifecycle handler raises RuntimeError (fail-fast, not wired)
    - Event bus callback deserializes bytes, wraps in envelope, dispatches
    - Event bus callback acks on success, nacks on failure
    - Topic alias mapping is correct

Related:
    - OMN-2215: Phase 4 -- MessageDispatchEngine integration for omnimemory
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnimemory.runtime.dispatch_handlers import (
    DISPATCH_ALIAS_ARCHIVE_MEMORY,
    DISPATCH_ALIAS_EXPIRE_MEMORY,
    DISPATCH_ALIAS_INTENT_CLASSIFIED,
    DISPATCH_ALIAS_INTENT_QUERY_REQUESTED,
    DISPATCH_ALIAS_RUNTIME_TICK,
    create_dispatch_callback,
    create_intent_classified_dispatch_handler,
    create_intent_query_dispatch_handler,
    create_lifecycle_dispatch_handler,
    create_memory_dispatch_engine,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def correlation_id() -> UUID:
    """Fixed correlation ID for deterministic tests."""
    return UUID("12345678-1234-1234-1234-123456789abc")


@pytest.fixture
def mock_intent_consumer() -> MagicMock:
    """Mock ProtocolIntentEventConsumer for dispatch handler tests."""
    consumer = MagicMock()
    consumer._handle_message = AsyncMock(return_value=None)
    return consumer


@pytest.fixture
def mock_intent_query_handler() -> MagicMock:
    """Mock ProtocolIntentQueryHandler for dispatch handler tests."""
    handler = MagicMock()
    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(return_value={"status": "success"})
    handler.execute = AsyncMock(return_value=mock_response)
    return handler


@pytest.fixture
def sample_intent_classified_payload() -> dict[str, object]:
    """Sample intent-classified event payload."""
    return {
        "correlation_id": "12345678-1234-1234-1234-123456789abc",
        "session_id": "test-session-001",
        "intent_category": "debugging",
        "confidence": 0.92,
        "keywords": ["error", "traceback"],
        "secondary_intents": [],
        "timestamp_utc": "2025-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_intent_query_payload() -> dict[str, object]:
    """Sample intent-query-requested event payload."""
    return {
        "query_id": "00000000-0000-0000-0000-000000000001",
        "query_type": "distribution",
        "correlation_id": "12345678-1234-1234-1234-123456789abc",
    }


@dataclass
class _MockEventMessage:
    """Mock event bus message implementing ProtocolEventMessage interface."""

    topic: str = "onex.evt.omniintelligence.intent-classified.v1"
    key: bytes | None = None
    value: bytes = b"{}"
    headers: dict[str, str] = field(default_factory=dict)

    _acked: bool = False
    _nacked: bool = False

    async def ack(self) -> None:
        self._acked = True

    async def nack(self) -> None:
        self._nacked = True


# =============================================================================
# Tests: Topic Alias Constants
# =============================================================================


@pytest.mark.unit
class TestTopicAliases:
    """Verify topic alias constants are correctly formed."""

    def test_intent_classified_alias_contains_events_segment(self) -> None:
        """Dispatch alias must contain .events. for from_topic() to work."""
        assert ".events." in DISPATCH_ALIAS_INTENT_CLASSIFIED

    def test_intent_classified_alias_references_omniintelligence(self) -> None:
        """Intent classified alias must reference omniintelligence (source)."""
        assert "omniintelligence" in DISPATCH_ALIAS_INTENT_CLASSIFIED

    def test_intent_classified_alias_preserves_event_name(self) -> None:
        """Alias must preserve the intent-classified event name."""
        assert "intent-classified" in DISPATCH_ALIAS_INTENT_CLASSIFIED

    def test_intent_query_alias_contains_commands_segment(self) -> None:
        """Dispatch alias must contain .commands. for from_topic() to work."""
        assert ".commands." in DISPATCH_ALIAS_INTENT_QUERY_REQUESTED

    def test_intent_query_alias_references_omnimemory(self) -> None:
        """Intent query alias must reference omnimemory domain."""
        assert "omnimemory" in DISPATCH_ALIAS_INTENT_QUERY_REQUESTED

    def test_intent_query_alias_preserves_event_name(self) -> None:
        """Alias must preserve the intent-query-requested name."""
        assert "intent-query-requested" in DISPATCH_ALIAS_INTENT_QUERY_REQUESTED

    def test_archive_memory_alias_contains_commands_segment(self) -> None:
        """Archive memory alias must contain .commands. segment."""
        assert ".commands." in DISPATCH_ALIAS_ARCHIVE_MEMORY

    def test_expire_memory_alias_contains_commands_segment(self) -> None:
        """Expire memory alias must contain .commands. segment."""
        assert ".commands." in DISPATCH_ALIAS_EXPIRE_MEMORY

    def test_runtime_tick_alias_contains_commands_segment(self) -> None:
        """Runtime tick alias must contain .commands. for from_topic() to work."""
        assert ".commands." in DISPATCH_ALIAS_RUNTIME_TICK


# =============================================================================
# Tests: Dispatch Engine Factory
# =============================================================================


@pytest.mark.unit
class TestCreateMemoryDispatchEngine:
    """Validate dispatch engine creation and configuration."""

    def test_engine_is_frozen(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Engine must be frozen after factory call."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )
        assert engine.is_frozen

    def test_engine_has_four_handlers(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """All 4 memory domain handlers must be registered."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )
        assert engine.handler_count == 4

    def test_engine_has_six_routes(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """All 6 memory domain routes must be registered."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )
        assert engine.route_count == 6

    def test_engine_accepts_publish_topics(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Engine factory should accept publish_topics without error."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
            publish_topics={
                "intent_query": "dev.onex.evt.omnimemory.intent-query-response.v1",
            },
        )
        assert engine.is_frozen


# =============================================================================
# Tests: Intent Classified Bridge Handler
# =============================================================================


@pytest.mark.unit
class TestIntentClassifiedDispatchHandler:
    """Validate the bridge handler for intent-classified events."""

    @pytest.mark.asyncio
    async def test_handler_delegates_to_consumer(
        self,
        sample_intent_classified_payload: dict[str, object],
        correlation_id: UUID,
        mock_intent_consumer: MagicMock,
    ) -> None:
        """Handler should delegate dict payload to consumer._handle_message()."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        handler = create_intent_classified_dispatch_handler(
            consumer=mock_intent_consumer,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=sample_intent_classified_payload,
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "event"},
            ),
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        result = await handler(envelope, context)
        assert isinstance(result, str)
        mock_intent_consumer._handle_message.assert_called_once_with(
            sample_intent_classified_payload, retry_count=0
        )

    @pytest.mark.asyncio
    async def test_handler_raises_for_non_dict_payload(
        self,
        correlation_id: UUID,
        mock_intent_consumer: MagicMock,
    ) -> None:
        """Handler should raise ValueError for non-dict payloads."""
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        handler = create_intent_classified_dispatch_handler(
            consumer=mock_intent_consumer,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload="not a dict payload",
            correlation_id=correlation_id,
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        with pytest.raises(ValueError, match="Unexpected payload type"):
            await handler(envelope, context)

    @pytest.mark.asyncio
    async def test_handler_propagates_consumer_exception(
        self,
        sample_intent_classified_payload: dict[str, object],
        correlation_id: UUID,
        mock_intent_consumer: MagicMock,
    ) -> None:
        """Handler should propagate exceptions from consumer._handle_message()."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        mock_intent_consumer._handle_message = AsyncMock(
            side_effect=RuntimeError("Storage failure")
        )

        handler = create_intent_classified_dispatch_handler(
            consumer=mock_intent_consumer,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=sample_intent_classified_payload,
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "event"},
            ),
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        with pytest.raises(RuntimeError, match="Storage failure"):
            await handler(envelope, context)


# =============================================================================
# Tests: Intent Query Bridge Handler
# =============================================================================


@pytest.mark.unit
class TestIntentQueryDispatchHandler:
    """Validate the bridge handler for intent-query-requested events."""

    @pytest.mark.asyncio
    async def test_handler_delegates_to_query_handler(
        self,
        sample_intent_query_payload: dict[str, object],
        correlation_id: UUID,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Handler should parse payload and delegate to query_handler.execute()."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        handler = create_intent_query_dispatch_handler(
            query_handler=mock_intent_query_handler,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=sample_intent_query_payload,
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "command"},
            ),
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        result = await handler(envelope, context)
        assert isinstance(result, str)
        mock_intent_query_handler.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_raises_for_non_dict_payload(
        self,
        correlation_id: UUID,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Handler should raise ValueError for non-dict payloads."""
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        handler = create_intent_query_dispatch_handler(
            query_handler=mock_intent_query_handler,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=42,
            correlation_id=correlation_id,
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        with pytest.raises(ValueError, match="Unexpected payload type"):
            await handler(envelope, context)

    @pytest.mark.asyncio
    async def test_handler_publishes_response_when_configured(
        self,
        sample_intent_query_payload: dict[str, object],
        correlation_id: UUID,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Handler should publish response when callback and topic are configured."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        publish_callback = MagicMock()
        publish_topic = "dev.onex.evt.omnimemory.intent-query-response.v1"

        handler = create_intent_query_dispatch_handler(
            query_handler=mock_intent_query_handler,
            publish_callback=publish_callback,
            publish_topic=publish_topic,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=sample_intent_query_payload,
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "command"},
            ),
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        await handler(envelope, context)
        publish_callback.assert_called_once()
        call_args = publish_callback.call_args
        assert call_args[0][0] == publish_topic

    @pytest.mark.asyncio
    async def test_handler_raises_for_invalid_query_payload(
        self,
        correlation_id: UUID,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Handler should raise ValueError for dict payloads that fail validation."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        handler = create_intent_query_dispatch_handler(
            query_handler=mock_intent_query_handler,
            correlation_id=correlation_id,
        )

        # Missing required fields
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload={"invalid_field": "value"},
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "command"},
            ),
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        with pytest.raises(ValueError, match="Failed to parse payload"):
            await handler(envelope, context)

    @pytest.mark.asyncio
    async def test_handler_awaits_async_publish_callback(
        self,
        sample_intent_query_payload: dict[str, object],
        correlation_id: UUID,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Handler should await an async publish_callback without error."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        async_publish_callback = AsyncMock()
        publish_topic = "dev.onex.evt.omnimemory.intent-query-response.v1"

        handler = create_intent_query_dispatch_handler(
            query_handler=mock_intent_query_handler,
            publish_callback=async_publish_callback,
            publish_topic=publish_topic,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=sample_intent_query_payload,
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "command"},
            ),
        )
        context = ModelEffectContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        await handler(envelope, context)

        async_publish_callback.assert_awaited_once()
        call_args = async_publish_callback.call_args
        assert call_args[0][0] == publish_topic
        assert isinstance(call_args[0][1], dict)


# =============================================================================
# Tests: Lifecycle Dispatch Handler (fail-fast)
# =============================================================================


@pytest.mark.unit
class TestLifecycleDispatchHandler:
    """Validate the fail-fast lifecycle dispatch handler."""

    @pytest.mark.asyncio
    async def test_handler_raises_runtime_error(
        self,
        correlation_id: UUID,
    ) -> None:
        """Lifecycle handler must raise RuntimeError (not silently drop)."""
        from omnibase_core.models.core.model_envelope_metadata import (
            ModelEnvelopeMetadata,
        )
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )
        from omnibase_core.models.orchestrator.model_orchestrator_context import (
            ModelOrchestratorContext,
        )

        handler = create_lifecycle_dispatch_handler(
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload={"tick_type": "scheduled"},
            correlation_id=correlation_id,
            metadata=ModelEnvelopeMetadata(
                tags={"message_category": "command"},
            ),
        )
        context = ModelOrchestratorContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        with pytest.raises(RuntimeError, match="Lifecycle dispatch handler not wired"):
            await handler(envelope, context)

    @pytest.mark.asyncio
    async def test_handler_includes_correlation_id_in_error(
        self,
        correlation_id: UUID,
    ) -> None:
        """RuntimeError message must include correlation_id for debugging."""
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )
        from omnibase_core.models.orchestrator.model_orchestrator_context import (
            ModelOrchestratorContext,
        )

        handler = create_lifecycle_dispatch_handler(
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload={"action": "archive"},
            correlation_id=correlation_id,
        )
        context = ModelOrchestratorContext(
            correlation_id=correlation_id,
            envelope_id=uuid4(),
        )

        with pytest.raises(RuntimeError, match=str(correlation_id)):
            await handler(envelope, context)


# =============================================================================
# Tests: Event Bus Dispatch Callback
# =============================================================================


@pytest.mark.unit
class TestCreateDispatchCallback:
    """Validate the event bus callback that bridges to the dispatch engine."""

    @pytest.mark.asyncio
    async def test_callback_dispatches_json_message_and_acks(
        self,
        sample_intent_classified_payload: dict[str, object],
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should deserialize bytes, dispatch, and ack on success."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic=DISPATCH_ALIAS_INTENT_CLASSIFIED,
        )

        msg = _MockEventMessage(
            value=json.dumps(sample_intent_classified_payload).encode("utf-8"),
        )

        await callback(msg)

        assert msg._acked, "Message should be acked after successful dispatch"
        assert not msg._nacked

    @pytest.mark.asyncio
    async def test_callback_nacks_on_invalid_json(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should nack the message if JSON parsing fails."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic=DISPATCH_ALIAS_INTENT_CLASSIFIED,
        )

        msg = _MockEventMessage(
            value=b"not valid json {{{",
        )

        await callback(msg)

        assert msg._nacked, "Message should be nacked on parse failure"
        assert not msg._acked

    @pytest.mark.asyncio
    async def test_callback_handles_dict_message(
        self,
        sample_intent_classified_payload: dict[str, object],
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should handle plain dict messages (inmemory event bus)."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic=DISPATCH_ALIAS_INTENT_CLASSIFIED,
        )

        metrics_before = engine.get_structured_metrics()

        # InMemoryEventBus may pass dicts directly
        await callback(sample_intent_classified_payload)

        metrics_after = engine.get_structured_metrics()
        assert metrics_after.total_dispatches == metrics_before.total_dispatches + 1

    @pytest.mark.asyncio
    async def test_callback_extracts_correlation_id_from_payload(
        self,
        sample_intent_classified_payload: dict[str, object],
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should extract correlation_id from payload if present."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic=DISPATCH_ALIAS_INTENT_CLASSIFIED,
        )

        msg = _MockEventMessage(
            value=json.dumps(sample_intent_classified_payload).encode("utf-8"),
        )

        await callback(msg)
        assert msg._acked

    @pytest.mark.asyncio
    async def test_callback_nacks_on_dispatch_failure(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should nack when dispatch result indicates failure."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        # Use a topic with no matching route to trigger dispatch failure
        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic="onex.commands.nonexistent.topic.v1",
        )

        msg = _MockEventMessage(
            value=json.dumps(
                {
                    "query_type": "distribution",
                    "query_id": "test-query",
                }
            ).encode("utf-8"),
        )

        await callback(msg)

        assert msg._nacked, "Message should be nacked on dispatch failure"
        assert not msg._acked

    @pytest.mark.asyncio
    async def test_callback_nacks_on_unexpected_value_type(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should nack when message value is an unexpected type."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic=DISPATCH_ALIAS_INTENT_CLASSIFIED,
        )

        # Message with integer value (unexpected)
        msg = _MockEventMessage()
        msg.value = 12345  # type: ignore[assignment]

        await callback(msg)

        assert msg._nacked, "Message should be nacked for unexpected value type"
        assert not msg._acked

    @pytest.mark.asyncio
    async def test_callback_nacks_on_non_dict_json_payload(
        self,
        mock_intent_consumer: MagicMock,
        mock_intent_query_handler: MagicMock,
    ) -> None:
        """Callback should nack when JSON payload deserializes to a non-dict (e.g. array)."""
        engine = create_memory_dispatch_engine(
            intent_consumer=mock_intent_consumer,
            intent_query_handler=mock_intent_query_handler,
        )

        callback = create_dispatch_callback(
            engine=engine,
            dispatch_topic=DISPATCH_ALIAS_INTENT_CLASSIFIED,
        )

        # JSON array is valid JSON but not a valid dispatch payload
        msg = _MockEventMessage(
            value=json.dumps([1, 2, 3]).encode("utf-8"),
        )

        await callback(msg)

        assert msg._nacked, "Message should be nacked for non-dict JSON payload"
        assert not msg._acked
