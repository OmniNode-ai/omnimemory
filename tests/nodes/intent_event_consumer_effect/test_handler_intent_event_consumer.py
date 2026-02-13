# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerIntentEventConsumer.

These tests verify the handler's integration behavior including:
- Initialization lifecycle
- Message processing with mocked storage adapter
- Circuit breaker behavior
- DLQ routing
- Health check semantics
- Retry logic with exponential backoff

All tests use mocked dependencies (no real Kafka/Memgraph).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnimemory.models.utils.model_health_status import HealthStatus
from omnimemory.nodes.intent_event_consumer_effect.handler_intent_event_consumer import (
    HandlerIntentEventConsumer,
)
from omnimemory.nodes.intent_event_consumer_effect.models import (
    ModelIntentEventConsumerConfig,
)
from omnimemory.nodes.intent_storage_effect.models.model_intent_storage_response import (
    ModelIntentStorageResponse,
)
from omnimemory.utils.concurrency import CircuitBreakerState

# Type aliases for Kafka message handling (avoiding Any per zero-Any policy)
type MessagePayload = dict[str, object]
type PublishCall = tuple[str, MessagePayload]
type SubscriptionEntry = tuple[str, object]


def create_valid_message(
    session_id: str = "test-session",
    intent_category: str = "debugging",
    confidence: float = 0.85,
    correlation_id: UUID | None = None,
) -> MessagePayload:
    """Create a valid intent-classified event message for testing.

    The model has strict=True, so we need to provide properly typed values
    that will pass Pydantic's strict validation mode.

    Args:
        session_id: Session identifier.
        intent_category: Classified intent category.
        confidence: Classification confidence score.
        correlation_id: Optional correlation ID, generated if not provided.

    Returns:
        Dictionary representing a valid Kafka message payload.
    """
    return {
        "event_type": "IntentClassified",
        "session_id": session_id,
        "correlation_id": correlation_id or uuid4(),
        "intent_category": intent_category,
        "confidence": confidence,
        "keywords": ["test", "keyword"],
        "timestamp": datetime.now(timezone.utc),
    }


def create_mock_storage_adapter(
    response: ModelIntentStorageResponse | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock storage adapter.

    Args:
        response: Optional response to return from execute().
        side_effect: Optional exception to raise from execute().

    Returns:
        MagicMock configured as a storage adapter.
    """
    mock = MagicMock()
    if side_effect:
        mock.execute = AsyncMock(side_effect=side_effect)
    else:
        default_response = response or ModelIntentStorageResponse(
            status="success",
            intent_id=uuid4(),
            created=True,
        )
        mock.execute = AsyncMock(return_value=default_response)
    return mock


def create_mock_subscribe() -> tuple[MagicMock, list[SubscriptionEntry]]:
    """Create a mock subscribe callback.

    Returns:
        Tuple of (subscribe function, list to capture subscriptions).
    """
    subscriptions: list[SubscriptionEntry] = []

    def subscribe(topic: str, handler: object) -> MagicMock:
        subscriptions.append((topic, handler))
        return MagicMock()  # unsubscribe function

    return MagicMock(side_effect=subscribe), subscriptions


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for consumer initialization lifecycle."""

    def test_consumer_not_initialized_before_initialize(self) -> None:
        """Consumer should report not initialized before initialize() is called."""
        config = ModelIntentEventConsumerConfig()
        storage = create_mock_storage_adapter()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)

        assert consumer.is_initialized is False

    @pytest.mark.asyncio
    async def test_consumer_initialized_after_initialize(self) -> None:
        """Consumer should report initialized after initialize() is called."""
        config = ModelIntentEventConsumerConfig()
        storage = create_mock_storage_adapter()
        subscribe_fn, subscriptions = create_mock_subscribe()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        assert consumer.is_initialized is True
        assert len(subscriptions) == 1
        assert (
            subscriptions[0][0] == "test.onex.evt.omniintelligence.intent-classified.v1"
        )

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """Calling initialize() twice should not break or re-subscribe."""
        config = ModelIntentEventConsumerConfig()
        storage = create_mock_storage_adapter()
        subscribe_fn, subscriptions = create_mock_subscribe()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)

        # First initialization
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")
        assert len(subscriptions) == 1

        # Second initialization should be a no-op
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")
        assert len(subscriptions) == 1  # Still 1, not 2
        assert consumer.is_initialized is True


# =============================================================================
# Message Processing Tests
# =============================================================================


class TestMessageProcessing:
    """Tests for message processing behavior."""

    @pytest.mark.asyncio
    async def test_successful_message_processing(self) -> None:
        """Valid message should be processed and stored successfully."""
        intent_id = uuid4()
        storage_response = ModelIntentStorageResponse(
            status="success",
            intent_id=intent_id,
            created=True,
        )
        storage = create_mock_storage_adapter(response=storage_response)
        config = ModelIntentEventConsumerConfig()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Verify storage was called
        storage.execute.assert_called_once()

        # Verify counters updated
        assert consumer._messages_consumed == 1
        assert consumer._messages_failed == 0
        assert consumer._last_consume_timestamp is not None

    @pytest.mark.asyncio
    async def test_validation_error_routes_to_dlq(self) -> None:
        """Invalid message format should route to DLQ without retry."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig(retry_max_attempts=3)
        publish_calls: list[PublishCall] = []

        def mock_publish(topic: str, msg: MessagePayload) -> None:
            publish_calls.append((topic, msg))

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(
            subscribe_callback=subscribe_fn,
            env_prefix="test",
            publish_callback=mock_publish,
        )

        # Invalid message - missing required fields
        invalid_message: MessagePayload = {
            "event_type": "IntentClassified",
            "session_id": "",
        }

        await consumer._handle_message(invalid_message, retry_count=0)

        # Verify storage was NOT called (validation failed first)
        storage.execute.assert_not_called()

        # Verify DLQ received the message
        assert consumer._messages_dlq == 1
        assert consumer._messages_failed == 1
        assert len(publish_calls) == 1
        assert "dlq" in publish_calls[0][0]

    @pytest.mark.asyncio
    async def test_storage_failure_routes_to_dlq_after_retries(self) -> None:
        """Storage failure should route to DLQ after exhausting retries."""
        storage = create_mock_storage_adapter(
            side_effect=RuntimeError("Storage unavailable")
        )
        config = ModelIntentEventConsumerConfig(
            retry_max_attempts=2,
            retry_backoff_base_seconds=0.1,  # Minimum allowed value
        )
        publish_calls: list[PublishCall] = []

        def mock_publish(topic: str, msg: MessagePayload) -> None:
            publish_calls.append((topic, msg))

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(
            subscribe_callback=subscribe_fn,
            env_prefix="test",
            publish_callback=mock_publish,
        )

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Storage called 3 times total: initial + 2 retries
        assert storage.execute.call_count == 3

        # Verify DLQ received the message
        assert consumer._messages_dlq == 1
        assert consumer._messages_failed == 1

        # Check DLQ message includes retry_count
        dlq_calls = [c for c in publish_calls if "dlq" in c[0]]
        assert len(dlq_calls) == 1
        assert dlq_calls[0][1]["retry_count"] == 2


# =============================================================================
# UUID Validation Tests
# =============================================================================


class TestUUIDValidation:
    """Tests for intent_id validation on successful storage."""

    @pytest.mark.asyncio
    async def test_storage_success_without_intent_id_raises_error(self) -> None:
        """Storage returning success but intent_id=None should raise RuntimeError."""
        # Storage returns success but with no intent_id (a bug)
        buggy_response = ModelIntentStorageResponse(
            status="success",
            intent_id=None,  # Bug: should have an ID on success
            created=True,
        )
        storage = create_mock_storage_adapter(response=buggy_response)
        config = ModelIntentEventConsumerConfig(
            retry_max_attempts=0,  # No retries - fail immediately
        )
        publish_calls: list[PublishCall] = []

        def mock_publish(topic: str, msg: MessagePayload) -> None:
            publish_calls.append((topic, msg))

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(
            subscribe_callback=subscribe_fn,
            env_prefix="test",
            publish_callback=mock_publish,
        )

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Should fail and route to DLQ
        assert consumer._messages_failed == 1
        assert consumer._messages_dlq == 1

        # Check that the error message mentions the storage adapter bug
        dlq_calls = [c for c in publish_calls if "dlq" in c[0]]
        assert len(dlq_calls) == 1
        assert "intent_id is None" in str(dlq_calls[0][1]["failure_reason"])


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry behavior with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_storage_failure(self) -> None:
        """Storage failures should trigger exponential backoff retries.

        Uses mocked asyncio.sleep for deterministic testing without timing flakiness.
        """
        call_count = 0
        sleep_durations: list[float] = []

        async def failing_execute(
            *args: object, **kwargs: object
        ) -> ModelIntentStorageResponse:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Storage failed")

        async def mock_sleep(duration: float) -> None:
            """Capture sleep durations without actually sleeping."""
            sleep_durations.append(duration)

        storage = MagicMock()
        storage.execute = failing_execute

        config = ModelIntentEventConsumerConfig(
            retry_max_attempts=2,
            retry_backoff_base_seconds=1.0,  # Use 1.0s base for clear math
        )

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        message = create_valid_message()

        # Patch asyncio.sleep to make test deterministic
        import unittest.mock

        with unittest.mock.patch("asyncio.sleep", mock_sleep):
            await consumer._handle_message(message, retry_count=0)

        # Should have 3 calls: initial + 2 retries
        assert call_count == 3

        # Verify exponential backoff durations (deterministic)
        # First retry: 1.0 * 2^0 = 1.0s
        # Second retry: 1.0 * 2^1 = 2.0s
        assert len(sleep_durations) == 2
        assert sleep_durations[0] == 1.0  # First backoff: base * 2^0
        assert sleep_durations[1] == 2.0  # Second backoff: base * 2^1

    @pytest.mark.asyncio
    async def test_no_retry_on_validation_error(self) -> None:
        """Validation errors should not trigger retries."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig(
            retry_max_attempts=5,  # Would retry if it tried
        )

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Invalid message
        invalid_message: MessagePayload = {"garbage": "data"}
        await consumer._handle_message(invalid_message, retry_count=0)

        # Storage should never be called
        storage.execute.assert_not_called()

        # Should fail immediately without retries
        assert consumer._messages_failed == 1

    @pytest.mark.asyncio
    async def test_no_retry_when_max_attempts_zero(self) -> None:
        """Config with retry_max_attempts=0 should not retry."""
        storage = create_mock_storage_adapter(
            side_effect=RuntimeError("Storage failed")
        )
        config = ModelIntentEventConsumerConfig(
            retry_max_attempts=0,
        )

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Should only call once, no retries
        storage.execute.assert_called_once()
        assert consumer._messages_failed == 1


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self) -> None:
        """Circuit breaker should open after reaching failure threshold."""
        # Storage adapter that always fails
        storage = create_mock_storage_adapter(
            side_effect=RuntimeError("Storage failed")
        )
        config = ModelIntentEventConsumerConfig(
            circuit_breaker_failure_threshold=3,
            retry_max_attempts=0,  # No retries to simplify test
        )

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Process messages until circuit opens (need 3 failures)
        # Each storage failure records a circuit breaker failure
        for i in range(5):
            message = create_valid_message(session_id=f"session-{i}")
            await consumer._handle_message(message, retry_count=0)

        # Circuit should be open after 3 storage failures
        assert consumer._circuit_breaker.state == CircuitBreakerState.OPEN
        # Verify storage was called 3 times before circuit opened
        # (then subsequent calls are blocked by the circuit)
        assert storage.execute.call_count >= 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_routes_to_dlq(self) -> None:
        """Messages should route to DLQ when circuit breaker is open."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig(
            circuit_breaker_failure_threshold=2,
            retry_max_attempts=0,
        )
        publish_calls: list[PublishCall] = []

        def mock_publish(topic: str, msg: MessagePayload) -> None:
            publish_calls.append((topic, msg))

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(
            subscribe_callback=subscribe_fn,
            env_prefix="test",
            publish_callback=mock_publish,
        )

        # Manually open the circuit breaker
        consumer._circuit_breaker.state = CircuitBreakerState.OPEN

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Storage should NOT be called (circuit is open)
        storage.execute.assert_not_called()

        # Should route to DLQ
        assert consumer._messages_dlq == 1
        dlq_calls = [c for c in publish_calls if "dlq" in c[0]]
        assert len(dlq_calls) == 1
        assert "Circuit breaker open" in str(dlq_calls[0][1]["failure_reason"])


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check behavior."""

    @pytest.mark.asyncio
    async def test_health_check_unknown_when_not_initialized(self) -> None:
        """Health check should return UNKNOWN status when not initialized."""
        config = ModelIntentEventConsumerConfig()
        storage = create_mock_storage_adapter()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        health = await consumer.health_check()

        assert health.status == HealthStatus.UNKNOWN
        assert health.initialized is False
        assert health.is_healthy is False
        assert "not initialized" in (health.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_health_check_healthy_after_consume(self) -> None:
        """Health check should return HEALTHY after successful consumption."""
        storage_response = ModelIntentStorageResponse(
            status="success",
            intent_id=uuid4(),
            created=True,
        )
        storage = create_mock_storage_adapter(response=storage_response)
        config = ModelIntentEventConsumerConfig(
            staleness_threshold_seconds=300.0,
        )

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Process a message successfully
        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        health = await consumer.health_check()

        assert health.status == HealthStatus.HEALTHY
        assert health.is_healthy is True
        assert health.initialized is True
        assert health.is_stale is False
        assert health.circuit_breaker_state == "closed"

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_stale(self) -> None:
        """Health check should return DEGRADED when no recent consumption."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig(
            staleness_threshold_seconds=1.0,  # 1 second threshold
        )

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Set last consume timestamp to be old
        consumer._last_consume_timestamp = datetime.now(timezone.utc) - timedelta(
            seconds=10
        )

        health = await consumer.health_check()

        assert health.status == HealthStatus.DEGRADED
        assert health.is_healthy is False
        assert health.is_stale is True
        assert health.staleness_seconds is not None
        assert health.staleness_seconds > 1.0

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_never_consumed(self) -> None:
        """Health check should return DEGRADED when initialized but never consumed."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Never consume any messages
        health = await consumer.health_check()

        assert health.status == HealthStatus.DEGRADED
        assert health.is_stale is True
        assert health.last_consume_timestamp is None

    @pytest.mark.asyncio
    async def test_health_check_circuit_open_status(self) -> None:
        """Health check should return CIRCUIT_OPEN when circuit breaker is open."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Manually open the circuit breaker
        consumer._circuit_breaker.state = CircuitBreakerState.OPEN
        consumer._circuit_breaker.failure_count = 5

        health = await consumer.health_check()

        assert health.status == HealthStatus.CIRCUIT_OPEN
        assert health.is_healthy is False
        assert health.circuit_breaker_state == "open"
        assert health.circuit_breaker_failure_count == 5


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for success/failure event emission."""

    @pytest.mark.asyncio
    async def test_stored_event_emitted_on_success(self) -> None:
        """Successful storage should emit an intent-stored event.

        Uses canonical ModelIntentStoredEvent from omnibase_core which has:
        - Versioned event_type: "onex.omnimemory.intent.stored.v1"
        - session_ref field (mapped from session_id at boundary)
        - status="success" for successful storage
        """
        intent_id = uuid4()
        storage_response = ModelIntentStorageResponse(
            status="success",
            intent_id=intent_id,
            created=True,
        )
        storage = create_mock_storage_adapter(response=storage_response)
        config = ModelIntentEventConsumerConfig()
        publish_calls: list[PublishCall] = []

        def mock_publish(topic: str, msg: MessagePayload) -> None:
            publish_calls.append((topic, msg))

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(
            subscribe_callback=subscribe_fn,
            env_prefix="test",
            publish_callback=mock_publish,
        )

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Check stored event was emitted with canonical event_type
        stored_events = [c for c in publish_calls if "intent-stored" in c[0]]
        assert len(stored_events) == 1
        assert stored_events[0][1]["event_type"] == "onex.omnimemory.intent.stored.v1"
        assert stored_events[0][1]["intent_id"] == str(intent_id)
        assert stored_events[0][1]["status"] == "success"
        # session_id is mapped to session_ref at boundary
        assert stored_events[0][1]["session_ref"] == message["session_id"]

    @pytest.mark.asyncio
    async def test_failed_event_emitted_on_failure(self) -> None:
        """Storage failure should emit an intent-stored event with status=error.

        Uses canonical ModelIntentStoredEvent from omnibase_core which encodes
        both success and error via the status field (not separate events).
        Error details are in error_message field.
        """
        storage = create_mock_storage_adapter(
            side_effect=RuntimeError("Storage failed")
        )
        config = ModelIntentEventConsumerConfig(
            retry_max_attempts=0,
        )
        publish_calls: list[PublishCall] = []

        def mock_publish(topic: str, msg: MessagePayload) -> None:
            publish_calls.append((topic, msg))

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(
            subscribe_callback=subscribe_fn,
            env_prefix="test",
            publish_callback=mock_publish,
        )

        message = create_valid_message()
        await consumer._handle_message(message, retry_count=0)

        # Check stored event (with error status) was emitted
        # Uses same topic as success events - status field distinguishes
        stored_events = [c for c in publish_calls if "intent-stored" in c[0]]
        assert len(stored_events) == 1
        assert stored_events[0][1]["event_type"] == "onex.omnimemory.intent.stored.v1"
        assert stored_events[0][1]["status"] == "error"
        assert "RuntimeError" in str(stored_events[0][1]["error_message"])
        assert "Storage failed" in str(stored_events[0][1]["error_message"])
        # session_id is mapped to session_ref at boundary
        assert stored_events[0][1]["session_ref"] == message["session_id"]


# =============================================================================
# Stop/Cleanup Tests
# =============================================================================


class TestStopCleanup:
    """Tests for consumer stop and cleanup behavior."""

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_and_resets(self) -> None:
        """stop() should unsubscribe and reset initialization state."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig()
        unsubscribe_mock = MagicMock()

        def mock_subscribe(topic: str, handler: object) -> MagicMock:
            return unsubscribe_mock

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        await consumer.initialize(subscribe_callback=mock_subscribe, env_prefix="test")

        assert consumer.is_initialized is True

        await consumer.stop()

        # Should have called unsubscribe
        unsubscribe_mock.assert_called_once()

        # Should reset state
        assert consumer.is_initialized is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        """Calling stop() multiple times should not raise errors."""
        storage = create_mock_storage_adapter()
        config = ModelIntentEventConsumerConfig()

        consumer = HandlerIntentEventConsumer(config=config, storage_adapter=storage)
        subscribe_fn, _ = create_mock_subscribe()
        await consumer.initialize(subscribe_callback=subscribe_fn, env_prefix="test")

        # Multiple stop calls should be safe
        await consumer.stop()
        await consumer.stop()
        await consumer.stop()

        assert consumer.is_initialized is False
