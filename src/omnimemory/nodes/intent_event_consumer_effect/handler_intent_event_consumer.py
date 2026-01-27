# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent event consumer handler.

Consumes intent-classified.v1 events from Kafka and persists to Memgraph.

Architecture:
    This handler subscribes to intent classification events from omniintelligence
    and routes them to HandlerIntentStorageAdapter for persistence. It includes:
    - Circuit breaker for storage failure protection
    - DLQ routing for unprocessable messages
    - Success/failure event emission for observability
    - Staleness detection for health monitoring

Consumer Group:
    Derived from node identity per ADR:
    {env}.omnimemory.intent_event_consumer_effect.consume.v1

Example::

    from omnimemory.nodes.intent_event_consumer_effect import (
        HandlerIntentEventConsumer,
        ModelIntentEventConsumerConfig,
    )
    from omnimemory.nodes.intent_storage_effect.adapters import (
        HandlerIntentStorageAdapter,
    )

    async def example():
        config = ModelIntentEventConsumerConfig()
        storage_adapter = HandlerIntentStorageAdapter()
        await storage_adapter.initialize(connection_uri="bolt://localhost:7687")

        consumer = HandlerIntentEventConsumer(
            config=config,
            storage_adapter=storage_adapter,
        )

        # Initialize with Kafka subscription callback
        await consumer.initialize(
            subscribe_callback=kafka_bus.subscribe,
            env_prefix="dev",
        )

        # Run until shutdown
        await consumer.stop()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1619.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from omnibase_core.models.events import ModelIntentStoredEvent
from pydantic import ValidationError

from omnimemory.models.events import ModelIntentClassifiedEvent
from omnimemory.models.utils.model_health_status import HealthStatus
from omnimemory.nodes.intent_event_consumer_effect.models import (
    ModelIntentEventConsumerConfig,
    ModelIntentEventConsumerHealth,
)
from omnimemory.nodes.intent_event_consumer_effect.utils import (
    map_event_to_storage_request,
)
from omnimemory.utils.concurrency import CircuitBreaker, CircuitBreakerState

if TYPE_CHECKING:
    from omnimemory.nodes.intent_storage_effect.adapters.adapter_intent_storage import (
        HandlerIntentStorageAdapter,
    )

__all__ = ["HandlerIntentEventConsumer"]

logger = logging.getLogger(__name__)

# Handler identifier for logging and registration
HANDLER_ID_INTENT_CONSUMER: str = "intent-event-consumer"


class HandlerIntentEventConsumer:
    """Kafka consumer for intent-classified events.

    Consumer group is derived from node identity per ADR:
    {env}.{service}.{node_name}.{purpose}.{version}

    Example: dev.omnimemory.intent_event_consumer_effect.consume.v1

    Attributes:
        _config: Consumer configuration.
        _storage_adapter: Adapter for storing intents to graph.
        _circuit_breaker: Circuit breaker for storage failures.
        _initialized: Whether the consumer is initialized.

    Example::

        config = ModelIntentEventConsumerConfig()
        storage = HandlerIntentStorageAdapter()
        await storage.initialize(connection_uri="bolt://localhost:7687")

        consumer = HandlerIntentEventConsumer(config, storage)
        await consumer.initialize(kafka_subscribe, "dev")
    """

    def __init__(
        self,
        config: ModelIntentEventConsumerConfig,
        storage_adapter: HandlerIntentStorageAdapter,
    ) -> None:
        """Initialize the consumer handler.

        Args:
            config: Consumer configuration including topic suffixes,
                circuit breaker settings, and staleness threshold.
            storage_adapter: Adapter for storing intents to graph.
                Must be initialized before calling consumer.initialize().
        """
        self._config = config
        self._storage_adapter = storage_adapter

        # Circuit breaker for storage failures
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=float(config.circuit_breaker_recovery_timeout_seconds),
            success_threshold=1,  # Close after 1 success in half-open
        )

        # State tracking
        self._initialized = False
        self._last_consume_timestamp: datetime | None = None
        self._messages_consumed = 0
        self._messages_failed = 0
        self._messages_dlq = 0
        self._unsubscribe: Callable[[], None] | None = None

        # Kafka publish callback (set during initialize)
        self._publish_callback: Callable[[str, dict[str, object]], None] | None = None
        self._env_prefix: str = "dev"

        # Track pending tasks to prevent garbage collection (RUF006)
        self._pending_tasks: set[asyncio.Task[None]] = set()

        logger.info(
            "HandlerIntentEventConsumer initialized",
            extra={
                "handler": HANDLER_ID_INTENT_CONSUMER,
                "subscribe_topic": config.subscribe_topic_suffix,
                "publish_stored_topic": config.publish_stored_topic_suffix,
            },
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the consumer has been initialized."""
        return self._initialized

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"HandlerIntentEventConsumer("
            f"initialized={self._initialized}, "
            f"consumed={self._messages_consumed}, "
            f"failed={self._messages_failed})"
        )

    async def initialize(
        self,
        subscribe_callback: Callable[
            [str, Callable[[dict[str, object]], None]], Callable[[], None]
        ],
        env_prefix: str = "dev",
        publish_callback: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        """Initialize Kafka subscription.

        Args:
            subscribe_callback: Function to subscribe to Kafka topic.
                Signature: (topic, handler) -> unsubscribe_fn
            env_prefix: Environment prefix for topic names (e.g., "dev", "staging").
            publish_callback: Optional callback for publishing events.
                Signature: (topic, message_dict) -> None
        """
        if self._initialized:
            logger.debug(
                "Consumer already initialized, skipping",
                extra={"handler": HANDLER_ID_INTENT_CONSUMER},
            )
            return

        self._env_prefix = env_prefix
        self._publish_callback = publish_callback

        full_topic = f"{env_prefix}.{self._config.subscribe_topic_suffix}"

        self._unsubscribe = subscribe_callback(full_topic, self._handle_message_sync)
        self._initialized = True

        logger.info(
            "Kafka subscription initialized",
            extra={
                "handler": HANDLER_ID_INTENT_CONSUMER,
                "topic": full_topic,
                "env_prefix": env_prefix,
            },
        )

    def _handle_message_sync(self, message: dict[str, object]) -> None:
        """Synchronous message handler wrapper for Kafka callback.

        Kafka callbacks are typically synchronous, so this wraps the
        async handler. In production, use an event loop runner.

        Threading Model:
            This method handles two scenarios:

            1. **No running event loop** (e.g., called from a synchronous Kafka
               consumer thread): Creates a new event loop via ``asyncio.run()``.
               Note: In high-throughput scenarios, consider using a dedicated
               event loop thread to avoid per-message loop creation overhead.

            2. **Existing event loop** (e.g., called from an async context or
               when the consumer is integrated with an async Kafka client):
               Schedules the handler as a task in the existing loop.

            For production deployments, prefer using an async Kafka client
            (e.g., aiokafka) that provides native async message handling,
            avoiding the need for this synchronous wrapper.

        Args:
            message: Raw Kafka message payload.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one for this callback.
            # Note: In high-throughput scenarios, this creates a new loop per
            # message. Consider using a dedicated event loop thread instead.
            asyncio.run(self._handle_message(message, retry_count=0))
        else:
            # Schedule in existing loop
            task = loop.create_task(self._handle_message(message, retry_count=0))
            # Store reference to prevent garbage collection
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def _handle_message(
        self, message: dict[str, object], *, retry_count: int = 0
    ) -> None:
        """Process a single Kafka message with retry support.

        On storage failures (not validation errors), retries with exponential
        backoff up to retry_max_attempts times before routing to DLQ.

        Args:
            message: Raw Kafka message payload.
            retry_count: Current retry attempt (0 = first attempt).
        """
        correlation_id: UUID | None = None
        session_id: str | None = None
        intent_category: str | None = None

        try:
            # Parse the event
            event = ModelIntentClassifiedEvent.model_validate(message)
            correlation_id = event.correlation_id
            session_id = event.session_id
            intent_category = event.intent_category

            logger.debug(
                "Processing intent-classified event",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "correlation_id": str(correlation_id),
                    "session_id": session_id,
                    "intent_category": intent_category,
                },
            )

            # Check circuit breaker
            if not self._circuit_breaker.should_allow_request():
                logger.warning(
                    "Circuit breaker open, routing to DLQ",
                    extra={
                        "handler": HANDLER_ID_INTENT_CONSUMER,
                        "correlation_id": str(correlation_id),
                        "circuit_state": self._circuit_breaker.state.value,
                        "failure_count": self._circuit_breaker.failure_count,
                    },
                )
                self._messages_failed += (
                    1  # Count circuit-open as failed for accurate metrics
                )

                # Emit failure event for observability (matches storage-error path)
                if session_id and intent_category and correlation_id:
                    failed_event = ModelIntentStoredEvent.from_error(
                        session_ref=session_id,  # Map at boundary
                        intent_category=intent_category,
                        error_message="Circuit breaker open",
                        correlation_id=correlation_id,
                    )
                    await self._emit_stored_event(failed_event)

                await self._route_to_dlq(message, "Circuit breaker open")
                return

            # Map to storage request
            storage_request = map_event_to_storage_request(event)

            # Store the intent
            start_time = datetime.now(timezone.utc)
            result = await self._storage_adapter.execute(storage_request)
            storage_latency_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            if result.status == "success":
                # Validate intent_id is present for successful store
                if result.intent_id is None:
                    raise RuntimeError(
                        "Storage returned success but intent_id is None - storage adapter bug"
                    )

                self._circuit_breaker.record_success()
                self._messages_consumed += 1
                self._last_consume_timestamp = datetime.now(timezone.utc)

                # Emit success event using canonical omnibase_core model
                # Maps session_id → session_ref at the emission boundary
                stored_event = ModelIntentStoredEvent.create(
                    session_ref=event.session_id,  # Map at boundary
                    intent_category=event.intent_category,
                    intent_id=result.intent_id,
                    confidence=event.confidence,
                    keywords=event.keywords,
                    created=result.created,
                    execution_time_ms=storage_latency_ms,
                    correlation_id=event.correlation_id,
                )
                await self._emit_stored_event(stored_event)

                logger.info(
                    "Intent stored successfully",
                    extra={
                        "handler": HANDLER_ID_INTENT_CONSUMER,
                        "correlation_id": str(correlation_id),
                        "intent_id": str(stored_event.intent_id),
                        "storage_latency_ms": storage_latency_ms,
                    },
                )
            else:
                # Storage returned error status
                error_msg = result.error_message or "Storage failed with error status"
                raise RuntimeError(error_msg)

        except ValidationError as e:
            # Validation errors are NOT retried - they indicate malformed messages
            # that will never succeed regardless of retry attempts. Route directly
            # to DLQ for manual inspection.
            logger.error(
                "Invalid event payload",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "error": str(e),
                    "message_preview": str(message)[:200],
                },
            )
            self._messages_failed += 1
            await self._route_to_dlq(message, f"Validation error: {e}")

        except Exception as e:
            self._circuit_breaker.record_failure()

            # Check if retries are available (storage failures only)
            max_attempts = self._config.retry_max_attempts
            if retry_count < max_attempts:
                # Calculate exponential backoff: base * 2^attempt
                backoff_seconds = self._config.retry_backoff_base_seconds * (
                    2**retry_count
                )
                next_retry = retry_count + 1

                logger.warning(
                    "Storage failure, scheduling retry",
                    extra={
                        "handler": HANDLER_ID_INTENT_CONSUMER,
                        "correlation_id": str(correlation_id)
                        if correlation_id
                        else None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "retry_count": retry_count,
                        "next_retry": next_retry,
                        "max_attempts": max_attempts,
                        "backoff_seconds": backoff_seconds,
                    },
                )

                # Wait with exponential backoff
                await asyncio.sleep(backoff_seconds)

                # Recursive retry with incremented count
                await self._handle_message(message, retry_count=next_retry)
                return

            # No retries left - route to DLQ
            self._messages_failed += 1

            logger.error(
                "Failed to process event after retries exhausted",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "retry_count": retry_count,
                    "max_attempts": max_attempts,
                    "circuit_state": self._circuit_breaker.state.value,
                    "failure_count": self._circuit_breaker.failure_count,
                },
            )

            # Emit failure event using canonical omnibase_core model
            # Uses status="error" pattern instead of separate failed event
            if session_id and intent_category and correlation_id:
                error_msg = f"{type(e).__name__}: {e} (retries: {retry_count})"
                failed_event = ModelIntentStoredEvent.from_error(
                    session_ref=session_id,  # Map at boundary
                    intent_category=intent_category,
                    error_message=error_msg,
                    correlation_id=correlation_id,
                )
                await self._emit_stored_event(failed_event)

            await self._route_to_dlq(message, str(e), retry_count=retry_count)

    async def _emit_stored_event(self, event: ModelIntentStoredEvent) -> None:
        """Emit intent-stored event to Kafka.

        Uses canonical omnibase_core.ModelIntentStoredEvent which supports
        both success (status="success") and error (status="error") cases
        via a single event type.

        Args:
            event: The stored event to emit (success or error).
        """
        if self._publish_callback is None:
            logger.debug(
                "No publish callback configured, skipping event emission",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "event_type": event.event_type,
                    "status": event.status,
                },
            )
            return

        topic = f"{self._env_prefix}.{self._config.publish_stored_topic_suffix}"
        try:
            self._publish_callback(topic, event.model_dump(mode="json"))
            logger.debug(
                "Emitted intent-stored event",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "topic": topic,
                    "intent_id": str(event.intent_id),
                    "session_ref": event.session_ref,
                    "status": event.status,
                },
            )
        except Exception as e:
            # Log but don't fail the main operation for publish errors
            logger.warning(
                "Failed to emit intent-stored event",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "topic": topic,
                    "error": str(e),
                },
            )

    async def _route_to_dlq(
        self, message: dict[str, object], reason: str, *, retry_count: int = 0
    ) -> None:
        """Route failed message to dead letter queue.

        Args:
            message: The original message.
            reason: Why the message failed.
            retry_count: Number of retry attempts made before DLQ routing.
        """
        self._messages_dlq += 1

        if self._publish_callback is None:
            logger.warning(
                "Message would be routed to DLQ (no publish callback)",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "reason": reason,
                    "retry_count": retry_count,
                    "dlq_topic": self._config.dlq_topic_suffix,
                },
            )
            return

        topic = f"{self._env_prefix}.{self._config.dlq_topic_suffix}"
        dlq_message = {
            "original_message": message,
            "failure_reason": reason,
            "retry_count": retry_count,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "handler": HANDLER_ID_INTENT_CONSUMER,
        }

        try:
            self._publish_callback(topic, dlq_message)
            logger.warning(
                "Message routed to DLQ",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "reason": reason,
                    "retry_count": retry_count,
                    "dlq_topic": topic,
                },
            )
        except Exception as e:
            # Log but don't fail for DLQ publish errors
            logger.error(
                "Failed to route message to DLQ",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "topic": topic,
                    "error": str(e),
                },
            )

    async def health_check(self) -> ModelIntentEventConsumerHealth:
        """Return consumer health status with meaningful signals.

        Avoids the "always returns OK" anti-pattern by tracking:
        - Initialization state
        - Staleness (time since last consumption)
        - Circuit breaker state
        - Dependency health (storage adapter)

        Returns:
            Health status with initialization, staleness, and circuit breaker info.
        """
        now = datetime.now(timezone.utc)

        # Calculate staleness
        staleness_seconds: float | None = None
        is_stale = False
        if self._last_consume_timestamp:
            staleness_seconds = (now - self._last_consume_timestamp).total_seconds()
            is_stale = staleness_seconds > self._config.staleness_threshold_seconds
        elif self._initialized:
            # Stale if initialized but never consumed
            is_stale = True

        # Determine status based on circuit breaker state
        cb_state = self._circuit_breaker.state
        error_message: str | None = None

        if not self._initialized:
            status = HealthStatus.UNKNOWN
            error_message = "Consumer not initialized"
        elif cb_state == CircuitBreakerState.OPEN:
            status = HealthStatus.CIRCUIT_OPEN
            error_message = (
                f"Circuit breaker is open "
                f"(failures: {self._circuit_breaker.failure_count})"
            )
        elif is_stale:
            status = HealthStatus.DEGRADED
            staleness_display = (
                f"{staleness_seconds:.0f}s"
                if staleness_seconds is not None
                else "never consumed"
            )
            error_message = f"No messages consumed: {staleness_display}"
        else:
            status = HealthStatus.HEALTHY

        # Map circuit breaker state to Literal type
        cb_state_str: Literal["closed", "open", "half_open"] | None = None
        if cb_state == CircuitBreakerState.CLOSED:
            cb_state_str = "closed"
        elif cb_state == CircuitBreakerState.OPEN:
            cb_state_str = "open"
        elif cb_state == CircuitBreakerState.HALF_OPEN:
            cb_state_str = "half_open"

        # Check storage adapter health via public interface
        storage_healthy: bool | None = None
        try:
            # Check if storage adapter has a public health_check method
            if hasattr(self._storage_adapter, "health_check"):
                adapter_health = await self._storage_adapter.health_check()
                storage_healthy = getattr(adapter_health, "is_healthy", True)
            elif hasattr(self._storage_adapter, "is_initialized"):
                # Fall back to checking initialization property
                storage_healthy = self._storage_adapter.is_initialized
            else:
                # Cannot determine health, assume healthy if we can execute
                storage_healthy = True
        except Exception as e:
            logger.warning(
                "Failed to check storage adapter health",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "error": str(e),
                },
            )
            storage_healthy = False

        return ModelIntentEventConsumerHealth(
            status=status,
            is_healthy=status == HealthStatus.HEALTHY,
            initialized=self._initialized,
            error_message=error_message,
            last_consume_timestamp=self._last_consume_timestamp,
            is_stale=is_stale,
            staleness_seconds=staleness_seconds,
            circuit_breaker_state=cb_state_str,
            circuit_breaker_failure_count=self._circuit_breaker.failure_count,
            messages_consumed_total=self._messages_consumed,
            messages_failed_total=self._messages_failed,
            messages_dlq_total=self._messages_dlq,
            storage_handler_healthy=storage_healthy,
            health_check_timestamp=now,
        )

    async def stop(self) -> None:
        """Graceful shutdown.

        Unsubscribes from Kafka topic, cancels pending message processing tasks,
        and resets initialization state. Does not shutdown the storage adapter
        (caller's responsibility).
        """
        # Unsubscribe FIRST to prevent new messages during shutdown
        if self._unsubscribe:
            try:
                self._unsubscribe()
            except Exception as e:
                logger.warning(
                    "Error during Kafka unsubscribe",
                    extra={
                        "handler": HANDLER_ID_INTENT_CONSUMER,
                        "error": str(e),
                    },
                )
            self._unsubscribe = None

        # THEN cancel pending tasks
        if self._pending_tasks:
            pending_count = len(self._pending_tasks)
            logger.debug(
                "Cancelling pending tasks",
                extra={
                    "handler": HANDLER_ID_INTENT_CONSUMER,
                    "pending_count": pending_count,
                },
            )
            for task in self._pending_tasks:
                task.cancel()
            # Wait for cancellation to complete, capturing any exceptions
            results = await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            # Log any unexpected errors (CancelledError is expected)
            for result in results:
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.debug(
                        "Task cancellation resulted in exception",
                        extra={
                            "handler": HANDLER_ID_INTENT_CONSUMER,
                            "error": str(result),
                            "error_type": type(result).__name__,
                        },
                    )
            self._pending_tasks.clear()

        self._initialized = False
        self._publish_callback = None

        logger.info(
            "Consumer stopped",
            extra={
                "handler": HANDLER_ID_INTENT_CONSUMER,
                "messages_consumed": self._messages_consumed,
                "messages_failed": self._messages_failed,
                "messages_dlq": self._messages_dlq,
            },
        )
