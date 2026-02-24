# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Dispatch bridge handlers for OmniMemory domain.

This module provides bridge handlers that adapt between the MessageDispatchEngine
handler signature and existing OmniMemory domain handlers. It also defines
topic alias mappings needed because ONEX canonical topic naming uses ``.cmd.``
and ``.evt.`` segments, which EnumMessageCategory.from_topic() does not yet
recognize (it expects ``.commands.`` and ``.events.``).

Design Decisions:
    - Topic aliases are a temporary bridge until EnumMessageCategory.from_topic()
      is updated to handle ``.cmd.`` / ``.evt.`` short forms.
    - Bridge handlers adapt (envelope, context) -> existing handler interfaces.
    - The dispatch engine is created per-plugin (not kernel-managed).
    - message_types=None on handler registration accepts all message types in
      the category -- correct when routing by topic, not type.
    - Unimplemented dispatch routes raise RuntimeError to prevent silent data loss.

Related:
    - OMN-2215: Phase 4 -- MessageDispatchEngine integration for omnimemory
    - OMN-934: MessageDispatchEngine implementation in omnibase_core
    - omniintelligence/runtime/dispatch_handlers.py (reference implementation)
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from uuid import UUID, uuid4

from omnibase_core.enums.enum_execution_shape import EnumMessageCategory
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.runtime.runtime_message_dispatch import MessageDispatchEngine

if TYPE_CHECKING:
    from omnibase_core.protocols.handler.protocol_handler_context import (
        ProtocolHandlerContext,
    )

logger = logging.getLogger(__name__)

# =============================================================================
# Dependency Protocols (structural typing for dispatch handler deps)
# =============================================================================
# Defined locally to avoid circular imports with handler modules.
# These mirror the handler interfaces that the dispatch handlers delegate to.


@runtime_checkable
class ProtocolIntentEventConsumer(Protocol):
    """Protocol for intent event consumer handler.

    Note on ``_handle_message`` naming:
        The underscore-prefixed method name is intentional and mirrors the actual
        implementation in omnibase_core's ``HandlerIntentEventConsumer``, where
        ``_handle_message`` is the concrete method called by the base-class
        ``on_message`` template.  Renaming it here would break structural typing
        compatibility.  The ``# noqa: SLF001`` suppression on call-sites
        acknowledges this cross-boundary private-method access.
    """

    async def _handle_message(
        self, message: dict[str, object], *, retry_count: int = 0
    ) -> None: ...


@runtime_checkable
class ProtocolIntentQueryHandler(Protocol):
    """Protocol for intent query handler.

    Matches HandlerIntentQuery.execute() signature.
    """

    async def execute(
        self,
        request: object,
    ) -> object: ...


# =============================================================================
# Topic Alias Mapping
# =============================================================================
# ONEX canonical topic naming uses `.cmd.` for commands and `.evt.` for events.
# MessageDispatchEngine.dispatch() uses EnumMessageCategory.from_topic() which
# only recognizes `.commands.` and `.events.` segments. These aliases bridge
# the naming gap until from_topic() is updated.
#
# Usage: when calling dispatch(), pass the alias instead of the raw topic.

DISPATCH_ALIAS_INTENT_CLASSIFIED = "onex.events.omniintelligence.intent-classified.v1"
"""Dispatch-compatible alias for intent-classified canonical topic."""

DISPATCH_ALIAS_INTENT_QUERY_REQUESTED = (
    "onex.commands.omnimemory.intent-query-requested.v1"
)
"""Dispatch-compatible alias for intent-query-requested canonical topic."""

DISPATCH_ALIAS_RUNTIME_TICK = "onex.commands.omnimemory.runtime-tick.v1"
"""Dispatch-compatible alias for runtime-tick command topic."""

DISPATCH_ALIAS_ARCHIVE_MEMORY = "onex.commands.omnimemory.archive-memory.v1"
"""Dispatch-compatible alias for archive-memory command topic."""

DISPATCH_ALIAS_EXPIRE_MEMORY = "onex.commands.omnimemory.expire-memory.v1"
"""Dispatch-compatible alias for expire-memory command topic."""

DISPATCH_ALIAS_MEMORY_RETRIEVAL_REQUESTED = (
    "onex.commands.omnimemory.memory-retrieval-requested.v1"
)
"""Dispatch-compatible alias for memory-retrieval-requested command topic."""


# =============================================================================
# Bridge Handler: Intent Classified Event
# =============================================================================


def create_intent_classified_dispatch_handler(
    *,
    consumer: ProtocolIntentEventConsumer,
    correlation_id: UUID | None = None,
) -> Callable[
    [ModelEventEnvelope[object], ProtocolHandlerContext],
    Awaitable[str],
]:
    """Create a dispatch engine handler for intent-classified events.

    Returns an async handler function compatible with MessageDispatchEngine's
    handler signature. The handler extracts the payload from the envelope
    and delegates to HandlerIntentEventConsumer._handle_message().

    Args:
        consumer: REQUIRED intent event consumer handler instance.
        correlation_id: Optional fixed correlation ID for tracing.

    Returns:
        Async handler function with signature (envelope, context) -> str.
    """

    async def _handle(
        envelope: ModelEventEnvelope[object],
        context: ProtocolHandlerContext,
    ) -> str:
        """Bridge handler: envelope -> HandlerIntentEventConsumer._handle_message()."""
        ctx_correlation_id = (
            correlation_id or getattr(context, "correlation_id", None) or uuid4()
        )

        payload = envelope.payload

        if not isinstance(payload, dict):
            msg = (
                f"Unexpected payload type {type(payload).__name__} "
                f"for intent-classified event "
                f"(correlation_id={ctx_correlation_id})"
            )
            logger.warning(msg)
            raise ValueError(msg)

        logger.info(
            "Dispatching intent-classified event via MessageDispatchEngine "
            "(correlation_id=%s)",
            ctx_correlation_id,
        )

        await consumer._handle_message(payload, retry_count=0)  # noqa: SLF001

        logger.info(
            "Intent-classified event processed via dispatch engine (correlation_id=%s)",
            ctx_correlation_id,
        )

        return ""

    return _handle


# =============================================================================
# Bridge Handler: Intent Query Requested
# =============================================================================


def create_intent_query_dispatch_handler(
    *,
    query_handler: ProtocolIntentQueryHandler,
    publish_callback: Callable[[str, dict[str, object]], Awaitable[None]]
    | Callable[[str, dict[str, object]], None]
    | None = None,
    publish_topic: str | None = None,
    correlation_id: UUID | None = None,
) -> Callable[
    [ModelEventEnvelope[object], ProtocolHandlerContext],
    Awaitable[str],
]:
    """Create a dispatch engine handler for intent-query-requested events.

    Returns an async handler function compatible with MessageDispatchEngine's
    handler signature. The handler extracts the payload from the envelope,
    validates it as a ModelIntentQueryRequestedEvent, delegates to
    HandlerIntentQuery.execute(), and optionally publishes the response.

    Args:
        query_handler: REQUIRED intent query handler instance.
        publish_callback: Optional callback for publishing response events.
            Accepts both sync and async callables; async results are awaited.
        publish_topic: Full topic for intent query response events (from contract).
        correlation_id: Optional fixed correlation ID for tracing.

    Returns:
        Async handler function with signature (envelope, context) -> str.

    Note:
        Response publishing is best-effort; publish failures are logged but do
        not cause the handler to fail. The query is considered successful if
        execute() completes without error.
    """

    async def _handle(
        envelope: ModelEventEnvelope[object],
        context: ProtocolHandlerContext,
    ) -> str:
        """Bridge handler: envelope -> HandlerIntentQuery.execute()."""
        from omnimemory.nodes.intent_query_effect.models import (
            ModelIntentQueryRequestedEvent,
        )

        ctx_correlation_id = (
            correlation_id or getattr(context, "correlation_id", None) or uuid4()
        )

        payload = envelope.payload

        # Parse payload into ModelIntentQueryRequestedEvent
        if isinstance(payload, ModelIntentQueryRequestedEvent):
            request = payload
        elif isinstance(payload, dict):
            try:
                request = ModelIntentQueryRequestedEvent(**payload)
            except Exception as e:
                msg = (
                    f"Failed to parse payload as ModelIntentQueryRequestedEvent: {e} "
                    f"(correlation_id={ctx_correlation_id})"
                )
                logger.warning(msg)
                raise ValueError(msg) from e
        else:
            msg = (
                f"Unexpected payload type {type(payload).__name__} "
                f"for intent-query-requested "
                f"(correlation_id={ctx_correlation_id})"
            )
            logger.warning(msg)
            raise ValueError(msg)

        logger.info(
            "Dispatching intent-query-requested via MessageDispatchEngine "
            "(query_type=%s, query_id=%s, correlation_id=%s)",
            request.query_type,
            request.query_id,
            ctx_correlation_id,
        )

        response = await query_handler.execute(request)

        logger.info(
            "Intent query processed via dispatch engine "
            "(query_type=%s, query_id=%s, correlation_id=%s)",
            request.query_type,
            request.query_id,
            ctx_correlation_id,
        )

        # Publish response if callback and topic are configured
        if publish_callback and publish_topic and hasattr(response, "model_dump"):
            try:
                publish_result = publish_callback(
                    publish_topic,
                    response.model_dump(mode="json"),
                )
                if inspect.isawaitable(publish_result):
                    await publish_result
                logger.debug(
                    "Published intent query response (topic=%s, query_id=%s)",
                    publish_topic,
                    request.query_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to publish intent query response: %s "
                    "(topic=%s, correlation_id=%s)",
                    e,
                    publish_topic,
                    ctx_correlation_id,
                )

        return ""

    return _handle


# =============================================================================
# Bridge Handler: Lifecycle Orchestrator (runtime-tick, archive, expire)
# =============================================================================


def create_lifecycle_noop_dispatch_handler(
    *,
    topic_label: str = "lifecycle",
    correlation_id: UUID | None = None,
) -> Callable[
    [ModelEventEnvelope[object], ProtocolHandlerContext],
    Awaitable[str],
]:
    """Create a no-op dispatch handler for lifecycle orchestrator topics.

    Receives runtime-tick, archive-memory, and expire-memory commands and
    acknowledges them with structured logging. This prevents crashes when
    upstream services send lifecycle commands before the full orchestrator
    integration (OMN-1453, OMN-1524) is complete.

    Args:
        topic_label: Human-readable label for this handler in log output.
        correlation_id: Optional fixed correlation ID for tracing.

    Returns:
        Async handler function with signature (envelope, context) -> str.
    """

    async def _handle(
        envelope: ModelEventEnvelope[object],
        context: ProtocolHandlerContext,
    ) -> str:
        """No-op handler: lifecycle orchestrator not yet wired."""
        ctx_correlation_id = (
            correlation_id or getattr(context, "correlation_id", None) or uuid4()
        )

        payload = envelope.payload
        payload_keys: list[str] = []
        if isinstance(payload, dict):
            payload_keys = list(payload.keys())

        logger.info(
            "Lifecycle handler received %s event — no-op (not yet implemented) "
            "(correlation_id=%s, payload_keys=%s)",
            topic_label,
            ctx_correlation_id,
            payload_keys,
        )
        return ""

    return _handle


def create_lifecycle_dispatch_handler(
    *,
    correlation_id: UUID | None = None,
) -> Callable[
    [ModelEventEnvelope[object], ProtocolHandlerContext],
    Awaitable[str],
]:
    """Create a no-op dispatch handler for lifecycle orchestrator topics.

    Previously a fail-fast handler that raised RuntimeError; now a no-op
    with structured logging so that upstream lifecycle commands (runtime-tick,
    archive-memory, expire-memory) are gracefully acknowledged instead of
    crashing the service (OMN-2437).

    Args:
        correlation_id: Optional fixed correlation ID for tracing.

    Returns:
        Async handler function with signature (envelope, context) -> str.
    """
    return create_lifecycle_noop_dispatch_handler(
        topic_label="lifecycle",
        correlation_id=correlation_id,
    )


# =============================================================================
# Bridge Handler: Memory Retrieval
# =============================================================================


def create_memory_retrieval_dispatch_handler(
    *,
    correlation_id: UUID | None = None,
) -> Callable[
    [ModelEventEnvelope[object], ProtocolHandlerContext],
    Awaitable[str],
]:
    """Create a dispatch handler for memory-retrieval-requested commands.

    Delegates to HandlerMemoryRetrieval (initialized with mock backends via
    use_mock_handlers=True). Returns a serialised JSON response string.

    The handler initialises HandlerMemoryRetrieval lazily on first call so
    that container startup is not blocked by backend initialisation.

    Args:
        correlation_id: Optional fixed correlation ID for tracing.

    Returns:
        Async handler function with signature (envelope, context) -> str.
    """
    from omnimemory.nodes.memory_retrieval_effect.handlers import HandlerMemoryRetrieval
    from omnimemory.nodes.memory_retrieval_effect.models import (
        ModelHandlerMemoryRetrievalConfig,
        ModelMemoryRetrievalRequest,
    )

    _retrieval_handler: HandlerMemoryRetrieval | None = None

    async def _get_retrieval_handler() -> HandlerMemoryRetrieval:
        nonlocal _retrieval_handler
        if _retrieval_handler is None:
            config = ModelHandlerMemoryRetrievalConfig(use_mock_handlers=True)
            _retrieval_handler = HandlerMemoryRetrieval(config)
            await _retrieval_handler.initialize()
        return _retrieval_handler

    async def _handle(
        envelope: ModelEventEnvelope[object],
        context: ProtocolHandlerContext,
    ) -> str:
        """Bridge handler: envelope -> HandlerMemoryRetrieval.execute()."""
        ctx_correlation_id = (
            correlation_id or getattr(context, "correlation_id", None) or uuid4()
        )

        payload = envelope.payload

        # Parse payload into ModelMemoryRetrievalRequest
        if isinstance(payload, ModelMemoryRetrievalRequest):
            request = payload
        elif isinstance(payload, dict):
            try:
                request = ModelMemoryRetrievalRequest(**payload)
            except Exception as exc:
                msg = (
                    f"Failed to parse payload as ModelMemoryRetrievalRequest: {exc} "
                    f"(correlation_id={ctx_correlation_id})"
                )
                logger.warning(msg)
                raise ValueError(msg) from exc
        else:
            msg = (
                f"Unexpected payload type {type(payload).__name__} "
                f"for memory-retrieval-requested "
                f"(correlation_id={ctx_correlation_id})"
            )
            logger.warning(msg)
            raise ValueError(msg)

        logger.info(
            "Dispatching memory-retrieval-requested via MessageDispatchEngine "
            "(operation=%s, correlation_id=%s)",
            request.operation,
            ctx_correlation_id,
        )

        handler = await _get_retrieval_handler()
        response = await handler.execute(request)

        logger.info(
            "Memory retrieval processed via dispatch engine "
            "(operation=%s, status=%s, result_count=%d, correlation_id=%s)",
            request.operation,
            response.status,
            len(response.results),
            ctx_correlation_id,
        )

        return response.model_dump_json()

    return _handle


# =============================================================================
# Dispatch Engine Factory
# =============================================================================


def create_memory_dispatch_engine(
    *,
    intent_consumer: ProtocolIntentEventConsumer,
    intent_query_handler: ProtocolIntentQueryHandler,
    publish_callback: Callable[[str, dict[str, object]], Awaitable[None]]
    | Callable[[str, dict[str, object]], None]
    | None = None,
    publish_topics: dict[str, str] | None = None,
) -> MessageDispatchEngine:
    """Create and configure a MessageDispatchEngine for OmniMemory domain.

    Creates the engine, registers all omnimemory domain handlers and routes,
    and freezes it. The engine is ready for dispatch after this call.

    Registers 4 handlers covering 6 routes:
        1. intent-classified handler (1 route: intent-classified.v1 events)
        2. intent-query handler (1 route: intent-query-requested.v1 commands)
        3. memory-retrieval handler (1 route: memory-retrieval-requested.v1 -- fail-fast)
        4. lifecycle handler (3 routes: runtime-tick, archive, expire -- fail-fast)

    Args:
        intent_consumer: REQUIRED intent event consumer handler.
        intent_query_handler: REQUIRED intent query handler.
        publish_callback: Optional callback for publishing response events.
            Accepts both sync and async callables; async results are awaited.
        publish_topics: Optional mapping of handler name to publish topic.
            Keys: "intent_query". Values: full topic strings from contract
            event_bus.publish_topics.

    Returns:
        Frozen MessageDispatchEngine ready for dispatch.
    """
    topics = publish_topics or {}

    engine = MessageDispatchEngine(
        logger=logging.getLogger(f"{__name__}.dispatch_engine"),
    )

    # --- Handler 1: intent-classified events ---
    intent_classified_handler = create_intent_classified_dispatch_handler(
        consumer=intent_consumer,
    )
    engine.register_handler(
        handler_id="memory-intent-classified-handler",
        handler=intent_classified_handler,
        category=EnumMessageCategory.EVENT,
        node_kind=EnumNodeKind.EFFECT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="memory-intent-classified-route",
            topic_pattern=DISPATCH_ALIAS_INTENT_CLASSIFIED,
            message_category=EnumMessageCategory.EVENT,
            handler_id="memory-intent-classified-handler",
            description=(
                "Routes intent-classified events to "
                "HandlerIntentEventConsumer for persistence."
            ),
        )
    )

    # --- Handler 2: intent-query-requested commands ---
    intent_query_dispatch_handler = create_intent_query_dispatch_handler(
        query_handler=intent_query_handler,
        publish_callback=publish_callback,
        publish_topic=topics.get("intent_query"),
    )
    engine.register_handler(
        handler_id="memory-intent-query-handler",
        handler=intent_query_dispatch_handler,
        category=EnumMessageCategory.COMMAND,
        node_kind=EnumNodeKind.EFFECT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="memory-intent-query-route",
            topic_pattern=DISPATCH_ALIAS_INTENT_QUERY_REQUESTED,
            message_category=EnumMessageCategory.COMMAND,
            handler_id="memory-intent-query-handler",
            description=(
                "Routes intent-query-requested commands to "
                "HandlerIntentQuery for query processing."
            ),
        )
    )

    # --- Handler 3: memory-retrieval-requested ---
    # Uses HandlerMemoryRetrieval with mock backends so retrieval commands
    # are served without external dependencies (OMN-2437).
    retrieval_handler = create_memory_retrieval_dispatch_handler()
    engine.register_handler(
        handler_id="memory-retrieval-handler",
        handler=retrieval_handler,
        category=EnumMessageCategory.COMMAND,
        node_kind=EnumNodeKind.EFFECT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="memory-retrieval-route",
            topic_pattern=DISPATCH_ALIAS_MEMORY_RETRIEVAL_REQUESTED,
            message_category=EnumMessageCategory.COMMAND,
            handler_id="memory-retrieval-handler",
            description=(
                "Routes memory-retrieval-requested commands to "
                "HandlerMemoryRetrieval (fail-fast until fully wired)."
            ),
        )
    )

    # --- Handler 4: lifecycle orchestrator (fail-fast) ---
    lifecycle_handler = create_lifecycle_dispatch_handler()
    engine.register_handler(
        handler_id="memory-lifecycle-handler",
        handler=lifecycle_handler,
        category=EnumMessageCategory.COMMAND,
        node_kind=EnumNodeKind.ORCHESTRATOR,
        message_types=None,
    )
    # Runtime-tick uses .commands. segment so EnumMessageCategory.from_topic()
    # returns COMMAND, consistent with the other lifecycle command routes.
    engine.register_route(
        ModelDispatchRoute(
            route_id="memory-lifecycle-tick-route",
            topic_pattern=DISPATCH_ALIAS_RUNTIME_TICK,
            message_category=EnumMessageCategory.COMMAND,
            handler_id="memory-lifecycle-handler",
            description="Routes runtime-tick to lifecycle orchestrator (fail-fast).",
        )
    )
    # Route: archive-memory command
    engine.register_route(
        ModelDispatchRoute(
            route_id="memory-lifecycle-archive-route",
            topic_pattern=DISPATCH_ALIAS_ARCHIVE_MEMORY,
            message_category=EnumMessageCategory.COMMAND,
            handler_id="memory-lifecycle-handler",
            description=(
                "Routes archive-memory commands to lifecycle orchestrator (fail-fast)."
            ),
        )
    )
    # Route: expire-memory command
    engine.register_route(
        ModelDispatchRoute(
            route_id="memory-lifecycle-expire-route",
            topic_pattern=DISPATCH_ALIAS_EXPIRE_MEMORY,
            message_category=EnumMessageCategory.COMMAND,
            handler_id="memory-lifecycle-handler",
            description=(
                "Routes expire-memory commands to lifecycle orchestrator (fail-fast)."
            ),
        )
    )

    engine.freeze()

    logger.info(
        "Memory dispatch engine created and frozen (routes=%d, handlers=%d)",
        engine.route_count,
        engine.handler_count,
    )

    return engine


# =============================================================================
# Event Bus Callback Factory
# =============================================================================


def create_dispatch_callback(
    engine: MessageDispatchEngine,
    dispatch_topic: str,
    *,
    correlation_id: UUID | None = None,
) -> Callable[[object], Awaitable[None]]:
    """Create an event bus callback that routes messages through the dispatch engine.

    The callback:
    1. Deserializes the raw message value from bytes to dict
    2. Wraps it in a ModelEventEnvelope with category derived from dispatch_topic
    3. Calls engine.dispatch() with the dispatch-compatible topic alias
    4. Acks the message on success, nacks on failure

    Args:
        engine: Frozen MessageDispatchEngine.
        dispatch_topic: Dispatch-compatible topic alias to pass to dispatch().
        correlation_id: Optional fixed correlation ID for tracing. Note that a
            correlation_id found in the message payload takes precedence over
            this value. Full precedence order (highest wins): payload
            correlation_id > caller-provided correlation_id > auto-generated
            UUID.

    Returns:
        Async callback compatible with event bus subscribe(on_message=...).
    """

    async def _on_message(msg: object) -> None:
        """Event bus callback: raw message -> dispatch engine."""
        msg_correlation_id = correlation_id or uuid4()

        try:
            # Extract raw value from message
            if hasattr(msg, "value"):
                raw_value = msg.value
                if isinstance(raw_value, bytes | bytearray):
                    payload_dict = json.loads(raw_value.decode("utf-8"))
                elif isinstance(raw_value, str):
                    payload_dict = json.loads(raw_value)
                elif isinstance(raw_value, dict):
                    payload_dict = raw_value
                else:
                    logger.warning(
                        "Unexpected message value type %s (correlation_id=%s)",
                        type(raw_value).__name__,
                        msg_correlation_id,
                    )
                    if hasattr(msg, "nack"):
                        await msg.nack()
                    return
            elif isinstance(msg, dict):
                payload_dict = msg
            else:
                logger.warning(
                    "Unexpected message type %s (correlation_id=%s)",
                    type(msg).__name__,
                    msg_correlation_id,
                )
                if hasattr(msg, "nack"):
                    await msg.nack()
                return

            # Guard: json.loads can return arrays, scalars, etc.
            # Only dict payloads are valid for dispatch envelope wrapping.
            if not isinstance(payload_dict, dict):
                logger.warning(
                    "Expected JSON object but got %s; nacking message on topic %s",
                    type(payload_dict).__name__,
                    dispatch_topic,
                )
                if hasattr(msg, "nack"):
                    await msg.nack()
                return

            # Narrow the type from Any (json.loads return) to dict[str, object].
            payload: dict[str, object] = cast("dict[str, object]", payload_dict)

            # Extract correlation_id from payload if available.
            # Precedence (highest wins):
            #   1. payload correlation_id  (from the message body)
            #   2. caller-provided fixed correlation_id  (passed to create_dispatch_callback)
            #   3. auto-generated UUID  (uuid4 fallback)
            # If the payload contains a valid UUID correlation_id, it overrides the
            # caller-supplied value. If parsing fails, the current value (caller-
            # supplied or auto-generated) is silently retained via suppress().
            payload_correlation_id = payload.get("correlation_id")
            if payload_correlation_id:
                with contextlib.suppress(ValueError, AttributeError):
                    msg_correlation_id = UUID(str(payload_correlation_id))

            # Derive message category from dispatch_topic so EVENT topics
            # produce EVENT envelopes (not hard-coded COMMAND).
            topic_category = EnumMessageCategory.from_topic(dispatch_topic)
            if topic_category is None:
                logger.warning(
                    "from_topic() returned None for dispatch_topic=%r; "
                    "falling back to 'command' category (correlation_id=%s)",
                    dispatch_topic,
                    msg_correlation_id,
                )
            envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                payload=payload,
                correlation_id=msg_correlation_id,
                metadata=ModelEnvelopeMetadata(
                    tags={
                        "message_category": topic_category.value
                        if topic_category
                        else "command",
                    },
                ),
            )

            # Dispatch through the engine
            result = await engine.dispatch(
                topic=dispatch_topic,
                envelope=envelope,
            )

            logger.debug(
                "Dispatch result: status=%s, handler=%s, duration=%.2fms "
                "(correlation_id=%s)",
                result.status,
                result.handler_id,
                result.duration_ms,
                msg_correlation_id,
            )

            # Gate ack/nack on dispatch status
            if result.is_successful():
                if hasattr(msg, "ack"):
                    await msg.ack()
            else:
                logger.warning(
                    "Dispatch failed (status=%s, error=%s), nacking message "
                    "(correlation_id=%s)",
                    result.status,
                    result.error_message,
                    msg_correlation_id,
                )
                if hasattr(msg, "nack"):
                    await msg.nack()

        except Exception:
            logger.exception(
                "Failed to dispatch message via engine (correlation_id=%s)",
                msg_correlation_id,
            )
            if hasattr(msg, "nack"):
                await msg.nack()

    return _on_message


__all__ = [
    "DISPATCH_ALIAS_ARCHIVE_MEMORY",
    "DISPATCH_ALIAS_EXPIRE_MEMORY",
    "DISPATCH_ALIAS_INTENT_CLASSIFIED",
    "DISPATCH_ALIAS_INTENT_QUERY_REQUESTED",
    "DISPATCH_ALIAS_MEMORY_RETRIEVAL_REQUESTED",
    "DISPATCH_ALIAS_RUNTIME_TICK",
    "create_dispatch_callback",
    "create_intent_classified_dispatch_handler",
    "create_intent_query_dispatch_handler",
    "create_lifecycle_dispatch_handler",
    "create_lifecycle_noop_dispatch_handler",
    "create_memory_dispatch_engine",
    "create_memory_retrieval_dispatch_handler",
]
