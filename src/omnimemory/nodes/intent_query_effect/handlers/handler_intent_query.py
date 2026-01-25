# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for intent query operations via Kafka events.

Processes intent query requests (distribution, session, recent) and returns
responses via the event bus. Part of the event-driven architecture where
OmniDash queries intent data without direct database access.

Example::

    from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
    from omnimemory.handlers.adapters import AdapterIntentGraph

    handler = HandlerIntentQuery()
    await handler.initialize(adapter)
    response = await handler.execute(request_event)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from omnibase_core.models.events import (
    ModelIntentQueryRequestedEvent,
    ModelIntentQueryResponseEvent,
)

from omnimemory.nodes.intent_query_effect.models import ModelHandlerIntentQueryConfig
from omnimemory.nodes.intent_query_effect.utils import map_intent_records

if TYPE_CHECKING:
    from omnimemory.handlers.adapters import AdapterIntentGraph

logger = logging.getLogger(__name__)

# Contract SLA: max response time in milliseconds
_CONTRACT_MAX_RESPONSE_TIME_MS = 100.0

__all__ = ["HandlerIntentQuery"]


class HandlerIntentQuery:
    """Handler for event-driven intent queries.

    Routes intent query requests to the appropriate adapter method and
    constructs response events with proper correlation tracking.

    This handler is designed for use in event-driven architectures where
    clients (e.g., dashboards) request intent data via Kafka events rather
    than direct database queries.

    Supported query types:
        - distribution: Get intent counts grouped by category
        - session: Get intents for a specific session
        - recent: Get recent intents across all sessions

    Attributes:
        config: Handler configuration controlling timeouts and defaults.
        is_initialized: Whether the handler has been initialized.

    Example::

        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
        from omnimemory.handlers.adapters import (
            AdapterIntentGraph,
            ModelAdapterIntentGraphConfig,
        )

        # Create and initialize adapter
        adapter_config = ModelAdapterIntentGraphConfig()
        adapter = AdapterIntentGraph(adapter_config)
        await adapter.initialize(connection_uri="bolt://localhost:7687")

        # Create and initialize handler
        handler = HandlerIntentQuery()
        await handler.initialize(adapter)

        # Execute a query
        request = ModelIntentQueryRequestedEvent(
            query_type="distribution",
            time_range_hours=24,
        )
        response = await handler.execute(request)
        if response.status == "success":
            print(f"Distribution: {response.distribution}")

        await handler.shutdown()
    """

    def __init__(self, config: ModelHandlerIntentQueryConfig | None = None) -> None:
        """Initialize the handler.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or ModelHandlerIntentQueryConfig()
        self._adapter: AdapterIntentGraph | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelHandlerIntentQueryConfig:
        """Get handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if handler is initialized."""
        return self._initialized

    async def initialize(self, adapter: AdapterIntentGraph) -> None:
        """Initialize with adapter dependency.

        Thread-safe initialization using asyncio.Lock.

        Args:
            adapter: The intent graph adapter for database operations.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            self._adapter = adapter
            self._initialized = True
            logger.info("HandlerIntentQuery initialized")

    async def execute(
        self,
        request: ModelIntentQueryRequestedEvent,
    ) -> ModelIntentQueryResponseEvent:
        """Execute an intent query request.

        Routes the request to the appropriate handler method based on query_type.
        This method never raises - all errors are captured in the response.

        Args:
            request: The incoming query request event.

        Returns:
            Response event with query results or error details.
        """
        if not self._initialized or self._adapter is None:
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type=request.query_type,
                error_message="Handler not initialized",
                correlation_id=request.correlation_id,
            )

        start = time.monotonic()

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                match request.query_type:
                    case "distribution":
                        return await self._handle_distribution(request, start)
                    case "session":
                        return await self._handle_session(request, start)
                    case "recent":
                        return await self._handle_recent(request, start)
                    case _:
                        return ModelIntentQueryResponseEvent.from_error(
                            query_id=request.query_id,
                            query_type=request.query_type,
                            error_message=f"Unknown query_type: {request.query_type}",
                            correlation_id=request.correlation_id,
                        )
        except TimeoutError:
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type=request.query_type,
                error_message=f"Handler timeout after {self._config.timeout_seconds}s",
                correlation_id=request.correlation_id,
            )
        except Exception as e:
            logger.exception("Error executing intent query: %s", request.query_type)
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type=request.query_type,
                error_message=str(e),
                correlation_id=request.correlation_id,
            )

    async def _handle_distribution(
        self,
        request: ModelIntentQueryRequestedEvent,
        start: float,
    ) -> ModelIntentQueryResponseEvent:
        """Handle distribution query - get intent counts by category.

        Args:
            request: The query request event.
            start: Start time for execution timing.

        Returns:
            Response event with distribution data or error.
        """
        assert self._adapter is not None

        result = await self._adapter.get_intent_distribution(
            time_range_hours=request.time_range_hours,
        )

        execution_time_ms = (time.monotonic() - start) * 1000

        if execution_time_ms > _CONTRACT_MAX_RESPONSE_TIME_MS:
            logger.warning(
                "Query %s exceeded SLA: %.2fms (max: %.2fms)",
                request.query_type,
                execution_time_ms,
                _CONTRACT_MAX_RESPONSE_TIME_MS,
            )

        if result.status == "error":
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type="distribution",
                error_message=result.error_message or "Unknown error",
                correlation_id=request.correlation_id,
            )

        return ModelIntentQueryResponseEvent.create_distribution_response(
            query_id=request.query_id,
            distribution=result.distribution,
            time_range_hours=request.time_range_hours,
            execution_time_ms=execution_time_ms,
            correlation_id=request.correlation_id,
        )

    async def _handle_session(
        self,
        request: ModelIntentQueryRequestedEvent,
        start: float,
    ) -> ModelIntentQueryResponseEvent:
        """Handle session query - get intents for a specific session.

        Args:
            request: The query request event.
            start: Start time for execution timing.

        Returns:
            Response event with session intents or error.
        """
        assert self._adapter is not None

        if not request.session_ref:
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type="session",
                error_message="session_ref is required for session queries",
                correlation_id=request.correlation_id,
            )

        result = await self._adapter.get_session_intents(
            session_id=request.session_ref,
            min_confidence=request.min_confidence
            if request.min_confidence is not None and request.min_confidence > 0
            else None,
            limit=request.limit,
        )

        execution_time_ms = (time.monotonic() - start) * 1000

        if execution_time_ms > _CONTRACT_MAX_RESPONSE_TIME_MS:
            logger.warning(
                "Query %s exceeded SLA: %.2fms (max: %.2fms)",
                request.query_type,
                execution_time_ms,
                _CONTRACT_MAX_RESPONSE_TIME_MS,
            )

        if result.status == "error":
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type="session",
                error_message=result.error_message or "Unknown error",
                correlation_id=request.correlation_id,
            )

        # Session queries need session_ref populated in records for mapping
        # Create new instances instead of mutating adapter results
        intents_with_ref = []
        for intent in result.intents:
            if intent.session_ref is None:
                intents_with_ref.append(
                    intent.model_copy(update={"session_ref": request.session_ref})
                )
            else:
                intents_with_ref.append(intent)

        try:
            payloads = map_intent_records(intents_with_ref)
        except ValueError as e:
            logger.warning("Failed to map intent records: %s", e)
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type="session",
                error_message=f"Failed to map intent records: {e}",
                correlation_id=request.correlation_id,
            )

        return ModelIntentQueryResponseEvent.create_session_response(
            query_id=request.query_id,
            intents=payloads,
            execution_time_ms=execution_time_ms,
            correlation_id=request.correlation_id,
        )

    async def _handle_recent(
        self,
        request: ModelIntentQueryRequestedEvent,
        start: float,
    ) -> ModelIntentQueryResponseEvent:
        """Handle recent query - get recent intents across all sessions.

        Args:
            request: The query request event.
            start: Start time for execution timing.

        Returns:
            Response event with recent intents or error.
        """
        assert self._adapter is not None

        result = await self._adapter.get_recent_intents(
            time_range_hours=request.time_range_hours,
            min_confidence=request.min_confidence
            if request.min_confidence is not None and request.min_confidence > 0
            else None,
            limit=request.limit,
        )

        execution_time_ms = (time.monotonic() - start) * 1000

        if execution_time_ms > _CONTRACT_MAX_RESPONSE_TIME_MS:
            logger.warning(
                "Query %s exceeded SLA: %.2fms (max: %.2fms)",
                request.query_type,
                execution_time_ms,
                _CONTRACT_MAX_RESPONSE_TIME_MS,
            )

        if result.status == "error":
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type="recent",
                error_message=result.error_message or "Unknown error",
                correlation_id=request.correlation_id,
            )

        try:
            payloads = map_intent_records(result.intents)
        except ValueError as e:
            logger.warning("Failed to map intent records: %s", e)
            return ModelIntentQueryResponseEvent.from_error(
                query_id=request.query_id,
                query_type="recent",
                error_message=f"Failed to map intent records: {e}",
                correlation_id=request.correlation_id,
            )

        return ModelIntentQueryResponseEvent.create_recent_response(
            query_id=request.query_id,
            intents=payloads,
            time_range_hours=request.time_range_hours,
            execution_time_ms=execution_time_ms,
            correlation_id=request.correlation_id,
        )

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources.

        Safe to call multiple times. Does not shutdown the adapter
        (caller is responsible for adapter lifecycle).
        """
        if self._initialized:
            self._adapter = None
            self._initialized = False
            logger.info("HandlerIntentQuery shutdown")
