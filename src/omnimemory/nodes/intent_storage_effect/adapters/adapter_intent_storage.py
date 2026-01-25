# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent Storage Handler Adapter for the intent_storage_effect node.

This adapter wraps `AdapterIntentGraph` to implement the node's execute() interface,
translating between the storage request/response models and the underlying
graph adapter operations.

Example::

    import asyncio
    from omnimemory.nodes.intent_storage_effect import (
        HandlerIntentStorageAdapter,
        ModelIntentStorageRequest,
    )

    from omnimemory.handlers.adapters.models import ModelIntentClassificationOutput

    async def example():
        adapter = HandlerIntentStorageAdapter()
        await adapter.initialize(connection_uri="bolt://localhost:7687")

        # Store an intent
        request = ModelIntentStorageRequest(
            operation="store",
            session_id="session_123",
            intent_data=ModelIntentClassificationOutput(
                intent_category="debugging",
                confidence=0.92,
                keywords=["error", "fix"],
            ),
        )
        response = await adapter.execute(request)
        print(f"Stored intent: {response.intent_id}")

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for demo critical path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from omnimemory.handlers.adapters.models import ModelAdapterIntentGraphConfig
from omnimemory.nodes.intent_storage_effect.models import (
    ModelIntentRecordResponse,
    ModelIntentStorageRequest,
    ModelIntentStorageResponse,
)

if TYPE_CHECKING:
    from omnimemory.handlers.adapters import AdapterIntentGraph

# Runtime conditional import
_ADAPTER_AVAILABLE = False
_ADAPTER_IMPORT_ERROR: str | None = None

try:
    from omnimemory.handlers.adapters import AdapterIntentGraph

    _ADAPTER_AVAILABLE = True
except ImportError as e:
    _ADAPTER_IMPORT_ERROR = str(e)

    class AdapterIntentGraph:  # type: ignore[no-redef]
        """Stub for AdapterIntentGraph when dependencies not installed."""

        def __init__(self, config: object) -> None:
            raise ImportError(
                f"AdapterIntentGraph dependencies not available. "
                f"Ensure omnibase_infra is installed. "
                f"Original error: {_ADAPTER_IMPORT_ERROR}"
            )


logger = logging.getLogger(__name__)

__all__ = ["HandlerIntentStorageAdapter"]


class HandlerIntentStorageAdapter:
    """Handler adapter for intent storage operations.

    Wraps AdapterIntentGraph to provide the execute() interface expected
    by the ONEX node framework.

    Attributes:
        _adapter: The underlying AdapterIntentGraph instance.
        _initialized: Whether the adapter has been initialized.
        _config: Configuration for the intent graph adapter.
    """

    def __init__(
        self,
        config: ModelAdapterIntentGraphConfig | None = None,
    ) -> None:
        """Initialize the handler adapter.

        Args:
            config: Optional configuration for the intent graph adapter.
                If not provided, uses default configuration.
        """
        self._config = config or ModelAdapterIntentGraphConfig()
        self._adapter: AdapterIntentGraph | None = None
        self._initialized = False

    async def initialize(
        self,
        connection_uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the underlying adapter.

        Args:
            connection_uri: Memgraph connection URI.
            auth: Optional (username, password) tuple for authentication.
            options: Additional options passed to the graph handler.
        """
        if self._initialized:
            return

        if not _ADAPTER_AVAILABLE:
            raise ImportError(
                f"AdapterIntentGraph dependencies not available: {_ADAPTER_IMPORT_ERROR}"
            )

        self._adapter = AdapterIntentGraph(self._config)
        await self._adapter.initialize(
            connection_uri=connection_uri,
            auth=auth,
            options=options,
        )
        self._initialized = True
        logger.info("HandlerIntentStorageAdapter initialized")

    async def shutdown(self) -> None:
        """Shutdown the underlying adapter."""
        if self._adapter is not None:
            await self._adapter.shutdown()
            self._adapter = None
            self._initialized = False
            logger.info("HandlerIntentStorageAdapter shutdown")

    async def execute(
        self,
        request: ModelIntentStorageRequest,
    ) -> ModelIntentStorageResponse:
        """Execute an intent storage operation.

        Routes to the appropriate method based on request.operation:
        - store: Store a classified intent
        - get_session: Get intents for a session
        - get_distribution: Get intent category distribution

        Args:
            request: The storage request.

        Returns:
            Response with operation results.

        Raises:
            RuntimeError: If adapter not initialized.
        """
        if not self._initialized or self._adapter is None:
            return ModelIntentStorageResponse(
                status="error",
                error_message="Adapter not initialized. Call initialize() first.",
            )

        start_time = time.perf_counter()

        try:
            if request.operation == "store":
                response = await self._store_intent(request)
            elif request.operation == "get_session":
                response = await self._get_session_intents(request)
            elif request.operation == "get_distribution":
                response = await self._get_distribution(request)
            else:
                response = ModelIntentStorageResponse(
                    status="error",
                    error_message=f"Unknown operation: {request.operation}",
                )
        except AssertionError as e:
            # AssertionError indicates a programming bug - missing validation or
            # incorrect adapter state. Log at ERROR level with full traceback.
            logger.exception(
                "Assertion failed in intent storage operation '%s'. "
                "This indicates a programming error - request validation may be incomplete.",
                request.operation,
            )
            response = ModelIntentStorageResponse(
                status="error",
                error_message=f"Internal error: assertion failed - {e}",
            )
        except RuntimeError as e:
            # RuntimeError typically indicates adapter initialization issues
            # or infrastructure problems that the underlying adapter couldn't handle
            logger.error(
                "Runtime error during intent storage operation '%s': %s",
                request.operation,
                e,
            )
            response = ModelIntentStorageResponse(
                status="error",
                error_message=f"Runtime error: {e}",
            )
        except ValueError as e:
            # ValueError indicates invalid input data or configuration
            logger.warning(
                "Validation error during intent storage operation '%s': %s",
                request.operation,
                e,
            )
            response = ModelIntentStorageResponse(
                status="error",
                error_message=f"Validation error: {e}",
            )
        except Exception as e:
            # Safety net for truly unexpected errors - log at ERROR level with
            # full traceback to aid debugging. These should be rare since the
            # underlying adapter handles most exceptions internally.
            logger.exception(
                "Unexpected error (%s) during intent storage operation '%s'. "
                "This may indicate a bug or unhandled edge case.",
                type(e).__name__,
                request.operation,
            )
            response = ModelIntentStorageResponse(
                status="error",
                error_message=f"Unexpected error ({type(e).__name__}): {e}",
            )

        # Set execution time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response = response.model_copy(update={"execution_time_ms": elapsed_ms})
        return response

    async def _store_intent(
        self,
        request: ModelIntentStorageRequest,
    ) -> ModelIntentStorageResponse:
        """Store a classified intent."""
        assert self._adapter is not None
        assert request.session_id is not None
        assert request.intent_data is not None

        # Generate correlation_id if not provided
        correlation_id = request.correlation_id or uuid4()

        # Call the adapter
        result = await self._adapter.store_intent(
            session_id=request.session_id,
            intent_data=request.intent_data,
            correlation_id=correlation_id,
            user_context=request.user_context,
        )

        if result.status == "success":
            return ModelIntentStorageResponse(
                status="success",
                intent_id=result.intent_id,
                session_id=request.session_id,
                created=result.created,
                execution_time_ms=result.execution_time_ms,
            )
        else:
            return ModelIntentStorageResponse(
                status="error",
                session_id=request.session_id,
                error_message=result.error_message,
            )

    async def _get_session_intents(
        self,
        request: ModelIntentStorageRequest,
    ) -> ModelIntentStorageResponse:
        """Get intents for a specific session."""
        assert self._adapter is not None
        assert request.session_id is not None

        result = await self._adapter.get_session_intents(
            session_id=request.session_id,
            min_confidence=request.min_confidence,
            limit=request.limit,
        )

        if result.status == "success":
            # Convert to response model format
            intents = [
                ModelIntentRecordResponse(
                    intent_id=intent.intent_id,
                    intent_category=intent.intent_category,
                    confidence=intent.confidence,
                    keywords=intent.keywords,
                    created_at_utc=intent.created_at_utc.isoformat(),
                    correlation_id=intent.correlation_id,
                )
                for intent in result.intents
            ]
            return ModelIntentStorageResponse(
                status="success",
                intents=intents,
                total_count=result.total_count,
                execution_time_ms=result.execution_time_ms,
            )
        elif result.status == "not_found":
            return ModelIntentStorageResponse(
                status="not_found",
                error_message=f"Session not found: {request.session_id}",
            )
        elif result.status == "no_results":
            return ModelIntentStorageResponse(
                status="no_results",
                intents=[],
                total_count=0,
            )
        else:
            return ModelIntentStorageResponse(
                status="error",
                error_message=result.error_message,
            )

    async def _get_distribution(
        self,
        request: ModelIntentStorageRequest,
    ) -> ModelIntentStorageResponse:
        """Get intent category distribution."""
        assert self._adapter is not None

        result = await self._adapter.get_intent_distribution(
            time_range_hours=request.time_range_hours,
        )

        if result.status == "success":
            return ModelIntentStorageResponse(
                status="success",
                distribution=result.distribution,
                total_intents=result.total_intents,
                time_range_hours=result.time_range_hours,
                execution_time_ms=result.execution_time_ms,
            )
        else:
            return ModelIntentStorageResponse(
                status="error",
                error_message=result.error_message,
            )
