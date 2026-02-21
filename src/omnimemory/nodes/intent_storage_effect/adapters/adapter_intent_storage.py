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
from uuid import uuid4

import structlog

from omnimemory.handlers.adapters import AdapterIntentGraph
from omnimemory.handlers.adapters.models import ModelAdapterIntentGraphConfig
from omnimemory.models.utils.model_circuit_breaker_config import (
    ModelCircuitBreakerConfig,
)
from omnimemory.nodes.intent_storage_effect.models import (
    ModelIntentRecordResponse,
    ModelIntentStorageRequest,
    ModelIntentStorageResponse,
)
from omnimemory.utils.concurrency import CircuitBreaker, CircuitBreakerOpenError
from omnimemory.utils.pii_detector import PIIDetector

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)

__all__ = ["HandlerIntentStorageAdapter"]

# Default circuit breaker configuration for external adapter calls
_DEFAULT_CIRCUIT_BREAKER_CONFIG = ModelCircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=3,
    timeout=30.0,
)


class HandlerIntentStorageAdapter:
    """Handler adapter for intent storage operations.

    Wraps AdapterIntentGraph to provide the execute() interface expected
    by the ONEX node framework. Includes circuit breaker protection to
    prevent cascading failures when the underlying adapter is unavailable.

    Attributes:
        _adapter: The underlying AdapterIntentGraph instance.
        _initialized: Whether the adapter has been initialized.
        _config: Configuration for the intent graph adapter.
        _circuit_breaker: Circuit breaker for external adapter calls.
        _pii_detector: PII detector for user input sanitization.
    """

    def __init__(
        self,
        config: ModelAdapterIntentGraphConfig | None = None,
        circuit_breaker_config: ModelCircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize the handler adapter.

        Args:
            config: Optional configuration for the intent graph adapter.
                If not provided, uses default configuration.
            circuit_breaker_config: Optional circuit breaker configuration.
                If not provided, uses default configuration with 5 failure
                threshold, 60s recovery timeout, and 3 success threshold.
        """
        self._config = config or ModelAdapterIntentGraphConfig()
        self._adapter: AdapterIntentGraph | None = None
        self._initialized = False

        # Initialize circuit breaker for external adapter calls
        cb_config = circuit_breaker_config or _DEFAULT_CIRCUIT_BREAKER_CONFIG
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.failure_threshold,
            recovery_timeout=float(cb_config.recovery_timeout),
            success_threshold=cb_config.success_threshold,
        )

        # Initialize PII detector for user input sanitization
        self._pii_detector = PIIDetector()

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter has been initialized.

        Returns:
            True if initialize() has been called successfully and
            shutdown() has not been called, False otherwise.
        """
        return self._initialized

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

        Uses circuit breaker protection to prevent cascading failures when
        the underlying adapter is unavailable.

        Args:
            request: The storage request.

        Returns:
            Response with operation results. If adapter is not initialized,
            returns a response with status="error" and error_message
            instructing to call initialize() first.

        Note:
            This method catches and handles the following exceptions:
            - CircuitBreakerOpenError: When circuit breaker is open
            - ValueError: Invalid input data or configuration
            - Exception: Any other unexpected errors
        """
        start_time = time.perf_counter()

        if not self._initialized or self._adapter is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ModelIntentStorageResponse(
                status="error",
                error_message="Adapter not initialized. Call initialize() first.",
                execution_time_ms=elapsed_ms,
            )
        response: ModelIntentStorageResponse

        try:
            # Route to appropriate handler with circuit breaker protection
            if request.operation == "store":
                response = await self._circuit_breaker.call(self._store_intent, request)
            elif request.operation == "get_session":
                response = await self._circuit_breaker.call(
                    self._get_session_intents, request
                )
            elif request.operation == "get_distribution":
                response = await self._circuit_breaker.call(
                    self._get_distribution, request
                )
            else:
                response = ModelIntentStorageResponse(
                    status="error",
                    error_message=f"Unknown operation: {request.operation}",
                )
        except CircuitBreakerOpenError as e:
            # Circuit breaker is open - external service unavailable
            structured_logger.warning(
                "circuit_breaker_open",
                operation=request.operation,
                failure_count=self._circuit_breaker.failure_count,
                state=self._circuit_breaker.state.value,
            )
            response = ModelIntentStorageResponse(
                status="error",
                error_message=f"Service temporarily unavailable: {e}",
            )
        except ValueError as e:
            # ValueError indicates invalid input data or configuration
            structured_logger.warning(
                "validation_error",
                operation=request.operation,
                error=str(e),
            )
            response = ModelIntentStorageResponse(
                status="error",
                error_message=f"Validation error: {e}",
            )
        except Exception as e:
            # Safety net for truly unexpected errors - log at ERROR level with
            # full traceback to aid debugging. These should be rare since the
            # underlying adapter handles most exceptions internally.
            structured_logger.exception(
                "unexpected_error",
                error_type=type(e).__name__,
                operation=request.operation,
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
        """Store a classified intent.

        Args:
            request: The storage request with session_id and intent_data.

        Returns:
            Response with storage result.

        Raises:
            ValueError: If adapter is not initialized or required fields are missing.
        """
        # Explicit validation (asserts can be disabled with python -O)
        if self._adapter is None:
            raise ValueError("Adapter not initialized")
        if request.session_id is None:
            raise ValueError("session_id is required for store operation")
        if request.intent_data is None:
            raise ValueError("intent_data is required for store operation")

        # PII detection/redaction for user_context before persistence
        sanitized_context = request.user_context
        if sanitized_context:
            pii_result = self._pii_detector.detect_pii(
                sanitized_context, sensitivity_level="medium"
            )
            if pii_result.has_pii:
                # Use sanitized content with PII redacted
                sanitized_context = pii_result.sanitized_content
                structured_logger.info(
                    "pii_redacted_before_storage",
                    pii_types=[t.value for t in pii_result.pii_types_detected],
                    session_id=request.session_id,
                )

        # Generate correlation_id if not provided
        correlation_id = (
            str(request.correlation_id) if request.correlation_id else str(uuid4())
        )

        # Call the adapter with classification output directly
        result = await self._adapter.store_intent(
            session_id=request.session_id,
            intent_data=request.intent_data,
            correlation_id=correlation_id,
        )

        if result.success:
            return ModelIntentStorageResponse(
                status="success",
                intent_id=result.intent_id,
                session_id=request.session_id,
                created=result.created,
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
        """Get intents for a specific session.

        Args:
            request: The query request with session_id.

        Returns:
            Response with list of intents for the session.

        Raises:
            ValueError: If adapter is not initialized or session_id is missing.
        """
        # Explicit validation (asserts can be disabled with python -O)
        if self._adapter is None:
            raise ValueError("Adapter not initialized")
        if request.session_id is None:
            raise ValueError("session_id is required for get_session operation")

        result = await self._adapter.get_session_intents(
            session_id=request.session_id,
            min_confidence=request.min_confidence,
            limit=request.limit,
        )

        if not result.success:
            return ModelIntentStorageResponse(
                status="error",
                error_message=result.error_message,
            )

        if not result.intents:
            return ModelIntentStorageResponse(
                status="no_results",
                intents=[],
                total_count=0,
            )

        # Convert local ModelIntentRecord to response model format
        intents: list[ModelIntentRecordResponse] = []
        for intent in result.intents:
            intents.append(
                ModelIntentRecordResponse(
                    intent_id=intent.intent_id,
                    intent_category=intent.intent_category,
                    confidence=intent.confidence,
                    keywords=list(intent.keywords),
                    created_at_utc=intent.created_at.isoformat(),
                    correlation_id=intent.correlation_id,
                )
            )
        return ModelIntentStorageResponse(
            status="success",
            intents=intents,
            total_count=len(intents),
        )

    async def _get_distribution(
        self,
        request: ModelIntentStorageRequest,
    ) -> ModelIntentStorageResponse:
        """Get intent category distribution.

        Args:
            request: The query request with time_range_hours.

        Returns:
            Response with intent category distribution.

        Raises:
            ValueError: If adapter is not initialized.
        """
        # Explicit validation (asserts can be disabled with python -O)
        if self._adapter is None:
            raise ValueError("Adapter not initialized")

        result = await self._adapter.get_intent_distribution(
            time_range_hours=request.time_range_hours,
        )

        if result.status == "success":
            return ModelIntentStorageResponse(
                status="success",
                distribution=dict(result.distribution),
                total_intents=result.total_intents,
                time_range_hours=result.time_range_hours,
                execution_time_ms=result.execution_time_ms,
            )
        else:
            return ModelIntentStorageResponse(
                status="error",
                error_message=result.error_message,
            )
