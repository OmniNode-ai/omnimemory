# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Direct protocol handler for intent storage and query operations.

Wraps AdapterIntentGraph following the omnibase_infra container-driven pattern.
Unlike event-driven handlers, this handler provides synchronous direct method
calls for API layer integration.

Protocol Compliance:
    Implements :class:`~omnimemory.protocols.ProtocolHandlerIntent` protocol
    for contract-driven development and type-safe dependency injection.

Resilience:
    Includes circuit breaker pattern for fault tolerance. The circuit breaker
    protects against cascading failures when the graph database is unavailable
    or experiencing issues. Configuration via initialize() options:
    - circuit_breaker_threshold: Failures before opening (default: 5)
    - circuit_breaker_reset_timeout: Seconds before half-open (default: 60.0)

Architecture:
    - Container-driven initialization (receives ModelONEXContainer)
    - Handler owns adapter lifecycle (creates on init, closes on shutdown)
    - Delegates all operations to AdapterIntentGraph
    - Uses structured logging with correlation IDs
    - Circuit breaker wraps all adapter operations

Operations:
    - store_intent(): Store an intent classification linked to a session
    - query_session(): Retrieve intents for a specific session
    - query_distribution(): Get intent category statistics

Example::

    from omnibase_core.container import ModelONEXContainer
    from omnimemory.handlers import HandlerIntent

    container = ModelONEXContainer()
    handler = HandlerIntent(container)

    await handler.initialize(
        connection_uri="bolt://localhost:7687",
    )

    # Store an intent
    from uuid import uuid4
    from omnimemory.handlers.adapters.models import ModelIntentClassificationOutput

    result = await handler.store_intent(
        session_id="session_123",
        intent_data=ModelIntentClassificationOutput(
            intent_category="debugging",
            confidence=0.92,
            keywords=["error", "traceback"],
        ),
        correlation_id=uuid4(),
    )

    # Query session intents
    query_result = await handler.query_session(
        session_id="session_123",
        min_confidence=0.5,
    )

    await handler.shutdown()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1536.

.. versionchanged:: 0.2.0
    Added ProtocolHandlerIntent compliance and circuit breaker resilience.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.models.intelligence import (
    ModelIntentClassificationOutput,
    ModelIntentQueryResult,
    ModelIntentStorageResult,
)
from pydantic import BaseModel, ConfigDict, Field

from omnimemory.handlers.adapters.adapter_intent_graph import AdapterIntentGraph
from omnimemory.handlers.adapters.models import (
    ModelAdapterIntentGraphConfig,
    ModelIntentDistributionResult,
    ModelIntentGraphHealth,
)
from omnimemory.utils.concurrency import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

__all__ = [
    "CircuitBreakerOpenError",
    "HandlerIntent",
    "ModelHandlerIntentMetadata",
]

logger = logging.getLogger(__name__)

# Handler identifier for logging and registration
HANDLER_ID_INTENT: str = "intent-handler"


class ModelHandlerIntentMetadata(  # omnimemory-model-exempt: handler metadata
    BaseModel
):
    """Metadata describing the intent handler capabilities.

    Returned by HandlerIntent.describe() to provide information about
    handler type, capabilities, and configuration.

    Attributes:
        handler_type: The handler type identifier ("intent").
        capabilities: List of supported operations.
        adapter_type: Type of adapter being wrapped.
        initialized: Whether the handler is currently initialized.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    handler_type: str = Field(
        ...,
        description="Handler type identifier",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of supported operations",
    )
    adapter_type: str = Field(
        default="AdapterIntentGraph",
        description="Type of adapter being wrapped",
    )
    initialized: bool = Field(
        default=False,
        description="Whether the handler is currently initialized",
    )


class HandlerIntent:
    """Direct protocol handler for intent storage and query operations.

    Implements: :class:`~omnimemory.protocols.ProtocolHandlerIntent`

    Wraps AdapterIntentGraph following the omnibase_infra container-driven pattern.
    Unlike HandlerIntentQuery (event-driven), this handler provides synchronous
    direct method calls for API layer integration.

    The handler owns the adapter lifecycle:
    - Creates adapter during initialize()
    - Closes adapter during shutdown()

    Protocol Compliance:
        Implements :class:`~omnimemory.protocols.ProtocolHandlerIntent` protocol:
        - handler_type property returning "intent"
        - is_initialized property for state checking
        - initialize(), shutdown() lifecycle methods
        - store_intent(), query_session(), query_distribution() operations
        - health_check(), describe() introspection methods

    Circuit Breaker:
        All adapter operations are protected by a circuit breaker pattern to
        prevent cascading failures. The circuit breaker:
        - Opens after consecutive failures (configurable threshold)
        - Automatically attempts recovery after reset timeout
        - Raises CircuitBreakerOpenError when open (fail-fast behavior)

        Configuration via initialize() options:
        - circuit_breaker_threshold: Failures before opening (default: 5)
        - circuit_breaker_reset_timeout: Seconds before half-open (default: 60.0)

    Thread Safety:
        Uses asyncio.Lock for initialization to prevent race conditions
        when multiple coroutines call initialize() concurrently.

    Error Handling:
        All business operation methods return error status in response
        models rather than raising exceptions. This allows API layers
        to handle errors gracefully without try/except blocks.

        Exception: CircuitBreakerOpenError is raised when the circuit is
        open to provide immediate feedback that the service is unavailable.

    Attributes:
        handler_type: Returns "intent" as handler type identifier.
        is_initialized: Returns True if handler is ready for operations.

    Example::

        from omnibase_core.container import ModelONEXContainer
        from omnimemory.handlers import HandlerIntent

        container = ModelONEXContainer()
        handler = HandlerIntent(container)

        await handler.initialize(connection_uri="bolt://localhost:7687")

        result = await handler.store_intent(
            session_id="sess_123",
            intent_data=ModelIntentClassificationOutput(
                intent_category="code_generation",
                confidence=0.95,
                keywords=["python", "function"],
            ),
            correlation_id=uuid4(),
        )

        await handler.shutdown()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerIntent with ONEX container for dependency injection.

        Args:
            container: ONEX container providing dependency injection for
                services, configuration, and runtime context.

        Note:
            The container is stored for interface compliance with the standard
            ONEX handler pattern and to enable future DI-based service resolution.
            The adapter is NOT created here; it is created during initialize().
            The circuit breaker is also created during initialize() with
            configuration from the options parameter.
        """
        self._container = container
        self._adapter: AdapterIntentGraph | None = None
        self._adapter_config: ModelAdapterIntentGraphConfig | None = None
        self._initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._connection_uri: str = ""
        # Circuit breaker for adapter operations (initialized in initialize())
        self._circuit_breaker: CircuitBreaker | None = None

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            String "intent" identifying this handler type.
        """
        return "intent"

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized.

        Returns:
            True if handler is initialized and ready for operations.
        """
        return self._initialized

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"HandlerIntent(initialized={self._initialized})"

    async def initialize(
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize handler and create adapter.

        Establishes connection to the graph database by creating and
        initializing the underlying AdapterIntentGraph. This method
        is idempotent - calling it multiple times after successful
        initialization is a no-op.

        Args:
            connection_uri: Graph database URI (e.g., "bolt://localhost:7687").
            auth: Optional (username, password) tuple for authentication.
            options: Additional configuration options:
                - timeout_seconds: Operation timeout (default: 30.0)
                - max_intents_per_session: Max intents per query (default: 100)
                - default_confidence_threshold: Min confidence filter (default: 0.0)
                - auto_create_indexes: Create indexes on init (default: True)

        Raises:
            RuntimeError: If initialization fails or times out.
            ValueError: If connection_uri is malformed.
        """
        init_correlation_id = uuid4()

        async with self._init_lock:
            if self._initialized:
                logger.debug(
                    "Handler already initialized, skipping",
                    extra={
                        "handler": HANDLER_ID_INTENT,
                        "correlation_id": str(init_correlation_id),
                    },
                )
                return

            logger.info(
                "Initializing %s",
                self.__class__.__name__,
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(init_correlation_id),
                },
            )

            try:
                # Build adapter config from options
                opts = dict(options) if options else {}

                timeout_raw = opts.get("timeout_seconds", 30.0)
                timeout_seconds = (
                    float(timeout_raw)
                    if isinstance(timeout_raw, int | float | str)
                    else 30.0
                )

                max_intents_raw = opts.get("max_intents_per_session", 100)
                max_intents_per_session = (
                    int(max_intents_raw)
                    if isinstance(max_intents_raw, int | float | str)
                    else 100
                )

                threshold_raw = opts.get("default_confidence_threshold", 0.0)
                default_confidence_threshold = (
                    float(threshold_raw)
                    if isinstance(threshold_raw, int | float | str)
                    else 0.0
                )

                auto_indexes_raw = opts.get("auto_create_indexes", True)
                auto_create_indexes = bool(auto_indexes_raw)

                # Circuit breaker configuration
                cb_threshold_raw = opts.get("circuit_breaker_threshold", 5)
                circuit_breaker_threshold = (
                    int(cb_threshold_raw)
                    if isinstance(cb_threshold_raw, int | float | str)
                    else 5
                )

                cb_reset_raw = opts.get("circuit_breaker_reset_timeout", 60.0)
                circuit_breaker_reset_timeout = (
                    float(cb_reset_raw)
                    if isinstance(cb_reset_raw, int | float | str)
                    else 60.0
                )

                # Create circuit breaker for adapter protection
                self._circuit_breaker = CircuitBreaker(
                    failure_threshold=circuit_breaker_threshold,
                    recovery_timeout=circuit_breaker_reset_timeout,
                    success_threshold=1,  # Close after 1 success in half-open
                )

                logger.debug(
                    "Circuit breaker configured",
                    extra={
                        "handler": HANDLER_ID_INTENT,
                        "correlation_id": str(init_correlation_id),
                        "failure_threshold": circuit_breaker_threshold,
                        "recovery_timeout": circuit_breaker_reset_timeout,
                    },
                )

                self._adapter_config = ModelAdapterIntentGraphConfig(
                    timeout_seconds=timeout_seconds,
                    max_intents_per_session=max_intents_per_session,
                    default_confidence_threshold=default_confidence_threshold,
                    auto_create_indexes=auto_create_indexes,
                )

                # Create adapter with config and container
                self._adapter = AdapterIntentGraph(
                    config=self._adapter_config,
                    container=self._container,
                )

                # Initialize adapter
                await self._adapter.initialize(
                    connection_uri=connection_uri,
                    auth=auth,
                    options=options,
                )

                self._connection_uri = connection_uri
                self._initialized = True

                logger.info(
                    "%s initialized successfully",
                    self.__class__.__name__,
                    extra={
                        "handler": HANDLER_ID_INTENT,
                        "correlation_id": str(init_correlation_id),
                    },
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize %s: %s",
                    self.__class__.__name__,
                    e,
                    extra={
                        "handler": HANDLER_ID_INTENT,
                        "correlation_id": str(init_correlation_id),
                    },
                )
                # Cleanup partial initialization
                if self._adapter is not None:
                    try:
                        await self._adapter.shutdown()
                    except Exception as cleanup_error:
                        logger.warning(
                            "Error during adapter cleanup: %s",
                            cleanup_error,
                            extra={
                                "handler": HANDLER_ID_INTENT,
                                "correlation_id": str(init_correlation_id),
                            },
                        )
                    self._adapter = None
                raise RuntimeError(f"Initialization failed: {e}") from e

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Shutdown handler and close adapter.

        Gracefully shuts down the handler by closing the underlying
        adapter and releasing resources. Safe to call multiple times.

        Args:
            timeout_seconds: Maximum time to wait for shutdown. Defaults to 30.0.
        """
        shutdown_correlation_id = uuid4()

        async with self._init_lock:
            if not self._initialized:
                logger.debug(
                    "Handler not initialized, skipping shutdown",
                    extra={
                        "handler": HANDLER_ID_INTENT,
                        "correlation_id": str(shutdown_correlation_id),
                    },
                )
                return

            logger.info(
                "Shutting down %s",
                self.__class__.__name__,
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(shutdown_correlation_id),
                    "timeout_seconds": timeout_seconds,
                },
            )

            if self._adapter is not None:
                try:
                    await asyncio.wait_for(
                        self._adapter.shutdown(),
                        timeout=timeout_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        "Adapter shutdown timed out after %ss",
                        timeout_seconds,
                        extra={
                            "handler": HANDLER_ID_INTENT,
                            "correlation_id": str(shutdown_correlation_id),
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Error during adapter shutdown: %s",
                        e,
                        extra={
                            "handler": HANDLER_ID_INTENT,
                            "correlation_id": str(shutdown_correlation_id),
                        },
                    )
                self._adapter = None

            self._initialized = False
            self._adapter_config = None
            self._circuit_breaker = None

            logger.info(
                "%s shutdown complete",
                self.__class__.__name__,
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(shutdown_correlation_id),
                },
            )

    def _ensure_initialized(self) -> tuple[AdapterIntentGraph, CircuitBreaker]:
        """Ensure handler is initialized and return adapter with circuit breaker.

        Returns:
            Tuple of (AdapterIntentGraph, CircuitBreaker) for protected operations.

        Raises:
            RuntimeError: If handler is not initialized.
        """
        if (
            not self._initialized
            or self._adapter is None
            or self._circuit_breaker is None
        ):
            raise RuntimeError(
                "HandlerIntent not initialized. Call initialize() first."
            )
        return self._adapter, self._circuit_breaker

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def store_intent(
        self,
        session_id: str,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: str,
    ) -> ModelIntentStorageResult:
        """Store an intent classification linked to a session.

        Delegates to AdapterIntentGraph.store_intent() using MERGE semantics
        to create or update the session and intent nodes.

        Args:
            session_id: Unique identifier for the session.
            intent_data: The intent classification output to store.
            correlation_id: Correlation ID for request tracing.
            user_context: Optional user context string for the session.

        Returns:
            ModelIntentStorageResult indicating success or failure.
            On success, includes the intent_id and whether a new
            intent was created vs merged.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open.

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead. However,
            CircuitBreakerOpenError is raised for immediate fail-fast
            behavior when the service is unavailable.
        """
        try:
            adapter, circuit_breaker = self._ensure_initialized()
        except RuntimeError as e:
            logger.error(
                "store_intent called on uninitialized handler",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                },
            )
            return ModelIntentStorageResult(
                success=False,
                error_message=str(e),
            )

        # Check circuit breaker before attempting operation
        if not circuit_breaker.should_allow_request():
            logger.warning(
                "Circuit breaker is open, failing fast",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "circuit_state": circuit_breaker.state.value,
                },
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker is {circuit_breaker.state.value}"
            )

        logger.debug(
            "Storing intent for session %s",
            session_id,
            extra={
                "handler": HANDLER_ID_INTENT,
                "correlation_id": correlation_id,
                "session_id": session_id,
                "intent_category": intent_data.intent_category.value
                if hasattr(intent_data.intent_category, "value")
                else intent_data.intent_category,
            },
        )

        try:
            result = await adapter.store_intent(
                session_id=session_id,
                intent_data=intent_data,
                correlation_id=correlation_id,
            )
            # Record success if operation completed without exception
            # Business errors (e.g., validation) are not circuit breaker failures
            if result.success:
                circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(
                "store_intent failed, recorded circuit breaker failure",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "error": str(e),
                    "circuit_state": circuit_breaker.state.value,
                    "failure_count": circuit_breaker.failure_count,
                },
            )
            raise

    async def query_session(
        self,
        session_id: str,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> ModelIntentQueryResult:
        """Retrieve intents for a specific session.

        Delegates to AdapterIntentGraph.get_session_intents() with optional
        filtering by confidence threshold and result limit.

        Args:
            session_id: The session identifier to query.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                Defaults to config.default_confidence_threshold.
            limit: Maximum number of results to return.
                Defaults to config.max_intents_per_session.

        Returns:
            ModelIntentQueryResult with the list of intents or error status.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open.

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead. However,
            CircuitBreakerOpenError is raised for immediate fail-fast
            behavior when the service is unavailable.
        """
        query_correlation_id = uuid4()

        try:
            adapter, circuit_breaker = self._ensure_initialized()
        except RuntimeError as e:
            logger.error(
                "query_session called on uninitialized handler",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(query_correlation_id),
                    "session_id": session_id,
                },
            )
            return ModelIntentQueryResult(
                success=False,
                error_message=str(e),
            )

        # Check circuit breaker before attempting operation
        if not circuit_breaker.should_allow_request():
            logger.warning(
                "Circuit breaker is open, failing fast",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(query_correlation_id),
                    "session_id": session_id,
                    "circuit_state": circuit_breaker.state.value,
                },
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker is {circuit_breaker.state.value}"
            )

        logger.debug(
            "Querying intents for session %s",
            session_id,
            extra={
                "handler": HANDLER_ID_INTENT,
                "correlation_id": str(query_correlation_id),
                "session_id": session_id,
                "min_confidence": min_confidence,
                "limit": limit,
            },
        )

        try:
            result = await adapter.get_session_intents(
                session_id=session_id,
                min_confidence=min_confidence if min_confidence is not None else 0.0,
                limit=limit,
            )
            # Record success if operation completed without exception
            if result.success:
                circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(
                "query_session failed, recorded circuit breaker failure",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(query_correlation_id),
                    "session_id": session_id,
                    "error": str(e),
                    "circuit_state": circuit_breaker.state.value,
                    "failure_count": circuit_breaker.failure_count,
                },
            )
            raise

    async def query_distribution(
        self,
        time_range_hours: int = 24,
    ) -> ModelIntentDistributionResult:
        """Get intent category distribution for analytics.

        Delegates to AdapterIntentGraph.get_intent_distribution() to return
        the count of intents per category within the specified time range.

        Args:
            time_range_hours: Number of hours to look back from now.
                Defaults to 24 hours.

        Returns:
            ModelIntentDistributionResult with distribution data or error status.

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open.

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead. However,
            CircuitBreakerOpenError is raised for immediate fail-fast
            behavior when the service is unavailable.
        """
        query_correlation_id = uuid4()

        try:
            adapter, circuit_breaker = self._ensure_initialized()
        except RuntimeError as e:
            logger.error(
                "query_distribution called on uninitialized handler",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(query_correlation_id),
                    "time_range_hours": time_range_hours,
                },
            )
            return ModelIntentDistributionResult(
                status="error",
                time_range_hours=time_range_hours,
                error_message=str(e),
            )

        # Check circuit breaker before attempting operation
        if not circuit_breaker.should_allow_request():
            logger.warning(
                "Circuit breaker is open, failing fast",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(query_correlation_id),
                    "time_range_hours": time_range_hours,
                    "circuit_state": circuit_breaker.state.value,
                },
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker is {circuit_breaker.state.value}"
            )

        logger.debug(
            "Querying intent distribution for last %d hours",
            time_range_hours,
            extra={
                "handler": HANDLER_ID_INTENT,
                "correlation_id": str(query_correlation_id),
                "time_range_hours": time_range_hours,
            },
        )

        try:
            result = await adapter.get_intent_distribution(
                time_range_hours=time_range_hours,
            )
            # Record success if operation completed without exception
            if result.status == "success":
                circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(
                "query_distribution failed, recorded circuit breaker failure",
                extra={
                    "handler": HANDLER_ID_INTENT,
                    "correlation_id": str(query_correlation_id),
                    "time_range_hours": time_range_hours,
                    "error": str(e),
                    "circuit_state": circuit_breaker.state.value,
                    "failure_count": circuit_breaker.failure_count,
                },
            )
            raise

    # =========================================================================
    # Introspection
    # =========================================================================

    async def health_check(self) -> ModelIntentGraphHealth:
        """Check if the handler and adapter are healthy.

        Delegates to AdapterIntentGraph.health_check() to verify
        connectivity and gather graph statistics. Also includes
        circuit breaker status in the response.

        Returns:
            ModelIntentGraphHealth with detailed health status including
            circuit breaker state. This method never raises - errors are
            captured in the result model.
        """
        timestamp = datetime.now(UTC)

        if not self._initialized or self._adapter is None:
            return ModelIntentGraphHealth(
                is_healthy=False,
                initialized=False,
                handler_healthy=None,
                error_message="Handler not initialized",
                last_check_timestamp=timestamp,
            )

        # Include circuit breaker status in health check
        circuit_breaker_info = ""
        if self._circuit_breaker is not None:
            cb = self._circuit_breaker
            circuit_breaker_info = (
                f"circuit_breaker={cb.state.value}, "
                f"failures={cb.failure_count}/{cb.failure_threshold}"
            )
            # If circuit breaker is open, mark as degraded but not unhealthy
            if cb.state == CircuitBreakerState.OPEN:
                return ModelIntentGraphHealth(
                    is_healthy=False,
                    initialized=True,
                    handler_healthy=True,  # Handler itself is healthy
                    error_message=f"Circuit breaker is open ({circuit_breaker_info})",
                    last_check_timestamp=timestamp,
                )

        try:
            adapter_healthy = await self._adapter.health_check()
            error_msg = None if adapter_healthy else "Adapter health check failed"
            if circuit_breaker_info and not adapter_healthy:
                error_msg = f"{error_msg} ({circuit_breaker_info})"
            return ModelIntentGraphHealth(
                is_healthy=adapter_healthy,
                initialized=True,
                handler_healthy=adapter_healthy,
                error_message=error_msg,
                last_check_timestamp=timestamp,
            )
        except Exception as e:
            logger.warning(
                "Health check failed: %s",
                e,
                extra={"handler": HANDLER_ID_INTENT},
            )
            error_msg = f"Health check failed: {e}"
            if circuit_breaker_info:
                error_msg = f"{error_msg} ({circuit_breaker_info})"
            return ModelIntentGraphHealth(
                is_healthy=False,
                initialized=True,
                handler_healthy=None,
                error_message=error_msg,
                last_check_timestamp=timestamp,
            )

    async def describe(self) -> ModelHandlerIntentMetadata:
        """Return handler metadata and capabilities.

        Provides information about the handler type, supported operations,
        and current initialization state.

        Returns:
            ModelHandlerIntentMetadata with handler information.
        """
        capabilities = [
            "store_intent",
            "query_session",
            "query_distribution",
            "health_check",
            "describe",
        ]

        return ModelHandlerIntentMetadata(
            handler_type=self.handler_type,
            capabilities=capabilities,
            adapter_type="AdapterIntentGraph",
            initialized=self._initialized,
        )
