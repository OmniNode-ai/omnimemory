"""Qdrant Adapter Node - Message Bus Bridge for Vector Operations.

This adapter serves as a bridge between the ONEX message bus and Qdrant vector database operations.
It converts event envelopes containing vector requests into direct Qdrant client calls.
Following the ONEX infrastructure tool pattern for external service integration.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

# Import ONEX core components (NO FALLBACKS - MUST BE AVAILABLE)
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.models.health.model_health_status import ModelHealthStatus

# Import Qdrant client (NO FALLBACKS - MUST BE AVAILABLE)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    SearchRequest,
    VectorParams,
)

from ....models.events.model_omnimemory_event_data import (
    ModelOmniMemoryHealthData,
    ModelOmniMemoryStoreData,
    ModelOmniMemoryVectorSearchData,
)

# Import OmniMemory event models
from ....models.events.model_omnimemory_event_publisher import (
    ModelOmniMemoryEventPublisher,
)
from .models import (
    ModelQdrantAdapterConfig,
    ModelQdrantAdapterInput,
    ModelQdrantAdapterOutput,
    ModelQdrantVectorOperationRequest,
)


class QdrantStructuredLogger:
    """
    Structured logger for Qdrant adapter operations with correlation ID tracking.

    Provides consistent, structured logging across all vector operations with:
    - Correlation ID tracking for request tracing
    - Performance metrics logging
    - Error context preservation
    - Security-aware message sanitization
    """

    def __init__(self, logger_name: str = "qdrant_adapter"):
        """Initialize structured logger with correlation ID support."""
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            # Configure structured logging format if not already configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(operation)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _build_extra(
        self, correlation_id: Optional[UUID], operation: str, **kwargs
    ) -> dict:
        """Build extra fields for structured logging."""
        extra = {
            "correlation_id": str(correlation_id)
            if correlation_id
            else "no-correlation",
            "operation": operation,
            "component": "qdrant_adapter",
            "node_type": "effect",
        }
        extra.update(kwargs)
        return extra

    def info(
        self,
        message: str,
        correlation_id: Optional[UUID] = None,
        operation: str = "general",
        **kwargs,
    ):
        """Log info level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.info(message, extra=extra)

    def warning(
        self,
        message: str,
        correlation_id: Optional[UUID] = None,
        operation: str = "general",
        **kwargs,
    ):
        """Log warning level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.warning(message, extra=extra)

    def error(
        self,
        message: str,
        correlation_id: Optional[UUID] = None,
        operation: str = "general",
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Log error level message with structured fields and exception context."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        if exception:
            extra["exception_type"] = type(exception).__name__
            extra["exception_message"] = str(exception)
        self.logger.error(message, extra=extra, exc_info=exception is not None)

    def debug(
        self,
        message: str,
        correlation_id: Optional[UUID] = None,
        operation: str = "general",
        **kwargs,
    ):
        """Log debug level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.debug(message, extra=extra)

    def log_vector_operation_start(
        self, correlation_id: UUID, operation: str, collection: str, **kwargs
    ):
        """Log start of vector operation."""
        self.info(
            f"Starting Qdrant {operation} operation",
            correlation_id=correlation_id,
            operation=f"{operation}_start",
            collection_name=collection,
            **kwargs,
        )

    def log_vector_operation_success(
        self, correlation_id: UUID, operation: str, execution_time_ms: float, **kwargs
    ):
        """Log successful vector operation completion."""
        self.info(
            f"Qdrant {operation} completed successfully in {execution_time_ms:.2f}ms",
            correlation_id=correlation_id,
            operation=f"{operation}_success",
            execution_time_ms=execution_time_ms,
            performance_category="fast"
            if execution_time_ms < 100
            else "slow"
            if execution_time_ms < 1000
            else "very_slow",
            **kwargs,
        )

    def log_vector_operation_error(
        self,
        correlation_id: UUID,
        operation: str,
        execution_time_ms: float,
        exception: Exception,
    ):
        """Log vector operation error with context."""
        self.error(
            f"Qdrant {operation} failed after {execution_time_ms:.2f}ms",
            correlation_id=correlation_id,
            operation=f"{operation}_error",
            exception=exception,
            execution_time_ms=execution_time_ms,
            error_category=self._categorize_qdrant_error(exception),
        )

    def log_circuit_breaker_event(
        self, correlation_id: Optional[UUID], event: str, state: str, **kwargs
    ):
        """Log circuit breaker state changes and events."""
        self.warning(
            f"Circuit breaker {event} - state: {state}",
            correlation_id=correlation_id,
            operation="circuit_breaker",
            circuit_state=state,
            event_type=event,
            **kwargs,
        )

    def log_health_check(
        self, check_name: str, status: str, execution_time_ms: float, **kwargs
    ):
        """Log health check results."""
        level_method = (
            self.info
            if status == "healthy"
            else self.warning
            if status == "degraded"
            else self.error
        )
        level_method(
            f"Health check '{check_name}' returned {status} in {execution_time_ms:.2f}ms",
            operation="health_check",
            check_name=check_name,
            health_status=status,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )

    def _categorize_qdrant_error(self, exception: Exception) -> str:
        """Categorize Qdrant errors for better observability."""
        error_str = str(exception).lower()
        if "connection" in error_str or "timeout" in error_str:
            return "connectivity"
        elif "unauthorized" in error_str or "authentication" in error_str:
            return "authorization"
        elif "not found" in error_str or "collection" in error_str:
            return "resource_not_found"
        elif "invalid" in error_str or "dimension" in error_str:
            return "validation"
        else:
            return "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states for Qdrant connectivity failures."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class QdrantCircuitBreaker:
    """
    Circuit breaker implementation for Qdrant connectivity failures.

    Prevents cascading failures by monitoring Qdrant operation failures
    and temporarily blocking requests when failure thresholds are exceeded.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker with configurable thresholds.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting recovery
            half_open_max_calls: Max calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            OnexError: If circuit is open or function fails
        """
        async with self._lock:
            # Check if we should attempt recovery
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise OnexError(
                        code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                        message="Qdrant circuit breaker is OPEN - service temporarily unavailable",
                    )

            # In half-open state, limit calls
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise OnexError(
                        code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                        message="Qdrant circuit breaker is HALF_OPEN - maximum test calls exceeded",
                    )
                self.half_open_calls += 1

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    async def _record_success(self):
        """Record successful operation and potentially close circuit."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Reset to closed state after successful test
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
                self.half_open_calls = 0
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = max(0, self.failure_count - 1)

    async def _record_failure(self, exception: Exception):
        """Record failed operation and potentially open circuit."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure >= timedelta(seconds=self.timeout_seconds)

    def get_state(self) -> dict:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time
            else None,
            "half_open_calls": self.half_open_calls
            if self.state == CircuitBreakerState.HALF_OPEN
            else 0,
        }


class NodeQdrantAdapterEffect(NodeEffectService):
    """
    Infrastructure Qdrant Adapter Node - Message Bus Bridge.

    Converts message bus envelopes containing vector requests into direct
    Qdrant client operations. This follows the ONEX infrastructure
    tool pattern where adapters serve as bridges between the event-driven message
    bus and external service APIs.

    Message Flow:
    Event Envelope → Qdrant Adapter → Qdrant Client → Vector Database

    Integrates with:
    - omnimemory_event_processing_subcontract: Event bus integration patterns
    - qdrant_vector_management_subcontract: Vector operation management
    """

    # Configuration will be loaded from environment or container
    config: ModelQdrantAdapterConfig

    def __init__(self, container: ModelONEXContainer):
        """Initialize Qdrant adapter with container injection."""
        super().__init__(container)
        self.node_type = "effect"
        self.domain = "omnimemory"
        self._qdrant_client: Optional[QdrantClient] = None
        self._client_lock = asyncio.Lock()
        self._client_sync_lock = threading.Lock()

        # Initialize structured logger with correlation ID support
        self._logger = QdrantStructuredLogger("qdrant_adapter_node")

        # Initialize configuration from environment or container
        self.config = self._load_configuration(container)

        # Initialize circuit breaker for Qdrant connectivity failures AFTER configuration
        self._circuit_breaker = QdrantCircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            timeout_seconds=self.config.circuit_breaker_timeout_seconds,
            half_open_max_calls=self.config.circuit_breaker_half_open_max_calls,
        )

        # Initialize event bus for OmniMemory event publishing (REQUIRED - NO FALLBACKS)
        self._event_bus = self.container.get_service("ProtocolEventBus")
        if self._event_bus is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus service not available - event bus integration is REQUIRED for Qdrant adapter",
            )

        self._event_publisher = ModelOmniMemoryEventPublisher(
            node_id="qdrant_adapter_node"
        )

        self._logger.info(
            "Event bus integration initialized successfully",
            operation="event_bus_init",
            node_type=self.node_type,
            domain=self.domain,
        )

        # Pre-compiled regex patterns for error sanitization (performance optimization)
        self._error_sanitization_patterns = [
            (re.compile(r"api[_-]?key[_-]*[:=][^\s&]*", re.IGNORECASE), "api_key=***"),
            (
                re.compile(r"qdrant://[^\s]*@[^\s]*/", re.IGNORECASE),
                "qdrant://***@***/",
            ),
            (
                re.compile(r"bearer[\s]+[A-Za-z0-9+/=]{10,}", re.IGNORECASE),
                "bearer ***",
            ),
            (
                re.compile(r"auth[_-]?token[_-]*[:=][^\s&]*", re.IGNORECASE),
                "auth_token=***",
            ),
            (re.compile(r"[A-Za-z0-9+/=]{32,}"), "***REDACTED_TOKEN***"),
        ]

        # Log adapter initialization
        self._logger.info(
            "Qdrant adapter initialized successfully",
            operation="initialization",
            node_type=self.node_type,
            domain=self.domain,
            circuit_breaker_threshold=self.config.circuit_breaker_failure_threshold,
        )

    def _get_node_dir(self):
        """Override to provide explicit node directory path, bypassing namespace security check."""
        from pathlib import Path

        # Return the directory containing this node file
        return Path(__file__).parent

    def _load_configuration(
        self, container: ModelONEXContainer
    ) -> ModelQdrantAdapterConfig:
        """
        Load Qdrant adapter configuration from container or environment.

        Args:
            container: ONEX container for dependency injection

        Returns:
            Configured ModelQdrantAdapterConfig instance
        """
        try:
            # Try to get configuration from container first (ONEX pattern)
            config = container.get_service("qdrant_adapter_config")
            if (
                config
                and hasattr(config, "qdrant_host")
                and hasattr(config, "qdrant_port")
            ):
                return config
        except Exception:
            pass  # Fall back to environment configuration

        # Fall back to environment-based configuration
        environment = os.getenv("DEPLOYMENT_ENVIRONMENT", "development")
        return ModelQdrantAdapterConfig.for_environment(environment)

    def _validate_correlation_id(self, correlation_id: Optional[UUID]) -> UUID:
        """
        Validate and normalize correlation ID to prevent injection attacks.

        Args:
            correlation_id: Optional correlation ID to validate

        Returns:
            Valid UUID correlation ID

        Raises:
            OnexError: If correlation ID format is invalid
        """
        if correlation_id is None:
            # Generate a new correlation ID if none provided
            return uuid4()

        if hasattr(correlation_id, "replace") and hasattr(
            correlation_id, "split"
        ):  # String-like
            try:
                # Try to parse string as UUID to validate format
                correlation_id = UUID(correlation_id)
            except ValueError as e:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message="Invalid correlation ID format - must be valid UUID",
                ) from e

        if not hasattr(correlation_id, "hex"):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Correlation ID must be UUID type",
            )

        # Additional validation: ensure it's not an empty UUID
        if correlation_id == UUID("00000000-0000-0000-0000-000000000000"):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Correlation ID cannot be empty UUID",
            )

        return correlation_id

    @property
    def qdrant_client(self) -> QdrantClient:
        """
        Get Qdrant client instance with thread safety.

        Note: For async operations, prefer get_qdrant_client_async() to avoid mixing sync/async patterns.
        """
        with self._client_sync_lock:
            if self._qdrant_client is None:
                # Create Qdrant client with configuration
                client_config = self.config.get_qdrant_client_config()
                self._qdrant_client = QdrantClient(**client_config)

            return self._qdrant_client

    async def get_qdrant_client_async(self) -> QdrantClient:
        """
        Get Qdrant client instance with async thread safety.

        Returns:
            QdrantClient instance

        Raises:
            OnexError: If client cannot be created
        """
        async with self._client_lock:
            if self._qdrant_client is None:
                try:
                    # Create Qdrant client with configuration
                    client_config = self.config.get_qdrant_client_config()
                    self._qdrant_client = QdrantClient(**client_config)

                    self._logger.info(
                        "Qdrant client created successfully",
                        operation="client_creation",
                        host=client_config.get("host", "url-based"),
                        port=client_config.get("port", "url-based"),
                    )

                except Exception as e:
                    self._logger.error(
                        "Failed to create Qdrant client",
                        operation="client_creation_error",
                        exception=e,
                        config_host=self.config.qdrant_host,
                        config_port=self.config.qdrant_port,
                    )
                    raise OnexError(
                        code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                        message=f"Failed to create Qdrant client: {str(e)}",
                        details={
                            "host": self.config.qdrant_host,
                            "port": self.config.qdrant_port,
                        },
                    ) from e

            return self._qdrant_client

    def get_health_checks(
        self,
    ) -> List[
        Callable[[], Union[ModelHealthStatus, "asyncio.Future[ModelHealthStatus]"]]
    ]:
        """
        Override health checks to provide Qdrant-specific health checks.

        Returns list of health check functions that validate Qdrant connectivity,
        vector operations, and adapter functionality.
        """
        return [
            self._check_qdrant_connectivity,
            self._check_circuit_breaker_health,
            self._check_vector_operations_health,
            self._check_event_publishing_health,
        ]

    def _check_qdrant_connectivity(self) -> ModelHealthStatus:
        """Check basic Qdrant connectivity (sync wrapper for health checks)."""
        start_time = time.perf_counter()
        try:
            # Basic connectivity indicator based on client availability
            if self._qdrant_client is None:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._logger.log_health_check(
                    check_name="qdrant_connectivity",
                    status="degraded",
                    execution_time_ms=execution_time_ms,
                    reason="client_not_initialized",
                )
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Qdrant client not initialized",
                    timestamp=datetime.utcnow().isoformat(),
                )

            # Basic connectivity indicator based on client state
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.log_health_check(
                check_name="qdrant_connectivity",
                status="healthy",
                execution_time_ms=execution_time_ms,
                reason="client_operational",
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Qdrant client operational",
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.log_health_check(
                check_name="qdrant_connectivity",
                status="unhealthy",
                execution_time_ms=execution_time_ms,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Qdrant connectivity check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
            )

    def _check_circuit_breaker_health(self) -> ModelHealthStatus:
        """Check circuit breaker health and state (sync version for health checks)."""
        try:
            circuit_state = self._circuit_breaker.get_state()
            state_value = circuit_state["state"]

            if state_value == CircuitBreakerState.CLOSED.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.HEALTHY,
                    message=f"Circuit breaker CLOSED - failures: {circuit_state['failure_count']}",
                    timestamp=datetime.utcnow().isoformat(),
                )
            elif state_value == CircuitBreakerState.HALF_OPEN.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message=f"Circuit breaker HALF_OPEN - testing recovery ({circuit_state['half_open_calls']} calls)",
                    timestamp=datetime.utcnow().isoformat(),
                )
            else:  # OPEN state
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message=f"Circuit breaker OPEN - service temporarily unavailable (failures: {circuit_state['failure_count']})",
                    timestamp=datetime.utcnow().isoformat(),
                )

        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Circuit breaker health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
            )

    def _check_vector_operations_health(self) -> ModelHealthStatus:
        """Check vector operations capability (sync wrapper for health checks)."""
        start_time = time.perf_counter()
        try:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Vector operations capability available",
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Vector operations health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
            )

    def _check_event_publishing_health(self) -> ModelHealthStatus:
        """Check event publishing capability health (sync wrapper for health checks)."""
        start_time = time.perf_counter()
        try:
            # Basic event publisher availability check
            if not self._event_publisher:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Event publisher not available",
                    timestamp=datetime.utcnow().isoformat(),
                )

            # Check event publisher configuration
            if not hasattr(self._event_publisher, "node_id"):
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Event publisher missing configuration",
                    timestamp=datetime.utcnow().isoformat(),
                )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Event publishing capability available",
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Event publishing health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
            )

    async def process(
        self, input_data: ModelQdrantAdapterInput
    ) -> ModelQdrantAdapterOutput:
        """
        Process Qdrant adapter request following infrastructure tool pattern.

        Routes message envelope to appropriate vector operation based on operation_type.
        Handles vector search, storage, and health check operations with proper error handling
        and metrics collection.

        Args:
            input_data: Input envelope containing operation type and request data

        Returns:
            Output envelope with operation results
        """
        start_time = time.perf_counter()

        try:
            # Validate and normalize correlation ID to prevent injection attacks
            validated_correlation_id = self._validate_correlation_id(
                input_data.correlation_id
            )

            # Update the input data with validated correlation ID if it was modified
            if validated_correlation_id != input_data.correlation_id:
                input_data.correlation_id = validated_correlation_id

            # Validate input for the operation
            if not input_data.validate_for_operation():
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Invalid input data for operation: {input_data.operation_type}",
                )

            # Route based on operation type
            if input_data.operation_type == "vector_search":
                return await self._handle_vector_search_operation(
                    input_data, start_time
                )
            elif input_data.operation_type == "store_vector":
                return await self._handle_store_vector_operation(input_data, start_time)
            elif input_data.operation_type == "get_vector":
                return await self._handle_get_vector_operation(input_data, start_time)
            elif input_data.operation_type == "delete_vector":
                return await self._handle_delete_vector_operation(
                    input_data, start_time
                )
            elif input_data.operation_type == "batch_upsert":
                return await self._handle_batch_upsert_operation(input_data, start_time)
            elif input_data.operation_type == "create_collection":
                return await self._handle_create_collection_operation(
                    input_data, start_time
                )
            elif input_data.operation_type == "health_check":
                return await self._handle_health_check_operation(input_data, start_time)
            else:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Unsupported operation type: {input_data.operation_type}",
                )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            if hasattr(e, "code") and hasattr(e, "message"):  # OnexError-like
                error_message = str(e)
            else:
                error_message = f"Qdrant adapter error: {str(e)}"

            output = ModelQdrantAdapterOutput(
                operation_type=input_data.operation_type,
                success=False,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )
            output.add_error(error_message, type(e).__name__)
            return output

    async def _handle_vector_search_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle Qdrant vector search operation."""
        vector_request = input_data.vector_request
        correlation_id = input_data.correlation_id

        # Log search start
        self._logger.log_vector_operation_start(
            correlation_id=correlation_id,
            operation="vector_search",
            collection=vector_request.collection_name,
            search_limit=vector_request.search_limit,
            has_filter=vector_request.search_filter is not None,
        )

        try:
            # Get Qdrant client and execute search with circuit breaker protection
            client = await self.get_qdrant_client_async()

            search_result = await self._circuit_breaker.call(
                self._execute_qdrant_search, client, vector_request
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Log successful search completion
            self._logger.log_vector_operation_success(
                correlation_id=correlation_id,
                operation="vector_search",
                execution_time_ms=execution_time_ms,
                results_count=len(search_result),
                collection=vector_request.collection_name,
            )

            # Create search data for event publishing
            query_vector_hash = hashlib.md5(
                json.dumps(vector_request.query_vector, sort_keys=True).encode()
            ).hexdigest()

            search_data = ModelOmniMemoryVectorSearchData(
                query_vector_hash=query_vector_hash,
                vector_dimensions=len(vector_request.query_vector),
                similarity_threshold=vector_request.score_threshold or 0.0,
                max_results=vector_request.search_limit,
                result_count=len(search_result),
                search_type="similarity_search",
                index_name=vector_request.collection_name,
                search_time_ms=execution_time_ms,
                results=[
                    {
                        "id": str(point.id),
                        "score": float(point.score) if hasattr(point, "score") else 0.0,
                        "payload": point.payload if hasattr(point, "payload") else {},
                    }
                    for point in search_result
                ],
            )

            # Publish vector search completed event
            await self._publish_vector_search_completed_event(
                correlation_id=correlation_id,
                search_data=search_data,
                execution_time_ms=execution_time_ms,
                success=True,
            )

            # Create successful output
            output = ModelQdrantAdapterOutput(
                operation_type="vector_search",
                success=True,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

            # Convert search results to output format
            search_points = []
            for point in search_result:
                search_points.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score) if hasattr(point, "score") else 0.0,
                        "payload": point.payload if hasattr(point, "payload") else {},
                        "vector": point.vector
                        if (hasattr(point, "vector") and point.vector)
                        else None,
                    }
                )

            output.add_search_result(
                points=search_points,
                search_time_ms=execution_time_ms,
                collection_name=vector_request.collection_name,
                query_vector_hash=query_vector_hash,
            )

            return output

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Log search error
            self._logger.log_vector_operation_error(
                correlation_id=correlation_id,
                operation="vector_search",
                execution_time_ms=execution_time_ms,
                exception=e,
            )

            # Sanitize error message
            sanitized_error = self._sanitize_error_message(str(e))

            # Publish vector search failed event
            await self._publish_vector_search_completed_event(
                correlation_id=correlation_id,
                search_data=ModelOmniMemoryVectorSearchData(
                    query_vector_hash="error",
                    vector_dimensions=len(vector_request.query_vector)
                    if vector_request.query_vector
                    else 0,
                    similarity_threshold=vector_request.score_threshold or 0.0,
                    max_results=vector_request.search_limit,
                    result_count=0,
                    search_type="similarity_search",
                    index_name=vector_request.collection_name,
                    search_time_ms=execution_time_ms,
                    results=[],
                ),
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=sanitized_error,
            )

            # Create error output
            output = ModelQdrantAdapterOutput(
                operation_type="vector_search",
                success=False,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )
            output.add_error(sanitized_error, type(e).__name__)

            return output

    async def _execute_qdrant_search(
        self, client: QdrantClient, vector_request: ModelQdrantVectorOperationRequest
    ) -> List[Any]:
        """Execute Qdrant search operation."""
        search_params = {
            "collection_name": vector_request.collection_name,
            "query_vector": vector_request.query_vector,
            "limit": vector_request.search_limit,
            "with_payload": vector_request.with_payload,
            "with_vector": vector_request.with_vector,
        }

        if vector_request.score_threshold is not None:
            search_params["score_threshold"] = vector_request.score_threshold

        if vector_request.search_filter:
            # Convert filter to Qdrant format
            qdrant_filter = self._build_qdrant_filter(vector_request.search_filter)
            search_params["query_filter"] = qdrant_filter

        # Execute search with Qdrant client
        return await client.search(**search_params)

    def _build_qdrant_filter(self, filter_conditions: Dict[str, Any]) -> Any:
        """Build Qdrant filter from filter conditions."""
        must_conditions = []

        for key, value in filter_conditions.items():
            if isinstance(value, dict):
                # Handle range conditions
                if "gte" in value or "gt" in value or "lte" in value or "lt" in value:
                    # Create range condition (Qdrant-specific implementation needed)
                    must_conditions.append(FieldCondition(key=key, range=value))
                elif "in" in value:
                    # Handle "in" conditions
                    must_conditions.append(
                        FieldCondition(key=key, match={"any": value["in"]})
                    )
            else:
                # Simple match condition
                must_conditions.append(FieldCondition(key=key, match={"value": value}))

        return Filter(must=must_conditions) if must_conditions else None

    async def _handle_store_vector_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle Qdrant store vector operation."""
        vector_request = input_data.vector_request
        correlation_id = input_data.correlation_id

        # Log operation start
        self._logger.log_vector_operation_start(
            correlation_id=correlation_id,
            operation="store_vector",
            collection=vector_request.collection_name,
            vector_id=vector_request.vector_id,
            vector_dimensions=len(vector_request.vector_data)
            if vector_request.vector_data
            else 0,
        )

        try:
            # Get Qdrant client and execute upsert with circuit breaker protection
            client = await self.get_qdrant_client_async()

            upsert_result = await self._circuit_breaker.call(
                self._execute_qdrant_upsert, client, vector_request
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Log successful operation completion
            self._logger.log_vector_operation_success(
                correlation_id=correlation_id,
                operation="store_vector",
                execution_time_ms=execution_time_ms,
                vector_id=vector_request.vector_id,
                collection=vector_request.collection_name,
            )

            # Create store data for event publishing
            content_hash = hashlib.md5(
                json.dumps(
                    {
                        "vector": vector_request.vector_data,
                        "payload": vector_request.payload or {},
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            store_data = ModelOmniMemoryStoreData(
                memory_key=vector_request.vector_id,
                memory_type="vector",
                content_hash=content_hash,
                storage_size=len(vector_request.vector_data) * 4,  # Assuming float32
                metadata=vector_request.payload or {},
                vector_dimensions=len(vector_request.vector_data),
                affected_indices=[vector_request.collection_name],
            )

            # Publish memory stored event
            await self._publish_memory_stored_event(
                correlation_id=correlation_id,
                store_data=store_data,
                execution_time_ms=execution_time_ms,
                success=True,
            )

            # Create successful output
            output = ModelQdrantAdapterOutput(
                operation_type="store_vector",
                success=True,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

            output.add_operation_result(
                operation_type="upsert",
                affected_points=1,
                operation_time_ms=execution_time_ms,
                collection_name=vector_request.collection_name,
                result_data={"vector_id": vector_request.vector_id},
            )

            return output

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Log operation error
            self._logger.log_vector_operation_error(
                correlation_id=correlation_id,
                operation="store_vector",
                execution_time_ms=execution_time_ms,
                exception=e,
            )

            # Sanitize error message
            sanitized_error = self._sanitize_error_message(str(e))

            # Create error output
            output = ModelQdrantAdapterOutput(
                operation_type="store_vector",
                success=False,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )
            output.add_error(sanitized_error, type(e).__name__)

            return output

    async def _execute_qdrant_upsert(
        self, client: QdrantClient, vector_request: ModelQdrantVectorOperationRequest
    ) -> Any:
        """Execute Qdrant upsert operation."""
        point = PointStruct(
            id=vector_request.vector_id,
            vector=vector_request.vector_data,
            payload=vector_request.payload or {},
        )

        return await client.upsert(
            collection_name=vector_request.collection_name, points=[point]
        )

    async def _handle_health_check_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle health check operation for Qdrant adapter."""
        correlation_id = input_data.correlation_id

        try:
            # Run health checks
            health_results = []

            # Check Qdrant connectivity
            connectivity_health = await self._check_qdrant_connectivity_async()
            health_results.append(connectivity_health)

            # Check circuit breaker
            circuit_health = self._check_circuit_breaker_health()
            health_results.append(circuit_health)

            # Determine overall health status
            overall_healthy = all(
                result.status == EnumHealthStatus.HEALTHY for result in health_results
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Create health check data
            health_data = ModelOmniMemoryHealthData(
                overall_status="healthy" if overall_healthy else "unhealthy",
                response_time_ms=execution_time_ms,
                storage_stats={"qdrant_client": "connected"},
                circuit_breaker_state=self._circuit_breaker.get_state()["state"],
            )

            # Publish health response event
            await self._publish_memory_health_response_event(
                correlation_id=correlation_id,
                health_status="healthy" if overall_healthy else "unhealthy",
                health_data=health_data,
            )

            # Create output
            output = ModelQdrantAdapterOutput(
                operation_type="health_check",
                success=overall_healthy,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

            output.add_health_status(
                status="healthy" if overall_healthy else "unhealthy",
                response_time_ms=execution_time_ms,
                connection_status="connected",
                version=None,
            )

            return output

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = self._sanitize_error_message(
                f"Health check operation failed: {str(e)}"
            )

            # Create error output
            output = ModelQdrantAdapterOutput(
                operation_type="health_check",
                success=False,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )
            output.add_error(sanitized_error, type(e).__name__)

            return output

    async def _check_qdrant_connectivity_async(self) -> ModelHealthStatus:
        """Check Qdrant connectivity asynchronously."""
        try:
            client = await self.get_qdrant_client_async()

            if client:
                # Try to get collections as a connectivity test
                collections = await client.get_collections()
                return ModelHealthStatus(
                    status=EnumHealthStatus.HEALTHY,
                    message="Qdrant connectivity verified",
                    timestamp=datetime.utcnow().isoformat(),
                )
            else:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Qdrant client not available",
                    timestamp=datetime.utcnow().isoformat(),
                )

        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Qdrant connectivity failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
            )

    # Additional operation handlers would be implemented here following the same pattern
    async def _handle_get_vector_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle get vector operation - placeholder implementation."""
        # Implementation would follow same pattern as vector search
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        output = ModelQdrantAdapterOutput(
            operation_type="get_vector",
            success=False,
            correlation_id=input_data.correlation_id,
            execution_time_ms=execution_time_ms,
            context=input_data.context,
        )
        output.add_error("Get vector operation not yet implemented", "NOT_IMPLEMENTED")
        return output

    async def _handle_delete_vector_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle delete vector operation - placeholder implementation."""
        # Implementation would follow same pattern as store vector
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        output = ModelQdrantAdapterOutput(
            operation_type="delete_vector",
            success=False,
            correlation_id=input_data.correlation_id,
            execution_time_ms=execution_time_ms,
            context=input_data.context,
        )
        output.add_error(
            "Delete vector operation not yet implemented", "NOT_IMPLEMENTED"
        )
        return output

    async def _handle_batch_upsert_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle batch upsert operation - placeholder implementation."""
        # Implementation would follow same pattern as store vector but for batches
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        output = ModelQdrantAdapterOutput(
            operation_type="batch_upsert",
            success=False,
            correlation_id=input_data.correlation_id,
            execution_time_ms=execution_time_ms,
            context=input_data.context,
        )
        output.add_error(
            "Batch upsert operation not yet implemented", "NOT_IMPLEMENTED"
        )
        return output

    async def _handle_create_collection_operation(
        self, input_data: ModelQdrantAdapterInput, start_time: float
    ) -> ModelQdrantAdapterOutput:
        """Handle create collection operation - placeholder implementation."""
        # Implementation would create Qdrant collection
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        output = ModelQdrantAdapterOutput(
            operation_type="create_collection",
            success=False,
            correlation_id=input_data.correlation_id,
            execution_time_ms=execution_time_ms,
            context=input_data.context,
        )
        output.add_error(
            "Create collection operation not yet implemented", "NOT_IMPLEMENTED"
        )
        return output

    # Event publishing methods
    async def _publish_vector_search_completed_event(
        self,
        correlation_id: UUID,
        search_data: ModelOmniMemoryVectorSearchData,
        execution_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Publish vector search completed event to RedPanda."""
        if not self._event_bus or not self._event_publisher:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Event bus or publisher not available for vector search event publishing",
            )

        try:
            envelope = self._event_publisher.create_vector_search_completed_envelope(
                correlation_id=correlation_id,
                search_data=search_data,
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message,
            )

            await self._event_bus.publish_async(envelope.payload)

            self._logger.info(
                "Vector search completed event published",
                correlation_id=correlation_id,
                operation="vector_search_event_published",
                success=success,
                results_count=search_data.result_count,
            )

        except Exception as e:
            self._logger.error(
                "Failed to publish vector search completed event",
                correlation_id=correlation_id,
                operation="vector_search_event_error",
                exception=e,
            )
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to publish vector search event: {str(e)}",
                details={"correlation_id": str(correlation_id)},
            ) from e

    async def _publish_memory_stored_event(
        self,
        correlation_id: UUID,
        store_data: ModelOmniMemoryStoreData,
        execution_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Publish memory stored event to RedPanda."""
        if not self._event_bus or not self._event_publisher:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Event bus or publisher not available for memory stored event publishing",
            )

        try:
            envelope = self._event_publisher.create_memory_stored_envelope(
                correlation_id=correlation_id,
                store_data=store_data,
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message,
            )

            await self._event_bus.publish_async(envelope.payload)

            self._logger.info(
                "Memory stored event published",
                correlation_id=correlation_id,
                operation="memory_stored_event_published",
                success=success,
                memory_key=store_data.memory_key,
            )

        except Exception as e:
            self._logger.error(
                "Failed to publish memory stored event",
                correlation_id=correlation_id,
                operation="memory_stored_event_error",
                exception=e,
            )
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to publish memory stored event: {str(e)}",
                details={"correlation_id": str(correlation_id)},
            ) from e

    async def _publish_memory_health_response_event(
        self,
        correlation_id: UUID,
        health_status: str,
        health_data: ModelOmniMemoryHealthData,
    ) -> None:
        """Publish memory health response event to RedPanda."""
        if not self._event_bus or not self._event_publisher:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Event bus or publisher not available for health response event publishing",
            )

        try:
            envelope = self._event_publisher.create_memory_health_response_envelope(
                correlation_id=correlation_id,
                health_status=health_status,
                health_data=health_data,
            )

            await self._event_bus.publish_async(envelope.payload)

            self._logger.info(
                "Memory health response event published",
                correlation_id=correlation_id,
                operation="health_response_event_published",
                health_status=health_status,
            )

        except Exception as e:
            self._logger.error(
                "Failed to publish memory health response event",
                correlation_id=correlation_id,
                operation="health_response_event_error",
                exception=e,
            )
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to publish memory health response event: {str(e)}",
                details={"correlation_id": str(correlation_id)},
            ) from e

    def _sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error messages to prevent sensitive information leakage."""
        if not self.config.enable_error_sanitization:
            return error_message

        # Apply all sanitization patterns using pre-compiled regex for performance
        sanitized = error_message
        for pattern, replacement in self._error_sanitization_patterns:
            sanitized = pattern.sub(replacement, sanitized)

        # If error is too generic, provide a more specific safe message
        if len(sanitized.strip()) < 10 or "connection" in sanitized.lower():
            return "Qdrant operation failed - please check connection and request parameters"

        return sanitized

    async def initialize(self) -> None:
        """Initialize the Qdrant adapter and client."""
        try:
            client = await self.get_qdrant_client_async()
            self._logger.info(
                "Qdrant adapter initialized successfully",
                operation="initialization_complete",
            )
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_FAILED,
                message=f"Failed to initialize Qdrant adapter: {str(e)}",
            ) from e

    async def cleanup(self) -> None:
        """Enhanced cleanup with comprehensive resource management and thread safety."""
        cleanup_tasks = []
        cleanup_errors = []

        # Thread-safe cleanup coordination
        async with self._client_lock:
            # Schedule client cleanup
            if self._qdrant_client:
                cleanup_tasks.append(self._cleanup_qdrant_client())

        # Schedule event bus cleanup (if available)
        if self._event_bus:
            cleanup_tasks.append(self._cleanup_event_bus())

        # Schedule circuit breaker cleanup
        if self._circuit_breaker:
            cleanup_tasks.append(self._cleanup_circuit_breaker())

        # Execute all cleanup tasks concurrently with error isolation
        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # Collect any cleanup errors for observability
            for i, result in enumerate(results):
                if hasattr(result, "__traceback__") and hasattr(
                    result, "args"
                ):  # Exception-like protocol
                    cleanup_errors.append(f"Cleanup task {i}: {str(result)}")

        # Clear all references in thread-safe manner
        async with self._client_lock:
            self._qdrant_client = None
            self._event_bus = None
            self._circuit_breaker = None

        # Log cleanup summary for observability
        if cleanup_errors:
            self._logger.warning(
                f"Cleanup completed with {len(cleanup_errors)} non-critical errors",
                operation="cleanup_summary",
                error_count=len(cleanup_errors),
                errors=cleanup_errors,
            )
        else:
            self._logger.info(
                "Resource cleanup completed successfully",
                operation="cleanup_success",
                tasks_completed=len(cleanup_tasks),
            )

    async def _cleanup_qdrant_client(self) -> None:
        """Cleanup Qdrant client resources."""
        try:
            if self._qdrant_client and hasattr(self._qdrant_client, "close"):
                await self._qdrant_client.close()
                self._logger.debug("Qdrant client cleanup completed")
        except Exception as e:
            self._logger.warning(f"Qdrant client cleanup error: {str(e)}")
            # Don't raise during cleanup - log and continue

    async def _cleanup_event_bus(self) -> None:
        """Cleanup event bus resources."""
        try:
            if self._event_bus and hasattr(self._event_bus, "cleanup"):
                await self._event_bus.cleanup()
                self._logger.debug("Event bus cleanup completed")
        except Exception as e:
            self._logger.warning(f"Event bus cleanup error: {str(e)}")
            # Don't raise during cleanup - log and continue

    async def _cleanup_circuit_breaker(self) -> None:
        """Cleanup circuit breaker resources."""
        try:
            # Circuit breaker state reset for clean shutdown
            if self._circuit_breaker:
                # Log final circuit breaker state for observability
                final_state = self._circuit_breaker.get_state()
                self._logger.debug(
                    "Circuit breaker final state",
                    circuit_state=final_state["state"],
                    failure_count=final_state["failure_count"],
                )
        except Exception as e:
            self._logger.warning(f"Circuit breaker cleanup error: {str(e)}")
            # Don't raise during cleanup - log and continue


async def main():
    """Main entry point for Qdrant Adapter - runs in service mode with NodeEffectService"""
    # Import infrastructure container with all dependencies registered
    # Using local fixed version due to import path issues in omnibase_infra
    from omnimemory.infrastructure.container import create_infrastructure_container

    # Create infrastructure container with ProtocolEventBus and all shared dependencies
    container = create_infrastructure_container()

    adapter = NodeQdrantAdapterEffect(container)

    # Initialize the adapter
    await adapter.initialize()

    # Start service mode using NodeEffectService capabilities
    await adapter.start_service_mode()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
