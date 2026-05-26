# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Correlation context tracking and ObservabilityManager for OmniMemory.

Extracted from observability.py (OMN-11580).

- ContextVar integration for correlation ID propagation
- ObservabilityManager for distributed tracing and performance monitoring
- Convenience decorators: inject_correlation_context, inject_correlation_context_async
- Module-level singletons: observability_manager, correlation_id_var, request_id_var
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import (
    TypeVar,
    cast,
)

import structlog

from ..models.foundation.model_typed_collections import (
    ModelKeyValuePair,
    ModelMetadata,
)
from ..models.utils.model_correlation_context import ModelCorrelationContext
from ..models.utils.model_structured_log_entry import TraceLevel
from .adapter_metrics_counter import (
    DEFAULT_MAX_ACTIVE_TRACES,
    MetadataValue,
    _sanitize_error,
    sanitize_metadata_value,
    validate_correlation_id,
)

# Alias for internal use
CorrelationContext = ModelCorrelationContext

# Type variable for generic function types
F = TypeVar("F", bound=Callable[..., object])

# Optional psutil import for memory tracking - gracefully degrade if unavailable
_psutil_available = False
try:
    import psutil  # type: ignore[import-untyped]  # Why: psutil ships without type stubs

    _psutil_available = True
except ImportError:
    psutil = None  # type: ignore[assignment]

# Context variables for correlation tracking
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
operation_var: ContextVar[str | None] = ContextVar("operation", default=None)

logger = structlog.get_logger(__name__)


class OperationType(Enum):
    """Operation type enumeration for categorizing operations."""

    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_SEARCH = "memory_search"
    INTELLIGENCE_PROCESS = "intelligence_process"
    HEALTH_CHECK = "health_check"
    MIGRATION = "migration"
    CLEANUP = "cleanup"
    EXTERNAL_API = "external_api"


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    start_time: float
    end_time: float | None = None
    duration: float | None = None
    memory_usage_start: float | None = None
    memory_usage_end: float | None = None
    memory_delta: float | None = None
    success: bool | None = None
    error_type: str | None = None


class ObservabilityManager:
    """Comprehensive observability manager for OmniMemory.

    Provides:
    - Correlation ID management and propagation
    - Distributed tracing support
    - Performance monitoring
    - Enhanced logging with context

    Thread-safety:
        All trace storage operations are protected by a lock to ensure
        thread-safe access to the active traces dictionary.

    Memory bounds:
        Active traces are stored in a bounded OrderedDict with LRU eviction.
    """

    def __init__(
        self,
        max_active_traces: int = DEFAULT_MAX_ACTIVE_TRACES,
    ) -> None:
        self.max_active_traces = max_active_traces
        self._active_traces: OrderedDict[str, PerformanceMetrics] = OrderedDict()
        self._traces_lock = threading.Lock()
        self._logger = structlog.get_logger(__name__)

    @asynccontextmanager
    async def correlation_context(
        self,
        correlation_id: str | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
        operation: str | None = None,
        trace_level: TraceLevel = TraceLevel.INFO,
        **metadata: MetadataValue,
    ) -> AsyncGenerator[CorrelationContext, None]:
        """Async context manager for correlation tracking."""
        if correlation_id and not validate_correlation_id(correlation_id):
            raise ValueError(f"Invalid correlation ID format: {correlation_id}")

        metadata_pairs = [
            ModelKeyValuePair(key=key, value=str(sanitize_metadata_value(value)))
            for key, value in metadata.items()
            if sanitize_metadata_value(value) is not None
        ]
        sanitized_metadata = ModelMetadata(pairs=metadata_pairs)

        context = CorrelationContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            request_id=request_id,
            user_id=user_id,
            operation=operation,
            parent_correlation_id=correlation_id_var.get(),
            trace_level=trace_level,
            metadata=sanitized_metadata,
        )

        correlation_token = correlation_id_var.set(context.correlation_id)
        request_token = request_id_var.set(context.request_id)
        user_token = user_id_var.set(context.user_id)
        operation_token = operation_var.set(context.operation)

        try:
            self._logger.info(
                "correlation_context_started",
                correlation_id=context.correlation_id,
                request_id=context.request_id,
                user_id=context.user_id,
                operation=context.operation,
                trace_level=context.trace_level.value,
                metadata=context.metadata,
            )

            yield context

        except Exception as e:
            self._logger.error(
                "correlation_context_error",
                correlation_id=context.correlation_id,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            correlation_id_var.reset(correlation_token)
            request_id_var.reset(request_token)
            user_id_var.reset(user_token)
            operation_var.reset(operation_token)

            self._logger.info(
                "correlation_context_ended",
                correlation_id=context.correlation_id,
                operation=context.operation,
            )

    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        operation_type: OperationType,
        trace_performance: bool = True,
        **additional_context: MetadataValue,
    ) -> AsyncGenerator[str, None]:
        """Async context manager for operation tracing."""
        trace_id = str(uuid.uuid4())
        correlation_id = correlation_id_var.get()

        start_memory: float | None = None
        if trace_performance:
            if _psutil_available and psutil is not None:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                    psutil.Error,
                    OSError,
                    AttributeError,
                ) as e:
                    self._logger.debug(
                        "psutil_memory_tracking_unavailable",
                        reason=type(e).__name__,
                        phase="start",
                    )
                    start_memory = None

            metrics = PerformanceMetrics(
                start_time=time.time(), memory_usage_start=start_memory
            )
            with self._traces_lock:
                while len(self._active_traces) >= self.max_active_traces:
                    evicted_id, evicted_metrics = self._active_traces.popitem(
                        last=False
                    )
                    self._logger.warning(
                        "trace_evicted_due_to_capacity",
                        evicted_trace_id=evicted_id,
                        evicted_duration=evicted_metrics.duration,
                        max_active_traces=self.max_active_traces,
                    )
                self._active_traces[trace_id] = metrics

        try:
            self._logger.info(
                "operation_started",
                trace_id=trace_id,
                correlation_id=correlation_id,
                operation_name=operation_name,
                operation_type=operation_type.value,
                **additional_context,
            )

            yield trace_id

            if trace_performance:
                with self._traces_lock:
                    if trace_id in self._active_traces:
                        self._active_traces[trace_id].success = True

        except Exception as e:
            if trace_performance:
                with self._traces_lock:
                    if trace_id in self._active_traces:
                        self._active_traces[trace_id].success = False
                        self._active_traces[trace_id].error_type = type(e).__name__

            self._logger.error(
                "operation_failed",
                trace_id=trace_id,
                correlation_id=correlation_id,
                operation_name=operation_name,
                operation_type=operation_type.value,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
                **additional_context,
            )
            raise
        finally:
            metrics_final: PerformanceMetrics | None = None
            if trace_performance:
                with self._traces_lock:
                    if trace_id in self._active_traces:
                        metrics_final = self._active_traces[trace_id]

                if metrics_final is not None:
                    metrics_final.end_time = time.time()
                    metrics_final.duration = (
                        metrics_final.end_time - metrics_final.start_time
                    )

                    if metrics_final.memory_usage_start is not None:
                        if _psutil_available and psutil is not None:
                            try:
                                process = psutil.Process()
                                end_memory = process.memory_info().rss / 1024 / 1024
                                metrics_final.memory_usage_end = end_memory
                                metrics_final.memory_delta = (
                                    end_memory - metrics_final.memory_usage_start
                                )
                            except (
                                psutil.NoSuchProcess,
                                psutil.AccessDenied,
                                psutil.ZombieProcess,
                                psutil.Error,
                                OSError,
                                AttributeError,
                            ) as e:
                                self._logger.debug(
                                    "psutil_memory_tracking_unavailable",
                                    reason=type(e).__name__,
                                    phase="end",
                                )

                    self._logger.info(
                        "operation_completed",
                        trace_id=trace_id,
                        correlation_id=correlation_id,
                        operation_name=operation_name,
                        operation_type=operation_type.value,
                        duration=metrics_final.duration,
                        memory_delta=metrics_final.memory_delta,
                        success=metrics_final.success,
                        error_type=metrics_final.error_type,
                        **additional_context,
                    )

                    with self._traces_lock:
                        if trace_id in self._active_traces:
                            del self._active_traces[trace_id]

    def get_current_context(self) -> dict[str, str | None]:
        """Get current correlation context."""
        return {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "operation": operation_var.get(),
        }

    def get_performance_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get current performance metrics for active traces (thread-safe)."""
        with self._traces_lock:
            return dict(self._active_traces)

    def log_with_context(
        self, level: str, message: str, **additional_fields: MetadataValue
    ) -> None:
        """Log a message with current correlation context."""
        context = self.get_current_context()
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message, **context, **additional_fields)


# Global observability manager instance
observability_manager = ObservabilityManager()


# Convenience functions for common patterns
@asynccontextmanager
async def correlation_context(
    correlation_id: str | None = None,
    request_id: str | None = None,
    user_id: str | None = None,
    operation: str | None = None,
    trace_level: TraceLevel = TraceLevel.INFO,
    **metadata: MetadataValue,
) -> AsyncGenerator[CorrelationContext, None]:
    """Convenience function for correlation context management."""
    async with observability_manager.correlation_context(
        correlation_id=correlation_id,
        request_id=request_id,
        user_id=user_id,
        operation=operation,
        trace_level=trace_level,
        **metadata,
    ) as ctx:
        yield ctx


@asynccontextmanager
async def trace_operation(
    operation_name: str,
    operation_type: OperationType | str,
    trace_performance: bool = True,
    **context: MetadataValue,
) -> AsyncGenerator[str, None]:
    """Convenience function for operation tracing."""
    if isinstance(operation_type, str):
        try:
            operation_type = OperationType(operation_type)
        except ValueError:
            operation_type = OperationType.EXTERNAL_API

    async with observability_manager.trace_operation(
        operation_name=operation_name,
        operation_type=operation_type,
        trace_performance=trace_performance,
        **context,
    ) as trace_id:
        yield trace_id


def get_correlation_id() -> str | None:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def get_request_id() -> str | None:
    """Get current request ID from context."""
    return request_id_var.get()


def log_with_correlation(level: str, message: str, **fields: MetadataValue) -> None:
    """Log a message with correlation context."""
    observability_manager.log_with_context(level, message, **fields)


def inject_correlation_context(func: F) -> F:
    """Decorator to inject correlation context into function logs."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        context = observability_manager.get_current_context()
        logger.info(
            f"function_called_{func.__name__}",
            **context,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )
        try:
            result = func(*args, **kwargs)
            logger.info(f"function_completed_{func.__name__}", **context, success=True)
            return result
        except Exception as e:
            logger.error(
                f"function_failed_{func.__name__}",
                **context,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise

    return wrapper  # type: ignore[return-value]  # Why: decorator preserves F via functools.wraps


def inject_correlation_context_async(func: F) -> F:
    """Async decorator to inject correlation context into function logs."""

    @functools.wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
        context = observability_manager.get_current_context()
        logger.info(
            f"async_function_called_{func.__name__}",
            **context,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )
        try:
            result = await cast("Awaitable[object]", func(*args, **kwargs))
            logger.info(
                f"async_function_completed_{func.__name__}", **context, success=True
            )
            return result
        except Exception as e:
            logger.error(
                f"async_function_failed_{func.__name__}",
                **context,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise

    return wrapper  # type: ignore[return-value]  # Why: decorator preserves F via functools.wraps
