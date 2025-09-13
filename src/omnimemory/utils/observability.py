"""
Observability utilities for OmniMemory ONEX architecture.

This module provides:
- ContextVar integration for correlation ID tracking
- Distributed tracing support
- Enhanced logging with correlation context
- Performance monitoring and metrics collection
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional, Union

from pydantic import BaseModel, Field
import structlog

# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
operation_var: ContextVar[Optional[str]] = ContextVar('operation', default=None)

logger = structlog.get_logger(__name__)

class TraceLevel(Enum):
    """Trace level enumeration for different types of operations."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

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
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage_start: Optional[float] = None
    memory_usage_end: Optional[float] = None
    memory_delta: Optional[float] = None
    success: Optional[bool] = None
    error_type: Optional[str] = None

class CorrelationContext(BaseModel):
    """Context information for correlation tracking."""
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    operation: Optional[str] = Field(default=None)
    parent_correlation_id: Optional[str] = Field(default=None)
    trace_level: TraceLevel = Field(default=TraceLevel.INFO)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class ObservabilityManager:
    """
    Comprehensive observability manager for OmniMemory.

    Provides:
    - Correlation ID management and propagation
    - Distributed tracing support
    - Performance monitoring
    - Enhanced logging with context
    """

    def __init__(self):
        self._active_traces: Dict[str, PerformanceMetrics] = {}
        self._logger = structlog.get_logger(__name__)

    @asynccontextmanager
    async def correlation_context(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        trace_level: TraceLevel = TraceLevel.INFO,
        **metadata
    ) -> AsyncGenerator[CorrelationContext, None]:
        """
        Async context manager for correlation tracking.

        Args:
            correlation_id: Unique correlation identifier
            request_id: Request identifier
            user_id: User identifier
            operation: Operation name
            trace_level: Tracing level
            **metadata: Additional metadata
        """
        # Create context
        context = CorrelationContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            request_id=request_id,
            user_id=user_id,
            operation=operation,
            parent_correlation_id=correlation_id_var.get(),
            trace_level=trace_level,
            metadata=metadata
        )

        # Set context variables
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
                metadata=context.metadata
            )

            yield context

        except Exception as e:
            self._logger.error(
                "correlation_context_error",
                correlation_id=context.correlation_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            # Reset context variables
            correlation_id_var.reset(correlation_token)
            request_id_var.reset(request_token)
            user_id_var.reset(user_token)
            operation_var.reset(operation_token)

            self._logger.info(
                "correlation_context_ended",
                correlation_id=context.correlation_id,
                operation=context.operation
            )

    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        operation_type: OperationType,
        trace_performance: bool = True,
        **additional_context
    ) -> AsyncGenerator[str, None]:
        """
        Async context manager for operation tracing.

        Args:
            operation_name: Name of the operation being traced
            operation_type: Type of operation
            trace_performance: Whether to track performance metrics
            **additional_context: Additional context for tracing
        """
        trace_id = str(uuid.uuid4())
        correlation_id = correlation_id_var.get()

        # Initialize performance metrics if requested
        if trace_performance:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            metrics = PerformanceMetrics(
                start_time=time.time(),
                memory_usage_start=start_memory
            )
            self._active_traces[trace_id] = metrics

        try:
            self._logger.info(
                "operation_started",
                trace_id=trace_id,
                correlation_id=correlation_id,
                operation_name=operation_name,
                operation_type=operation_type.value,
                **additional_context
            )

            yield trace_id

            # Mark as successful
            if trace_performance and trace_id in self._active_traces:
                self._active_traces[trace_id].success = True

        except Exception as e:
            # Mark as failed and log error
            if trace_performance and trace_id in self._active_traces:
                self._active_traces[trace_id].success = False
                self._active_traces[trace_id].error_type = type(e).__name__

            self._logger.error(
                "operation_failed",
                trace_id=trace_id,
                correlation_id=correlation_id,
                operation_name=operation_name,
                operation_type=operation_type.value,
                error=str(e),
                error_type=type(e).__name__,
                **additional_context
            )
            raise
        finally:
            # Complete performance metrics if requested
            if trace_performance and trace_id in self._active_traces:
                metrics = self._active_traces[trace_id]
                metrics.end_time = time.time()
                metrics.duration = metrics.end_time - metrics.start_time

                if metrics.memory_usage_start:
                    import psutil
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    metrics.memory_usage_end = end_memory
                    metrics.memory_delta = end_memory - metrics.memory_usage_start

                self._logger.info(
                    "operation_completed",
                    trace_id=trace_id,
                    correlation_id=correlation_id,
                    operation_name=operation_name,
                    operation_type=operation_type.value,
                    duration=metrics.duration,
                    memory_delta=metrics.memory_delta,
                    success=metrics.success,
                    error_type=metrics.error_type,
                    **additional_context
                )

                # Clean up completed trace
                del self._active_traces[trace_id]

    def get_current_context(self) -> Dict[str, Optional[str]]:
        """Get current correlation context."""
        return {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "operation": operation_var.get()
        }

    def get_performance_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get current performance metrics for active traces."""
        return self._active_traces.copy()

    def log_with_context(
        self,
        level: str,
        message: str,
        **additional_fields
    ):
        """Log a message with current correlation context."""
        context = self.get_current_context()

        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(
            message,
            **context,
            **additional_fields
        )

# Global observability manager instance
observability_manager = ObservabilityManager()

# Convenience functions for common patterns
@asynccontextmanager
async def correlation_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    operation: Optional[str] = None,
    **metadata
):
    """Convenience function for correlation context management."""
    async with observability_manager.correlation_context(
        correlation_id=correlation_id,
        request_id=request_id,
        user_id=user_id,
        operation=operation,
        **metadata
    ) as ctx:
        yield ctx

@asynccontextmanager
async def trace_operation(
    operation_name: str,
    operation_type: Union[OperationType, str],
    **context
):
    """Convenience function for operation tracing."""
    if isinstance(operation_type, str):
        # Try to convert string to OperationType
        try:
            operation_type = OperationType(operation_type)
        except ValueError:
            # Default to external API if unknown
            operation_type = OperationType.EXTERNAL_API

    async with observability_manager.trace_operation(
        operation_name=operation_name,
        operation_type=operation_type,
        **context
    ) as trace_id:
        yield trace_id

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id_var.get()

def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()

def log_with_correlation(level: str, message: str, **fields):
    """Log a message with correlation context."""
    observability_manager.log_with_context(level, message, **fields)

def inject_correlation_context(func):
    """Decorator to inject correlation context into function logs."""
    def wrapper(*args, **kwargs):
        context = observability_manager.get_current_context()
        logger.info(
            f"function_called_{func.__name__}",
            **context,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        try:
            result = func(*args, **kwargs)
            logger.info(
                f"function_completed_{func.__name__}",
                **context,
                success=True
            )
            return result
        except Exception as e:
            logger.error(
                f"function_failed_{func.__name__}",
                **context,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    return wrapper

async def inject_correlation_context_async(func):
    """Async decorator to inject correlation context into function logs."""
    async def wrapper(*args, **kwargs):
        context = observability_manager.get_current_context()
        logger.info(
            f"async_function_called_{func.__name__}",
            **context,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        try:
            result = await func(*args, **kwargs)
            logger.info(
                f"async_function_completed_{func.__name__}",
                **context,
                success=True
            )
            return result
        except Exception as e:
            logger.error(
                f"async_function_failed_{func.__name__}",
                **context,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    return wrapper