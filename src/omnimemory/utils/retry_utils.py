"""
Retry utilities with exponential backoff following ONEX standards.

This module provides retry decorators and utilities for handling transient
failures in OmniMemory operations with configurable backoff strategies.
"""

from __future__ import annotations

__all__ = [
    "RetryConfig",
    "RetryAttemptInfo",
    "RetryStatistics",
    "is_retryable_exception",
    "retry_with_backoff",
    "retry_decorator",
    "RetryManager",
]

import asyncio
import functools
import logging
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

from .error_sanitizer import ErrorSanitizer, SanitizationLevel

logger = logging.getLogger(__name__)

# Initialize error sanitizer for secure logging
_error_sanitizer = ErrorSanitizer(
    default_level=SanitizationLevel.STANDARD, enable_stack_trace_filter=True
)

T = TypeVar("T")


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum number of retry attempts"
    )
    base_delay_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Base delay between attempts in milliseconds",
    )
    max_delay_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Maximum delay between attempts in milliseconds",
    )
    exponential_multiplier: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Exponential backoff multiplier"
    )
    jitter: bool = Field(
        default=True, description="Whether to add random jitter to delays"
    )
    retryable_exceptions: List[str] = Field(
        default_factory=lambda: [
            "ConnectionError",
            "TimeoutError",
            "HTTPError",
            "TemporaryFailure",
        ],
        description="Exception types that should trigger retries",
    )


class RetryAttemptInfo(BaseModel):
    """Information about a retry attempt."""

    attempt_number: int = Field(description="Current attempt number (1-indexed)")
    delay_ms: int = Field(description="Delay before this attempt in milliseconds")
    exception: Optional[str] = Field(
        default=None, description="Exception that triggered the retry"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the attempt was made"
    )
    correlation_id: Optional[UUID] = Field(
        default=None, description="Request correlation ID"
    )


class RetryStatistics(BaseModel):
    """Statistics about retry operations."""

    total_operations: int = Field(
        default=0, description="Total number of operations attempted"
    )
    successful_operations: int = Field(
        default=0, description="Number of successful operations"
    )
    failed_operations: int = Field(
        default=0, description="Number of permanently failed operations"
    )
    total_retries: int = Field(default=0, description="Total number of retry attempts")
    average_attempts: float = Field(
        default=0.0, description="Average number of attempts per operation"
    )
    common_exceptions: Dict[str, int] = Field(
        default_factory=dict, description="Count of common exceptions encountered"
    )


def is_retryable_exception(
    exception: Exception, retryable_exceptions: List[str]
) -> bool:
    """
    Check if an exception should trigger a retry.

    Args:
        exception: The exception to check
        retryable_exceptions: List of retryable exception type names

    Returns:
        True if the exception should trigger a retry
    """
    exception_name = type(exception).__name__

    # Check exact match
    if exception_name in retryable_exceptions:
        return True

    # Check inheritance (common patterns)
    for retryable in retryable_exceptions:
        if retryable in exception_name:
            return True

    return False


def calculate_delay(attempt: int, config: RetryConfig) -> int:
    """
    Calculate delay for a retry attempt with exponential backoff.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        Delay in milliseconds
    """
    if attempt <= 1:
        return 0

    # Exponential backoff: base_delay * multiplier^(attempt-2)
    delay = config.base_delay_ms * (config.exponential_multiplier ** (attempt - 2))

    # Cap at maximum delay
    delay = min(delay, config.max_delay_ms)

    # Add jitter if enabled (±25% random variation)
    if config.jitter:
        jitter_range = delay * 0.25
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay + jitter)

    return int(delay)


async def retry_with_backoff(
    operation: Callable[..., Any],
    config: RetryConfig,
    correlation_id: Optional[UUID] = None,
    *args,
    **kwargs,
) -> T:
    """
    Execute an operation with retry and exponential backoff.

    Args:
        operation: The operation to execute
        config: Retry configuration
        correlation_id: Optional correlation ID for tracking
        *args: Positional arguments for the operation
        **kwargs: Keyword arguments for the operation

    Returns:
        The result of the successful operation

    Raises:
        The last exception if all retry attempts fail
    """
    last_exception = None
    attempts: List[RetryAttemptInfo] = []

    for attempt in range(1, config.max_attempts + 1):
        try:
            delay_ms = calculate_delay(attempt, config)

            if delay_ms > 0:
                logger.debug(
                    f"Retry attempt {attempt}/{config.max_attempts} "
                    f"after {delay_ms}ms delay (correlation_id: {correlation_id})"
                )
                await asyncio.sleep(delay_ms / 1000.0)

            # Record attempt
            attempt_info = RetryAttemptInfo(
                attempt_number=attempt, delay_ms=delay_ms, correlation_id=correlation_id
            )
            attempts.append(attempt_info)

            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)

            # Success - log if there were retries
            if attempt > 1:
                logger.info(
                    f"Operation succeeded on attempt {attempt}/{config.max_attempts} "
                    f"(correlation_id: {correlation_id})"
                )

            return result

        except Exception as e:
            last_exception = e

            # Update attempt info with exception
            attempts[-1].exception = type(e).__name__

            # Check if we should retry
            if attempt < config.max_attempts and is_retryable_exception(
                e, config.retryable_exceptions
            ):
                # Sanitize error message to prevent information disclosure
                sanitized_error = _error_sanitizer.sanitize_error_message(str(e))
                logger.warning(
                    f"Attempt {attempt}/{config.max_attempts} failed with {type(e).__name__}: {sanitized_error} "
                    f"(correlation_id: {correlation_id})"
                )
                continue
            else:
                # Final failure or non-retryable exception
                # Use stricter sanitization for final failures
                sanitized_error = _error_sanitizer.sanitize_error_message(
                    str(e), level=SanitizationLevel.STRICT
                )
                logger.error(
                    f"Operation failed permanently after {attempt} attempts "
                    f"with {type(e).__name__}: {sanitized_error} "
                    f"(correlation_id: {correlation_id})"
                )
                break

    # All attempts failed
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("Operation failed without exception")


def retry_decorator(
    config: Optional[RetryConfig] = None,
    max_attempts: int = 3,
    base_delay_ms: int = 1000,
    max_delay_ms: int = 30000,
    exponential_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator for adding retry behavior to functions.

    Args:
        config: Retry configuration (if provided, other params ignored)
        max_attempts: Maximum retry attempts
        base_delay_ms: Base delay between attempts in milliseconds
        max_delay_ms: Maximum delay between attempts in milliseconds
        exponential_multiplier: Exponential backoff multiplier
        jitter: Whether to add random jitter
        retryable_exceptions: List of retryable exception names

    Returns:
        Decorated function with retry behavior
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay_ms=base_delay_ms,
            max_delay_ms=max_delay_ms,
            exponential_multiplier=exponential_multiplier,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions or [],
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            correlation_id = kwargs.pop("correlation_id", None)
            return await retry_with_backoff(
                func, config, correlation_id, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # For sync functions, run in event loop
            correlation_id = kwargs.pop("correlation_id", None)

            async def async_operation():
                return await retry_with_backoff(
                    func, config, correlation_id, *args, **kwargs
                )

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a task
                    task = loop.create_task(async_operation())
                    return loop.run_until_complete(task)
                else:
                    return loop.run_until_complete(async_operation())
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(async_operation())

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RetryManager:
    """
    Manager for retry operations with statistics tracking.
    """

    def __init__(self, default_config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.

        Args:
            default_config: Default retry configuration
        """
        self.default_config = default_config or RetryConfig()
        self.statistics = RetryStatistics()
        self._operation_attempts: Dict[str, int] = {}

    async def execute_with_retry(
        self,
        operation: Callable[..., T],
        operation_name: str,
        config: Optional[RetryConfig] = None,
        correlation_id: Optional[UUID] = None,
        *args,
        **kwargs,
    ) -> T:
        """
        Execute an operation with retry and track statistics.

        Args:
            operation: The operation to execute
            operation_name: Name for tracking purposes
            config: Optional retry configuration (uses default if not provided)
            correlation_id: Optional correlation ID
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result
        """
        retry_config = config or self.default_config

        try:
            result = await retry_with_backoff(
                operation, retry_config, correlation_id, *args, **kwargs
            )

            # Update success statistics
            self.statistics.total_operations += 1
            self.statistics.successful_operations += 1

            return result

        except Exception as e:
            # Update failure statistics
            self.statistics.total_operations += 1
            self.statistics.failed_operations += 1

            exception_name = type(e).__name__
            if exception_name in self.statistics.common_exceptions:
                self.statistics.common_exceptions[exception_name] += 1
            else:
                self.statistics.common_exceptions[exception_name] = 1

            raise

    def get_statistics(self) -> RetryStatistics:
        """
        Get current retry statistics.

        Returns:
            Current statistics
        """
        # Calculate average attempts
        if self.statistics.total_operations > 0:
            self.statistics.average_attempts = (
                self.statistics.total_operations + self.statistics.total_retries
            ) / self.statistics.total_operations

        return self.statistics

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.statistics = RetryStatistics()
        self._operation_attempts.clear()


# Global retry manager instance
default_retry_manager = RetryManager()
