# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Retry utilities with exponential backoff following ONEX standards.

failures in OmniMemory operations with configurable backoff strategies.
"""

from __future__ import annotations

__all__ = [
    "ModelRetryConfig",
    "ModelRetryAttemptInfo",
    "ModelRetryStatistics",
    "RetryManager",
    "calculate_delay",
    "is_retryable_exception",
    "retry_decorator",
    "retry_with_backoff",
    "default_retry_manager",
]

import asyncio
import concurrent.futures
import functools
import logging
import random
from collections.abc import Callable
from typing import Any, TypeVar, cast
from uuid import UUID

from ..models.utils.model_retry_attempt_info import ModelRetryAttemptInfo
from ..models.utils.model_retry_config import ModelRetryConfig
from ..models.utils.model_retry_statistics import ModelRetryStatistics
from .error_sanitizer import SanitizationLevel, sanitize_error

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_retryable_exception(
    exception: Exception, retryable_exceptions: list[str]
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


def calculate_delay(attempt: int, config: ModelRetryConfig) -> int:
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
    operation: Callable[..., T],
    config: ModelRetryConfig,
    correlation_id: UUID | None = None,
    *args: Any,
    **kwargs: Any,
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
    attempts: list[ModelRetryAttemptInfo] = []

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
            attempt_info = ModelRetryAttemptInfo(
                attempt_number=attempt, delay_ms=delay_ms, correlation_id=correlation_id
            )
            attempts.append(attempt_info)

            # Execute operation
            # Note: asyncio.iscoroutinefunction() narrows the type. For async
            # functions we await to get T, for sync functions we get T directly.
            if asyncio.iscoroutinefunction(operation):
                result: T = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)  # pyright: ignore[reportAssignmentType]

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
                # Sanitize error message for logging
                sanitized_msg = sanitize_error(e)
                logger.warning(
                    f"Attempt {attempt}/{config.max_attempts} failed: "
                    f"{type(e).__name__}: {sanitized_msg} "
                    f"(cid: {correlation_id})"
                )
                continue
            # Final failure or non-retryable exception
            # Use stricter sanitization for final failures
            sanitized_msg = sanitize_error(e, level=SanitizationLevel.STRICT)
            logger.error(
                f"Operation failed permanently after {attempt} attempts "
                f"with {type(e).__name__}: {sanitized_msg} "
                f"(correlation_id: {correlation_id})"
            )
            break

    # All attempts failed
    if last_exception:
        raise last_exception
    raise RuntimeError("Operation failed without exception")


def retry_decorator(
    config: ModelRetryConfig | None = None,
    max_attempts: int = 3,
    base_delay_ms: int = 1000,
    max_delay_ms: int = 30000,
    exponential_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: list[str] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
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
    # Build config with proper defaults - honor ModelRetryConfig defaults when
    # retryable_exceptions is None
    effective_config: ModelRetryConfig
    if config is not None:
        effective_config = config
    else:
        config_kwargs: dict[str, Any] = {
            "max_attempts": max_attempts,
            "base_delay_ms": base_delay_ms,
            "max_delay_ms": max_delay_ms,
            "exponential_multiplier": exponential_multiplier,
            "jitter": jitter,
        }
        # Only override retryable_exceptions if explicitly provided
        if retryable_exceptions is not None:
            config_kwargs["retryable_exceptions"] = retryable_exceptions
        effective_config = ModelRetryConfig(**config_kwargs)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            correlation_id = cast("UUID | None", kwargs.pop("correlation_id", None))
            return await retry_with_backoff(
                func, effective_config, correlation_id, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # For sync functions, run in event loop
            correlation_id = cast("UUID | None", kwargs.pop("correlation_id", None))

            async def async_operation() -> T:
                return await retry_with_backoff(
                    func, effective_config, correlation_id, *args, **kwargs
                )

            try:
                asyncio.get_running_loop()
                # Loop is already running - run in separate thread to avoid
                # blocking the current event loop. This handles the case where
                # sync code is called from within an async context.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, async_operation())
                    return future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run directly
                return asyncio.run(async_operation())

        if asyncio.iscoroutinefunction(func):
            return cast("Callable[..., T]", async_wrapper)
        else:
            return cast("Callable[..., T]", sync_wrapper)

    return decorator


class RetryManager:
    """
    Manager for retry operations with statistics tracking.
    """

    def __init__(self, default_config: ModelRetryConfig | None = None) -> None:
        """
        Initialize retry manager.

        Args:
            default_config: Default retry configuration
        """
        self.default_config = default_config or ModelRetryConfig()
        self.statistics = ModelRetryStatistics()
        self._operation_attempts: dict[str, int] = {}

    async def execute_with_retry(
        self,
        operation: Callable[..., T],
        operation_name: str,
        config: ModelRetryConfig | None = None,
        correlation_id: UUID | None = None,
        *args: Any,
        **kwargs: Any,
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
            result: T = await retry_with_backoff(
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

    def get_statistics(self) -> ModelRetryStatistics:
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
        self.statistics = ModelRetryStatistics()
        self._operation_attempts.clear()


# Global retry manager instance
default_retry_manager = RetryManager()
