"""
Resource management utilities for OmniMemory ONEX architecture.

This module provides:
- Async context managers for proper resource cleanup
- Circuit breaker patterns for external service resilience
- Connection pool management and exhaustion handling
- Timeout configurations for all async operations
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Optional, TypeVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


def _sanitize_error(error: Exception) -> str:
    """
    Sanitize error messages to prevent information disclosure in logs.

    Args:
        error: Exception to sanitize

    Returns:
        Safe error message without sensitive information
    """
    error_type = type(error).__name__
    # Only include safe, generic error information
    if isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return f"{error_type}: Connection or timeout issue"
    elif isinstance(error, ValueError):
        return f"{error_type}: Invalid value"
    elif isinstance(error, KeyError):
        return f"{error_type}: Missing key"
    elif isinstance(error, AttributeError):
        return f"{error_type}: Missing attribute"
    else:
        return f"{error_type}: Operation failed"


T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states following resilience patterns."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = Field(
        default=5, description="Number of failures before opening circuit"
    )
    recovery_timeout: int = Field(
        default=60, description="Seconds to wait before trying half-open"
    )
    recovery_timeout_jitter: float = Field(
        default=0.1, description="Jitter factor (0.0-1.0) to prevent thundering herd"
    )
    success_threshold: int = Field(
        default=3, description="Successful calls needed to close circuit"
    )
    timeout: float = Field(default=30.0, description="Default timeout for operations")


@dataclass
class CircuitBreakerStats:
    """Statistics tracking for circuit breaker behavior."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state_changed_at: datetime = field(default_factory=datetime.now)
    total_calls: int = 0
    total_timeouts: int = 0


class CircuitBreakerStatsResponse(BaseModel):
    """Typed response model for circuit breaker statistics."""

    state: str = Field(description="Current circuit breaker state")
    failure_count: int = Field(description="Number of failures recorded")
    success_count: int = Field(description="Number of successful calls")
    total_calls: int = Field(description="Total number of calls attempted")
    total_timeouts: int = Field(description="Total number of timeout failures")
    last_failure_time: Optional[str] = Field(
        description="ISO timestamp of last failure"
    )
    state_changed_at: str = Field(description="ISO timestamp when state last changed")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, service_name: str, state: CircuitState):
        self.service_name = service_name
        self.state = state
        super().__init__(f"Circuit breaker for {service_name} is {state.value}")


class AsyncCircuitBreaker:
    """
    Async circuit breaker for external service resilience.

    Implements the circuit breaker pattern to handle external service failures
    gracefully and provide fast failure when services are known to be down.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute a function call through the circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(self.name, self.state)

        try:
            # Apply timeout to the operation
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout
            )
            await self._on_success()
            return result

        except asyncio.TimeoutError as e:
            self.stats.total_timeouts += 1
            await self._on_failure(e)
            raise
        except Exception as e:
            await self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset with jitter."""
        if self.stats.last_failure_time is None:
            return True

        # Calculate recovery timeout with jitter to prevent thundering herd
        base_timeout = self.config.recovery_timeout
        jitter_range = base_timeout * self.config.recovery_timeout_jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        effective_timeout = base_timeout + jitter

        time_since_failure = datetime.now() - self.stats.last_failure_time
        return time_since_failure.total_seconds() >= effective_timeout

    async def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.stats.state_changed_at = datetime.now()
        self.stats.success_count = 0

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            new_state="half_open",
            reason="recovery_timeout_reached",
        )

    async def _on_success(self):
        """Handle successful operation result."""
        async with self._lock:
            self.stats.total_calls += 1

            if self.state == CircuitState.HALF_OPEN:
                self.stats.success_count += 1
                if self.stats.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.stats.failure_count = 0  # Reset failure count on success

    async def _on_failure(self, error: Exception):
        """Handle failed operation result."""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.failure_count += 1
            self.stats.last_failure_time = datetime.now()

            if (
                self.state == CircuitState.CLOSED
                and self.stats.failure_count >= self.config.failure_threshold
            ):
                await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    async def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.stats.state_changed_at = datetime.now()
        self.stats.failure_count = 0

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            new_state="closed",
            reason="success_threshold_reached",
        )

    async def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.stats.state_changed_at = datetime.now()

        logger.warning(
            "circuit_breaker_state_change",
            name=self.name,
            new_state="open",
            reason="failure_threshold_reached",
            failure_count=self.stats.failure_count,
        )


class AsyncResourceManager:
    """
    Comprehensive async resource manager for OmniMemory.

    Provides:
    - Circuit breakers for external services
    - Semaphores for rate-limited operations
    - Timeout management
    - Resource cleanup
    """

    def __init__(self):
        self._circuit_breakers: Dict[str, AsyncCircuitBreaker] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> AsyncCircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = AsyncCircuitBreaker(name, config)
        return self._circuit_breakers[name]

    def get_semaphore(self, name: str, limit: int) -> asyncio.Semaphore:
        """Get or create a semaphore for rate limiting."""
        if name not in self._semaphores:
            self._semaphores[name] = asyncio.Semaphore(limit)
        return self._semaphores[name]

    def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a lock for resource synchronization."""
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    @contextlib.asynccontextmanager
    async def managed_resource(
        self,
        resource_name: str,
        acquire_func: Callable[..., Any],
        release_func: Optional[Callable[[Any], None]] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        semaphore_limit: Optional[int] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """
        Async context manager for comprehensive resource management.

        Args:
            resource_name: Unique identifier for the resource
            acquire_func: Function to acquire the resource
            release_func: Function to release the resource
            circuit_breaker_config: Circuit breaker configuration
            semaphore_limit: Semaphore limit for rate limiting
            *args, **kwargs: Arguments passed to acquire_func
        """
        circuit_breaker = self.get_circuit_breaker(
            resource_name, circuit_breaker_config
        )
        semaphore = (
            self.get_semaphore(resource_name, semaphore_limit)
            if semaphore_limit
            else None
        )

        resource = None
        try:
            # Apply semaphore if configured
            if semaphore:
                await semaphore.acquire()

            # Acquire resource through circuit breaker
            resource = await circuit_breaker.call(acquire_func, *args, **kwargs)

            logger.debug(
                "resource_acquired",
                resource_name=resource_name,
                circuit_state=circuit_breaker.state.value,
            )

            yield resource

        except Exception as e:
            logger.error(
                "resource_management_error",
                resource_name=resource_name,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            # Clean up resource
            if resource is not None and release_func:
                try:
                    if asyncio.iscoroutinefunction(release_func):
                        await release_func(resource)
                    else:
                        release_func(resource)

                    logger.debug("resource_released", resource_name=resource_name)
                except Exception as e:
                    logger.error(
                        "resource_cleanup_error",
                        resource_name=resource_name,
                        error=_sanitize_error(e),
                    )

            # Release semaphore if acquired
            if semaphore:
                semaphore.release()

    def get_circuit_breaker_stats(self) -> Dict[str, CircuitBreakerStatsResponse]:
        """Get typed statistics for all circuit breakers."""
        stats = {}
        for name, cb in self._circuit_breakers.items():
            stats[name] = CircuitBreakerStatsResponse(
                state=cb.state.value,
                failure_count=cb.stats.failure_count,
                success_count=cb.stats.success_count,
                total_calls=cb.stats.total_calls,
                total_timeouts=cb.stats.total_timeouts,
                last_failure_time=cb.stats.last_failure_time.isoformat()
                if cb.stats.last_failure_time
                else None,
                state_changed_at=cb.stats.state_changed_at.isoformat(),
            )
        return stats


# Global resource manager instance
resource_manager = AsyncResourceManager()


# Convenience functions for common patterns
async def with_circuit_breaker(
    service_name: str,
    func: Callable[..., Any],
    config: Optional[CircuitBreakerConfig] = None,
    *args,
    **kwargs,
) -> Any:
    """Execute a function with circuit breaker protection."""
    circuit_breaker = resource_manager.get_circuit_breaker(service_name, config)
    return await circuit_breaker.call(func, *args, **kwargs)


@contextlib.asynccontextmanager
async def with_semaphore(name: str, limit: int):
    """Context manager for semaphore-based rate limiting."""
    semaphore = resource_manager.get_semaphore(name, limit)
    async with semaphore:
        yield


@contextlib.asynccontextmanager
async def with_timeout(timeout: float):
    """Context manager for timeout operations."""
    try:
        async with asyncio.timeout(timeout):
            yield
    except asyncio.TimeoutError:
        logger.warning("operation_timeout", timeout=timeout)
        raise
