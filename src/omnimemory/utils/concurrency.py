# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Concurrency utilities for OmniMemory ONEX architecture.

This module provides:
- Advanced semaphore patterns for rate-limited operations
- Proper locking mechanisms for shared resources
- Connection pool management and exhaustion handling
- Fair scheduling and priority-based access control

NOTE ON Any TYPES:
This module intentionally uses 'Any' types in the connection pool implementation:
- validate_connection: Callable[[Any], bool] - Validates connections of arbitrary types
- close_connection: Callable[[Any], None] - Closes connections of arbitrary types
- Connection pool methods returning Any - Pools manage generic connection objects

This design allows the same pool implementation to work with any connection type
(database connections, HTTP sessions, etc.) without requiring generic type parameters
throughout the codebase.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import functools
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

# Type variable for generic function return types
F = TypeVar("F", bound=Callable[..., Any])

from uuid import uuid4

import structlog

from ..models.foundation.model_connection_metadata import (
    ConnectionMetadata,
)
from .error_sanitizer import SanitizationLevel
from .error_sanitizer import sanitize_error as _base_sanitize_error
from .observability import (
    OperationType,
    correlation_context,
    trace_operation,
)

if TYPE_CHECKING:
    from ..models.utils.model_concurrency import ModelConnectionPoolConfig

logger = structlog.get_logger(__name__)


def _sanitize_error(error: Exception) -> str:
    """
    Sanitize error messages to prevent information disclosure in logs.

    Uses the centralized error sanitizer for consistent security handling.

    Args:
        error: Exception to sanitize

    Returns:
        Safe error message without sensitive information
    """
    return _base_sanitize_error(
        error, context="connection_pool", level=SanitizationLevel.STANDARD
    )


class LockPriority(Enum):
    """Priority levels for lock acquisition."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class PoolStatus(Enum):
    """Connection pool status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


def _create_default_connection_metadata() -> ConnectionMetadata:
    """Create a default ConnectionMetadata with a generated connection_id."""
    return ConnectionMetadata(connection_id=str(uuid4()))


@dataclass
class LockRequest:
    """Request for lock acquisition with priority and metadata."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    priority: LockPriority = LockPriority.NORMAL
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Monotonic timestamp for timeout tracking - immune to wall clock adjustments
    requested_at_monotonic: float = field(default_factory=time.monotonic)
    correlation_id: str | None = None
    timeout: float | None = None
    metadata: ConnectionMetadata = field(
        default_factory=_create_default_connection_metadata
    )


@dataclass
class SemaphoreStats:
    """Statistics for semaphore usage."""

    total_permits: int
    available_permits: int
    waiting_count: int
    total_acquisitions: int = 0
    total_releases: int = 0
    total_timeouts: int = 0
    average_hold_time: float = 0.0
    max_hold_time: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PoolMetrics:
    """Metrics for connection pool monitoring."""

    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_created: int = 0
    total_destroyed: int = 0
    pool_exhaustions: int = 0
    average_wait_time: float = 0.0
    last_exhaustion: datetime | None = None


class PriorityLock:
    """
    Async lock with priority-based fair scheduling.

    Provides priority-based access to shared resources with fairness
    guarantees and timeout support.
    """

    def __init__(self, name: str):
        self.name = name
        self._lock = asyncio.Lock()
        self._queue: list[LockRequest] = []
        self._current_holder: LockRequest | None = None
        self._stats = {
            "total_acquisitions": 0,
            "total_releases": 0,
            "total_timeouts": 0,
            "average_hold_time": 0.0,
            "max_hold_time": 0.0,
        }

    @asynccontextmanager
    async def acquire(
        self,
        priority: LockPriority = LockPriority.NORMAL,
        timeout: float | None = None,
        correlation_id: str | None = None,
        **metadata: Any,
    ) -> AsyncGenerator[None, None]:
        """
        Acquire the lock with priority and timeout support.

        Args:
            priority: Priority level for lock acquisition
            timeout: Maximum time to wait for lock
            correlation_id: Correlation ID for tracing
            **metadata: Additional metadata for the lock request
        """
        # Create metadata with a unique connection_id, merging any additional metadata
        connection_metadata = (
            ConnectionMetadata(
                connection_id=str(uuid4()),
                **metadata,
            )
            if metadata
            else _create_default_connection_metadata()
        )
        request = LockRequest(
            priority=priority,
            timeout=timeout,
            correlation_id=correlation_id,
            metadata=connection_metadata,
        )

        acquired_at: datetime | None = None

        async with correlation_context(correlation_id=correlation_id):
            async with trace_operation(
                f"priority_lock_acquire_{self.name}",
                OperationType.EXTERNAL_API,  # Using as generic operation type
                lock_name=self.name,
                priority=priority.name,
            ):
                try:
                    # Add request to priority queue
                    await self._enqueue_request(request)

                    # Wait for our turn
                    await self._wait_for_turn(request)

                    acquired_at = datetime.now(timezone.utc)
                    self._current_holder = request
                    self._stats["total_acquisitions"] += 1

                    logger.debug(
                        "priority_lock_acquired",
                        lock_name=self.name,
                        request_id=request.request_id,
                        priority=priority.name,
                        wait_time=(acquired_at - request.requested_at).total_seconds(),
                    )

                    yield

                except TimeoutError:
                    self._stats["total_timeouts"] += 1
                    logger.warning(
                        "priority_lock_timeout",
                        lock_name=self.name,
                        request_id=request.request_id,
                        timeout=timeout,
                    )
                    raise
                finally:
                    # Always clean up
                    await self._cleanup_request(request, acquired_at)

    async def _enqueue_request(self, request: LockRequest) -> None:
        """Add request to priority queue maintaining order."""
        async with self._lock:
            # Insert request maintaining priority order (higher priority first)
            inserted = False
            for i, queued_request in enumerate(self._queue):
                if request.priority.value > queued_request.priority.value:
                    self._queue.insert(i, request)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(request)

    async def _wait_for_turn(self, request: LockRequest) -> None:
        """Wait until it's this request's turn to acquire the lock."""
        while True:
            async with self._lock:
                # Check if we're at the front of the queue
                if self._queue and self._queue[0].request_id == request.request_id:
                    # Check if lock is available
                    if self._current_holder is None:
                        # Remove from queue and proceed
                        self._queue.pop(0)
                        return

            # Apply timeout if specified using monotonic clock
            # (immune to system clock adjustments like NTP, daylight saving, etc.)
            if request.timeout:
                elapsed = time.monotonic() - request.requested_at_monotonic
                if elapsed >= request.timeout:
                    raise TimeoutError(
                        f"Lock acquisition timeout after {request.timeout}s"
                    )

            # Wait a bit before checking again
            await asyncio.sleep(0.001)  # 1ms

    async def _cleanup_request(
        self, request: LockRequest, acquired_at: datetime | None
    ) -> None:
        """Clean up after lock release."""
        async with self._lock:
            # Calculate hold time if lock was acquired
            if acquired_at:
                hold_time = (datetime.now(timezone.utc) - acquired_at).total_seconds()
                self._stats["total_releases"] += 1

                # Update average hold time
                current_avg = self._stats["average_hold_time"]
                releases = self._stats["total_releases"]
                self._stats["average_hold_time"] = (
                    (current_avg * (releases - 1)) + hold_time
                ) / releases

                # Update max hold time
                self._stats["max_hold_time"] = max(
                    self._stats["max_hold_time"], hold_time
                )

            # Remove from queue if still there (timeout case)
            self._queue = [r for r in self._queue if r.request_id != request.request_id]

            # Clear current holder
            if (
                self._current_holder
                and self._current_holder.request_id == request.request_id
            ):
                self._current_holder = None


class FairSemaphore:
    """
    Fair semaphore with statistics and priority support.

    Provides fair access to limited resources with comprehensive
    monitoring and priority-based scheduling.
    """

    def __init__(self, value: int, name: str):
        self.name = name
        self._semaphore = asyncio.Semaphore(value)
        self._total_permits = value
        self._waiting_queue: deque[LockRequest] = deque()
        self._active_holders: dict[str, datetime] = {}
        self._stats = SemaphoreStats(
            total_permits=value, available_permits=value, waiting_count=0
        )
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(
        self, timeout: float | None = None, correlation_id: str | None = None
    ) -> AsyncGenerator[None, None]:
        """
        Acquire semaphore permit with timeout and tracking.

        Args:
            timeout: Maximum time to wait for permit
            correlation_id: Correlation ID for tracing
        """
        holder_id = str(uuid4())
        acquired_at: datetime | None = None

        async with correlation_context(correlation_id=correlation_id):
            async with trace_operation(
                f"semaphore_acquire_{self.name}",
                OperationType.EXTERNAL_API,
                semaphore_name=self.name,
                holder_id=holder_id,
            ):
                try:
                    # Update waiting count
                    async with self._lock:
                        self._stats.waiting_count += 1

                    # Acquire with timeout
                    if timeout:
                        await asyncio.wait_for(
                            self._semaphore.acquire(), timeout=timeout
                        )
                    else:
                        await self._semaphore.acquire()

                    acquired_at = datetime.now(timezone.utc)

                    # Update statistics
                    async with self._lock:
                        self._active_holders[holder_id] = acquired_at
                        self._stats.waiting_count -= 1
                        self._stats.available_permits -= 1
                        self._stats.total_acquisitions += 1

                    logger.debug(
                        "semaphore_acquired",
                        semaphore_name=self.name,
                        holder_id=holder_id,
                        available_permits=self._stats.available_permits,
                    )

                    yield

                except TimeoutError:
                    async with self._lock:
                        self._stats.waiting_count -= 1
                        self._stats.total_timeouts += 1

                    logger.warning(
                        "semaphore_timeout",
                        semaphore_name=self.name,
                        holder_id=holder_id,
                        timeout=timeout,
                    )
                    raise
                finally:
                    # Always release and update stats
                    if acquired_at:
                        hold_time = (
                            datetime.now(timezone.utc) - acquired_at
                        ).total_seconds()

                        async with self._lock:
                            self._active_holders.pop(holder_id, None)
                            self._stats.available_permits += 1
                            self._stats.total_releases += 1

                            # Update hold time statistics (optimized calculation)
                            releases = self._stats.total_releases
                            if releases == 1:
                                # First release, set average directly
                                self._stats.average_hold_time = hold_time
                            else:
                                # Use exponential moving average for better performance
                                alpha = min(
                                    0.1, 2.0 / (releases + 1)
                                )  # Adaptive smoothing factor
                                self._stats.average_hold_time = (
                                    1 - alpha
                                ) * self._stats.average_hold_time + alpha * hold_time
                            self._stats.max_hold_time = max(
                                self._stats.max_hold_time, hold_time
                            )

                        self._semaphore.release()

                        logger.debug(
                            "semaphore_released",
                            semaphore_name=self.name,
                            holder_id=holder_id,
                            hold_time=hold_time,
                            available_permits=self._stats.available_permits,
                        )

    def get_stats(self) -> SemaphoreStats:
        """Get current semaphore statistics.

        Returns a copy to prevent external mutation of internal state.
        """
        return replace(self._stats)


class AsyncConnectionPool:
    """
    Advanced async connection pool with health checking and metrics.

    Provides robust connection management with:
    - Health checking and automatic recovery
    - Pool exhaustion handling
    - Connection lifecycle management
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        config: ModelConnectionPoolConfig,
        create_connection: Callable[[], Any],
        validate_connection: Callable[[Any], bool] | None = None,
        close_connection: Callable[[Any], None] | None = None,
    ):
        self.config = config
        self._create_connection = create_connection
        self._validate_connection = validate_connection or (lambda conn: True)
        self._close_connection = close_connection or (lambda conn: None)

        # Connection availability queue with built-in synchronization.
        # asyncio.Queue provides thread-safe put/get operations with internal
        # event signaling - no explicit _available_event needed. When a connection
        # is returned via put_nowait(), any coroutines awaiting get() are
        # automatically woken up by the queue's internal notification mechanism.
        self._available: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=config.max_connections
        )
        # Maps connection_id (str) -> connection object (Any for generic pool support).
        # See module docstring for rationale on Any type usage.
        self._active: dict[str, Any] = {}
        self._metrics = PoolMetrics()
        self._status = PoolStatus.HEALTHY
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task[None] | None = None

        # Start health check task
        self._start_health_check()

    @asynccontextmanager
    async def acquire(
        self,
        timeout: float | None = None,
        correlation_id: str | None = None,
        _retry_count: int = 0,
    ) -> AsyncGenerator[Any, None]:
        """
        Acquire a connection from the pool.

        Args:
            timeout: Maximum time to wait for connection
            correlation_id: Correlation ID for tracing
            _retry_count: Internal retry counter to prevent infinite recursion

        Yields:
            Connection object from the pool

        Raises:
            RuntimeError: If maximum retry attempts exceeded
        """
        connection_id = str(uuid4())
        connection = None
        acquired_at = datetime.now(timezone.utc)

        async with correlation_context(correlation_id=correlation_id):
            async with trace_operation(
                f"connection_pool_acquire_{self.config.name}",
                OperationType.EXTERNAL_API,
                pool_name=self.config.name,
                connection_id=connection_id,
            ):
                max_retries = 3
                current_retry = _retry_count

                try:
                    # Use iterative retry loop instead of recursion
                    while current_retry <= max_retries:
                        try:
                            # Try to get existing connection first
                            try:
                                connection = self._available.get_nowait()
                                logger.debug(
                                    "connection_reused",
                                    pool_name=self.config.name,
                                    connection_id=connection_id,
                                )
                            except asyncio.QueueEmpty:
                                # No available connections, check capacity
                                async with self._lock:
                                    total_connections = (
                                        len(self._active) + self._available.qsize()
                                    )

                                    if total_connections < self.config.max_connections:
                                        # Create new connection
                                        connection = await self._create_new_connection()
                                        logger.debug(
                                            "connection_created",
                                            pool_name=self.config.name,
                                            connection_id=connection_id,
                                            total_connections=total_connections + 1,
                                        )
                                    else:
                                        # Pool at capacity, wait for available
                                        self._metrics.pool_exhaustions += 1
                                        self._metrics.last_exhaustion = datetime.now(
                                            timezone.utc
                                        )
                                        self._status = PoolStatus.EXHAUSTED

                                        logger.warning(
                                            "connection_pool_exhausted",
                                            pool_name=self.config.name,
                                            max_connections=self.config.max_connections,
                                        )

                                        # Wait for connection with timeout
                                        wait_timeout = (
                                            timeout or self.config.connection_timeout
                                        )
                                        connection = await asyncio.wait_for(
                                            self._available.get(), timeout=wait_timeout
                                        )

                            # Validate connection before use
                            if not self._validate_connection(connection):
                                logger.warning(
                                    "connection_invalid",
                                    pool_name=self.config.name,
                                    connection_id=connection_id,
                                    retry_count=current_retry,
                                )
                                await self._destroy_connection(connection)

                                # Check retry limit
                                if current_retry >= max_retries:
                                    logger.error(
                                        "connection_validation_max_retries_exceeded",
                                        pool_name=self.config.name,
                                        connection_id=connection_id,
                                        max_retries=max_retries,
                                    )
                                    msg = (
                                        f"Failed to acquire connection "
                                        f"after {max_retries} attempts"
                                    )
                                    raise RuntimeError(msg)

                                # Increment retry counter and continue the loop
                                current_retry += 1
                                continue

                            # Connection is valid, break out of retry loop
                            break

                        except Exception:
                            # Handle exceptions during connection acquisition
                            if connection:
                                await self._destroy_connection(connection)
                            raise

                    # Track active connection
                    async with self._lock:
                        self._active[connection_id] = connection

                    # Update metrics
                    wait_time = (
                        datetime.now(timezone.utc) - acquired_at
                    ).total_seconds()
                    if wait_time > 0:
                        current_avg = self._metrics.average_wait_time
                        acquisitions = len(self._active)
                        self._metrics.average_wait_time = (
                            (current_avg * (acquisitions - 1)) + wait_time
                        ) / acquisitions

                    yield connection

                except TimeoutError:
                    logger.error(
                        "connection_acquisition_timeout",
                        pool_name=self.config.name,
                        connection_id=connection_id,
                        timeout=timeout or self.config.connection_timeout,
                    )
                    raise
                except Exception as e:
                    self._metrics.failed_connections += 1
                    logger.error(
                        "connection_acquisition_failed",
                        pool_name=self.config.name,
                        connection_id=connection_id,
                        error=_sanitize_error(e),
                        error_type=type(e).__name__,
                    )
                    raise
                finally:
                    # Return connection to pool with shielded cleanup
                    if connection:
                        try:
                            # Shield cleanup to ensure it completes if cancelled
                            await asyncio.shield(
                                self._return_connection(connection_id, connection)
                            )
                        except Exception as cleanup_error:
                            # Log cleanup errors but don't propagate them
                            logger.error(
                                "connection_cleanup_failed",
                                pool_name=self.config.name,
                                connection_id=connection_id,
                                error=_sanitize_error(cleanup_error),
                            )

    async def _create_new_connection(self) -> Any:
        """Create a new connection."""
        try:
            connection = await self._create_connection()
            self._metrics.total_created += 1
            return connection
        except Exception as e:
            self._metrics.failed_connections += 1
            logger.error(
                "connection_creation_failed",
                pool_name=self.config.name,
                error=_sanitize_error(e),
            )
            raise

    async def _return_connection(self, connection_id: str, connection: Any) -> None:
        """Return a connection to the pool."""
        try:
            async with self._lock:
                # Remove from active connections
                self._active.pop(connection_id, None)

            # Validate connection before returning to pool
            if self._validate_connection(connection):
                try:
                    self._available.put_nowait(connection)
                    logger.debug(
                        "connection_returned",
                        pool_name=self.config.name,
                        connection_id=connection_id,
                    )
                except asyncio.QueueFull:
                    # Pool is full, destroy excess connection
                    await self._destroy_connection(connection)
            else:
                # Connection is invalid, destroy it
                await self._destroy_connection(connection)

        except Exception as e:
            logger.error(
                "connection_return_failed",
                pool_name=self.config.name,
                connection_id=connection_id,
                error=_sanitize_error(e),
            )
            # Try to destroy the connection on error
            try:
                await self._destroy_connection(connection)
            except Exception:
                pass  # Ignore cleanup errors

    async def _destroy_connection(self, connection: Any) -> None:
        """Destroy a connection."""
        try:
            if asyncio.iscoroutinefunction(self._close_connection):
                await self._close_connection(connection)
            else:
                self._close_connection(connection)

            self._metrics.total_destroyed += 1
        except Exception as e:
            logger.error(
                "connection_destruction_failed",
                pool_name=self.config.name,
                error=_sanitize_error(e),
            )

    def _start_health_check(self) -> None:
        """Start the health check background task."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "health_check_error",
                    pool_name=self.config.name,
                    error=_sanitize_error(e),
                )

    async def _perform_health_check(self) -> None:
        """Perform health check on pool connections."""
        # Simple health check - could be enhanced based on specific needs
        total_connections = len(self._active) + self._available.qsize()

        if total_connections == 0 and self._metrics.pool_exhaustions > 0:
            self._status = PoolStatus.FAILED
        elif self._available.qsize() < self.config.min_connections:
            self._status = PoolStatus.DEGRADED
        else:
            self._status = PoolStatus.HEALTHY

        logger.debug(
            "pool_health_check",
            pool_name=self.config.name,
            status=self._status.value,
            active_connections=len(self._active),
            available_connections=self._available.qsize(),
            total_connections=total_connections,
        )

    def get_metrics(self) -> PoolMetrics:
        """Get current pool metrics.

        Returns a copy to prevent external mutation of internal state.
        """
        self._metrics.active_connections = len(self._active)
        self._metrics.idle_connections = self._available.qsize()
        return replace(self._metrics)

    def get_status(self) -> PoolStatus:
        """Get current pool status."""
        return self._status

    async def close(self) -> None:
        """Close the connection pool and all connections."""
        if self._health_check_task:
            self._health_check_task.cancel()

        # Close all active connections
        for connection in self._active.values():
            await self._destroy_connection(connection)

        # Close all available connections
        while not self._available.empty():
            try:
                connection = self._available.get_nowait()
                await self._destroy_connection(connection)
            except asyncio.QueueEmpty:
                break

        self._active.clear()


# Global managers
_locks: dict[str, PriorityLock] = {}
_semaphores: dict[str, FairSemaphore] = {}
_pools: dict[str, AsyncConnectionPool] = {}
_manager_lock = asyncio.Lock()

# Shared ThreadPoolExecutor for sync function timeout enforcement
# Using a module-level executor avoids the overhead of creating a new executor per call
_shared_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=10, thread_name_prefix="omnimemory_timeout_"
)


def _cleanup_shared_executor() -> None:
    """Clean up the shared executor on module unload.

    Registered via atexit to ensure proper resource cleanup when the
    Python interpreter exits.
    """
    _shared_executor.shutdown(wait=False)


# Register cleanup handler for the shared executor
atexit.register(_cleanup_shared_executor)


async def get_priority_lock(name: str) -> PriorityLock:
    """Get or create a priority lock by name."""
    async with _manager_lock:
        if name not in _locks:
            _locks[name] = PriorityLock(name)
        return _locks[name]


async def get_fair_semaphore(name: str, permits: int) -> FairSemaphore:
    """Get or create a fair semaphore by name."""
    async with _manager_lock:
        if name not in _semaphores:
            _semaphores[name] = FairSemaphore(permits, name)
        return _semaphores[name]


async def register_connection_pool(
    name: str,
    config: ModelConnectionPoolConfig,
    create_connection: Callable[[], Any],
    validate_connection: Callable[[Any], bool] | None = None,
    close_connection: Callable[[Any], None] | None = None,
) -> AsyncConnectionPool:
    """Register a new connection pool."""
    async with _manager_lock:
        if name in _pools:
            await _pools[name].close()

        pool = AsyncConnectionPool(
            config=config,
            create_connection=create_connection,
            validate_connection=validate_connection,
            close_connection=close_connection,
        )
        _pools[name] = pool
        return pool


async def get_connection_pool(name: str) -> AsyncConnectionPool | None:
    """Get a connection pool by name."""
    return _pools.get(name)


# Convenience functions
@asynccontextmanager
async def with_priority_lock(
    name: str,
    priority: LockPriority = LockPriority.NORMAL,
    timeout: float | None = None,
) -> AsyncGenerator[None, None]:
    """Context manager for priority lock acquisition."""
    lock = await get_priority_lock(name)
    async with lock.acquire(priority=priority, timeout=timeout):
        yield


@asynccontextmanager
async def with_fair_semaphore(
    name: str, permits: int, timeout: float | None = None
) -> AsyncGenerator[None, None]:
    """Context manager for fair semaphore acquisition."""
    semaphore = await get_fair_semaphore(name, permits)
    async with semaphore.acquire(timeout=timeout):
        yield


@asynccontextmanager
async def with_connection_pool(
    name: str, timeout: float | None = None
) -> AsyncGenerator[Any, None]:
    """Context manager for connection pool usage."""
    pool = await get_connection_pool(name)
    if not pool:
        raise ValueError(f"Connection pool '{name}' not found")

    async with pool.acquire(timeout=timeout) as connection:
        yield connection


# === CIRCUIT BREAKER IMPLEMENTATION ===
# These classes provide the circuit breaker pattern expected by tests


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker for external service resilience.

    Implements the circuit breaker pattern to prevent cascading failures
    when external services are unavailable.

    Thread-safe: All state modifications are protected by a threading.Lock
    to ensure safe concurrent access from multiple threads/coroutines.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            success_threshold: Successful calls needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.total_timeouts = 0
        self.last_failure_time: datetime | None = None
        self.state_changed_at: datetime = datetime.now(timezone.utc)
        # Thread-safe lock for synchronous record methods
        # Note: We use threading.Lock (not asyncio.Lock) because record_*
        # methods are synchronous and may be called from sync or async contexts
        self._sync_lock = threading.Lock()

    def record_success(self) -> None:
        """
        Record a successful call.

        Thread-safe: Uses threading.Lock to ensure atomic updates to
        counters and state transitions.
        """
        with self._sync_lock:
            self.total_calls += 1
            self.success_count += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed call.

        Thread-safe: Uses threading.Lock to ensure atomic updates to
        counters and state transitions.
        """
        with self._sync_lock:
            self.total_calls += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()

    def record_timeout(self) -> None:
        """
        Record a timeout failure.

        Thread-safe: Uses threading.Lock to ensure atomic updates.
        Note: We use a separate lock acquisition here rather than calling
        record_failure() to avoid nested lock acquisition.
        """
        with self._sync_lock:
            self.total_timeouts += 1
            self.total_calls += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self.state = CircuitBreakerState.OPEN
        self.state_changed_at = datetime.now(timezone.utc)
        logger.warning(
            "circuit_breaker_opened",
            failure_count=self.failure_count,
            threshold=self.failure_threshold,
        )

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.state_changed_at = datetime.now(timezone.utc)
        self.failure_count = 0
        self.success_count = 0
        logger.info("circuit_breaker_closed")

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.state_changed_at = datetime.now(timezone.utc)
        self.success_count = 0
        logger.info("circuit_breaker_half_open")

    def should_allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Thread-safe: Uses threading.Lock to ensure atomic state checks
        and transitions.

        Returns:
            bool: True if request is allowed, False if blocked
        """
        with self._sync_lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = (
                        datetime.now(timezone.utc) - self.last_failure_time
                    ).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._transition_to_half_open()
                        return True
                return False
            # CLOSED and HALF_OPEN both allow requests through
            return True

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            timeout: Optional timeout in seconds
            **kwargs: Keyword arguments for function

        Returns:
            Result of the function

        Raises:
            CircuitBreakerOpenError: If circuit is open
            TimeoutError: If operation times out
        """
        if not self.should_allow_request():
            raise CircuitBreakerOpenError(f"Circuit breaker is {self.state.value}")

        try:
            if asyncio.iscoroutinefunction(func):
                # Async function - use asyncio.wait_for for timeout
                if timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout
                    )
                else:
                    result = await func(*args, **kwargs)
            else:
                # Synchronous function - use shared ThreadPoolExecutor for efficiency
                # Using functools.partial for proper argument binding
                loop = asyncio.get_running_loop()
                bound_func = functools.partial(func, *args, **kwargs)
                if timeout:
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(_shared_executor, bound_func),
                            timeout=timeout,
                        )
                    except TimeoutError:
                        # Re-raise as TimeoutError for consistent handling
                        raise
                else:
                    # No timeout - run in executor to avoid blocking loop
                    result = await loop.run_in_executor(_shared_executor, bound_func)

            self.record_success()
            return result

        except TimeoutError:
            self.record_timeout()
            raise

        except Exception:
            self.record_failure()
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when an operation is attempted on an open circuit breaker.

    This exception signals that the circuit breaker is in OPEN state and
    refusing requests to allow the downstream service to recover.
    """


# === SIMPLE CONNECTION POOL ===
# A simpler connection pool implementation for basic use cases


class ConnectionPool:
    """
    Simple connection pool for basic connection management.

    This provides a simpler interface than AsyncConnectionPool for
    cases where full configuration and callbacks are not needed.
    """

    def __init__(self, max_size: int = 10, timeout: float = 30.0):
        """
        Initialize connection pool.

        Args:
            max_size: Maximum number of connections in the pool
            timeout: Timeout for connection operations in seconds
        """
        self.max_size = max_size
        self.timeout = timeout
        self._connections: list[Any] = []
        # Maps connection_id -> actual connection object (Any type for generic pool)
        self._active: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._connection_counter = 0
        # Thread-safe lock for synchronous accessor methods
        self._sync_lock = threading.Lock()

    @property
    def active_connections(self) -> int:
        """Return the number of currently active connections.

        Thread-safe: Uses threading.Lock to ensure atomic read of active count.
        """
        with self._sync_lock:
            return len(self._active)

    @property
    def available_connections(self) -> int:
        """Return the number of available connections in the pool.

        Thread-safe: Uses threading.Lock to ensure atomic read of pool size.
        """
        with self._sync_lock:
            return len(self._connections)

    @property
    def total_connections(self) -> int:
        """Return total connections (active + available).

        Thread-safe: Uses threading.Lock to ensure atomic read.
        """
        with self._sync_lock:
            return len(self._active) + len(self._connections)

    def _create_connection(self) -> Any:
        """
        Create a new connection.

        This method can be overridden or mocked in tests.

        WARNING: This is a synchronous method called within an async lock context.
        The base implementation is trivial (creates an object), but if you override
        this method with blocking I/O (e.g., database connection), consider:
        1. Using AsyncConnectionPool instead (recommended for async I/O)
        2. Making the connection creation non-blocking
        3. Using a pre-warmed pool with min_connections

        Returns:
            A new connection object
        """
        self._connection_counter += 1
        return object()

    @asynccontextmanager
    async def acquire(
        self, timeout: float | None = None, max_retries: int = 3
    ) -> AsyncGenerator[Any, None]:
        """
        Acquire a connection from the pool.

        Uses iterative retry (not recursive) to prevent stack overflow
        when connection creation fails.

        Args:
            timeout: Optional timeout override
            max_retries: Maximum number of retry attempts for connection creation

        Yields:
            Connection object
        """
        connection_id = str(uuid4())
        connection = None
        effective_timeout = timeout if timeout is not None else self.timeout
        start_time = datetime.now(timezone.utc)

        try:
            # Iterative retry loop to prevent recursion/stack overflow
            for attempt in range(max_retries):
                try:
                    # Check if timeout has elapsed before attempting
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    if elapsed >= effective_timeout:
                        raise TimeoutError(
                            f"Connection pool timeout after {elapsed:.2f}s"
                        )

                    # Acquire async lock with timeout support
                    remaining_timeout = effective_timeout - elapsed
                    try:
                        await asyncio.wait_for(
                            self._lock.acquire(), timeout=remaining_timeout
                        )
                    except TimeoutError as lock_timeout:
                        msg = f"Pool lock timeout after {effective_timeout:.2f}s"
                        raise TimeoutError(msg) from lock_timeout

                    try:
                        # Thread-safe state access within async lock context
                        with self._sync_lock:
                            # Try to get from pool first
                            if self._connections:
                                connection = self._connections.pop()
                            elif len(self._active) < self.max_size:
                                # Create new connection (may fail and trigger retry)
                                connection = self._create_connection()
                            else:
                                # Pool exhausted - would need to wait
                                msg = f"Pool exhausted (max={self.max_size})"
                                raise TimeoutError(msg)

                            self._active[connection_id] = connection
                    finally:
                        self._lock.release()

                    # Successfully acquired connection, break out of retry loop
                    break

                except (ConnectionError, OSError) as e:
                    # Connection creation failed, retry if attempts remain
                    _ = e  # Capture error for debugging if needed
                    if attempt < max_retries - 1:
                        logger.debug(
                            "connection_creation_retry",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            error=str(e),
                        )
                        # Small delay before retry with exponential backoff
                        await asyncio.sleep(0.01 * (2**attempt))
                        continue
                    else:
                        # Max retries exhausted, re-raise the last error
                        raise

            yield connection

        finally:
            if connection is not None:
                async with self._lock:
                    # Thread-safe state modification
                    with self._sync_lock:
                        # Remove from active
                        self._active.pop(connection_id, None)
                        # Validate pool size before returning connection
                        # Total pool size = available connections + active connections
                        total_pool_size = len(self._connections) + len(self._active)
                        if (
                            total_pool_size < self.max_size
                            and len(self._connections) < self.max_size
                        ):
                            self._connections.append(connection)
                            logger.debug(
                                "connection_returned_to_pool",
                                connection_id=connection_id,
                                pool_size=len(self._connections),
                                active_connections=len(self._active),
                            )
                        else:
                            # Pool is at capacity, discard excess connection
                            logger.debug(
                                "connection_discarded_pool_full",
                                connection_id=connection_id,
                                pool_size=len(self._connections),
                                max_size=self.max_size,
                            )


# === DECORATOR UTILITIES ===


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = (
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
                    if attempt < max_attempts - 1:
                        error_to_sanitize = (
                            exc if isinstance(exc, Exception) else Exception(str(exc))
                        )
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=current_delay,
                            error=_sanitize_error(error_to_sanitize),
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_multiplier

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit without exception") from None

        return wrapper  # type: ignore[return-value]

    return decorator


def with_timeout(timeout: float) -> Callable[[F], F]:
    """
    Decorator for adding timeout to async and sync functions.

    For async functions, uses asyncio.wait_for for timeout enforcement.
    For sync functions, uses ThreadPoolExecutor with timeout enforcement.

    Args:
        timeout: Timeout in seconds

    Returns:
        Decorated function with timeout

    Raises:
        TimeoutError: If the function execution exceeds the timeout
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            # Async function - use asyncio.wait_for
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

            return async_wrapper  # type: ignore[return-value]
        else:
            # Sync function - use ThreadPoolExecutor for timeout enforcement
            @functools.wraps(func)
            async def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                loop = asyncio.get_running_loop()
                # Use module-level executor for efficiency
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            _shared_executor, functools.partial(func, *args, **kwargs)
                        ),
                        timeout=timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        "sync_function_timeout", function=func.__name__, timeout=timeout
                    )
                    raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def with_circuit_breaker(
    circuit_breaker_or_threshold: CircuitBreaker | int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 1,
) -> Callable[[F], F]:
    """
    Decorator for adding circuit breaker pattern to async functions.

    Can be used in two ways:
    1. With a CircuitBreaker instance: @with_circuit_breaker(my_breaker)
    2. With parameters: @with_circuit_breaker(failure_threshold=5)

    Args:
        circuit_breaker_or_threshold: Either a CircuitBreaker instance or
            the failure threshold (int) for creating a new CircuitBreaker
        recovery_timeout: Seconds to wait before trying half-open (only used
            when creating a new CircuitBreaker)
        success_threshold: Successful calls needed to close circuit (only used
            when creating a new CircuitBreaker)

    Returns:
        Decorated function with circuit breaker protection
    """
    # Check if a CircuitBreaker instance was passed directly
    if isinstance(circuit_breaker_or_threshold, CircuitBreaker):
        breaker = circuit_breaker_or_threshold
    else:
        # Create a new CircuitBreaker with the provided parameters
        breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_or_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.call(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
