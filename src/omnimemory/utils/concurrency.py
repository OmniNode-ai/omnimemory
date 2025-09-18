"""
Concurrency utilities for OmniMemory ONEX architecture.

This module provides:
- Advanced semaphore patterns for rate-limited operations
- Proper locking mechanisms for shared resources
- Connection pool management and exhaustion handling
- Fair scheduling and priority-based access control
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from ..models.foundation.model_connection_metadata import ConnectionMetadata
from .observability import OperationType, correlation_context, trace_operation

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


@dataclass
class LockRequest:
    """Request for lock acquisition with priority and metadata."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    priority: LockPriority = LockPriority.NORMAL
    requested_at: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    timeout: Optional[float] = None
    metadata: ConnectionMetadata = field(default_factory=ConnectionMetadata)


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
    created_at: datetime = field(default_factory=datetime.now)


class ConnectionPoolConfig(BaseModel):
    """Configuration for connection pools."""

    name: str = Field(description="Pool name")
    min_connections: int = Field(default=1, ge=0, description="Minimum connections")
    max_connections: int = Field(
        default=50,
        ge=1,
        description="Maximum connections (increased for production load)",
    )
    connection_timeout: float = Field(
        default=30.0, gt=0, description="Connection timeout"
    )
    idle_timeout: float = Field(
        default=300.0, gt=0, description="Idle connection timeout"
    )
    health_check_interval: float = Field(
        default=60.0, gt=0, description="Health check interval"
    )
    retry_attempts: int = Field(
        default=3, ge=0, description="Retry attempts for failed connections"
    )


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
    last_exhaustion: Optional[datetime] = None


class PriorityLock:
    """
    Async lock with priority-based fair scheduling.

    Provides priority-based access to shared resources with fairness
    guarantees and timeout support.
    """

    def __init__(self, name: str):
        self.name = name
        self._lock = asyncio.Lock()
        self._queue: List[LockRequest] = []
        self._current_holder: Optional[LockRequest] = None
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
        timeout: Optional[float] = None,
        correlation_id: Optional[str] = None,
        **metadata,
    ) -> AsyncGenerator[None, None]:
        """
        Acquire the lock with priority and timeout support.

        Args:
            priority: Priority level for lock acquisition
            timeout: Maximum time to wait for lock
            correlation_id: Correlation ID for tracing
            **metadata: Additional metadata for the lock request
        """
        request = LockRequest(
            priority=priority,
            timeout=timeout,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        acquired_at: Optional[datetime] = None

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

                    acquired_at = datetime.now()
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

                except asyncio.TimeoutError:
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

    async def _enqueue_request(self, request: LockRequest):
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

    async def _wait_for_turn(self, request: LockRequest):
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

            # Apply timeout if specified
            if request.timeout:
                elapsed = (datetime.now() - request.requested_at).total_seconds()
                if elapsed >= request.timeout:
                    raise asyncio.TimeoutError(
                        f"Lock acquisition timeout after {request.timeout}s"
                    )

            # Wait a bit before checking again
            await asyncio.sleep(0.001)  # 1ms

    async def _cleanup_request(
        self, request: LockRequest, acquired_at: Optional[datetime]
    ):
        """Clean up after lock release."""
        async with self._lock:
            # Calculate hold time if lock was acquired
            if acquired_at:
                hold_time = (datetime.now() - acquired_at).total_seconds()
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
        self._waiting_queue: deque = deque()
        self._active_holders: Dict[str, datetime] = {}
        self._stats = SemaphoreStats(
            total_permits=value, available_permits=value, waiting_count=0
        )
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(
        self, timeout: Optional[float] = None, correlation_id: Optional[str] = None
    ) -> AsyncGenerator[None, None]:
        """
        Acquire semaphore permit with timeout and tracking.

        Args:
            timeout: Maximum time to wait for permit
            correlation_id: Correlation ID for tracing
        """
        holder_id = str(uuid4())
        acquired_at: Optional[datetime] = None

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

                    acquired_at = datetime.now()

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

                except asyncio.TimeoutError:
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
                        hold_time = (datetime.now() - acquired_at).total_seconds()

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
                                    (1 - alpha) * self._stats.average_hold_time
                                    + alpha * hold_time
                                )
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
        """Get current semaphore statistics."""
        return self._stats


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
        config: ConnectionPoolConfig,
        create_connection: Callable[[], Any],
        validate_connection: Optional[Callable[[Any], bool]] = None,
        close_connection: Optional[Callable[[Any], None]] = None,
    ):
        self.config = config
        self._create_connection = create_connection
        self._validate_connection = validate_connection or (lambda conn: True)
        self._close_connection = close_connection or (lambda conn: None)

        self._available: asyncio.Queue = asyncio.Queue(maxsize=config.max_connections)
        self._active: Dict[str, ConnectionMetadata] = {}
        self._metrics = PoolMetrics()
        self._status = PoolStatus.HEALTHY
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None

        # Start health check task
        self._start_health_check()

    @asynccontextmanager
    async def acquire(
        self,
        timeout: Optional[float] = None,
        correlation_id: Optional[str] = None,
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
        acquired_at = datetime.now()

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
                    # Use iterative retry loop instead of recursion to prevent stack overflow
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
                                # No available connections, check if we can create new one
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
                                        # Pool is at capacity, wait for available connection
                                        self._metrics.pool_exhaustions += 1
                                        self._metrics.last_exhaustion = datetime.now()
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
                                    raise RuntimeError(
                                        f"Failed to acquire valid connection after {max_retries} attempts"
                                    )

                                # Increment retry counter and continue the loop
                                current_retry += 1
                                continue

                            # Connection is valid, break out of retry loop
                            break

                        except Exception:
                            # Handle unexpected exceptions during connection acquisition
                            if connection:
                                await self._destroy_connection(connection)
                            raise

                    # Track active connection
                    async with self._lock:
                        self._active[connection_id] = connection

                    # Update metrics
                    wait_time = (datetime.now() - acquired_at).total_seconds()
                    if wait_time > 0:
                        current_avg = self._metrics.average_wait_time
                        acquisitions = len(self._active)
                        self._metrics.average_wait_time = (
                            (current_avg * (acquisitions - 1)) + wait_time
                        ) / acquisitions

                    yield connection

                except asyncio.TimeoutError:
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
                    # Return connection to pool with shielded cleanup to prevent resource leaks
                    if connection:
                        try:
                            # Shield the cleanup operation to ensure it completes even if cancelled
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

    async def _return_connection(self, connection_id: str, connection: Any):
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

    async def _destroy_connection(self, connection: Any):
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

    def _start_health_check(self):
        """Start the health check background task."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self):
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

    async def _perform_health_check(self):
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
        """Get current pool metrics."""
        self._metrics.active_connections = len(self._active)
        self._metrics.idle_connections = self._available.qsize()
        return self._metrics

    def get_status(self) -> PoolStatus:
        """Get current pool status."""
        return self._status

    async def close(self):
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
_locks: Dict[str, PriorityLock] = {}
_semaphores: Dict[str, FairSemaphore] = {}
_pools: Dict[str, AsyncConnectionPool] = {}
_manager_lock = asyncio.Lock()


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
    config: ConnectionPoolConfig,
    create_connection: Callable[[], Any],
    validate_connection: Optional[Callable[[Any], bool]] = None,
    close_connection: Optional[Callable[[Any], None]] = None,
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


async def get_connection_pool(name: str) -> Optional[AsyncConnectionPool]:
    """Get a connection pool by name."""
    return _pools.get(name)


# Convenience functions
@asynccontextmanager
async def with_priority_lock(
    name: str,
    priority: LockPriority = LockPriority.NORMAL,
    timeout: Optional[float] = None,
):
    """Context manager for priority lock acquisition."""
    lock = await get_priority_lock(name)
    async with lock.acquire(priority=priority, timeout=timeout):
        yield


@asynccontextmanager
async def with_fair_semaphore(name: str, permits: int, timeout: Optional[float] = None):
    """Context manager for fair semaphore acquisition."""
    semaphore = await get_fair_semaphore(name, permits)
    async with semaphore.acquire(timeout=timeout):
        yield


@asynccontextmanager
async def with_connection_pool(name: str, timeout: Optional[float] = None):
    """Context manager for connection pool usage."""
    pool = await get_connection_pool(name)
    if not pool:
        raise ValueError(f"Connection pool '{name}' not found")

    async with pool.acquire(timeout=timeout) as connection:
        yield connection
