"""
Resource management utilities for OmniMemory ONEX architecture.

This module provides:
- Async context managers for proper resource cleanup
- Circuit breaker patterns for external service resilience
- Connection pool management and exhaustion handling
- Timeout configurations for all async operations

NOTE ON Any TYPES:
This module intentionally uses 'Any' types for generic resource management:
- ResourcePool manages arbitrary resource types (connections, handles, etc.)
- resource_id/resource fields use Any for generic resource references
- config: dict[str, Any] - Resource pool configs contain various value types
- Pool health/stats methods return dict[str, Any] for flexible monitoring data

This design enables a single resource pool implementation to work with any
resource type without complex generic type hierarchies.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

import structlog

from ..models.utils.model_circuit_breaker_config import ModelCircuitBreakerConfig
from ..models.utils.model_circuit_breaker_stats_response import (
    ModelCircuitBreakerStatsResponse,
)

# Re-export ModelCircuitBreakerConfig for public API
__all__ = ["ModelCircuitBreakerConfig"]

from .error_sanitizer import SanitizationLevel
from .error_sanitizer import sanitize_error as _base_sanitize_error

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
        error, context="resource_manager", level=SanitizationLevel.STANDARD
    )


class CircuitState(Enum):
    """Circuit breaker states following resilience patterns."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerStats:
    """Statistics tracking for circuit breaker behavior."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    state_changed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_calls: int = 0
    total_timeouts: int = 0


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

    def __init__(self, name: str, config: ModelCircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or ModelCircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function call through the circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(self.name, self.state)

        try:
            # Apply timeout to the operation
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout
            )
            await self._on_success()
            return result

        except TimeoutError as e:
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

        time_since_failure = datetime.now(timezone.utc) - self.stats.last_failure_time
        return time_since_failure.total_seconds() >= effective_timeout

    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state.

        Note: This is a synchronous method as it only modifies internal state.
        No I/O operations are performed.
        """
        self.state = CircuitState.HALF_OPEN
        self.stats.state_changed_at = datetime.now(timezone.utc)
        self.stats.success_count = 0

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            new_state="half_open",
            reason="recovery_timeout_reached",
        )

    async def _on_success(self) -> None:
        """Handle successful operation result."""
        async with self._lock:
            self.stats.total_calls += 1

            if self.state == CircuitState.HALF_OPEN:
                self.stats.success_count += 1
                if self.stats.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.stats.failure_count = 0  # Reset failure count on success

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed operation result."""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.failure_count += 1
            self.stats.last_failure_time = datetime.now(timezone.utc)

            if (
                self.state == CircuitState.CLOSED
                and self.stats.failure_count >= self.config.failure_threshold
            ) or self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()

    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state.

        Note: This is a synchronous method as it only modifies internal state.
        No I/O operations are performed.
        """
        self.state = CircuitState.CLOSED
        self.stats.state_changed_at = datetime.now(timezone.utc)
        self.stats.failure_count = 0

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            new_state="closed",
            reason="success_threshold_reached",
        )

    def _transition_to_open(self) -> None:
        """Transition circuit breaker to open state.

        Note: This is a synchronous method as it only modifies internal state.
        No I/O operations are performed.
        """
        self.state = CircuitState.OPEN
        self.stats.state_changed_at = datetime.now(timezone.utc)

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

    def __init__(self) -> None:
        self._circuit_breakers: dict[str, AsyncCircuitBreaker] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_circuit_breaker(
        self, name: str, config: ModelCircuitBreakerConfig | None = None
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
        release_func: Callable[[Any], None] | None = None,
        circuit_breaker_config: ModelCircuitBreakerConfig | None = None,
        semaphore_limit: int | None = None,
        *args: Any,
        **kwargs: Any,
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

    def get_circuit_breaker_stats(self) -> dict[str, ModelCircuitBreakerStatsResponse]:
        """Get typed statistics for all circuit breakers."""
        stats = {}
        for name, cb in self._circuit_breakers.items():
            stats[name] = ModelCircuitBreakerStatsResponse(
                state=cb.state.value,
                failure_count=cb.stats.failure_count,
                success_count=cb.stats.success_count,
                total_calls=cb.stats.total_calls,
                total_timeouts=cb.stats.total_timeouts,
                last_failure_time=(
                    cb.stats.last_failure_time.isoformat()
                    if cb.stats.last_failure_time
                    else None
                ),
                state_changed_at=cb.stats.state_changed_at.isoformat(),
            )
        return stats


# Global resource manager instance
resource_manager = AsyncResourceManager()


# Convenience functions for common patterns
async def with_circuit_breaker(
    service_name: str,
    func: Callable[..., Any],
    config: ModelCircuitBreakerConfig | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a function with circuit breaker protection."""
    circuit_breaker = resource_manager.get_circuit_breaker(service_name, config)
    return await circuit_breaker.call(func, *args, **kwargs)


@contextlib.asynccontextmanager
async def with_semaphore(name: str, limit: int) -> AsyncGenerator[None, None]:
    """Context manager for semaphore-based rate limiting."""
    semaphore = resource_manager.get_semaphore(name, limit)
    async with semaphore:
        yield


@contextlib.asynccontextmanager
async def with_timeout(timeout: float) -> AsyncGenerator[None, None]:
    """Context manager for timeout operations."""
    try:
        async with asyncio.timeout(timeout):
            yield
    except TimeoutError:
        logger.warning("operation_timeout", timeout=timeout)
        raise


# === TEST-COMPATIBLE INTERFACES ===
# These classes provide the interface expected by test_resource_manager.py


class ResourceType(Enum):
    """Types of resources that can be managed."""

    DATABASE = "database"
    MEMORY = "memory"
    CACHE = "cache"
    NETWORK = "network"
    STORAGE = "storage"
    FILE = "file"


class ResourceStatus(Enum):
    """Status of a resource handle."""

    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED = "failed"


class ResourceAllocationError(Exception):
    """Exception raised when resource allocation fails."""


class ResourceTimeoutError(Exception):
    """Exception raised when resource acquisition times out."""


@dataclass
class ResourceHandle:
    """
    Handle to an acquired resource.

    Provides resource lifecycle management and context tracking.
    """

    resource_id: Any
    resource: Any
    resource_type: ResourceType
    status: ResourceStatus = ResourceStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: float | None = None
    _context: dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """
        Check if the resource is healthy.

        Returns:
            bool: True if resource is healthy
        """
        if self.status != ResourceStatus.ACTIVE:
            return False

        if hasattr(self.resource, "is_healthy"):
            result: bool = self.resource.is_healthy()
            return result

        return True

    def is_expired(self) -> bool:
        """
        Check if the resource has expired.

        Returns:
            bool: True if resource is expired
        """
        if self.ttl is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed >= self.ttl

    def set_context(self, key: str, value: Any) -> None:
        """
        Set context data on the handle.

        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value

    def get_context(self, key: str) -> Any | None:
        """
        Get context data from the handle.

        Args:
            key: Context key

        Returns:
            Context value or None if not found
        """
        return self._context.get(key)

    def clear_context(self) -> None:
        """Clear all context data."""
        self._context.clear()


class ResourcePool:
    """
    Pool of resources of a specific type.

    Manages resource creation, pooling, and lifecycle.
    """

    def __init__(self, resource_type: ResourceType, config: dict[str, Any]):
        """
        Initialize resource pool.

        Args:
            resource_type: Type of resources in this pool
            config: Pool configuration
        """
        self.resource_type = resource_type
        self.min_size = config.get("min_size", 1)
        self.max_size = config.get("max_size", 10)
        self._factory = config.get("factory")
        self._timeout = config.get("timeout", 30.0)
        self._ttl = config.get("resource_ttl")
        self._scale_threshold = config.get("scale_threshold", 0.8)
        self._scale_increment = config.get("scale_increment", 1)
        self._health_check_interval = config.get("health_check_interval", 60.0)

        # Store (resource, added_time) tuples to enable TTL tracking
        self.available_resources: list[tuple[Any, float]] = []
        self.active_resources: dict[Any, ResourceHandle] = {}
        self.current_size = 0

        self._lock = asyncio.Lock()
        self._available_event = asyncio.Event()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the pool with minimum resources."""
        async with self._lock:
            for _ in range(self.min_size):
                resource = self._create_resource()
                if resource:
                    self.available_resources.append((resource, time.time()))
                    self.current_size += 1

            self._initialized = True
            if self.available_resources:
                self._available_event.set()

    def _create_resource(self) -> Any | None:
        """Create a new resource."""
        if self._factory:
            try:
                return self._factory()
            except Exception as e:
                logger.error(
                    "resource_creation_failed",
                    resource_type=self.resource_type.value,
                    error=_sanitize_error(e),
                )
                return None
        return None

    async def acquire(self, timeout: float | None = None) -> ResourceHandle:
        """
        Acquire a resource from the pool.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            ResourceHandle: Handle to the acquired resource

        Raises:
            ResourceTimeoutError: If acquisition times out
            ResourceAllocationError: If resource creation fails
        """
        # Resolve effective timeout upfront for type safety in arithmetic operations
        effective_timeout: float = timeout if timeout is not None else self._timeout
        start_time = time.time()

        while True:
            async with self._lock:
                # Try to get an available resource
                if self.available_resources:
                    resource, _added_time = self.available_resources.pop()
                    resource_id = id(resource)
                    handle = ResourceHandle(
                        resource_id=resource_id,
                        resource=resource,
                        resource_type=self.resource_type,
                        ttl=self._ttl,
                    )
                    self.active_resources[resource_id] = handle
                    return handle

                # Try to create a new resource if under max
                if self.current_size < self.max_size:
                    resource = self._create_resource()
                    if resource:
                        self.current_size += 1
                        resource_id = id(resource)
                        handle = ResourceHandle(
                            resource_id=resource_id,
                            resource=resource,
                            resource_type=self.resource_type,
                            ttl=self._ttl,
                        )
                        self.active_resources[resource_id] = handle
                        return handle
                    else:
                        raise ResourceAllocationError(
                            f"Failed to create {self.resource_type.value} resource"
                        )

                # Clear event inside lock to prevent missed wakeups
                # This must happen before releasing the lock so we don't miss
                # a signal from release() that happens between lock release and wait
                self._available_event.clear()

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= effective_timeout:
                raise ResourceTimeoutError(
                    f"Timeout acquiring {self.resource_type.value} resource"
                )
            try:
                await asyncio.wait_for(
                    self._available_event.wait(), timeout=effective_timeout - elapsed
                )
            except TimeoutError as e:
                raise ResourceTimeoutError(
                    f"Timeout acquiring {self.resource_type.value} resource"
                ) from e

    async def release(self, handle: ResourceHandle) -> None:
        """
        Release a resource back to the pool.

        Args:
            handle: Handle to the resource to release
        """
        async with self._lock:
            if handle.resource_id in self.active_resources:
                del self.active_resources[handle.resource_id]

            # Check health and expiration BEFORE changing status
            # (is_healthy() checks status == ACTIVE, so must check first)
            should_return_to_pool = handle.is_healthy() and not handle.is_expired()

            handle.status = ResourceStatus.RELEASED

            # Return to pool if was healthy and not expired
            if should_return_to_pool:
                self.available_resources.append((handle.resource, time.time()))
                self._available_event.set()


class ResourceManager:
    """
    Resource manager with test-compatible interface.

    Provides:
    - Resource pool management
    - Resource acquisition and release
    - Health monitoring
    - Metrics collection
    """

    def __init__(self) -> None:
        """Initialize resource manager."""
        self.resource_pools: dict[ResourceType, ResourcePool] = {}
        self._metrics: dict[str, int] = {
            "total_operations": 0,
            "total_acquisitions": 0,
            "total_releases": 0,
            "total_errors": 0,
        }

    async def register_pool(
        self, resource_type: ResourceType, config: dict[str, Any]
    ) -> None:
        """
        Register a resource pool.

        Args:
            resource_type: Type of resource for this pool
            config: Pool configuration
        """
        pool = ResourcePool(resource_type, config)
        await pool.initialize()  # Pre-create min_size resources
        self.resource_pools[resource_type] = pool
        logger.info(
            "resource_pool_registered",
            resource_type=resource_type.value,
            min_size=config.get("min_size", 1),
            max_size=config.get("max_size", 10),
        )

    async def acquire(
        self, resource_type: ResourceType, timeout: float | None = None
    ) -> ResourceHandle:
        """
        Acquire a resource of the specified type.

        Args:
            resource_type: Type of resource to acquire
            timeout: Optional timeout in seconds

        Returns:
            ResourceHandle: Handle to the acquired resource

        Raises:
            ValueError: If pool not registered
            ResourceTimeoutError: If acquisition times out
        """
        if resource_type not in self.resource_pools:
            raise ValueError(f"No pool registered for {resource_type.value}")

        pool = self.resource_pools[resource_type]

        # Ensure pool is initialized
        if not pool._initialized:  # noqa: SLF001  # Intentional - manager needs pool internals
            await pool.initialize()

        self._metrics["total_operations"] += 1
        self._metrics["total_acquisitions"] += 1

        return await pool.acquire(timeout=timeout)

    async def release(self, handle: ResourceHandle) -> None:
        """
        Release a resource handle.

        Args:
            handle: Handle to release
        """
        if handle.resource_type not in self.resource_pools:
            return

        pool = self.resource_pools[handle.resource_type]
        self._metrics["total_operations"] += 1
        self._metrics["total_releases"] += 1

        await pool.release(handle)

    @contextlib.asynccontextmanager
    async def acquire_context(
        self, resource_type: ResourceType, timeout: float | None = None
    ) -> AsyncGenerator[ResourceHandle, None]:
        """
        Context manager for resource acquisition.

        Args:
            resource_type: Type of resource to acquire
            timeout: Optional timeout in seconds

        Yields:
            ResourceHandle: Handle to the acquired resource
        """
        handle = await self.acquire(resource_type, timeout=timeout)
        try:
            yield handle
        finally:
            await self.release(handle)

    def get_pool_health(self, resource_type: ResourceType) -> dict[str, Any]:
        """
        Get health status of a resource pool.

        Args:
            resource_type: Type of resource pool

        Returns:
            Dict with health information
        """
        if resource_type not in self.resource_pools:
            return {"error": f"Pool not found: {resource_type.value}"}

        pool = self.resource_pools[resource_type]
        return {
            "resource_type": resource_type.value,
            "total_created": pool.current_size,
            "active_resources": len(pool.active_resources),
            "available_resources": len(pool.available_resources),
            "health_check_failures": 0,
        }

    def get_pool_stats(self, resource_type: ResourceType) -> dict[str, Any]:
        """
        Get statistics for a resource pool.

        Args:
            resource_type: Type of resource pool

        Returns:
            Dict with pool statistics
        """
        if resource_type not in self.resource_pools:
            return {"error": f"Pool not found: {resource_type.value}"}

        pool = self.resource_pools[resource_type]
        return {
            "resource_type": resource_type.value,
            "current_size": pool.current_size,
            "min_size": pool.min_size,
            "max_size": pool.max_size,
            "active_count": len(pool.active_resources),
            "available_count": len(pool.available_resources),
        }

    def get_metrics(self) -> dict[str, Any]:
        """
        Get resource manager metrics.

        Returns:
            Dict with metrics
        """
        return {
            "total_pools": len(self.resource_pools),
            "total_resources": sum(
                p.current_size for p in self.resource_pools.values()
            ),
            "resource_types": [rt.value for rt in self.resource_pools.keys()],
            "total_operations": self._metrics["total_operations"],
            "total_acquisitions": self._metrics["total_acquisitions"],
            "total_releases": self._metrics["total_releases"],
        }

    async def _close_resource(self, resource: Any) -> None:
        """
        Close a resource if it has a close method.

        Handles both sync and async close methods. Errors during close
        are logged but not raised to ensure cleanup continues.

        Args:
            resource: The resource to close
        """
        if resource is None:
            return

        if hasattr(resource, "close"):
            try:
                close_method = resource.close
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
                logger.debug("resource_closed", resource_type=type(resource).__name__)
            except Exception as e:
                logger.warning(
                    "resource_close_error",
                    resource_type=type(resource).__name__,
                    error=_sanitize_error(e),
                )

    async def _check_resource_expiration(self, resource_type: ResourceType) -> None:
        """
        Check and handle expired resources in a pool.

        Filters out resources that have exceeded the pool's TTL configuration.
        Resources without TTL configured are never expired. Expired resources
        are explicitly closed if they have a close() method.

        Args:
            resource_type: Type of resource pool to check
        """
        if resource_type not in self.resource_pools:
            return

        pool = self.resource_pools[resource_type]
        expired_resources: list[Any] = []

        async with pool._lock:  # noqa: SLF001  # Intentional - manager needs pool internals
            # If no TTL configured, all resources are valid
            if pool._ttl is None:  # noqa: SLF001
                return

            current_time = time.time()
            valid_resources: list[tuple[Any, float]] = []

            for resource, added_time in pool.available_resources:
                elapsed = current_time - added_time
                if elapsed < pool._ttl:  # noqa: SLF001
                    valid_resources.append((resource, added_time))
                else:
                    expired_resources.append(resource)
                    pool.current_size -= 1

            if expired_resources:
                logger.debug(
                    "expired_resources_cleaned",
                    resource_type=resource_type.value,
                    expired_count=len(expired_resources),
                    remaining_count=len(valid_resources),
                )

            pool.available_resources = valid_resources

        # Close expired resources outside the lock to avoid blocking
        for resource in expired_resources:
            await self._close_resource(resource)

    async def shutdown(self) -> None:
        """Shut down the resource manager and close all resources.

        Explicitly closes all active and available resources that have
        a close() method before clearing the pools.
        """
        for resource_type, pool in self.resource_pools.items():
            resources_to_close: list[Any] = []

            async with pool._lock:  # noqa: SLF001  # Intentional - shutdown needs pool internals
                # Collect active resources for closing
                for handle in list(pool.active_resources.values()):
                    handle.status = ResourceStatus.RELEASED
                    resources_to_close.append(handle.resource)

                # Collect available resources for closing
                for resource, _added_time in pool.available_resources:
                    resources_to_close.append(resource)

                pool.active_resources.clear()
                pool.available_resources.clear()
                pool.current_size = 0

            # Close resources outside the lock to avoid blocking
            for resource in resources_to_close:
                await self._close_resource(resource)

            logger.debug(
                "resource_pool_shutdown",
                resource_type=resource_type.value,
                resources_closed=len(resources_to_close),
            )

        logger.info("resource_manager_shutdown_completed")
