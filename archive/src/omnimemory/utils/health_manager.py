"""
Comprehensive health check manager for OmniMemory ONEX architecture.

This module provides:
- Aggregated health checks from all dependencies
- Async gathering with failure isolation
- Circuit breaker integration for health checks
- Performance monitoring and alerting
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import psutil
import structlog
from pydantic import BaseModel, Field

from ..models.foundation.model_health_metadata import (
    AggregateHealthMetadata,
    ConfigurationChangeMetadata,
    HealthCheckMetadata,
)
from ..models.foundation.model_health_response import (
    ModelCircuitBreakerStats,
    ModelCircuitBreakerStatsCollection,
    ModelRateLimitedHealthCheckResponse,
)
from .error_sanitizer import SanitizationLevel, sanitize_error

# === RATE LIMITING ===


class RateLimiter:
    """Simple rate limiter for API endpoints."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for the given identifier.

        Args:
            identifier: Client identifier (IP, user ID, etc.)

        Returns:
            bool: True if request is allowed, False if rate limited
        """
        async with self._lock:
            current_time = time.time()

            # Initialize or get existing requests list
            if identifier not in self._requests:
                self._requests[identifier] = []

            requests = self._requests[identifier]

            # Remove old requests outside the window
            cutoff_time = current_time - self.window_seconds
            self._requests[identifier] = [
                req_time for req_time in requests if req_time > cutoff_time
            ]

            # Check if we can accept this request
            if len(self._requests[identifier]) >= self.max_requests:
                return False

            # Add current request
            self._requests[identifier].append(current_time)
            return True


def _sanitize_error(error: Exception) -> str:
    """
    Sanitize error messages to prevent information disclosure in logs.

    Uses the enhanced centralized error sanitizer for improved security.
    """
    return sanitize_error(
        error, context="health_check", level=SanitizationLevel.STANDARD
    )


def _get_package_version() -> str:
    """Get package version from metadata or fallback to default."""
    try:
        # Try to get version from package metadata
        from importlib.metadata import version

        return version("omnimemory")
    except ImportError:
        # Fallback for older Python versions
        try:
            import pkg_resources

            return pkg_resources.get_distribution("omnimemory").version
        except Exception:
            return "0.1.0"  # Fallback version
    except Exception:
        return "0.1.0"  # Fallback version


def _get_environment() -> str:
    """Detect current environment from environment variables."""
    # Check common environment variables
    env = os.getenv(
        "ENVIRONMENT", os.getenv("ENV", os.getenv("NODE_ENV", "development"))
    )

    # Normalize environment names
    if env.lower() in ("prod", "production"):
        return "production"
    elif env.lower() in ("stage", "staging"):
        return "staging"
    elif env.lower() in ("test", "testing"):
        return "testing"
    else:
        return "development"


from ..models.foundation.model_health_response import (
    ModelDependencyStatus,
    ModelHealthResponse,
    ModelResourceMetrics,
)
from .observability import OperationType, correlation_context, trace_operation
from .resource_manager import AsyncCircuitBreaker, CircuitBreakerConfig

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Enhanced health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DependencyType(Enum):
    """Types of system dependencies."""

    DATABASE = "database"
    CACHE = "cache"
    VECTOR_DB = "vector_db"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"


class HealthCheckConfig(BaseModel):
    """Configuration for individual health checks."""

    name: str = Field(description="Dependency name")
    dependency_type: DependencyType = Field(description="Type of dependency")
    timeout: float = Field(default=5.0, description="Health check timeout in seconds")
    critical: bool = Field(
        default=True, description="Whether failure affects overall health"
    )
    circuit_breaker_config: Optional[CircuitBreakerConfig] = Field(default=None)
    metadata: HealthCheckMetadata = Field(default_factory=HealthCheckMetadata)


class HealthCheckResult(BaseModel):
    """Result of an individual health check."""

    config: HealthCheckConfig = Field(description="Health check configuration")
    status: HealthStatus = Field(description="Health status")
    latency_ms: float = Field(description="Check latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None)
    metadata: HealthCheckMetadata = Field(default_factory=HealthCheckMetadata)

    def to_dependency_status(self) -> ModelDependencyStatus:
        """Convert to ModelDependencyStatus for API response."""
        return ModelDependencyStatus(
            name=self.config.name,
            status=self.status.value,
            latency_ms=self.latency_ms,
            last_check=self.timestamp,
            error_message=self.error_message,
        )


class HealthCheckManager:
    """
    Comprehensive health check manager for OmniMemory.

    Provides:
    - Aggregated health checks from all dependencies
    - Circuit breaker integration for resilient health checking
    - Resource monitoring and performance tracking
    - Failure isolation to prevent cascade failures
    """

    def __init__(self):
        self._health_checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self._configs: Dict[str, HealthCheckConfig] = {}
        self._circuit_breakers: Dict[str, AsyncCircuitBreaker] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._results_lock = asyncio.Lock()  # Prevent race conditions on metric updates
        self._rate_limiter = RateLimiter(
            max_requests=30, window_seconds=60
        )  # Rate limit health checks
        self._system_start_time = time.time()

    def register_health_check(
        self,
        config: HealthCheckConfig,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
    ):
        """
        Register a health check function with configuration.

        Args:
            config: Health check configuration
            check_func: Async function that performs the health check
        """
        self._configs[config.name] = config
        self._health_checks[config.name] = check_func

        # Create circuit breaker if configured
        if config.circuit_breaker_config:
            self._circuit_breakers[config.name] = AsyncCircuitBreaker(
                config.name, config.circuit_breaker_config
            )

        logger.info(
            "health_check_registered",
            dependency_name=config.name,
            dependency_type=config.dependency_type.value,
            critical=config.critical,
        )

    async def check_single_dependency(self, name: str) -> HealthCheckResult:
        """
        Perform health check for a single dependency.

        Args:
            name: Name of the dependency to check

        Returns:
            HealthCheckResult: Result of the health check
        """
        if name not in self._health_checks:
            return HealthCheckResult(
                config=HealthCheckConfig(
                    name=name, dependency_type=DependencyType.EXTERNAL_API
                ),
                status=HealthStatus.UNKNOWN,
                latency_ms=0.0,
                error_message="Health check not registered",
            )

        config = self._configs[name]
        check_func = self._health_checks[name]

        async with correlation_context(operation=f"health_check_{name}"):
            async with trace_operation(
                f"health_check_{name}",
                OperationType.HEALTH_CHECK,
                dependency=name,
                dependency_type=config.dependency_type.value,
            ):
                start_time = time.time()

                try:
                    # Use circuit breaker if configured
                    if name in self._circuit_breakers:
                        circuit_breaker = self._circuit_breakers[name]
                        result = await circuit_breaker.call(check_func)
                    else:
                        # Apply timeout directly
                        result = await asyncio.wait_for(
                            check_func(), timeout=config.timeout
                        )

                    # Ensure result has correct latency
                    result.latency_ms = (time.time() - start_time) * 1000

                    # Thread-safe update of results to prevent race conditions
                    async with self._results_lock:
                        self._last_results[name] = result

                    logger.debug(
                        "health_check_completed",
                        dependency_name=name,
                        status=result.status.value,
                        latency_ms=result.latency_ms,
                    )

                    return result

                except asyncio.TimeoutError:
                    latency_ms = (time.time() - start_time) * 1000
                    result = HealthCheckResult(
                        config=config,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error_message=f"Health check timeout after {config.timeout}s",
                    )

                    # Thread-safe update of results to prevent race conditions
                    async with self._results_lock:
                        self._last_results[name] = result
                    logger.warning(
                        "health_check_timeout",
                        dependency_name=name,
                        timeout=config.timeout,
                        latency_ms=latency_ms,
                    )

                    return result

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    result = HealthCheckResult(
                        config=config,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error_message=_sanitize_error(e),
                    )

                    # Thread-safe update of results to prevent race conditions
                    async with self._results_lock:
                        self._last_results[name] = result
                    logger.error(
                        "health_check_failed",
                        dependency_name=name,
                        error=_sanitize_error(e),
                        error_type=type(e).__name__,
                        latency_ms=latency_ms,
                    )

                    return result

    async def check_all_dependencies(self) -> List[HealthCheckResult]:
        """
        Perform health checks for all registered dependencies.

        Uses asyncio.gather with return_exceptions=True to ensure
        individual dependency failures don't crash the overall health check.

        Returns:
            List[HealthCheckResult]: Results for all dependencies
        """
        if not self._health_checks:
            return []

        async with correlation_context(operation="health_check_all"):
            async with trace_operation(
                "health_check_all",
                OperationType.HEALTH_CHECK,
                dependency_count=len(self._health_checks),
            ):
                # Use asyncio.gather with return_exceptions=True for failure isolation
                tasks = [
                    self.check_single_dependency(name)
                    for name in self._health_checks.keys()
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and handle exceptions
                health_results = []
                for i, result in enumerate(results):
                    dependency_name = list(self._health_checks.keys())[i]

                    if isinstance(result, Exception):
                        # Create error result for exceptions
                        config = self._configs[dependency_name]
                        error_result = HealthCheckResult(
                            config=config,
                            status=HealthStatus.UNHEALTHY,
                            latency_ms=0.0,
                            error_message=f"Health check exception: {str(result)}",
                        )
                        health_results.append(error_result)

                        logger.error(
                            "health_check_gather_exception",
                            dependency_name=dependency_name,
                            error=_sanitize_error(result),
                            error_type=type(result).__name__,
                        )
                    else:
                        health_results.append(result)

                logger.info(
                    "health_check_all_completed",
                    total_dependencies=len(health_results),
                    healthy_count=len(
                        [r for r in health_results if r.status == HealthStatus.HEALTHY]
                    ),
                    degraded_count=len(
                        [r for r in health_results if r.status == HealthStatus.DEGRADED]
                    ),
                    unhealthy_count=len(
                        [
                            r
                            for r in health_results
                            if r.status == HealthStatus.UNHEALTHY
                        ]
                    ),
                )

                return health_results

    def get_resource_metrics(self) -> ModelResourceMetrics:
        """Get current system resource metrics."""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024

            # Get disk usage for root partition
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Get network stats (simplified)
            network_stats = psutil.net_io_counters()
            # Simple approximation of throughput (bytes per second converted to Mbps)
            network_mbps = (
                (network_stats.bytes_sent + network_stats.bytes_recv) / 1024 / 1024 * 8
            )

            return ModelResourceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory_mb,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk_percent,
                network_throughput_mbps=network_mbps,
            )

        except Exception as e:
            logger.error(
                "resource_metrics_error",
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            # Return default metrics on error
            return ModelResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_throughput_mbps=0.0,
            )

    def calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """
        Calculate overall system health based on dependency results.

        Args:
            results: List of health check results

        Returns:
            HealthStatus: Overall system health status
        """
        if not results:
            return HealthStatus.UNKNOWN

        critical_results = [r for r in results if r.config.critical]
        non_critical_results = [r for r in results if not r.config.critical]

        # Check critical dependencies
        critical_unhealthy = [
            r for r in critical_results if r.status == HealthStatus.UNHEALTHY
        ]
        critical_degraded = [
            r for r in critical_results if r.status == HealthStatus.DEGRADED
        ]

        # If any critical dependency is unhealthy, system is unhealthy
        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        # If any critical dependency is degraded, system is degraded
        if critical_degraded:
            return HealthStatus.DEGRADED

        # Check non-critical dependencies for degradation signals
        non_critical_unhealthy = [
            r for r in non_critical_results if r.status == HealthStatus.UNHEALTHY
        ]

        # If more than half of non-critical dependencies are unhealthy, system is degraded
        if (
            non_critical_results
            and len(non_critical_unhealthy) > len(non_critical_results) / 2
        ):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    async def get_comprehensive_health(self) -> ModelHealthResponse:
        """
        Get comprehensive health response including all dependencies and metrics.

        Returns:
            ModelHealthResponse: Complete health check response
        """
        start_time = time.time()

        async with correlation_context(operation="comprehensive_health_check"):
            # Get all dependency health results
            dependency_results = await self.check_all_dependencies()

            # Calculate overall status
            overall_status = self.calculate_overall_status(dependency_results)

            # Get resource metrics
            resource_metrics = self.get_resource_metrics()

            # Calculate uptime
            uptime_seconds = int(time.time() - self._system_start_time)

            # Calculate total latency
            total_latency_ms = (time.time() - start_time) * 1000

            # Convert results to dependency status objects
            dependencies = [
                result.to_dependency_status() for result in dependency_results
            ]

            response = ModelHealthResponse(
                status=overall_status.value,
                latency_ms=total_latency_ms,
                timestamp=datetime.now(),
                resource_usage=resource_metrics,
                dependencies=dependencies,
                uptime_seconds=uptime_seconds,
                version=_get_package_version(),
                environment=_get_environment(),
            )

            logger.info(
                "comprehensive_health_completed",
                overall_status=overall_status.value,
                dependency_count=len(dependencies),
                latency_ms=total_latency_ms,
                uptime_seconds=uptime_seconds,
            )

            return response

    def get_circuit_breaker_stats(self) -> ModelCircuitBreakerStatsCollection:
        """Get circuit breaker statistics for all dependencies."""
        stats = {}
        for name, circuit_breaker in self._circuit_breakers.items():
            stats[name] = ModelCircuitBreakerStats(
                state=circuit_breaker.state.value,
                failure_count=circuit_breaker.stats.failure_count,
                success_count=circuit_breaker.stats.success_count,
                total_calls=circuit_breaker.stats.total_calls,
                total_timeouts=circuit_breaker.stats.total_timeouts,
                last_failure_time=circuit_breaker.stats.last_failure_time,
                state_changed_at=circuit_breaker.stats.state_changed_at,
            )
        return ModelCircuitBreakerStatsCollection(stats=stats)

    async def rate_limited_health_check(
        self, client_identifier: str
    ) -> ModelRateLimitedHealthCheckResponse:
        """
        Rate-limited health check endpoint for API exposure.

        Args:
            client_identifier: Client identifier (IP address, API key, etc.)

        Returns:
            Rate-limited health check response with proper typing
        """
        # Check rate limit
        if not await self._rate_limiter.is_allowed(client_identifier):
            return ModelRateLimitedHealthCheckResponse(
                health_check=None,
                rate_limited=True,
                error_message=f"Rate limit exceeded for client: {client_identifier}",
                rate_limit_reset_time=None,  # Could be calculated from rate limiter
                remaining_requests=0,
            )

        # Perform health check
        health_check_result = await self.comprehensive_health_check()
        return ModelRateLimitedHealthCheckResponse(
            health_check=health_check_result,
            rate_limited=False,
            rate_limit_reset_time=None,
            remaining_requests=None,
        )


# Global health manager instance
health_manager = HealthCheckManager()


# Convenience functions for registering common health checks
async def create_postgresql_health_check(
    connection_string: str,
) -> Callable[[], Awaitable[HealthCheckResult]]:
    """Create a PostgreSQL health check function."""

    async def check_postgresql() -> HealthCheckResult:
        import asyncpg

        config = HealthCheckConfig(
            name="postgresql", dependency_type=DependencyType.DATABASE
        )

        try:
            conn = await asyncpg.connect(connection_string)
            await conn.execute("SELECT 1")
            await conn.close()

            return HealthCheckResult(
                config=config,
                status=HealthStatus.HEALTHY,
                latency_ms=0.0,  # Will be set by the manager
            )
        except Exception as e:
            return HealthCheckResult(
                config=config,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                error_message=_sanitize_error(e),
            )

    return check_postgresql


async def create_redis_health_check(
    redis_url: str,
) -> Callable[[], Awaitable[HealthCheckResult]]:
    """Create a Redis health check function."""

    async def check_redis() -> HealthCheckResult:
        import redis.asyncio as redis

        config = HealthCheckConfig(name="redis", dependency_type=DependencyType.CACHE)

        try:
            client = redis.from_url(redis_url)
            await client.ping()
            await client.close()

            return HealthCheckResult(
                config=config, status=HealthStatus.HEALTHY, latency_ms=0.0
            )
        except Exception as e:
            return HealthCheckResult(
                config=config,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                error_message=_sanitize_error(e),
            )

    return check_redis


async def create_pinecone_health_check(
    api_key: str, environment: str
) -> Callable[[], Awaitable[HealthCheckResult]]:
    """Create a Pinecone health check function."""

    async def check_pinecone() -> HealthCheckResult:
        config = HealthCheckConfig(
            name="pinecone", dependency_type=DependencyType.VECTOR_DB
        )

        try:
            # Simple connection test - this would need to be adapted based on Pinecone client
            # For now, return healthy as a placeholder
            return HealthCheckResult(
                config=config,
                status=HealthStatus.HEALTHY,
                latency_ms=0.0,
                metadata={"environment": environment},
            )
        except Exception as e:
            return HealthCheckResult(
                config=config,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                error_message=_sanitize_error(e),
            )

    return check_pinecone
