"""
Utility modules for OmniMemory ONEX architecture.

This package provides common utilities used across the OmniMemory system:
- Retry logic with exponential backoff
- Resource management with circuit breakers and async context managers
- Observability with ContextVar correlation tracking
- Concurrency utilities with priority locks and fair semaphores
- Health checking with comprehensive dependency monitoring
- Performance monitoring helpers
- Common validation patterns
"""

from .retry_utils import (
    RetryConfig,
    RetryAttemptInfo,
    RetryStatistics,
    RetryManager,
    default_retry_manager,
    retry_decorator,
    retry_with_backoff,
    is_retryable_exception,
    calculate_delay,
)

from .resource_manager import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerError,
    AsyncCircuitBreaker,
    AsyncResourceManager,
    resource_manager,
    with_circuit_breaker,
    with_semaphore,
    with_timeout,
)

from .observability import (
    TraceLevel,
    OperationType,
    CorrelationContext,
    ObservabilityManager,
    observability_manager,
    correlation_context,
    trace_operation,
    get_correlation_id,
    get_request_id,
    log_with_correlation,
    inject_correlation_context,
    inject_correlation_context_async,
    validate_correlation_id,
    sanitize_metadata_value,
)

from .concurrency import (
    LockPriority,
    PoolStatus,
    ConnectionPoolConfig,
    PriorityLock,
    FairSemaphore,
    AsyncConnectionPool,
    get_priority_lock,
    get_fair_semaphore,
    register_connection_pool,
    get_connection_pool,
    with_priority_lock,
    with_fair_semaphore,
    with_connection_pool,
)

from .health_manager import (
    HealthStatus,
    DependencyType,
    HealthCheckConfig,
    HealthCheckResult,
    HealthCheckManager,
    health_manager,
    RateLimiter,
    create_postgresql_health_check,
    create_redis_health_check,
    create_pinecone_health_check,
)

__all__ = [
    # Retry utilities
    "RetryConfig",
    "RetryAttemptInfo",
    "RetryStatistics",
    "RetryManager",
    "default_retry_manager",
    "retry_decorator",
    "retry_with_backoff",
    "is_retryable_exception",
    "calculate_delay",

    # Resource management
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "AsyncCircuitBreaker",
    "AsyncResourceManager",
    "resource_manager",
    "with_circuit_breaker",
    "with_semaphore",
    "with_timeout",

    # Observability
    "TraceLevel",
    "OperationType",
    "CorrelationContext",
    "ObservabilityManager",
    "observability_manager",
    "correlation_context",
    "trace_operation",
    "get_correlation_id",
    "get_request_id",
    "log_with_correlation",
    "inject_correlation_context",
    "inject_correlation_context_async",
    "validate_correlation_id",
    "sanitize_metadata_value",

    # Concurrency
    "LockPriority",
    "PoolStatus",
    "ConnectionPoolConfig",
    "PriorityLock",
    "FairSemaphore",
    "AsyncConnectionPool",
    "get_priority_lock",
    "get_fair_semaphore",
    "register_connection_pool",
    "get_connection_pool",
    "with_priority_lock",
    "with_fair_semaphore",
    "with_connection_pool",

    # Health management
    "HealthStatus",
    "DependencyType",
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthCheckManager",
    "health_manager",
    "RateLimiter",
    "create_postgresql_health_check",
    "create_redis_health_check",
    "create_pinecone_health_check",
]