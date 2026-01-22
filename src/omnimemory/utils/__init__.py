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

from ..models.utils.model_concurrency import ModelConnectionPoolConfig
from .concurrency import (
    AsyncConnectionPool,
    FairSemaphore,
    LockPriority,
    PoolStatus,
    PriorityLock,
    get_connection_pool,
    get_fair_semaphore,
    get_priority_lock,
    register_connection_pool,
    with_connection_pool,
    with_fair_semaphore,
    with_priority_lock,
)
from .error_sanitizer import (
    ErrorSanitizer,
    SanitizationLevel,
    sanitize_dict,
    sanitize_error,
)
from .health_manager import (
    DependencyType,
    HealthCheckConfig,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
    RateLimiter,
    create_pinecone_health_check,
    create_postgresql_health_check,
    create_redis_health_check,
    health_manager,
)
from .observability import (
    CorrelationContext,
    Counter,
    Gauge,
    HandlerMetrics,
    HandlerObservabilityWrapper,
    Histogram,
    MetricsRegistry,
    ObservabilityManager,
    OperationType,
    TraceLevel,
    correlation_context,
    get_correlation_id,
    get_request_id,
    inject_correlation_context,
    inject_correlation_context_async,
    log_with_correlation,
    metrics_registry,
    observability_manager,
    sanitize_metadata_value,
    trace_operation,
    validate_correlation_id,
)
from .pii_detector import (
    ModelPIIDetectionResult,
    ModelPIIDetectorConfig,
    ModelPIIMatch,
    ModelPIIPatternConfig,
    PIIDetector,
    PIIType,
)
from .resource_manager import (
    AsyncCircuitBreaker,
    AsyncResourceManager,
    CircuitBreakerError,
    CircuitState,
    ModelCircuitBreakerConfig,
    resource_manager,
    with_circuit_breaker,
    with_semaphore,
    with_timeout,
)
from .retry_utils import (
    ModelRetryAttemptInfo,
    ModelRetryConfig,
    ModelRetryStatistics,
    RetryManager,
    calculate_delay,
    default_retry_manager,
    is_retryable_exception,
    retry_decorator,
    retry_with_backoff,
)

__all__ = [
    # Retry utilities
    "ModelRetryConfig",
    "ModelRetryAttemptInfo",
    "ModelRetryStatistics",
    "RetryManager",
    "default_retry_manager",
    "retry_decorator",
    "retry_with_backoff",
    "is_retryable_exception",
    "calculate_delay",
    # Resource management
    "CircuitState",
    "ModelCircuitBreakerConfig",
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
    # In-process metrics (P1C)
    "Counter",
    "Histogram",
    "Gauge",
    "MetricsRegistry",
    "metrics_registry",
    "HandlerMetrics",
    "HandlerObservabilityWrapper",
    # Concurrency
    "LockPriority",
    "PoolStatus",
    "ModelConnectionPoolConfig",
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
    # PII Detection
    "ModelPIIDetectionResult",
    "ModelPIIDetectorConfig",
    "ModelPIIMatch",
    "ModelPIIPatternConfig",
    "PIIDetector",
    "PIIType",
    # Error Sanitization
    "SanitizationLevel",
    "ErrorSanitizer",
    "sanitize_error",
    "sanitize_dict",
]
