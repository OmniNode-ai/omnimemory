"""
Utils Model Package - Pydantic models for utility modules.

This submodule contains models that were moved from the utils/ directory
to comply with ONEX standards requiring all Pydantic models in models/.

Models are organized by functional domain:
- model_retry: Retry configuration and statistics
- model_concurrency: Connection pool configuration
- model_resource_manager: Circuit breaker configuration and stats
- model_observability: Structured logging and correlation tracking
- model_audit: Audit event tracking
- model_health: Health check configuration and results
- model_pii: PII detection configuration and results

All models provide backward compatibility aliases matching original names.
"""

# Audit models
from .model_audit import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    ModelAuditEvent,
)

# Concurrency models
from .model_concurrency import (
    ConnectionPoolConfig,
    ModelConnectionPoolConfig,
)

# Health models
from .model_health import (
    DependencyType,
    HealthCheckConfig,
    HealthCheckDetails,
    HealthCheckResult,
    HealthStatus,
    ModelHealthCheckConfig,
    ModelHealthCheckDetails,
    ModelHealthCheckResult,
    ModelResourceHealthCheck,
    ModelSystemHealth,
    ResourceHealthCheck,
    SystemHealth,
)

# Observability models
from .model_observability import (
    CorrelationContext,
    ModelCorrelationContext,
    ModelStructuredLogEntry,
    StructuredLogEntry,
    TraceLevel,
)

# PII models
from .model_pii import (
    ModelPIIDetectionResult,
    ModelPIIDetectorConfig,
    ModelPIIMatch,
    ModelPIIPatternConfig,
    PIIDetectionResult,
    PIIDetectorConfig,
    PIIMatch,
    PIIPatternConfig,
    PIIType,
)

# Resource manager models
from .model_resource_manager import (
    CircuitBreakerConfig,
    CircuitBreakerStatsResponse,
    ModelCircuitBreakerConfig,
    ModelCircuitBreakerStatsResponse,
)

# Retry models
from .model_retry import (
    ModelRetryAttemptInfo,
    ModelRetryConfig,
    ModelRetryStatistics,
    RetryAttemptInfo,
    RetryConfig,
    RetryStatistics,
)

__all__ = [
    # Audit
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "ModelAuditEvent",
    # Concurrency
    "ConnectionPoolConfig",
    "ModelConnectionPoolConfig",
    # Health
    "DependencyType",
    "HealthCheckConfig",
    "HealthCheckDetails",
    "HealthCheckResult",
    "HealthStatus",
    "ModelHealthCheckConfig",
    "ModelHealthCheckDetails",
    "ModelHealthCheckResult",
    "ModelResourceHealthCheck",
    "ModelSystemHealth",
    "ResourceHealthCheck",
    "SystemHealth",
    # Observability
    "CorrelationContext",
    "ModelCorrelationContext",
    "ModelStructuredLogEntry",
    "StructuredLogEntry",
    "TraceLevel",
    # PII
    "ModelPIIDetectionResult",
    "ModelPIIDetectorConfig",
    "ModelPIIMatch",
    "ModelPIIPatternConfig",
    "PIIDetectionResult",
    "PIIDetectorConfig",
    "PIIMatch",
    "PIIPatternConfig",
    "PIIType",
    # Resource manager
    "CircuitBreakerConfig",
    "CircuitBreakerStatsResponse",
    "ModelCircuitBreakerConfig",
    "ModelCircuitBreakerStatsResponse",
    # Retry
    "ModelRetryAttemptInfo",
    "ModelRetryConfig",
    "ModelRetryStatistics",
    "RetryAttemptInfo",
    "RetryConfig",
    "RetryStatistics",
]
