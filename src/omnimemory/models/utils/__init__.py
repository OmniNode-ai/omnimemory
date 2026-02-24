# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Utils Model Package - Pydantic models for utility modules.

This submodule contains models that were moved from the utils/ directory
to comply with ONEX standards requiring all Pydantic models in models/.

Models are organized by functional domain:
- model_retry_*: Retry configuration and statistics
- model_concurrency: Connection pool configuration
- model_circuit_breaker_config: Circuit breaker configuration
- model_circuit_breaker_stats_response: Circuit breaker statistics
- model_structured_log_entry: Structured logging with trace levels
- model_correlation_context: Correlation tracking
- model_audit: Audit event tracking
- model_health_*: Health check configuration and results
- model_pii_*: PII detection configuration and results
"""

# Audit models
from .model_audit import (
    AuditEventType,
    AuditSeverity,
    ModelAuditEvent,
)

# Circuit breaker models
from .model_circuit_breaker_config import ModelCircuitBreakerConfig
from .model_circuit_breaker_stats_response import ModelCircuitBreakerStatsResponse

# Concurrency models
from .model_concurrency import (
    ModelConnectionPoolConfig,
)

# Correlation context models
from .model_correlation_context import (
    ModelCorrelationContext,
)

# Health models
from .model_health_check_config import (
    DependencyType,
    ModelHealthCheckConfig,
)
from .model_health_check_details import (
    ModelHealthCheckDetails,
)
from .model_health_check_result import (
    ModelHealthCheckResult,
)
from .model_health_status import (
    HealthStatus,
)

# PII models
from .model_pii_detection_result import ModelPIIDetectionResult
from .model_pii_detector_config import ModelPIIDetectorConfig
from .model_pii_match import ModelPIIMatch
from .model_pii_pattern_config import ModelPIIPatternConfig
from .model_pii_type import PIIType
from .model_resource_health_check import (
    ModelResourceHealthCheck,
)

# Retry models
from .model_retry_attempt_info import ModelRetryAttemptInfo
from .model_retry_config import ModelRetryConfig
from .model_retry_statistics import ModelRetryStatistics

# Structured log entry models
from .model_structured_log_entry import (
    ModelStructuredLogEntry,
    TraceLevel,
)
from .model_system_health import (
    ModelSystemHealth,
)

__all__ = [
    # Audit
    "AuditEventType",
    "AuditSeverity",
    "ModelAuditEvent",
    # Concurrency
    "ModelConnectionPoolConfig",
    # Health
    "DependencyType",
    "HealthStatus",
    "ModelHealthCheckConfig",
    "ModelHealthCheckDetails",
    "ModelHealthCheckResult",
    "ModelResourceHealthCheck",
    "ModelSystemHealth",
    # Observability
    "ModelCorrelationContext",
    "ModelStructuredLogEntry",
    "TraceLevel",
    # PII
    "ModelPIIDetectionResult",
    "ModelPIIDetectorConfig",
    "ModelPIIMatch",
    "ModelPIIPatternConfig",
    "PIIType",
    # Circuit breaker
    "ModelCircuitBreakerConfig",
    "ModelCircuitBreakerStatsResponse",
    # Retry
    "ModelRetryAttemptInfo",
    "ModelRetryConfig",
    "ModelRetryStatistics",
]
