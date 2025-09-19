"""
ONEX-compliant typed model for performance audit details.

This module provides strongly typed replacement for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.
"""

from typing import Optional

from pydantic import BaseModel, Field

from ...enums.service.enum_circuit_breaker_state import EnumCircuitBreakerState


class ModelPerformanceAuditDetails(BaseModel):
    """Strongly typed performance audit information."""

    operation_latency_ms: float = Field(description="Operation latency in milliseconds")

    throughput_ops_per_second: Optional[float] = Field(
        default=None, description="Throughput in operations per second"
    )

    queue_depth: Optional[int] = Field(
        default=None, description="Queue depth at operation time"
    )

    connection_pool_usage: Optional[float] = Field(
        default=None, description="Connection pool usage percentage"
    )

    circuit_breaker_state: Optional[EnumCircuitBreakerState] = Field(
        default=None, description="Circuit breaker state during operation"
    )

    retry_count: int = Field(default=0, description="Number of retries attempted")

    cache_efficiency: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cache hit ratio (0.0-1.0 where 1.0 is 100% hit rate)",
    )
