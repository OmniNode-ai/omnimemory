"""
Circuit breaker statistics collection model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from .model_circuit_breaker_stats import ModelCircuitBreakerStats


class ModelCircuitBreakerStatsCollection(BaseModel):
    """Collection of circuit breaker statistics for all dependencies."""

    stats: dict[str, ModelCircuitBreakerStats] = Field(
        description="Circuit breaker statistics keyed by dependency name"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the statistics were collected",
    )
