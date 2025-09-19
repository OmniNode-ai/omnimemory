"""
Circuit breaker state enum for ONEX standards.
"""

from enum import Enum


class EnumCircuitBreakerState(str, Enum):
    """Circuit breaker states for resilience patterns."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    DISABLED = "disabled"
