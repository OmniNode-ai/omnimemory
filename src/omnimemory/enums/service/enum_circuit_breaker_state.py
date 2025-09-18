"""
Circuit breaker state enum for OmniMemory following ONEX standards.

Defines states for circuit breaker pattern in service communication.
"""

from enum import Enum


class EnumCircuitBreakerState(str, Enum):
    """Circuit breaker states for resilient service communication."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def default(cls) -> "EnumCircuitBreakerState":
        """Return default state."""
        return cls.CLOSED

    @property
    def allows_requests(self) -> bool:
        """Check if state allows requests to pass through."""
        return self in {self.CLOSED, self.HALF_OPEN}

    @property
    def is_failure_state(self) -> bool:
        """Check if state indicates failures."""
        return self == self.OPEN

    def next_state_on_success(self) -> "EnumCircuitBreakerState":
        """Get next state on successful request."""
        if self == self.HALF_OPEN:
            return self.CLOSED
        return self

    def next_state_on_failure(self) -> "EnumCircuitBreakerState":
        """Get next state on failed request."""
        if self == self.CLOSED:
            return self.OPEN
        elif self == self.HALF_OPEN:
            return self.OPEN
        return self
