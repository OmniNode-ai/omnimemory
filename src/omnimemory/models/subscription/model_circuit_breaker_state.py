"""
Circuit breaker state model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from ...enums.enum_subscription_status import EnumCircuitBreakerState
from .constants import (
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN_MS,
    DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
)


class ModelCircuitBreakerState(BaseModel):
    """Circuit breaker state for a webhook endpoint following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid", strict=True)

    endpoint: HttpUrl = Field(
        description="Webhook endpoint URL being monitored",
    )
    state: EnumCircuitBreakerState = Field(
        default=EnumCircuitBreakerState.CLOSED,
        description="Current circuit breaker state: closed, open, or half_open",
    )
    failure_count: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive failures",
    )
    success_count: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive successes (for half_open state)",
    )
    failure_threshold: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        ge=1,
        le=100,
        description="Number of failures before circuit opens",
    )
    success_threshold: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        ge=1,
        le=100,
        description="Number of successes in half_open before closing",
    )
    last_failure_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last failure",
    )
    last_success_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last success",
    )
    open_until: datetime | None = Field(
        default=None,
        description="When the circuit breaker will transition to half_open",
    )
    cooldown_ms: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_COOLDOWN_MS,
        ge=1000,
        le=300000,
        description="Cooldown period in milliseconds before transitioning to half_open",
    )
    last_error_message: str | None = Field(
        default=None,
        max_length=1024,
        description="Last error message for debugging",
    )
    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total number of requests through this circuit",
    )
    total_failures: int = Field(
        default=0,
        ge=0,
        description="Total number of failures (lifetime)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this circuit breaker was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this circuit breaker was last updated",
    )

    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through the circuit breaker."""
        if self.state == EnumCircuitBreakerState.CLOSED:
            return True
        if self.state == EnumCircuitBreakerState.OPEN:
            # Check if cooldown has passed
            if self.open_until and datetime.now(timezone.utc) >= self.open_until:
                # Transition to half_open and allow the test request
                self.state = EnumCircuitBreakerState.HALF_OPEN
                self.success_count = 0  # Reset for half_open tracking
                self.updated_at = datetime.now(timezone.utc)
                return True
            return False
        # HALF_OPEN: allow limited requests for testing
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.failure_count = 0
        self.last_success_at = datetime.now(timezone.utc)
        self.total_requests += 1
        self.updated_at = datetime.now(timezone.utc)

        if self.state == EnumCircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self.state = EnumCircuitBreakerState.CLOSED
                self.success_count = 0

    def record_failure(self, error_message: str | None = None) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_at = datetime.now(timezone.utc)
        self.total_requests += 1
        self.total_failures += 1
        self.updated_at = datetime.now(timezone.utc)
        if error_message:
            self.last_error_message = error_message[:1024]

        if self.state == EnumCircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = EnumCircuitBreakerState.OPEN
                self.open_until = datetime.now(timezone.utc) + timedelta(
                    milliseconds=self.cooldown_ms
                )
        elif self.state == EnumCircuitBreakerState.HALF_OPEN:
            # Any failure in half_open immediately opens the circuit
            self.state = EnumCircuitBreakerState.OPEN
            self.open_until = datetime.now(timezone.utc) + timedelta(
                milliseconds=self.cooldown_ms
            )
