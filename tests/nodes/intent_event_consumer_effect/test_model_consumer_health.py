"""Tests for consumer health model."""

from datetime import datetime, timezone

import pytest

from omnimemory.models.utils.model_health_status import HealthStatus
from omnimemory.nodes.intent_event_consumer_effect.models import (
    ModelIntentEventConsumerHealth,
)


class TestModelIntentEventConsumerHealth:
    """Tests for the health status model."""

    def test_healthy_consumer(self) -> None:
        """Test creating a healthy consumer status."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            initialized=True,
            last_consume_timestamp=datetime.now(timezone.utc),
            circuit_breaker_state="closed",
            messages_consumed_total=100,
        )

        assert health.is_healthy is True
        assert health.status == HealthStatus.HEALTHY
        assert health.circuit_breaker_state == "closed"

    def test_degraded_consumer_stale(self) -> None:
        """Test degraded status due to staleness."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.DEGRADED,
            is_healthy=False,
            initialized=True,
            is_stale=True,
            staleness_seconds=600.0,
            error_message="No messages consumed in 600s",
        )

        assert health.is_healthy is False
        assert health.status == HealthStatus.DEGRADED
        assert health.is_stale is True
        assert health.staleness_seconds == 600.0

    def test_circuit_breaker_open(self) -> None:
        """Test circuit open status."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.CIRCUIT_OPEN,
            is_healthy=False,
            initialized=True,
            circuit_breaker_state="open",
            circuit_breaker_failure_count=5,
            error_message="Circuit breaker is open",
        )

        assert health.status == HealthStatus.CIRCUIT_OPEN
        assert health.circuit_breaker_state == "open"
        assert health.circuit_breaker_failure_count == 5

    def test_unknown_when_not_initialized(self) -> None:
        """Test unknown status when not initialized."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.UNKNOWN,
            is_healthy=False,
            initialized=False,
            error_message="Consumer not initialized",
        )

        assert health.status == HealthStatus.UNKNOWN
        assert health.initialized is False
        assert health.error_message == "Consumer not initialized"

    def test_metrics_tracking(self) -> None:
        """Test that all metrics are tracked."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            initialized=True,
            messages_consumed_total=1000,
            messages_failed_total=5,
            messages_dlq_total=3,
        )

        assert health.messages_consumed_total == 1000
        assert health.messages_failed_total == 5
        assert health.messages_dlq_total == 3

    def test_validates_non_negative_consumed_count(self) -> None:
        """Test that messages_consumed_total cannot be negative."""
        with pytest.raises(ValueError):
            ModelIntentEventConsumerHealth(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                initialized=True,
                messages_consumed_total=-1,  # Invalid
            )

    def test_validates_non_negative_failed_count(self) -> None:
        """Test that messages_failed_total cannot be negative."""
        with pytest.raises(ValueError):
            ModelIntentEventConsumerHealth(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                initialized=True,
                messages_failed_total=-1,  # Invalid
            )

    def test_validates_non_negative_dlq_count(self) -> None:
        """Test that messages_dlq_total cannot be negative."""
        with pytest.raises(ValueError):
            ModelIntentEventConsumerHealth(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                initialized=True,
                messages_dlq_total=-1,  # Invalid
            )

    def test_validates_non_negative_failure_count(self) -> None:
        """Test that circuit_breaker_failure_count cannot be negative."""
        with pytest.raises(ValueError):
            ModelIntentEventConsumerHealth(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                initialized=True,
                circuit_breaker_failure_count=-1,  # Invalid
            )

    def test_validates_non_negative_staleness_seconds(self) -> None:
        """Test that staleness_seconds cannot be negative."""
        with pytest.raises(ValueError):
            ModelIntentEventConsumerHealth(
                status=HealthStatus.DEGRADED,
                is_healthy=False,
                initialized=True,
                staleness_seconds=-1.0,  # Invalid
            )

    def test_circuit_breaker_half_open_state(self) -> None:
        """Test circuit breaker in half-open state."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.DEGRADED,
            is_healthy=False,
            initialized=True,
            circuit_breaker_state="half_open",
            circuit_breaker_failure_count=3,
            error_message="Circuit breaker testing recovery",
        )

        assert health.circuit_breaker_state == "half_open"
        assert health.status == HealthStatus.DEGRADED

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            initialized=True,
        )

        # Check defaults
        assert health.error_message is None
        assert health.last_consume_timestamp is None
        assert health.is_stale is False
        assert health.staleness_seconds is None
        assert health.circuit_breaker_state is None
        assert health.circuit_breaker_failure_count == 0
        assert health.messages_consumed_total == 0
        assert health.messages_failed_total == 0
        assert health.messages_dlq_total == 0
        assert health.storage_handler_healthy is None

    def test_health_check_timestamp_auto_set(self) -> None:
        """Test that health_check_timestamp is auto-populated."""
        before = datetime.now(timezone.utc)
        health = ModelIntentEventConsumerHealth(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            initialized=True,
        )
        after = datetime.now(timezone.utc)

        assert health.health_check_timestamp is not None
        assert before <= health.health_check_timestamp <= after

    def test_storage_handler_healthy_tracking(self) -> None:
        """Test that storage handler health is tracked."""
        # Healthy storage handler
        health_ok = ModelIntentEventConsumerHealth(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            initialized=True,
            storage_handler_healthy=True,
        )
        assert health_ok.storage_handler_healthy is True

        # Unhealthy storage handler
        health_bad = ModelIntentEventConsumerHealth(
            status=HealthStatus.UNHEALTHY,
            is_healthy=False,
            initialized=True,
            storage_handler_healthy=False,
            error_message="Storage handler unavailable",
        )
        assert health_bad.storage_handler_healthy is False

    def test_all_health_status_values(self) -> None:
        """Test that all HealthStatus enum values are valid."""
        valid_statuses = [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
            HealthStatus.TIMEOUT,
            HealthStatus.RATE_LIMITED,
            HealthStatus.CIRCUIT_OPEN,
        ]

        for status in valid_statuses:
            health = ModelIntentEventConsumerHealth(
                status=status,
                is_healthy=(status == HealthStatus.HEALTHY),
                initialized=True,
            )
            assert health.status == status
