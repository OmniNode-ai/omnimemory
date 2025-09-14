"""
Tests for health manager utilities following ONEX standards.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from uuid import uuid4

from omnimemory.utils.health_manager import (
    HealthManager,
    HealthStatus,
    ResourceHealthCheck,
    SystemHealth,
)
from omnimemory.utils.concurrency import CircuitBreaker, CircuitBreakerState
from omnimemory.models.foundation.model_health_response import (
    ModelCircuitBreakerStats,
    ModelCircuitBreakerStatsCollection,
    ModelRateLimitedHealthCheckResponse,
)


class TestHealthManager:
    """Test health manager functionality."""

    def test_health_manager_creation(self):
        """Test health manager can be created with valid parameters."""
        hm = HealthManager()
        assert hm is not None
        assert isinstance(hm.circuit_breakers, dict)

    def test_register_health_check(self):
        """Test registering health checks."""
        hm = HealthManager()

        async def mock_health_check():
            return {"status": "healthy", "details": "All systems operational"}

        hm.register_health_check("database", mock_health_check)
        assert "database" in hm.health_checks

    @pytest.mark.asyncio
    async def test_check_resource_health_success(self):
        """Test resource health check with successful result."""
        hm = HealthManager()

        async def healthy_check():
            return {"status": "healthy", "response_time": 0.05}

        hm.register_health_check("api", healthy_check)
        result = await hm.check_resource_health("api")

        assert result.status == HealthStatus.HEALTHY
        assert result.response_time < 1.0
        assert "status" in result.details

    @pytest.mark.asyncio
    async def test_check_resource_health_failure(self):
        """Test resource health check with failure."""
        hm = HealthManager()

        async def failing_check():
            raise ConnectionError("Database connection failed")

        hm.register_health_check("database", failing_check)
        result = await hm.check_resource_health("database")

        assert result.status == HealthStatus.UNHEALTHY
        assert "error" in result.details
        assert "Database connection failed" in result.details["error"]

    @pytest.mark.asyncio
    async def test_check_resource_health_timeout(self):
        """Test resource health check with timeout."""
        hm = HealthManager(default_timeout=0.1)

        async def slow_check():
            import asyncio
            await asyncio.sleep(0.5)  # Longer than timeout
            return {"status": "healthy"}

        hm.register_health_check("slow_service", slow_check)
        result = await hm.check_resource_health("slow_service")

        assert result.status == HealthStatus.TIMEOUT
        assert result.response_time >= 0.1

    @pytest.mark.asyncio
    async def test_get_system_health(self):
        """Test getting overall system health."""
        hm = HealthManager()

        async def healthy_check():
            return {"status": "healthy"}

        async def unhealthy_check():
            raise ValueError("Service down")

        hm.register_health_check("service1", healthy_check)
        hm.register_health_check("service2", unhealthy_check)

        system_health = await hm.get_system_health()

        assert isinstance(system_health, SystemHealth)
        assert system_health.overall_status == HealthStatus.DEGRADED
        assert len(system_health.resource_statuses) == 2

        # Check individual statuses
        service1_status = system_health.resource_statuses.get("service1")
        service2_status = system_health.resource_statuses.get("service2")

        assert service1_status.status == HealthStatus.HEALTHY
        assert service2_status.status == HealthStatus.UNHEALTHY

    def test_get_circuit_breaker_stats(self):
        """Test getting circuit breaker statistics."""
        hm = HealthManager()

        # Add some circuit breakers
        cb1 = CircuitBreaker(failure_threshold=3)
        cb1.success_count = 10
        cb1.failure_count = 1

        cb2 = CircuitBreaker(failure_threshold=5)
        cb2.success_count = 5
        cb2.failure_count = 2
        cb2.state = CircuitBreakerState.OPEN

        hm.circuit_breakers["service1"] = cb1
        hm.circuit_breakers["service2"] = cb2

        stats = hm.get_circuit_breaker_stats()

        assert isinstance(stats, ModelCircuitBreakerStatsCollection)
        assert "service1" in stats.circuit_breakers
        assert "service2" in stats.circuit_breakers

        # Check service1 stats
        service1_stats = stats.circuit_breakers["service1"]
        assert service1_stats.state == "closed"
        assert service1_stats.success_count == 10
        assert service1_stats.failure_count == 1

        # Check service2 stats
        service2_stats = stats.circuit_breakers["service2"]
        assert service2_stats.state == "open"
        assert service2_stats.success_count == 5
        assert service2_stats.failure_count == 2

    @pytest.mark.asyncio
    async def test_rate_limited_health_check(self):
        """Test rate-limited health check functionality."""
        hm = HealthManager(rate_limit_window=1.0, max_checks_per_window=2)

        call_count = 0
        async def counting_check():
            nonlocal call_count
            call_count += 1
            return {"status": "healthy", "call_count": call_count}

        hm.register_health_check("counted_service", counting_check)

        # First check should execute
        result1 = await hm.check_resource_health("counted_service")
        assert result1.status == HealthStatus.HEALTHY
        assert call_count == 1

        # Second check should execute
        result2 = await hm.check_resource_health("counted_service")
        assert result2.status == HealthStatus.HEALTHY
        assert call_count == 2

        # Third check should be rate limited
        result3 = await hm.check_resource_health("counted_service")
        assert result3.status == HealthStatus.RATE_LIMITED
        assert call_count == 2  # Should not have incremented

    def test_get_rate_limited_health_response(self):
        """Test getting rate-limited health check response."""
        hm = HealthManager()

        response = hm.get_rate_limited_health_response()

        assert isinstance(response, ModelRateLimitedHealthCheckResponse)
        assert response.status == "rate_limited"
        assert response.message == "Health check rate limited"
        assert "retry_after" in response.details
        assert "current_window_requests" in response.details

    @pytest.mark.asyncio
    async def test_health_check_with_circuit_breaker(self):
        """Test health check integrated with circuit breaker."""
        hm = HealthManager()

        # Register circuit breaker for resource
        cb = CircuitBreaker(failure_threshold=2)
        hm.circuit_breakers["flaky_service"] = cb

        failure_count = 0
        async def flaky_check():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise ConnectionError(f"Failure {failure_count}")
            return {"status": "healthy"}

        hm.register_health_check("flaky_service", flaky_check)

        # First failure
        result1 = await hm.check_resource_health("flaky_service")
        assert result1.status == HealthStatus.UNHEALTHY
        assert cb.state == CircuitBreakerState.CLOSED

        # Second failure - should open circuit
        result2 = await hm.check_resource_health("flaky_service")
        assert result2.status == HealthStatus.UNHEALTHY
        assert cb.state == CircuitBreakerState.OPEN

        # Third attempt should be blocked by circuit breaker
        result3 = await hm.check_resource_health("flaky_service")
        assert result3.status == HealthStatus.CIRCUIT_OPEN

    def test_sanitize_error_details(self):
        """Test error sanitization in health checks."""
        hm = HealthManager()

        # Test with sensitive information
        error = Exception("Connection failed: password=secret123, token=abc456")
        sanitized = hm._sanitize_error(error)

        assert "secret123" not in sanitized
        assert "abc456" not in sanitized
        assert "Connection failed" in sanitized
        assert "[REDACTED]" in sanitized

    @pytest.mark.asyncio
    async def test_health_check_correlation_tracking(self):
        """Test health checks include correlation tracking."""
        hm = HealthManager()

        async def tracked_check():
            return {"status": "healthy", "service": "test"}

        hm.register_health_check("tracked_service", tracked_check)

        correlation_id = str(uuid4())
        result = await hm.check_resource_health(
            "tracked_service",
            correlation_id=correlation_id
        )

        assert result.correlation_id == correlation_id
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_bulk_health_check(self):
        """Test checking multiple resources in parallel."""
        hm = HealthManager()

        async def service1_check():
            return {"status": "healthy", "service": "service1"}

        async def service2_check():
            import asyncio
            await asyncio.sleep(0.1)
            return {"status": "healthy", "service": "service2"}

        async def service3_check():
            raise ValueError("Service3 is down")

        hm.register_health_check("service1", service1_check)
        hm.register_health_check("service2", service2_check)
        hm.register_health_check("service3", service3_check)

        results = await hm.check_multiple_resources(
            ["service1", "service2", "service3"]
        )

        assert len(results) == 3
        assert results["service1"].status == HealthStatus.HEALTHY
        assert results["service2"].status == HealthStatus.HEALTHY
        assert results["service3"].status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_manager_cleanup(self):
        """Test health manager resource cleanup."""
        hm = HealthManager()

        # Add some resources
        async def test_check():
            return {"status": "healthy"}

        hm.register_health_check("cleanup_test", test_check)
        cb = CircuitBreaker(failure_threshold=3)
        hm.circuit_breakers["cleanup_test"] = cb

        assert "cleanup_test" in hm.health_checks
        assert "cleanup_test" in hm.circuit_breakers

        # Cleanup
        await hm.cleanup()

        # Resources should be cleared
        assert len(hm.health_checks) == 0
        assert len(hm.circuit_breakers) == 0


@pytest.mark.integration
class TestHealthManagerIntegration:
    """Integration tests for health manager."""

    @pytest.mark.asyncio
    async def test_complete_health_monitoring_workflow(self):
        """Test complete health monitoring workflow."""
        hm = HealthManager(
            default_timeout=1.0,
            rate_limit_window=2.0,
            max_checks_per_window=5
        )

        # Simulate different types of services
        async def stable_service():
            return {"status": "healthy", "uptime": "99.9%"}

        async def intermittent_service():
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise ConnectionError("Intermittent failure")
            return {"status": "healthy", "load": "normal"}

        async def slow_service():
            import asyncio
            await asyncio.sleep(0.5)
            return {"status": "healthy", "response_time": "slow"}

        # Register services
        hm.register_health_check("stable", stable_service)
        hm.register_health_check("intermittent", intermittent_service)
        hm.register_health_check("slow", slow_service)

        # Add circuit breakers
        hm.circuit_breakers["intermittent"] = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1.0
        )

        # Perform multiple health checks
        results = []
        for i in range(10):
            system_health = await hm.get_system_health()
            results.append(system_health)

            # Small delay between checks
            import asyncio
            await asyncio.sleep(0.1)

        # Verify we got results
        assert len(results) == 10

        # Check that we have data for all services
        for result in results:
            assert "stable" in result.resource_statuses
            assert "intermittent" in result.resource_statuses
            assert "slow" in result.resource_statuses

        # Get final circuit breaker stats
        cb_stats = hm.get_circuit_breaker_stats()
        assert "intermittent" in cb_stats.circuit_breakers

        # Cleanup
        await hm.cleanup()