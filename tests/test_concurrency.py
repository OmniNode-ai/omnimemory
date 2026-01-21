"""
Tests for concurrency utilities following ONEX standards.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from omnimemory.utils.concurrency import (
    CircuitBreaker,
    CircuitBreakerState,
    ConnectionPool,
    with_circuit_breaker,
    with_retry,
    with_timeout,
)


class TestConnectionPool:
    """Test connection pool functionality."""

    @pytest.mark.asyncio
    async def test_connection_pool_creation(self) -> None:
        """Test connection pool can be created with valid parameters."""
        pool = ConnectionPool(max_size=5, timeout=30.0)
        assert pool.max_size == 5
        assert pool.timeout == 30.0

    @pytest.mark.asyncio
    async def test_connection_pool_acquire_release_cycle(self) -> None:
        """Test connection acquisition and release."""
        pool = ConnectionPool(max_size=2, timeout=1.0)

        # Mock connection factory
        mock_conn = Mock()
        pool._create_connection = Mock(return_value=mock_conn)

        # Acquire connection
        async with pool.acquire() as conn:
            assert conn is mock_conn
            assert pool.active_connections == 1

        # Connection should be released
        assert pool.active_connections == 0

    @pytest.mark.asyncio
    async def test_connection_pool_max_size_limit(self) -> None:
        """Test connection pool respects max size limit."""
        pool = ConnectionPool(max_size=1, timeout=0.1)
        pool._create_connection = Mock(return_value=Mock())

        # First connection should work
        async with pool.acquire():
            # Second connection should timeout
            with pytest.raises(asyncio.TimeoutError):
                async with pool.acquire():
                    pass

    @pytest.mark.asyncio
    async def test_connection_pool_iterative_retry_prevents_recursion(self) -> None:
        """Test that connection pool uses iterative retry to prevent stack overflow."""
        pool = ConnectionPool(max_size=1, timeout=1.0)

        # Mock connection factory that fails first few times
        call_count = 0

        def create_failing_connection() -> Mock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise ConnectionError("Connection failed")
            return Mock()

        pool._create_connection = create_failing_connection

        # This should succeed on 3rd attempt using iterative retry
        async with pool.acquire() as conn:
            assert conn is not None
            assert call_count == 3


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_creation(self) -> None:
        """Test circuit breaker can be created with valid parameters."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60.0
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_flow(self) -> None:
        """Test circuit breaker allows successful operations."""
        cb = CircuitBreaker(failure_threshold=2)

        @with_circuit_breaker(cb)
        async def successful_operation() -> str:
            return "success"

        result = await successful_operation()
        assert result == "success"
        assert cb.success_count == 1
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self) -> None:
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        @with_circuit_breaker(cb)
        async def failing_operation() -> None:
            raise ValueError("Operation failed")

        # First failure
        with pytest.raises(ValueError):
            await failing_operation()
        assert cb.state == CircuitBreakerState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await failing_operation()
        assert cb.state == CircuitBreakerState.OPEN

        # Third call should be blocked
        with pytest.raises(Exception):  # Circuit breaker exception
            await failing_operation()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self) -> None:
        """Test circuit breaker recovery through half-open state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        call_count = 0

        @with_circuit_breaker(cb)
        async def recovering_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Initial failure")
            return "recovered"

        # Cause failure to open circuit
        with pytest.raises(ValueError):
            await recovering_operation()
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Next call should succeed and close circuit
        result = await recovering_operation()
        assert result == "recovered"
        assert cb.state == CircuitBreakerState.CLOSED


class TestTimeoutDecorator:
    """Test timeout decorator functionality."""

    @pytest.mark.asyncio
    async def test_with_timeout_success(self) -> None:
        """Test timeout decorator allows fast operations."""

        @with_timeout(1.0)
        async def fast_operation() -> str:
            await asyncio.sleep(0.1)
            return "completed"

        result = await fast_operation()
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_with_timeout_failure(self) -> None:
        """Test timeout decorator cancels slow operations."""

        @with_timeout(0.1)
        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "should not complete"

        with pytest.raises(asyncio.TimeoutError):
            await slow_operation()


class TestRetryDecorator:
    """Test retry decorator functionality."""

    @pytest.mark.asyncio
    async def test_with_retry_success(self) -> None:
        """Test retry decorator allows successful operations."""

        @with_retry(max_attempts=3, delay=0.1)
        async def successful_operation() -> str:
            return "success"

        result = await successful_operation()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_retry_eventual_success(self) -> None:
        """Test retry decorator retries until success."""
        call_count = 0

        @with_retry(max_attempts=3, delay=0.01)
        async def eventually_successful() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        result = await eventually_successful()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_max_attempts_exceeded(self) -> None:
        """Test retry decorator respects max attempts."""
        call_count = 0

        @with_retry(max_attempts=2, delay=0.01)
        async def always_failing() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        with pytest.raises(ValueError, match="Attempt 2 failed"):
            await always_failing()
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_retry_exponential_backoff(self) -> None:
        """Test retry decorator uses exponential backoff."""
        call_times: list[float] = []

        @with_retry(max_attempts=3, delay=0.1, backoff_multiplier=2.0)
        async def timing_operation() -> str:
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise ValueError("Not yet")
            return "success"

        _start_time = asyncio.get_event_loop().time()
        await timing_operation()

        # Check that delays increased exponentially
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Use lenient tolerances to avoid CI flakiness
        # delay1 should be ~0.1s, delay2 should be ~0.2s (50% tolerance)
        assert (
            0.05 < delay1 < 0.20
        ), f"First delay {delay1:.3f}s outside 0.05-0.20s range"
        assert (
            0.10 < delay2 < 0.40
        ), f"Second delay {delay2:.3f}s outside 0.10-0.40s range"

        # More importantly, verify exponential relationship: delay2 ~= 2x delay1
        backoff_ratio = delay2 / delay1
        assert (
            1.5 < backoff_ratio < 3.0
        ), f"Backoff ratio {backoff_ratio:.2f} not in expected range 1.5-3.0"


@pytest.mark.integration
class TestConcurrencyIntegration:
    """Integration tests for concurrency utilities."""

    @pytest.mark.asyncio
    async def test_connection_pool_with_circuit_breaker(self) -> None:
        """Test connection pool integrated with circuit breaker."""
        pool = ConnectionPool(max_size=2, timeout=1.0)
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Mock connection that fails first several times
        # The pool has internal retry logic (default 3 retries per acquire call)
        # So we need to fail enough times to exhaust retries for 2 calls:
        # 3 retries * 2 calls = 6 failures needed
        fail_count = 0

        def create_connection() -> Mock:
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 6:
                raise ConnectionError("Connection failed")
            return Mock()

        pool._create_connection = create_connection

        @with_circuit_breaker(cb)
        async def get_connection() -> Mock:
            async with pool.acquire() as conn:
                return conn

        # First two attempts should fail (each exhausts 3 pool retries)
        with pytest.raises(ConnectionError):
            await get_connection()

        with pytest.raises(ConnectionError):
            await get_connection()

        # Circuit should now be open
        assert cb.state == CircuitBreakerState.OPEN

        # Next attempt should be blocked by circuit breaker
        with pytest.raises(Exception):
            await get_connection()

    @pytest.mark.asyncio
    async def test_retry_timeout_circuit_breaker_combined(self) -> None:
        """Test retry combined with timeout and circuit breaker."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        attempt_count = 0

        @with_timeout(0.5)
        @with_retry(max_attempts=5, delay=0.05)
        @with_circuit_breaker(cb)
        async def complex_operation() -> str:
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")

            await asyncio.sleep(0.02)  # Small delay
            return f"Success on attempt {attempt_count}"

        result = await complex_operation()
        assert result == "Success on attempt 3"
        assert attempt_count == 3
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.success_count == 1
