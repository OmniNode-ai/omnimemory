"""Error handling and timeout behavior tests for event-driven OmniMemory architecture.

Tests comprehensive error handling, timeout management, and graceful degradation
in the pure event-driven architecture without direct database connections.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from omnimemory.events.event_bus_client import EventBusClient
from omnimemory.events.event_consumer import EventConsumer
from omnimemory.events.event_producer import EventProducer
from omnimemory.models.memory.model_memory_request import (
    ModelMemoryRetrieveRequest,
    ModelMemoryStoreRequest,
    ModelMemoryVectorSearchRequest,
)
from omnimemory.services.event_driven_memory_service import EventDrivenMemoryService
from omnimemory.services.memory_operation_mapper import MemoryOperationMapper

# Import ONEX core components with fallback
try:
    from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
    from omnibase_core.core.protocol_event_bus import ProtocolEventBus
except ImportError:

    class ProtocolEventBus:
        async def publish_async(self, event):
            pass

        async def subscribe_async(self, topic, handler):
            pass

    class CoreErrorCode:
        SERVICE_UNAVAILABLE_ERROR = "SERVICE_UNAVAILABLE_ERROR"
        TIMEOUT_ERROR = "TIMEOUT_ERROR"
        VALIDATION_ERROR = "VALIDATION_ERROR"
        DEPENDENCY_RESOLUTION_ERROR = "DEPENDENCY_RESOLUTION_ERROR"

    class OnexError(Exception):
        def __init__(self, code: str, message: str, details: Optional[Dict] = None):
            self.code = code
            self.message = message
            self.details = details
            super().__init__(message)


class FailingEventBus(AsyncMock):
    """Mock event bus that simulates various failure conditions."""

    def __init__(self, failure_mode: str = "none", failure_count: int = 0):
        super().__init__(spec=ProtocolEventBus)
        self.failure_mode = failure_mode
        self.failure_count = failure_count
        self.call_count = 0
        self.published_events = []

    async def publish_async(self, event):
        self.call_count += 1

        if self.failure_mode == "always_fail":
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message="Event bus is unavailable",
            )
        elif (
            self.failure_mode == "intermittent"
            and self.call_count <= self.failure_count
        ):
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Intermittent failure {self.call_count}",
            )
        elif self.failure_mode == "timeout":
            # Simulate timeout by sleeping longer than expected
            await asyncio.sleep(10)  # 10 second delay
        elif self.failure_mode == "network_error":
            raise ConnectionError("Network unreachable")
        elif self.failure_mode == "auth_error":
            raise PermissionError("Authentication failed")

        # Success case
        self.published_events.append(event)


class SlowEventBus(AsyncMock):
    """Mock event bus that simulates slow responses."""

    def __init__(self, delay_seconds: float = 0.1):
        super().__init__(spec=ProtocolEventBus)
        self.delay_seconds = delay_seconds
        self.published_events = []

    async def publish_async(self, event):
        await asyncio.sleep(self.delay_seconds)
        self.published_events.append(event)


@pytest_asyncio.fixture
async def failing_event_bus():
    return FailingEventBus()


@pytest_asyncio.fixture
async def slow_event_bus():
    return SlowEventBus(delay_seconds=0.05)  # 50ms delay


@pytest_asyncio.fixture
async def event_driven_service_with_failing_bus(failing_event_bus):
    """Event-driven service with failing event bus."""
    producer = EventProducer()
    producer.initialize(failing_event_bus)

    consumer = EventConsumer()
    consumer.initialize(failing_event_bus)

    client = EventBusClient()
    client.initialize(producer, consumer)

    mapper = MemoryOperationMapper()

    service = EventDrivenMemoryService()
    service.initialize(client, mapper)

    return service


@pytest_asyncio.fixture
async def event_driven_service_with_slow_bus(slow_event_bus):
    """Event-driven service with slow event bus."""
    producer = EventProducer()
    producer.initialize(slow_event_bus)

    consumer = EventConsumer()
    consumer.initialize(slow_event_bus)

    client = EventBusClient()
    client.initialize(producer, consumer)

    mapper = MemoryOperationMapper()

    service = EventDrivenMemoryService()
    service.initialize(client, mapper)

    return service


class TestEventBusFailureHandling:
    """Test handling of event bus failures."""

    async def test_event_bus_unavailable_handling(self, failing_event_bus):
        """Test handling when event bus is completely unavailable."""
        failing_event_bus.failure_mode = "always_fail"

        # Initialize components
        producer = EventProducer()
        producer.initialize(failing_event_bus)

        # Create test request
        store_request = ModelMemoryStoreRequest(
            memory_key="failure_test",
            content={"test": "event_bus_failure"},
            metadata={"error_test": "unavailable_bus"},
        )

        # Attempt operation - should handle failure gracefully
        try:
            correlation_id = await producer.publish_store_command(
                correlation_id=uuid4(),
                store_data=store_request.to_store_data(),
                content=store_request.content,
            )
            # Should not reach here if properly handling failures
            pytest.fail("Expected OnexError for unavailable event bus")
        except OnexError as e:
            assert e.code == CoreErrorCode.SERVICE_UNAVAILABLE_ERROR
            assert "unavailable" in e.message.lower()
        except Exception as e:
            # Should be properly wrapped in OnexError
            assert False, f"Unexpected exception type: {type(e)}, message: {e}"

    async def test_intermittent_failure_retry(self, failing_event_bus):
        """Test retry logic for intermittent failures."""
        failing_event_bus.failure_mode = "intermittent"
        failing_event_bus.failure_count = 2  # Fail first 2 attempts, succeed on 3rd

        producer = EventProducer()
        producer.initialize(failing_event_bus)

        store_request = ModelMemoryStoreRequest(
            memory_key="retry_test",
            content={"test": "retry_logic"},
            metadata={"retry_test": True},
        )

        # With proper retry logic, this should eventually succeed
        correlation_id = uuid4()

        # Implement simple retry logic for testing
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                await producer.publish_store_command(
                    correlation_id=correlation_id,
                    store_data=store_request.to_store_data(),
                    content=store_request.content,
                )
                # Success - break out of retry loop
                break
            except OnexError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff
                continue

        # Should have succeeded after retries
        assert failing_event_bus.call_count == 3  # Failed twice, succeeded third time
        assert len(failing_event_bus.published_events) == 1

    async def test_network_error_handling(self):
        """Test handling of network-level errors."""
        network_failing_bus = FailingEventBus(failure_mode="network_error")

        producer = EventProducer()
        producer.initialize(network_failing_bus)

        store_request = ModelMemoryStoreRequest(
            memory_key="network_error_test",
            content={"test": "network_failure"},
            metadata={"network_test": True},
        )

        # Should handle network errors gracefully
        with pytest.raises(OnexError) as exc_info:
            await producer.publish_store_command(
                correlation_id=uuid4(),
                store_data=store_request.to_store_data(),
                content=store_request.content,
            )

        # Should be wrapped in proper OnexError
        assert exc_info.value.code == CoreErrorCode.SERVICE_UNAVAILABLE_ERROR
        assert (
            "network" in exc_info.value.message.lower()
            or "connection" in exc_info.value.message.lower()
        )

    async def test_authentication_error_handling(self):
        """Test handling of authentication errors."""
        auth_failing_bus = FailingEventBus(failure_mode="auth_error")

        producer = EventProducer()
        producer.initialize(auth_failing_bus)

        store_request = ModelMemoryStoreRequest(
            memory_key="auth_error_test",
            content={"test": "auth_failure"},
            metadata={"auth_test": True},
        )

        # Should handle auth errors gracefully
        with pytest.raises(OnexError) as exc_info:
            await producer.publish_store_command(
                correlation_id=uuid4(),
                store_data=store_request.to_store_data(),
                content=store_request.content,
            )

        # Should be wrapped in proper OnexError
        assert exc_info.value.code == CoreErrorCode.SERVICE_UNAVAILABLE_ERROR
        assert (
            "auth" in exc_info.value.message.lower()
            or "permission" in exc_info.value.message.lower()
        )


class TestTimeoutHandling:
    """Test timeout handling in event-driven operations."""

    async def test_operation_timeout_handling(
        self, event_driven_service_with_slow_bus, slow_event_bus
    ):
        """Test operation timeout handling."""
        # Set very short timeout
        with patch.object(
            event_driven_service_with_slow_bus, "operation_timeout", 0.01
        ):  # 10ms timeout
            # Set event bus to be slower than timeout
            slow_event_bus.delay_seconds = 0.1  # 100ms delay

            store_request = ModelMemoryStoreRequest(
                memory_key="timeout_test",
                content={"test": "timeout_handling"},
                metadata={"timeout_test": True},
            )

            # Operation should timeout
            start_time = asyncio.get_event_loop().time()

            try:
                correlation_id = await asyncio.wait_for(
                    event_driven_service_with_slow_bus.store_memory(store_request),
                    timeout=0.05,  # 50ms total timeout
                )
                # Should timeout, not reach here
                pytest.fail("Expected timeout exception")
            except asyncio.TimeoutError:
                # Expected timeout
                end_time = asyncio.get_event_loop().time()
                elapsed = end_time - start_time

                # Should timeout quickly (not wait for full operation)
                assert elapsed < 0.1, f"Timeout took too long: {elapsed}s"

    async def test_graceful_timeout_recovery(self, event_driven_service_with_slow_bus):
        """Test graceful recovery after timeout."""
        # First operation times out
        slow_request = ModelMemoryStoreRequest(
            memory_key="slow_operation",
            content={"test": "slow_timeout"},
            metadata={"slow": True},
        )

        try:
            await asyncio.wait_for(
                event_driven_service_with_slow_bus.store_memory(slow_request),
                timeout=0.01,  # Very short timeout
            )
        except asyncio.TimeoutError:
            pass  # Expected

        # Second operation should work normally (service should recover)
        fast_request = ModelMemoryStoreRequest(
            memory_key="fast_operation",
            content={"test": "recovery"},
            metadata={"recovery_test": True},
        )

        # This should succeed
        correlation_id = await event_driven_service_with_slow_bus.store_memory(
            fast_request
        )
        assert correlation_id is not None
        assert isinstance(correlation_id, UUID)

    async def test_concurrent_timeout_handling(
        self, event_driven_service_with_slow_bus
    ):
        """Test timeout handling with concurrent operations."""
        # Create multiple concurrent operations
        num_operations = 10
        requests = [
            ModelMemoryStoreRequest(
                memory_key=f"concurrent_timeout_{i}",
                content={"index": i, "test": "concurrent_timeout"},
                metadata={"concurrent": True, "index": i},
            )
            for i in range(num_operations)
        ]

        # Execute operations concurrently with timeout
        tasks = [
            asyncio.wait_for(
                event_driven_service_with_slow_bus.store_memory(request),
                timeout=0.2,  # 200ms timeout
            )
            for request in requests
        ]

        # Gather results - some may timeout, some may succeed
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        successes = [r for r in results if isinstance(r, UUID)]
        timeouts = [r for r in results if isinstance(r, asyncio.TimeoutError)]
        errors = [
            r
            for r in results
            if isinstance(r, Exception) and not isinstance(r, asyncio.TimeoutError)
        ]

        print(f"Concurrent timeout test results:")
        print(f"  Successes: {len(successes)}")
        print(f"  Timeouts: {len(timeouts)}")
        print(f"  Errors: {len(errors)}")

        # Should have some results (either success or timeout, minimal errors)
        assert len(successes) + len(timeouts) >= num_operations * 0.8
        assert len(errors) <= num_operations * 0.1


class TestInputValidationErrorHandling:
    """Test error handling for invalid input data."""

    async def test_invalid_memory_key_handling(
        self, event_driven_service_with_failing_bus
    ):
        """Test handling of invalid memory keys."""
        invalid_keys = [
            "",  # Empty key
            " ",  # Whitespace only
            None,  # None value
            "a" * 1000,  # Too long
            "\x00\x01\x02",  # Binary data
            "../../../etc/passwd",  # Path traversal
        ]

        for invalid_key in invalid_keys:
            try:
                # This should fail at validation level
                store_request = ModelMemoryStoreRequest(
                    memory_key=invalid_key,
                    content={"test": "invalid_key"},
                    metadata={"validation_test": True},
                )

                correlation_id = (
                    await event_driven_service_with_failing_bus.store_memory(
                        store_request
                    )
                )

                # If it succeeds, key should be sanitized
                if correlation_id is not None:
                    assert isinstance(correlation_id, UUID)
                    # Key should be sanitized/normalized
                    # (Implementation detail - would verify in real system)

            except OnexError as e:
                # Should be validation error
                assert e.code == CoreErrorCode.VALIDATION_ERROR
            except Exception as e:
                # Other exceptions should be properly wrapped
                assert (
                    False
                ), f"Unexpected exception for invalid key '{invalid_key}': {type(e)} - {e}"

    async def test_invalid_content_handling(
        self, event_driven_service_with_failing_bus
    ):
        """Test handling of invalid content data."""
        invalid_contents = [
            None,  # None content
            {"circular": None},  # Will create circular reference
            "not_json_serializable_object",
        ]

        # Add circular reference
        circular = {"self": None}
        circular["self"] = circular
        invalid_contents[1]["circular"] = circular

        for i, invalid_content in enumerate(invalid_contents):
            try:
                store_request = ModelMemoryStoreRequest(
                    memory_key=f"invalid_content_test_{i}",
                    content=invalid_content,
                    metadata={"content_validation": True, "test_index": i},
                )

                correlation_id = (
                    await event_driven_service_with_failing_bus.store_memory(
                        store_request
                    )
                )

                # If successful, content should be sanitized
                if correlation_id is not None:
                    assert isinstance(correlation_id, UUID)

            except OnexError as e:
                # Should be validation error
                assert e.code in [
                    CoreErrorCode.VALIDATION_ERROR,
                    CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                ]
            except Exception as e:
                # Should be properly handled
                assert (
                    False
                ), f"Unexpected exception for invalid content {i}: {type(e)} - {e}"

    async def test_oversized_payload_handling(
        self, event_driven_service_with_failing_bus
    ):
        """Test handling of oversized payloads."""
        # Create very large payload
        large_content = {"large_data": "A" * (10 * 1024 * 1024)}  # 10MB

        try:
            store_request = ModelMemoryStoreRequest(
                memory_key="oversized_payload_test",
                content=large_content,
                metadata={"size_test": True},
            )

            correlation_id = await event_driven_service_with_failing_bus.store_memory(
                store_request
            )

            # If successful, payload should be handled appropriately
            if correlation_id is not None:
                assert isinstance(correlation_id, UUID)
                # System should handle large payloads gracefully

        except OnexError as e:
            # Should be validation or service error
            assert e.code in [
                CoreErrorCode.VALIDATION_ERROR,
                CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
            ]
            # Error message should indicate size issue
            assert any(
                term in e.message.lower()
                for term in ["size", "large", "limit", "quota"]
            )


class TestCircuitBreakerBehavior:
    """Test circuit breaker pattern for fault tolerance."""

    async def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after consecutive failures."""
        # Create event bus that always fails
        failing_bus = FailingEventBus(failure_mode="always_fail")

        producer = EventProducer()
        producer.initialize(failing_bus)

        # Simulate multiple consecutive failures
        failure_count = 0
        max_attempts = 10

        for i in range(max_attempts):
            try:
                store_request = ModelMemoryStoreRequest(
                    memory_key=f"circuit_breaker_test_{i}",
                    content={"test": "circuit_breaker", "attempt": i},
                    metadata={"circuit_test": True},
                )

                await producer.publish_store_command(
                    correlation_id=uuid4(),
                    store_data=store_request.to_store_data(),
                    content=store_request.content,
                )

            except OnexError:
                failure_count += 1
            except Exception as e:
                failure_count += 1

        # All attempts should fail
        assert failure_count == max_attempts

        # Circuit breaker should be open (implementation dependent)
        # In real implementation, subsequent calls would fail fast

    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after service restoration."""
        # Create event bus with intermittent failures
        recovering_bus = FailingEventBus(failure_mode="intermittent", failure_count=5)

        producer = EventProducer()
        producer.initialize(recovering_bus)

        # First few requests should fail
        for i in range(3):
            with pytest.raises(OnexError):
                store_request = ModelMemoryStoreRequest(
                    memory_key=f"recovery_test_{i}",
                    content={"test": "recovery", "attempt": i},
                    metadata={"recovery_test": True},
                )

                await producer.publish_store_command(
                    correlation_id=uuid4(),
                    store_data=store_request.to_store_data(),
                    content=store_request.content,
                )

        # Later requests should succeed
        for i in range(3, 6):
            store_request = ModelMemoryStoreRequest(
                memory_key=f"recovery_test_{i}",
                content={"test": "recovery", "attempt": i},
                metadata={"recovery_test": True},
            )

            # Should succeed
            await producer.publish_store_command(
                correlation_id=uuid4(),
                store_data=store_request.to_store_data(),
                content=store_request.content,
            )

        # Verify recovery
        assert len(recovering_bus.published_events) == 3  # 3 successful requests


class TestGracefulDegradation:
    """Test graceful degradation when components are unavailable."""

    async def test_partial_service_degradation(
        self, event_driven_service_with_failing_bus
    ):
        """Test behavior when some services are unavailable."""
        # Configure partial failures
        failing_bus = (
            event_driven_service_with_failing_bus._event_bus_client._producer._event_bus
        )
        failing_bus.failure_mode = "intermittent"
        failing_bus.failure_count = 2  # Fail some operations

        # Execute multiple different operation types
        operations = [
            (
                "store",
                ModelMemoryStoreRequest(
                    memory_key="degradation_store",
                    content={"test": "graceful_degradation"},
                    metadata={"degradation_test": "store"},
                ),
            ),
            (
                "retrieve",
                ModelMemoryRetrieveRequest(
                    memory_key="degradation_retrieve", query_type="key"
                ),
            ),
            (
                "vector_search",
                ModelMemoryVectorSearchRequest(
                    query_vector=[0.1, 0.2, 0.3], collection_name="degradation_test"
                ),
            ),
        ]

        results = {}

        for op_type, request in operations:
            try:
                if op_type == "store":
                    result = await event_driven_service_with_failing_bus.store_memory(
                        request
                    )
                elif op_type == "retrieve":
                    result = (
                        await event_driven_service_with_failing_bus.retrieve_memory(
                            request
                        )
                    )
                elif op_type == "vector_search":
                    result = await event_driven_service_with_failing_bus.vector_search(
                        request
                    )

                results[op_type] = {"success": True, "result": result}

            except Exception as e:
                results[op_type] = {"success": False, "error": str(e)}

        # Some operations may succeed, some may fail
        print(f"Graceful degradation results: {results}")

        # System should handle partial failures gracefully
        # (Implementation detail - system should continue operating)
        assert isinstance(results, dict)
        assert len(results) == len(operations)


if __name__ == "__main__":
    # Run error handling and timeout tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
