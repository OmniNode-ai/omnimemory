# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerIntent with mocked adapter.

This module provides comprehensive testing for the HandlerIntent handler,
using mock implementations to test the handler logic in isolation without
requiring real database connections.

Test Categories:
    1. TestLifecycle: Handler initialization and shutdown
    2. TestCircuitBreaker: Circuit breaker state transitions
    3. TestOperations: Core operation delegation tests
    4. TestIntrospection: Health check and describe tests
    5. TestProperties: Handler property tests

Usage:
    pytest tests/handlers/test_handler_intent.py -v
    pytest tests/handlers/test_handler_intent.py -v -k "lifecycle"
    pytest tests/handlers/test_handler_intent.py -v -k "circuit"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1536.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnimemory.handlers.adapters.models import (
    ModelIntentClassificationOutput,
    ModelIntentDistributionResult,
    ModelIntentGraphHealth,
    ModelIntentQueryResult,
    ModelIntentRecord,
    ModelIntentStorageResult,
)
from omnimemory.handlers.handler_intent import (
    CircuitBreakerOpenError,
    HandlerIntent,
    ModelHandlerIntentMetadata,
)
from omnimemory.utils.concurrency import CircuitBreakerState

# =============================================================================
# Mock Container
# =============================================================================


def create_mock_container() -> MagicMock:
    """Create a mock ModelONEXContainer for testing."""
    return MagicMock(spec=["__class__"])


# =============================================================================
# Mock Adapter
# =============================================================================


class MockAdapterIntentGraph:
    """Mock adapter for testing HandlerIntent without real database connections.

    Provides controllable behavior for testing handler logic.
    """

    def __init__(
        self,
        *,
        store_result: ModelIntentStorageResult | None = None,
        query_result: ModelIntentQueryResult | None = None,
        distribution_result: ModelIntentDistributionResult | None = None,
        health_result: ModelIntentGraphHealth | None = None,
        fail_on_store: bool = False,
        fail_on_query: bool = False,
        fail_on_distribution: bool = False,
        fail_on_health: bool = False,
    ) -> None:
        self._initialized = False
        self._store_result = store_result
        self._query_result = query_result
        self._distribution_result = distribution_result
        self._health_result = health_result
        self._fail_on_store = fail_on_store
        self._fail_on_query = fail_on_query
        self._fail_on_distribution = fail_on_distribution
        self._fail_on_health = fail_on_health
        self.store_intent_calls: list[dict[str, object]] = []
        self.get_session_intents_calls: list[dict[str, object]] = []
        self.get_intent_distribution_calls: list[dict[str, object]] = []
        self.health_check_calls: int = 0
        self.shutdown_calls: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: dict[str, object] | None = None,
    ) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._initialized = False

    async def store_intent(
        self,
        session_id: str,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: UUID,
        user_context: str = "",
    ) -> ModelIntentStorageResult:
        self.store_intent_calls.append(
            {
                "session_id": session_id,
                "intent_data": intent_data,
                "correlation_id": correlation_id,
                "user_context": user_context,
            }
        )

        if self._fail_on_store:
            raise RuntimeError("Simulated store failure")

        if self._store_result:
            return self._store_result

        return ModelIntentStorageResult(
            status="success",
            intent_id=uuid4(),
            session_id=session_id,
            created=True,
            execution_time_ms=10.5,
        )

    async def get_session_intents(
        self,
        session_id: str,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> ModelIntentQueryResult:
        self.get_session_intents_calls.append(
            {
                "session_id": session_id,
                "min_confidence": min_confidence,
                "limit": limit,
            }
        )

        if self._fail_on_query:
            raise RuntimeError("Simulated query failure")

        if self._query_result:
            return self._query_result

        return ModelIntentQueryResult(
            status="success",
            intents=[],
            total_count=0,
            execution_time_ms=5.0,
        )

    async def get_intent_distribution(
        self,
        time_range_hours: int = 24,
    ) -> ModelIntentDistributionResult:
        self.get_intent_distribution_calls.append(
            {"time_range_hours": time_range_hours}
        )

        if self._fail_on_distribution:
            raise RuntimeError("Simulated distribution query failure")

        if self._distribution_result:
            return self._distribution_result

        return ModelIntentDistributionResult(
            status="success",
            distribution={"debugging": 10, "code_generation": 5},
            total_intents=15,
            time_range_hours=time_range_hours,
            execution_time_ms=8.0,
        )

    async def health_check(self) -> ModelIntentGraphHealth:
        self.health_check_calls += 1

        if self._fail_on_health:
            raise RuntimeError("Simulated health check failure")

        if self._health_result:
            return self._health_result

        return ModelIntentGraphHealth(
            is_healthy=True,
            initialized=True,
            handler_healthy=True,
            session_count=5,
            intent_count=15,
            last_check_timestamp=datetime.now(UTC),
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock container for testing."""
    return create_mock_container()


@pytest.fixture
def mock_adapter() -> MockAdapterIntentGraph:
    """Create a mock adapter for testing."""
    return MockAdapterIntentGraph()


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerIntent:
    """Create an uninitialized handler for testing."""
    return HandlerIntent(mock_container)


@pytest.fixture
def intent_data() -> ModelIntentClassificationOutput:
    """Create sample intent classification data."""
    return ModelIntentClassificationOutput(
        intent_category="debugging",
        confidence=0.92,
        keywords=["error", "traceback"],
    )


@pytest.fixture
def correlation_id() -> UUID:
    """Create a sample correlation ID."""
    return uuid4()


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for handler initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_creates_adapter(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Initialize should create and configure the adapter."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
            )

            assert handler.is_initialized is True
            assert mock_adapter.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Multiple initialize calls should not re-initialize."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ) as mock_class:
            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.initialize(connection_uri="bolt://localhost:7687")

            # AdapterIntentGraph constructor should only be called once
            assert mock_class.call_count == 1
            assert handler.is_initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_adapter(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Shutdown should close adapter and release resources."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")
            assert handler.is_initialized is True

            await handler.shutdown()

            assert handler.is_initialized is False
            assert mock_adapter.shutdown_calls == 1

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Multiple shutdown calls should be safe."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            await handler.shutdown()
            await handler.shutdown()
            await handler.shutdown()

            # Only one shutdown should actually clean up (when initialized)
            assert handler.is_initialized is False
            assert mock_adapter.shutdown_calls == 1

    @pytest.mark.asyncio
    async def test_shutdown_on_uninitialized_handler_is_noop(
        self, handler: HandlerIntent
    ) -> None:
        """Shutdown on uninitialized handler should be a no-op."""
        # Should not raise
        await handler.shutdown()
        assert handler.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_with_options(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Initialize should pass options to adapter."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={
                    "timeout_seconds": 60.0,
                    "circuit_breaker_threshold": 10,
                },
            )

            assert handler.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_auth(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Initialize should pass auth credentials to adapter."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                auth=("neo4j", "password"),
            )

            assert handler.is_initialized is True


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(
        self, handler: HandlerIntent
    ) -> None:
        """Circuit breaker should open after threshold failures."""
        # Create adapter that always fails
        failing_adapter = MockAdapterIntentGraph(fail_on_store=True)

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=failing_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={
                    "circuit_breaker_threshold": 3,
                    "circuit_breaker_reset_timeout": 60.0,
                },
            )

            intent_data = ModelIntentClassificationOutput(
                intent_category="test",
                confidence=0.9,
                keywords=[],
            )

            # Make enough calls to trip the circuit breaker
            for i in range(3):
                with pytest.raises(RuntimeError, match="Simulated store failure"):
                    await handler.store_intent(
                        session_id=f"session_{i}",
                        intent_data=intent_data,
                        correlation_id=uuid4(),
                    )

            # Next call should raise CircuitBreakerOpenError
            with pytest.raises(CircuitBreakerOpenError):
                await handler.store_intent(
                    session_id="session_final",
                    intent_data=intent_data,
                    correlation_id=uuid4(),
                )

    @pytest.mark.asyncio
    async def test_circuit_breaker_raises_when_open(
        self, handler: HandlerIntent
    ) -> None:
        """Operations should raise CircuitBreakerOpenError when circuit is open."""
        mock_adapter = MockAdapterIntentGraph()

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={"circuit_breaker_threshold": 1},
            )

            # Manually set circuit breaker to open state via internal access
            assert handler._circuit_breaker is not None
            handler._circuit_breaker.state = CircuitBreakerState.OPEN
            handler._circuit_breaker.last_failure_time = datetime.now(UTC)

            intent_data = ModelIntentClassificationOutput(
                intent_category="test",
                confidence=0.9,
                keywords=[],
            )

            # All operations should raise when circuit is open
            with pytest.raises(CircuitBreakerOpenError):
                await handler.store_intent(
                    session_id="session_1",
                    intent_data=intent_data,
                    correlation_id=uuid4(),
                )

            with pytest.raises(CircuitBreakerOpenError):
                await handler.query_session(session_id="session_1")

            with pytest.raises(CircuitBreakerOpenError):
                await handler.query_distribution(time_range_hours=24)

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """Successful operations should record success in circuit breaker."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={"circuit_breaker_threshold": 5},
            )

            intent_data = ModelIntentClassificationOutput(
                intent_category="debugging",
                confidence=0.9,
                keywords=[],
            )

            # Successful operation
            result = await handler.store_intent(
                session_id="session_1",
                intent_data=intent_data,
                correlation_id=uuid4(),
            )

            assert result.status == "success"
            assert handler._circuit_breaker is not None
            assert handler._circuit_breaker.state == CircuitBreakerState.CLOSED
            assert handler._circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_failure_count_on_success(
        self, handler: HandlerIntent
    ) -> None:
        """Success after failures should reset failure count."""
        # Adapter that fails twice then succeeds
        call_count = 0

        class PartiallyFailingAdapter(MockAdapterIntentGraph):
            async def store_intent(
                self, *args: object, **kwargs: object
            ) -> ModelIntentStorageResult:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RuntimeError("Temporary failure")
                return await super().store_intent(*args, **kwargs)  # type: ignore[arg-type]

        adapter = PartiallyFailingAdapter()

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={"circuit_breaker_threshold": 5},
            )

            intent_data = ModelIntentClassificationOutput(
                intent_category="test",
                confidence=0.9,
                keywords=[],
            )

            # Two failures
            for _ in range(2):
                with pytest.raises(RuntimeError):
                    await handler.store_intent(
                        session_id="session",
                        intent_data=intent_data,
                        correlation_id=uuid4(),
                    )

            assert handler._circuit_breaker is not None
            assert handler._circuit_breaker.failure_count == 2

            # Third call succeeds
            result = await handler.store_intent(
                session_id="session",
                intent_data=intent_data,
                correlation_id=uuid4(),
            )

            assert result.status == "success"
            assert handler._circuit_breaker.failure_count == 0


# =============================================================================
# Operation Tests
# =============================================================================


class TestOperations:
    """Tests for core handler operations."""

    @pytest.mark.asyncio
    async def test_store_intent_delegates_to_adapter(
        self,
        handler: HandlerIntent,
        mock_adapter: MockAdapterIntentGraph,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: UUID,
    ) -> None:
        """store_intent should delegate to adapter correctly."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.store_intent(
                session_id="session_123",
                intent_data=intent_data,
                correlation_id=correlation_id,
                user_context="test context",
            )

            assert result.status == "success"
            assert len(mock_adapter.store_intent_calls) == 1

            call = mock_adapter.store_intent_calls[0]
            assert call["session_id"] == "session_123"
            assert call["intent_data"] == intent_data
            assert call["correlation_id"] == correlation_id
            assert call["user_context"] == "test context"

    @pytest.mark.asyncio
    async def test_store_intent_returns_error_when_uninitialized(
        self,
        handler: HandlerIntent,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: UUID,
    ) -> None:
        """store_intent on uninitialized handler should return error result."""
        result = await handler.store_intent(
            session_id="session_123",
            intent_data=intent_data,
            correlation_id=correlation_id,
        )

        assert result.status == "error"
        assert result.error_message is not None
        assert "not initialized" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_query_session_delegates_to_adapter(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """query_session should delegate to adapter correctly."""
        # Set up expected result
        expected_intents = [
            ModelIntentRecord(
                intent_id=uuid4(),
                intent_category="debugging",
                confidence=0.92,
                keywords=["error"],
                created_at_utc=datetime.now(UTC),
            )
        ]
        mock_adapter._query_result = ModelIntentQueryResult(
            status="success",
            intents=expected_intents,
            total_count=1,
            execution_time_ms=5.0,
        )

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.query_session(
                session_id="session_123",
                min_confidence=0.5,
                limit=10,
            )

            assert result.status == "success"
            assert result.total_count == 1
            assert len(mock_adapter.get_session_intents_calls) == 1

            call = mock_adapter.get_session_intents_calls[0]
            assert call["session_id"] == "session_123"
            assert call["min_confidence"] == 0.5
            assert call["limit"] == 10

    @pytest.mark.asyncio
    async def test_query_session_returns_error_when_uninitialized(
        self, handler: HandlerIntent
    ) -> None:
        """query_session on uninitialized handler should return error result."""
        result = await handler.query_session(session_id="session_123")

        assert result.status == "error"
        assert result.error_message is not None
        assert "not initialized" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_query_distribution_delegates_to_adapter(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """query_distribution should delegate to adapter correctly."""
        expected_distribution = {
            "debugging": 50,
            "code_generation": 30,
            "explanation": 20,
        }
        mock_adapter._distribution_result = ModelIntentDistributionResult(
            status="success",
            distribution=expected_distribution,
            total_intents=100,
            time_range_hours=48,
            execution_time_ms=12.0,
        )

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.query_distribution(time_range_hours=48)

            assert result.status == "success"
            assert result.distribution == expected_distribution
            assert result.total_intents == 100
            assert len(mock_adapter.get_intent_distribution_calls) == 1

            call = mock_adapter.get_intent_distribution_calls[0]
            assert call["time_range_hours"] == 48

    @pytest.mark.asyncio
    async def test_query_distribution_returns_error_when_uninitialized(
        self, handler: HandlerIntent
    ) -> None:
        """query_distribution on uninitialized handler should return error result."""
        result = await handler.query_distribution(time_range_hours=24)

        assert result.status == "error"
        assert result.error_message is not None
        assert "not initialized" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_query_distribution_uses_default_time_range(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """query_distribution should use default 24 hours if not specified."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            await handler.query_distribution()

            call = mock_adapter.get_intent_distribution_calls[0]
            assert call["time_range_hours"] == 24


# =============================================================================
# Introspection Tests
# =============================================================================


class TestIntrospection:
    """Tests for health check and describe functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_health_status(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """health_check should return detailed health status."""
        mock_adapter._health_result = ModelIntentGraphHealth(
            is_healthy=True,
            initialized=True,
            handler_healthy=True,
            session_count=10,
            intent_count=50,
            last_check_timestamp=datetime.now(UTC),
        )

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            health = await handler.health_check()

            assert health.is_healthy is True
            assert health.initialized is True
            assert health.handler_healthy is True
            assert health.session_count == 10
            assert health.intent_count == 50

    @pytest.mark.asyncio
    async def test_health_check_returns_unhealthy_when_uninitialized(
        self, handler: HandlerIntent
    ) -> None:
        """health_check on uninitialized handler should return unhealthy."""
        health = await handler.health_check()

        assert health.is_healthy is False
        assert health.initialized is False
        assert health.error_message is not None
        assert "not initialized" in health.error_message.lower()

    @pytest.mark.asyncio
    async def test_health_check_includes_circuit_breaker_status(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """health_check should include circuit breaker state."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={"circuit_breaker_threshold": 5},
            )

            # Set circuit breaker to open state
            assert handler._circuit_breaker is not None
            handler._circuit_breaker.state = CircuitBreakerState.OPEN
            handler._circuit_breaker.last_failure_time = datetime.now(UTC)

            health = await handler.health_check()

            # When circuit breaker is open, should report unhealthy
            assert health.is_healthy is False
            assert health.error_message is not None
            assert "circuit breaker" in health.error_message.lower()
            assert "open" in health.error_message.lower()

    @pytest.mark.asyncio
    async def test_health_check_handles_adapter_failure(
        self, handler: HandlerIntent
    ) -> None:
        """health_check should capture adapter health check failures."""
        failing_adapter = MockAdapterIntentGraph(fail_on_health=True)

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=failing_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            health = await handler.health_check()

            assert health.is_healthy is False
            assert health.error_message is not None
            assert "health check failed" in health.error_message.lower()

    @pytest.mark.asyncio
    async def test_describe_returns_metadata(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """describe should return handler metadata."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            metadata = await handler.describe()

            assert isinstance(metadata, ModelHandlerIntentMetadata)
            assert metadata.handler_type == "intent"
            assert metadata.initialized is True
            assert metadata.adapter_type == "AdapterIntentGraph"
            assert "store_intent" in metadata.capabilities
            assert "query_session" in metadata.capabilities
            assert "query_distribution" in metadata.capabilities
            assert "health_check" in metadata.capabilities
            assert "describe" in metadata.capabilities

    @pytest.mark.asyncio
    async def test_describe_before_init(self, handler: HandlerIntent) -> None:
        """describe should work even before initialization."""
        metadata = await handler.describe()

        assert isinstance(metadata, ModelHandlerIntentMetadata)
        assert metadata.handler_type == "intent"
        assert metadata.initialized is False


# =============================================================================
# Properties Tests
# =============================================================================


class TestProperties:
    """Tests for handler properties."""

    def test_handler_type_returns_intent(self, handler: HandlerIntent) -> None:
        """handler_type property should return 'intent'."""
        assert handler.handler_type == "intent"

    def test_is_initialized_false_before_init(self, handler: HandlerIntent) -> None:
        """is_initialized should be False before initialization."""
        assert handler.is_initialized is False

    @pytest.mark.asyncio
    async def test_is_initialized_true_after_init(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """is_initialized should be True after initialization."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            assert handler.is_initialized is True

    @pytest.mark.asyncio
    async def test_is_initialized_false_after_shutdown(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """is_initialized should be False after shutdown."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.shutdown()

            assert handler.is_initialized is False

    def test_repr_before_init(self, handler: HandlerIntent) -> None:
        """__repr__ should work before initialization."""
        repr_str = repr(handler)
        assert "HandlerIntent" in repr_str
        assert "initialized=False" in repr_str

    @pytest.mark.asyncio
    async def test_repr_after_init(
        self, handler: HandlerIntent, mock_adapter: MockAdapterIntentGraph
    ) -> None:
        """__repr__ should reflect initialization state."""
        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            repr_str = repr(handler)
            assert "HandlerIntent" in repr_str
            assert "initialized=True" in repr_str


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling behavior."""

    @pytest.mark.asyncio
    async def test_store_intent_propagates_adapter_exception(
        self, handler: HandlerIntent, intent_data: ModelIntentClassificationOutput
    ) -> None:
        """store_intent should propagate adapter exceptions (and record failure)."""
        failing_adapter = MockAdapterIntentGraph(fail_on_store=True)

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=failing_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            with pytest.raises(RuntimeError, match="Simulated store failure"):
                await handler.store_intent(
                    session_id="session_123",
                    intent_data=intent_data,
                    correlation_id=uuid4(),
                )

            # Circuit breaker should record the failure
            assert handler._circuit_breaker is not None
            assert handler._circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_query_session_propagates_adapter_exception(
        self, handler: HandlerIntent
    ) -> None:
        """query_session should propagate adapter exceptions."""
        failing_adapter = MockAdapterIntentGraph(fail_on_query=True)

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=failing_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            with pytest.raises(RuntimeError, match="Simulated query failure"):
                await handler.query_session(session_id="session_123")

    @pytest.mark.asyncio
    async def test_query_distribution_propagates_adapter_exception(
        self, handler: HandlerIntent
    ) -> None:
        """query_distribution should propagate adapter exceptions."""
        failing_adapter = MockAdapterIntentGraph(fail_on_distribution=True)

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=failing_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            with pytest.raises(
                RuntimeError, match="Simulated distribution query failure"
            ):
                await handler.query_distribution(time_range_hours=24)


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests combining multiple operations."""

    @pytest.mark.asyncio
    async def test_full_workflow_store_and_query(
        self,
        handler: HandlerIntent,
        mock_adapter: MockAdapterIntentGraph,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: UUID,
    ) -> None:
        """Test storing intent then querying it back."""
        # Set up query to return the stored intent
        stored_intent = ModelIntentRecord(
            intent_id=uuid4(),
            intent_category=intent_data.intent_category,
            confidence=intent_data.confidence,
            keywords=intent_data.keywords,
            created_at_utc=datetime.now(UTC),
            correlation_id=correlation_id,
        )
        mock_adapter._query_result = ModelIntentQueryResult(
            status="success",
            intents=[stored_intent],
            total_count=1,
            execution_time_ms=5.0,
        )

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            return_value=mock_adapter,
        ):
            await handler.initialize(connection_uri="bolt://localhost:7687")

            # Store intent
            store_result = await handler.store_intent(
                session_id="session_123",
                intent_data=intent_data,
                correlation_id=correlation_id,
            )
            assert store_result.status == "success"

            # Query it back
            query_result = await handler.query_session(
                session_id="session_123",
                min_confidence=0.5,
            )
            assert query_result.status == "success"
            assert query_result.total_count == 1
            assert query_result.intents[0].intent_category == "debugging"

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(self, handler: HandlerIntent) -> None:
        """Handler should be reusable after shutdown."""
        mock_adapter1 = MockAdapterIntentGraph()
        mock_adapter2 = MockAdapterIntentGraph()
        adapters = [mock_adapter1, mock_adapter2]
        adapter_iter = iter(adapters)

        def create_adapter(*args: object, **kwargs: object) -> MockAdapterIntentGraph:
            return next(adapter_iter)

        with patch(
            "omnimemory.handlers.handler_intent.AdapterIntentGraph",
            side_effect=create_adapter,
        ):
            # First initialization
            await handler.initialize(connection_uri="bolt://localhost:7687")
            assert handler.is_initialized is True

            # Shutdown
            await handler.shutdown()
            assert handler.is_initialized is False

            # Re-initialize with new adapter
            await handler.initialize(connection_uri="bolt://localhost:7687")
            assert handler.is_initialized is True
            assert mock_adapter2.is_initialized is True
