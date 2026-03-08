# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Boundary integration tests for HandlerIntentStorageAdapter.

This module tests the integration boundary between HandlerIntentStorageAdapter
and AdapterIntentGraph, verifying that:
1. Requests are correctly routed to the appropriate adapter methods
2. Adapter call arguments are correct
3. Response models are correctly populated from adapter results
4. Error cases are properly handled

These are NOT full Memgraph integration tests - the AdapterIntentGraph is mocked
to isolate the handler behavior. For real database integration tests, see
tests with @pytest.mark.memgraph marker.

Test Categories:
    - Store operation: Verify store_intent() calls and response mapping
    - Get session operation: Verify get_session_intents() calls and response mapping
    - Get distribution operation: Verify get_intent_distribution() calls and response
    - Error handling: Verify error states (not initialized, unknown operation, exceptions, timing)
    - Correlation ID: Verify auto-generation when not provided

Prerequisites:
    - No external services required (all adapters mocked)

Related Tickets:
    - OMN-1510: Integration test for intent_storage_effect node
    - OMN-1514: intent_storage_effect ONEX node (implementation)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import UUID

import pytest

# Dependency availability check
_DEPENDENCIES_AVAILABLE = False
_SKIP_REASON = "Required dependencies not installed"

try:
    from omnibase_core.enums.intelligence import EnumIntentCategory
    from omnibase_core.models.intelligence import (
        ModelIntentClassificationOutput,
        ModelIntentStorageResult,
    )
    from omnibase_core.models.intelligence import (
        ModelIntentQueryResult as CoreIntentQueryResult,
    )
    from omnibase_core.models.intelligence import (
        ModelIntentRecord as CoreIntentRecord,
    )

    from omnimemory.handlers.adapters.models import (
        ModelAdapterIntentGraphConfig,
        ModelIntentDistributionResult,
    )
    from omnimemory.models.utils import ModelPIIDetectionResult, ModelPIIMatch, PIIType
    from omnimemory.nodes.node_intent_storage_effect import (
        HandlerIntentStorageAdapter,
        ModelIntentStorageRequest,
        ModelIntentStorageResponse,
    )
    from omnimemory.nodes.node_intent_storage_effect.adapters.adapter_intent_storage import (
        _DEFAULT_CIRCUIT_BREAKER_CONFIG,
    )

    _DEPENDENCIES_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as e:
    _SKIP_REASON = f"Required dependencies not available: {e}"

# Module-level pytest markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(not _DEPENDENCIES_AVAILABLE, reason=_SKIP_REASON),
]


# =============================================================================
# Test Constants
# =============================================================================

TEST_SESSION_ID = "test_session_abc123"
TEST_INTENT_ID = UUID("11111111-1111-1111-1111-111111111111")
TEST_INTENT_ID_2 = UUID("22222222-2222-2222-2222-222222222222")
TEST_CORRELATION_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
TEST_CREATED_AT = datetime(2025, 1, 25, 10, 30, 0, tzinfo=UTC)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def adapter_config() -> ModelAdapterIntentGraphConfig:
    """Create default adapter configuration."""
    return ModelAdapterIntentGraphConfig(
        timeout_seconds=30.0,
        session_node_label="Session",
        intent_node_label="Intent",
        relationship_type="HAD_INTENT",
        max_intents_per_session=100,
        default_confidence_threshold=0.0,
    )


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock AdapterIntentGraph with async methods."""
    adapter = MagicMock()
    adapter.store_intent = AsyncMock()
    adapter.get_session_intents = AsyncMock()
    adapter.get_intent_distribution = AsyncMock()
    adapter.initialize = AsyncMock()
    adapter.shutdown = AsyncMock()
    adapter.health_check = AsyncMock()
    return adapter


@pytest.fixture
def handler_with_mock(
    adapter_config: ModelAdapterIntentGraphConfig,
    mock_adapter: MagicMock,
) -> HandlerIntentStorageAdapter:
    """Create a handler with mock adapter injected.

    This bypasses the real initialize() flow and directly injects
    the mock adapter for testing.
    """
    handler = HandlerIntentStorageAdapter(config=adapter_config)
    handler._adapter = mock_adapter
    handler._initialized = True
    return handler


@pytest.fixture
def sample_intent_data() -> ModelIntentClassificationOutput:
    """Create sample intent classification data for testing."""
    return ModelIntentClassificationOutput(
        success=True,
        intent_category=EnumIntentCategory.DEBUGGING,
        confidence=0.92,
        keywords=["error", "traceback", "fix"],
    )


@pytest.fixture
def core_intent_record() -> CoreIntentRecord:
    """Create a core intent record as returned by the adapter."""
    return CoreIntentRecord(
        intent_id=TEST_INTENT_ID,
        session_id=TEST_SESSION_ID,
        intent_category=EnumIntentCategory.DEBUGGING,
        confidence=0.92,
        keywords=["error", "traceback"],
        created_at=TEST_CREATED_AT,
        correlation_id=TEST_CORRELATION_ID,
    )


# =============================================================================
# Store Operation Tests
# =============================================================================


class TestStoreOperation:
    """Tests for the store operation."""

    async def test_store_calls_adapter_with_correct_args(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify store_intent() is called with correct arguments."""
        # Arrange
        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=True,
            intent_id=TEST_INTENT_ID,
            created=True,
        )

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
            correlation_id=TEST_CORRELATION_ID,
        )

        # Act
        await handler_with_mock.execute(request)

        # Assert - adapter called exactly once with correct args
        mock_adapter.store_intent.assert_called_once()
        call_kwargs = mock_adapter.store_intent.call_args.kwargs
        assert call_kwargs["session_id"] == TEST_SESSION_ID
        assert call_kwargs["intent_data"] == sample_intent_data
        assert call_kwargs["correlation_id"] == str(TEST_CORRELATION_ID)

    async def test_store_returns_correct_response_model(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify store operation returns correctly populated response."""
        # Arrange
        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=True,
            intent_id=TEST_INTENT_ID,
            created=True,
        )

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
            correlation_id=TEST_CORRELATION_ID,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert - response model populated correctly
        assert isinstance(response, ModelIntentStorageResponse)
        assert response.status == "success"
        assert response.intent_id == TEST_INTENT_ID
        assert response.session_id == TEST_SESSION_ID
        assert response.created is True
        assert response.execution_time_ms > 0  # Handler adds its own timing

    async def test_store_generates_correlation_id_when_not_provided(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify correlation_id is auto-generated when not provided."""
        # Arrange
        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=True,
            intent_id=TEST_INTENT_ID,
            created=True,
        )

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
            # correlation_id intentionally omitted
        )

        # Act
        await handler_with_mock.execute(request)

        # Assert - a UUID was generated and passed
        mock_adapter.store_intent.assert_called_once()
        call_kwargs = mock_adapter.store_intent.call_args.kwargs
        generated_correlation_id = call_kwargs["correlation_id"]
        # correlation_id is now a string UUID
        assert isinstance(generated_correlation_id, str)
        # Verify it's a valid UUID (not the test constant)
        assert generated_correlation_id != str(TEST_CORRELATION_ID)

    async def test_store_propagates_user_context(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify user_context field is accepted (but no longer passed to adapter).

        Note: user_context is no longer passed to the adapter in omnibase-core 0.13.1.
        This test verifies the request is still accepted with user_context.
        """
        # Arrange
        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=True,
            intent_id=TEST_INTENT_ID,
            created=True,
        )

        user_context = "User is working on a Python web app"
        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
            user_context=user_context,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert - request still works, adapter called without user_context
        assert response.status == "success"
        mock_adapter.store_intent.assert_called_once()
        call_kwargs = mock_adapter.store_intent.call_args.kwargs
        assert "user_context" not in call_kwargs  # No longer passed

    async def test_store_redacts_pii_from_user_context(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify PII detection still runs on user_context.

        Note: user_context is no longer passed to the adapter in omnibase-core 0.13.1,
        but PII detection still runs and logs warnings for audit purposes.

        This test uses model_construct() to bypass the model's own PII
        validation, allowing us to test the handler's PII detection as a
        second line of defense.
        """
        # Arrange - setup PII detector mock to indicate PII was found
        original_context = "User email is test@example.com"
        sanitized_context = "User email is ***@***.***"

        pii_detection_result = ModelPIIDetectionResult(
            has_pii=True,
            matches=[
                ModelPIIMatch(
                    pii_type=PIIType.EMAIL,
                    value="test@example.com",
                    start_index=14,
                    end_index=30,
                    confidence=0.95,
                    masked_value="***@***.***",
                )
            ],
            sanitized_content=sanitized_context,
            pii_types_detected={PIIType.EMAIL},
            scan_duration_ms=1.5,
        )

        handler_with_mock._pii_detector.detect_pii = MagicMock(
            return_value=pii_detection_result
        )

        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=True,
            intent_id=TEST_INTENT_ID,
            created=True,
        )

        # Use model_construct to bypass model-level PII validation
        # This tests the handler's PII detection as a second line of defense
        request = ModelIntentStorageRequest.model_construct(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
            user_context=original_context,
            correlation_id=None,
            min_confidence=0.0,
            limit=100,
            time_range_hours=24,
        )

        # Act
        await handler_with_mock.execute(request)

        # Assert - PII detector was called with the original content
        # Use ANY for sensitivity_level as it's an implementation detail
        handler_with_mock._pii_detector.detect_pii.assert_called_once_with(
            original_context, sensitivity_level=ANY
        )

        # Assert - adapter was called (user_context no longer passed)
        mock_adapter.store_intent.assert_called_once()
        call_kwargs = mock_adapter.store_intent.call_args.kwargs
        assert "user_context" not in call_kwargs

    async def test_store_handles_adapter_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify error response when adapter returns error status."""
        # Arrange
        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=False,
            error_message="Database connection failed",
        )

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "error"
        assert response.error_message == "Database connection failed"


# =============================================================================
# Get Session Operation Tests
# =============================================================================


class TestGetSessionOperation:
    """Tests for the get_session operation."""

    async def test_get_session_calls_adapter_with_correct_args(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        core_intent_record: CoreIntentRecord,
    ) -> None:
        """Verify get_session_intents() is called with correct arguments."""
        # Arrange
        mock_adapter.get_session_intents.return_value = CoreIntentQueryResult(
            success=True,
            intents=[core_intent_record],
        )

        request = ModelIntentStorageRequest(
            operation="get_session",
            session_id=TEST_SESSION_ID,
            min_confidence=0.5,
            limit=50,
        )

        # Act
        await handler_with_mock.execute(request)

        # Assert
        mock_adapter.get_session_intents.assert_called_once_with(
            session_id=TEST_SESSION_ID,
            min_confidence=0.5,
            limit=50,
        )

    async def test_get_session_returns_correct_response_model(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        core_intent_record: CoreIntentRecord,
    ) -> None:
        """Verify get_session returns correctly populated response."""
        # Arrange
        mock_adapter.get_session_intents.return_value = CoreIntentQueryResult(
            success=True,
            intents=[core_intent_record],
        )

        request = ModelIntentStorageRequest(
            operation="get_session",
            session_id=TEST_SESSION_ID,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert isinstance(response, ModelIntentStorageResponse)
        assert response.status == "success"
        assert response.total_count == 1
        assert len(response.intents) == 1

        # Verify intent record mapping
        intent = response.intents[0]
        assert intent.intent_id == TEST_INTENT_ID
        assert intent.intent_category == "debugging"
        assert intent.confidence == 0.92
        assert intent.keywords == ["error", "traceback"]
        assert intent.correlation_id == TEST_CORRELATION_ID

    async def test_get_session_handles_not_found(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
    ) -> None:
        """Verify no_results status when adapter returns empty results."""
        # Arrange - adapter returns success with no intents
        mock_adapter.get_session_intents.return_value = CoreIntentQueryResult(
            success=True,
            intents=[],
        )

        request = ModelIntentStorageRequest(
            operation="get_session",
            session_id="nonexistent_session",
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "no_results"
        assert response.error_message is None

    async def test_get_session_handles_no_results(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
    ) -> None:
        """Verify no_results status when adapter returns empty list."""
        # Arrange - adapter returns success=True with empty intents
        mock_adapter.get_session_intents.return_value = CoreIntentQueryResult(
            success=True,
            intents=[],
        )

        request = ModelIntentStorageRequest(
            operation="get_session",
            session_id=TEST_SESSION_ID,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "no_results"
        assert response.intents == []
        assert response.total_count == 0

    async def test_get_session_without_session_id_returns_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
    ) -> None:
        """Verify error response when session_id is missing for get_session operation."""
        # Arrange - use model_construct to bypass Pydantic validation for testing
        request = ModelIntentStorageRequest.model_construct(
            operation="get_session",
            session_id=None,
            intent_data=None,
            min_confidence=0.0,
            limit=100,
            time_range_hours=24,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "error"
        assert "session_id" in (response.error_message or "").lower()


# =============================================================================
# Get Distribution Operation Tests
# =============================================================================


class TestGetDistributionOperation:
    """Tests for the get_distribution operation."""

    async def test_get_distribution_calls_adapter_with_correct_args(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
    ) -> None:
        """Verify get_intent_distribution() is called with correct arguments."""
        # Arrange
        mock_adapter.get_intent_distribution.return_value = (
            ModelIntentDistributionResult(
                status="success",
                distribution={"debugging": 10, "code_generation": 5},
                total_intents=15,
                time_range_hours=48,
                execution_time_ms=2.0,
            )
        )

        request = ModelIntentStorageRequest(
            operation="get_distribution",
            time_range_hours=48,
        )

        # Act
        await handler_with_mock.execute(request)

        # Assert
        mock_adapter.get_intent_distribution.assert_called_once_with(
            time_range_hours=48,
        )

    async def test_get_distribution_returns_correct_response_model(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
    ) -> None:
        """Verify get_distribution returns correctly populated response."""
        # Arrange
        expected_distribution = {
            "debugging": 10,
            "code_generation": 5,
            "explanation": 3,
        }
        mock_adapter.get_intent_distribution.return_value = (
            ModelIntentDistributionResult(
                status="success",
                distribution=expected_distribution,
                total_intents=18,
                time_range_hours=24,
                execution_time_ms=2.0,
            )
        )

        request = ModelIntentStorageRequest(
            operation="get_distribution",
            time_range_hours=24,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert isinstance(response, ModelIntentStorageResponse)
        assert response.status == "success"
        assert response.distribution == expected_distribution
        assert response.total_intents == 18
        assert response.time_range_hours == 24

    async def test_get_distribution_uses_default_time_range(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
    ) -> None:
        """Verify default time_range_hours (24) is used when not specified."""
        # Arrange
        mock_adapter.get_intent_distribution.return_value = (
            ModelIntentDistributionResult(
                status="success",
                distribution={},
                total_intents=0,
                time_range_hours=24,
                execution_time_ms=1.0,
            )
        )

        request = ModelIntentStorageRequest(
            operation="get_distribution",
            # time_range_hours not specified, should use default 24
        )

        # Act
        await handler_with_mock.execute(request)

        # Assert
        mock_adapter.get_intent_distribution.assert_called_once_with(
            time_range_hours=24,  # Default value
        )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    async def test_execute_without_initialize_returns_error(
        self,
        adapter_config: ModelAdapterIntentGraphConfig,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify error response when handler is not initialized."""
        # Arrange - create handler without calling initialize
        handler = HandlerIntentStorageAdapter(config=adapter_config)

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
        )

        # Act
        response = await handler.execute(request)

        # Assert
        assert response.status == "error"
        assert "not initialized" in (response.error_message or "").lower()

    async def test_unknown_operation_returns_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
    ) -> None:
        """Verify error response for unknown operation type."""
        # NOTE: Using model_construct() intentionally bypasses Pydantic validation
        # to test handler's internal routing logic with an invalid operation value
        # that would normally be rejected at model instantiation.
        request = ModelIntentStorageRequest.model_construct(
            operation="invalid_op",
            session_id=None,
            intent_data=None,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "error"
        assert "unknown operation" in (response.error_message or "").lower()

    async def test_adapter_exception_returns_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify error response when adapter raises unexpected exception."""
        # Arrange
        mock_adapter.store_intent.side_effect = RuntimeError("Unexpected DB error")

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "error"
        assert "RuntimeError" in (response.error_message or "")

    async def test_execution_time_always_set(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify execution_time_ms is always populated in response."""
        # Arrange
        mock_adapter.store_intent.return_value = ModelIntentStorageResult(
            success=True,
            intent_id=TEST_INTENT_ID,
            created=True,
        )

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert - handler adds its own timing
        assert response.execution_time_ms > 0

    async def test_store_without_session_id_returns_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify error response when session_id is missing for store operation."""
        # Arrange - use model_construct to bypass Pydantic validation for testing
        request = ModelIntentStorageRequest.model_construct(
            operation="store",
            session_id=None,
            intent_data=sample_intent_data,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "error"
        assert "session_id" in (response.error_message or "").lower()

    async def test_store_without_intent_data_returns_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
    ) -> None:
        """Verify error response when intent_data is missing for store operation."""
        # Arrange - use model_construct to bypass Pydantic validation for testing
        request = ModelIntentStorageRequest.model_construct(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=None,
        )

        # Act
        response = await handler_with_mock.execute(request)

        # Assert
        assert response.status == "error"
        assert "intent_data" in (response.error_message or "").lower()

    async def test_circuit_breaker_open_returns_error(
        self,
        handler_with_mock: HandlerIntentStorageAdapter,
        mock_adapter: MagicMock,
        sample_intent_data: ModelIntentClassificationOutput,
    ) -> None:
        """Verify error response when circuit breaker is open.

        The handler wraps adapter calls with a circuit breaker that opens
        after consecutive failures reaching the configured threshold. When open,
        requests fail fast with a "temporarily unavailable" error instead of
        attempting the underlying operation.
        """
        # Arrange - configure mock to always fail
        mock_adapter.store_intent.side_effect = RuntimeError("DB connection lost")

        request = ModelIntentStorageRequest(
            operation="store",
            session_id=TEST_SESSION_ID,
            intent_data=sample_intent_data,
        )

        # Trigger enough failures to open circuit breaker using the actual threshold
        failure_threshold = _DEFAULT_CIRCUIT_BREAKER_CONFIG.failure_threshold
        for _ in range(failure_threshold):
            response = await handler_with_mock.execute(request)
            # Verify failures are being recorded (not circuit breaker errors yet)
            assert response.status == "error"
            assert "RuntimeError" in (response.error_message or "")

        # Act - next call should get circuit breaker open error
        response = await handler_with_mock.execute(request)

        # Assert - circuit breaker is now open
        assert response.status == "error"
        assert "temporarily unavailable" in (response.error_message or "").lower()


# =============================================================================
# Unit Tests (Explicit markers override module-level)
# =============================================================================


@pytest.mark.unit
class TestHandlerIntentStorageAdapterUnit:
    """Unit tests that don't require integration markers."""

    async def test_handler_initialization(self) -> None:
        """Verify handler can be instantiated with default config."""
        if not _DEPENDENCIES_AVAILABLE:
            pytest.skip(_SKIP_REASON)

        handler = HandlerIntentStorageAdapter()
        assert handler._initialized is False
        assert handler._adapter is None

    async def test_handler_with_custom_config(self) -> None:
        """Verify handler accepts custom configuration."""
        if not _DEPENDENCIES_AVAILABLE:
            pytest.skip(_SKIP_REASON)

        config = ModelAdapterIntentGraphConfig(
            timeout_seconds=60.0,
            max_intents_per_session=500,
        )
        handler = HandlerIntentStorageAdapter(config=config)

        assert handler._config.timeout_seconds == 60.0
        assert handler._config.max_intents_per_session == 500

    async def test_shutdown_clears_state(
        self,
        adapter_config: ModelAdapterIntentGraphConfig,
    ) -> None:
        """Verify shutdown properly clears adapter state."""
        if not _DEPENDENCIES_AVAILABLE:
            pytest.skip(_SKIP_REASON)

        # Create and manually initialize handler with mock
        handler = HandlerIntentStorageAdapter(config=adapter_config)

        # Inject mock adapter to simulate initialized state
        mock_adapter = MagicMock()
        mock_adapter.shutdown = AsyncMock()
        handler._adapter = mock_adapter
        handler._initialized = True

        # Act
        await handler.shutdown()

        # Assert
        mock_adapter.shutdown.assert_called_once()
        assert handler._adapter is None
        assert handler._initialized is False
