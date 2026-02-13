"""
Foundation Tests for OmniMemory ONEX Architecture

This module tests the foundational components of the OmniMemory system
to ensure ONEX compliance and proper implementation of the ModelONEXContainer
patterns, protocols, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.core.model_base_result import ModelBaseResult
from omnibase_core.models.core.model_error_details import ModelErrorDetails
from omnibase_core.models.results.model_simple_metadata import ModelGenericMetadata

from omnimemory import (  # Protocols; Data models; Error handling
    AccessLevel,
    ContentType,
    EnumOmniMemoryErrorCode,
    MemoryPriority,
    MemoryRecord,
    MemoryStoreRequest,
    MemoryStoreResponse,
    ProtocolMemoryStorage,
    ProtocolOmniMemoryError,
    ProtocolValidationError,
)


class MockMemoryStorageNode:
    """Mock implementation of memory storage service for testing."""

    async def _check_storage_connectivity(self) -> bool:
        """Mock storage connectivity check."""
        return True

    async def _get_storage_operation_count(self) -> int:
        """Mock storage operation count."""
        return 42

    async def _get_cache_hit_rate(self) -> float:
        """Mock cache hit rate."""
        return 0.85

    async def _get_storage_utilization(self) -> dict[str, float]:
        """Mock storage utilization."""
        return {"disk": 0.60, "memory": 0.45}

    async def _validate_configuration(self, config: dict[str, str]) -> bool:
        """Mock configuration validation."""
        return "invalid_key" not in config

    async def _apply_configuration(self, config: dict[str, str]) -> None:
        """Mock configuration application."""

    async def store_memory(
        self,
        request: MemoryStoreRequest,
    ) -> ModelBaseResult:
        """Mock memory storage operation."""
        try:
            # Simulate storage operation
            response = MemoryStoreResponse(
                correlation_id=request.correlation_id,
                status="success",
                execution_time_ms=25,
                provenance=["mock_storage.store"],
                trust_score=1.0,
                memory_id=request.memory.memory_id,
                storage_location="/mock/storage/location",
                indexing_status="completed",
                embedding_generated=True,
                duplicate_detected=False,
                storage_size_bytes=len(request.memory.content),
            )

            return ModelBaseResult(
                success=True,
                exit_code=0,
                errors=[],
                metadata=ModelGenericMetadata(
                    custom_fields={
                        "response": response.model_dump(),
                        "service": "mock_storage",
                    }
                ),
            )

        except Exception as e:
            return ModelBaseResult(
                success=False,
                exit_code=1,
                errors=[
                    ModelErrorDetails(
                        error_message=f"Mock storage failed: {e!s}",
                        error_code="STORAGE_ERROR",
                        error_type="runtime",
                        component="mock_storage.store_memory.failed",
                    )
                ],
            )


class TestFoundationArchitecture:
    """Test suite for ONEX foundation architecture."""

    @pytest.fixture
    def container(self) -> ModelONEXContainer:
        """Create a test container instance."""
        return ModelONEXContainer()

    @pytest.fixture
    def sample_memory_record(self) -> MemoryRecord:
        """Create a sample memory record for testing."""
        return MemoryRecord(
            content="This is a test memory record for ONEX validation",
            content_type=ContentType.TEXT,
            priority=MemoryPriority.NORMAL,
            source_agent="test_agent",
            access_level=AccessLevel.INTERNAL,
            tags=["test", "validation", "onex"],
        )

    def test_container_initialization(self, container: ModelONEXContainer) -> None:
        """Test that the ONEX container initializes properly."""
        assert container is not None
        # omnibase_core container uses service registry pattern
        assert hasattr(container, "get_service")
        assert hasattr(container, "get_service_optional")
        assert hasattr(container, "service_registry")

    def test_container_service_registry(self, container: ModelONEXContainer) -> None:
        """Test ONEX container service registry access."""
        # omnibase_core uses service registry pattern
        # Test that service registry is accessible
        assert container.service_registry is not None

        # Test get_service_optional returns None for unregistered services
        result = container.get_service_optional(
            ProtocolMemoryStorage  # type: ignore[type-abstract]
        )
        # Unregistered service should return None (not raise)
        assert result is None

    def test_memory_record_validation(self, sample_memory_record: MemoryRecord) -> None:
        """Test memory record creation and validation."""
        assert sample_memory_record.memory_id is not None
        assert (
            sample_memory_record.content
            == "This is a test memory record for ONEX validation"
        )
        assert sample_memory_record.content_type == ContentType.TEXT
        assert sample_memory_record.priority == MemoryPriority.NORMAL
        assert sample_memory_record.source_agent == "test_agent"
        assert sample_memory_record.access_level == AccessLevel.INTERNAL
        assert "test" in sample_memory_record.tags
        assert "validation" in sample_memory_record.tags
        assert "onex" in sample_memory_record.tags
        assert sample_memory_record.created_at is not None
        assert sample_memory_record.updated_at is not None

    def test_memory_store_request_creation(
        self, sample_memory_record: MemoryRecord
    ) -> None:
        """Test memory store request creation and validation."""
        request = MemoryStoreRequest(
            memory=sample_memory_record,
            generate_embedding=True,
            index_immediately=True,
        )

        assert request.memory == sample_memory_record
        assert request.generate_embedding is True
        assert request.index_immediately is True
        assert request.correlation_id is not None
        assert request.timestamp is not None

    def test_error_handling_creation(self) -> None:
        """Test ONEX error handling patterns."""
        # Test basic ProtocolOmniMemoryError
        error = ProtocolOmniMemoryError(
            error_code=EnumOmniMemoryErrorCode.INVALID_INPUT,
            message="Test error message",
            context={"test_key": "test_value"},
        )

        assert error.omnimemory_error_code == EnumOmniMemoryErrorCode.INVALID_INPUT
        assert error.message == "Test error message"
        # ModelOnexError stores context dict under "additional_context.context" path
        assert (
            error.context["additional_context"]["context"]["test_key"] == "test_value"
        )
        assert error.is_recoverable() is False  # Validation errors are not recoverable

        # Test ProtocolValidationError
        validation_error = ProtocolValidationError(
            message="Invalid field value",
            field_name="test_field",
            field_value="invalid_value",
        )

        # Context is stored under additional_context.context by ModelOnexError
        assert (
            validation_error.context["additional_context"]["context"]["field_name"]
            == "test_field"
        )
        assert (
            validation_error.context["additional_context"]["context"]["field_value"]
            == "invalid_value"
        )
        assert "Review and correct" in validation_error.recovery_hint

    def test_error_categorization(self) -> None:
        """Test error categorization and metadata."""
        from omnimemory.protocols.error_models import get_error_category

        # Test validation error category
        validation_category = get_error_category(EnumOmniMemoryErrorCode.INVALID_INPUT)
        assert validation_category is not None
        assert validation_category.recoverable is False
        assert validation_category.default_retry_count == 0

        # Test storage error category
        storage_category = get_error_category(
            EnumOmniMemoryErrorCode.STORAGE_UNAVAILABLE
        )
        assert storage_category is not None
        assert storage_category.recoverable is True
        assert storage_category.default_retry_count > 0

    def test_model_base_result_success_failure(self) -> None:
        """Test ModelBaseResult success and failure patterns."""
        # Test successful ModelBaseResult
        success_result = ModelBaseResult(
            success=True,
            exit_code=0,
            errors=[],
            metadata=ModelGenericMetadata(
                custom_fields={
                    "value": "test_value",
                    "provenance": ["test.operation"],
                    "trust_score": 1.0,
                }
            ),
        )

        assert success_result.success is True
        assert success_result.exit_code == 0
        assert len(success_result.errors) == 0
        assert success_result.metadata is not None
        assert success_result.metadata.custom_fields["value"] == "test_value"
        assert "test.operation" in success_result.metadata.custom_fields["provenance"]
        assert success_result.metadata.custom_fields["trust_score"] == 1.0

        # Test failure ModelBaseResult
        failure_result = ModelBaseResult(
            success=False,
            exit_code=1,
            errors=[
                ModelErrorDetails(
                    error_message="Test failure",
                    error_code="SYSTEM_ERROR",
                    error_type="system",
                    component="test_component",
                )
            ],
            metadata=ModelGenericMetadata(
                custom_fields={"provenance": ["test.operation.failed"]}
            ),
        )

        assert failure_result.success is False
        assert failure_result.exit_code == 1
        assert len(failure_result.errors) == 1
        assert failure_result.errors[0].error_message == "Test failure"
        assert failure_result.metadata is not None
        assert (
            "test.operation.failed"
            in failure_result.metadata.custom_fields["provenance"]
        )

    def test_contract_compliance(self) -> None:
        """Test that the implementation follows contract specifications.

        Uses Path(__file__) for CWD-independent path resolution.
        Skips gracefully if contract.yaml doesn't exist.
        """
        # Verify contract.yaml can be loaded
        # Use __file__ relative path for CWD independence
        contract_path = Path(__file__).parent.parent / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"contract.yaml not found at {contract_path}")

        with open(contract_path, encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)

        # Verify contract structure
        assert "contract" in contract_data
        assert "protocols" in contract_data
        assert "schemas" in contract_data
        assert "error_handling" in contract_data

        # Verify ONEX architecture mapping
        architecture = contract_data["contract"]["architecture"]
        assert architecture["pattern"] == "onex_4_node"
        assert "effect" in architecture["nodes"]
        assert "compute" in architecture["nodes"]
        assert "reducer" in architecture["nodes"]
        assert "orchestrator" in architecture["nodes"]

    @pytest.mark.asyncio
    async def test_memory_operation_e2e(
        self, container: ModelONEXContainer, sample_memory_record: MemoryRecord
    ) -> None:
        """Test end-to-end memory operation using ONEX nodes."""
        # Verify container is available (omnibase_core pattern doesn't use register/resolve)
        assert container is not None

        # Directly instantiate mock storage node for testing
        storage_node = MockMemoryStorageNode()

        # Create store request
        store_request = MemoryStoreRequest(
            memory=sample_memory_record,
            generate_embedding=True,
            index_immediately=True,
        )

        # Perform store operation
        store_result = await storage_node.store_memory(store_request)

        assert store_result.success
        assert store_result.metadata is not None
        response_data = store_result.metadata.custom_fields["response"]
        # model_dump() may return UUID objects, so convert both to string for comparison
        assert str(response_data["memory_id"]) == str(sample_memory_record.memory_id)
        assert response_data["storage_location"] == "/mock/storage/location"
        assert response_data["indexing_status"] == "completed"
        assert response_data["embedding_generated"] is True


class TestMockMemoryStorageNode:
    """Test suite for MockMemoryStorageNode helper methods.

    These tests ensure all mock helper methods are exercised and work correctly.
    """

    @pytest.fixture
    def mock_node(self) -> MockMemoryStorageNode:
        """Create a mock storage node for testing."""
        return MockMemoryStorageNode()

    @pytest.mark.asyncio
    async def test_check_storage_connectivity(
        self, mock_node: MockMemoryStorageNode
    ) -> None:
        """Test mock storage connectivity check returns True."""
        result = await mock_node._check_storage_connectivity()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_storage_operation_count(
        self, mock_node: MockMemoryStorageNode
    ) -> None:
        """Test mock storage operation count returns expected value."""
        result = await mock_node._get_storage_operation_count()
        assert result == 42

    @pytest.mark.asyncio
    async def test_get_cache_hit_rate(self, mock_node: MockMemoryStorageNode) -> None:
        """Test mock cache hit rate returns expected value."""
        result = await mock_node._get_cache_hit_rate()
        assert result == 0.85

    @pytest.mark.asyncio
    async def test_get_storage_utilization(
        self, mock_node: MockMemoryStorageNode
    ) -> None:
        """Test mock storage utilization returns expected structure."""
        result = await mock_node._get_storage_utilization()
        assert isinstance(result, dict)
        assert "disk" in result
        assert "memory" in result
        assert result["disk"] == 0.60
        assert result["memory"] == 0.45

    @pytest.mark.asyncio
    async def test_validate_configuration_valid(
        self, mock_node: MockMemoryStorageNode
    ) -> None:
        """Test mock configuration validation accepts valid config."""
        valid_config = {"key1": "value1", "key2": "value2"}
        result = await mock_node._validate_configuration(valid_config)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_configuration_invalid(
        self, mock_node: MockMemoryStorageNode
    ) -> None:
        """Test mock configuration validation rejects invalid config."""
        invalid_config = {"invalid_key": "should_fail"}
        result = await mock_node._validate_configuration(invalid_config)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_configuration(self, mock_node: MockMemoryStorageNode) -> None:
        """Test mock configuration application completes without error."""
        config = {"setting1": "value1"}
        # Should not raise
        await mock_node._apply_configuration(config)


if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v"])
