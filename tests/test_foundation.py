"""
Foundation Tests for OmniMemory ONEX Architecture

This module tests the foundational components of the OmniMemory system
to ensure ONEX compliance and proper implementation of the ModelOnexContainer
patterns, protocols, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
    ProtocolSystemError,
    ProtocolValidationError,
)

# Use compat modules until omnibase_core components are available
from omnimemory.compat import ModelOnexContainer, NodeResult


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
    ) -> NodeResult[MemoryStoreResponse]:
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

            return NodeResult.success(
                value=response,
                provenance=["mock_storage.store_memory"],
                trust_score=1.0,
                metadata={"service": "mock_storage"},
            )

        except Exception as e:
            return NodeResult.failure(
                error=ProtocolSystemError(
                    message=f"Mock storage failed: {e!s}",
                    system_component="mock_storage",
                ),
                provenance=["mock_storage.store_memory.failed"],
            )


class TestFoundationArchitecture:
    """Test suite for ONEX foundation architecture."""

    @pytest.fixture
    def container(self) -> ModelOnexContainer:
        """Create a test container instance."""
        return ModelOnexContainer()

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

    def test_container_initialization(self, container: ModelOnexContainer) -> None:
        """Test that the ONEX container initializes properly."""
        assert container is not None
        assert hasattr(container, "register_singleton")
        assert hasattr(container, "register_transient")
        assert hasattr(container, "resolve")

    def test_container_node_registration_resolution(
        self, container: ModelOnexContainer
    ) -> None:
        """Test ONEX node registration and resolution functionality."""
        # Register mock storage node
        container.register_singleton(ProtocolMemoryStorage, MockMemoryStorageNode)

        # Resolve node
        storage_node = container.resolve(ProtocolMemoryStorage)

        assert storage_node is not None
        assert isinstance(storage_node, MockMemoryStorageNode)

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
        assert error.context["test_key"] == "test_value"
        assert error.is_recoverable() is False  # Validation errors are not recoverable

        # Test ProtocolValidationError
        validation_error = ProtocolValidationError(
            message="Invalid field value",
            field_name="test_field",
            field_value="invalid_value",
        )

        assert validation_error.context["field_name"] == "test_field"
        assert validation_error.context["field_value"] == "invalid_value"
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

    def test_node_result_success_failure_composition(self) -> None:
        """Test monadic patterns and NodeResult composition."""
        # Test successful NodeResult
        success_result = NodeResult.success(
            value="test_value",
            provenance=["test.operation"],
            trust_score=1.0,
        )

        assert success_result.is_success is True
        assert success_result.is_failure is False
        assert success_result.value == "test_value"
        assert "test.operation" in success_result.provenance
        assert success_result.trust_score == 1.0

        # Test failure NodeResult
        error = ProtocolSystemError(
            message="Test failure",
            system_component="test_component",
        )

        failure_result = NodeResult.failure(
            error=error,
            provenance=["test.operation.failed"],
        )

        assert failure_result.is_success is False
        assert failure_result.is_failure is True
        assert failure_result.error is not None
        assert "test.operation.failed" in failure_result.provenance

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

        with open(contract_path) as f:
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

    async def test_memory_operation_e2e(
        self, container: ModelOnexContainer, sample_memory_record: MemoryRecord
    ) -> None:
        """Test end-to-end memory operation using ONEX nodes."""
        # Register mock storage node
        container.register_singleton(ProtocolMemoryStorage, MockMemoryStorageNode)

        # Resolve storage node
        storage_node = container.resolve(ProtocolMemoryStorage)

        # Create store request
        store_request = MemoryStoreRequest(
            memory=sample_memory_record,
            generate_embedding=True,
            index_immediately=True,
        )

        # Perform store operation
        store_result = await storage_node.store_memory(store_request)

        assert store_result.is_success
        response = store_result.value
        assert response.memory_id == sample_memory_record.memory_id
        assert response.storage_location == "/mock/storage/location"
        assert response.indexing_status == "completed"
        assert response.embedding_generated is True


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
