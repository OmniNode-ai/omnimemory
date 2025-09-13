"""
Foundation Tests for OmniMemory ONEX Architecture

This module tests the foundational components of the OmniMemory system
to ensure ONEX compliance and proper implementation of the ModelOnexContainer
patterns, protocols, and error handling.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any
from uuid import UUID, uuid4

from omnimemory import (
    # Container and services
    OmniMemoryContainer,
    create_omnimemory_container,
    OmniMemoryServiceProvider,
    MemoryServiceRegistry,
    
    # Base implementations
    BaseMemoryService,
    BaseEffectService,
    
    # Protocols
    ProtocolMemoryBase,
    ProtocolMemoryStorage,
    
    # Data models
    MemoryRecord,
    ContentType,
    MemoryPriority,
    AccessLevel,
    MemoryStoreRequest,
    MemoryStoreResponse,
    
    # Error handling
    OmniMemoryError,
    OmniMemoryErrorCode,
    ValidationError,
    SystemError,
)

from omnibase_core.core.monadic.model_node_result import NodeResult
from omnibase_spi import ProtocolLogger


class MockMemoryStorageService(BaseEffectService):
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
    
    async def _get_storage_utilization(self) -> Dict[str, float]:
        """Mock storage utilization."""
        return {"disk": 0.60, "memory": 0.45}
    
    async def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Mock configuration validation."""
        return "invalid_key" not in config
    
    async def _apply_configuration(self, config: Dict[str, Any]) -> None:
        """Mock configuration application."""
        pass
    
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
                error=SystemError(
                    message=f"Mock storage failed: {str(e)}",
                    system_component="mock_storage",
                ),
                provenance=["mock_storage.store_memory.failed"],
            )


class TestFoundationArchitecture:
    """Test suite for ONEX foundation architecture."""
    
    @pytest.fixture
    async def container(self) -> OmniMemoryContainer:
        """Create a test container instance."""
        return await create_omnimemory_container()
    
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
    
    def test_container_initialization(self, container: OmniMemoryContainer):
        """Test that the container initializes properly."""
        assert container is not None
        assert container.service_provider is not None
        assert container.settings is not None
        assert hasattr(container, 'get_service_async')
        assert hasattr(container, 'get_service_sync')
        assert hasattr(container, 'health_check')
    
    def test_container_settings(self, container: OmniMemoryContainer):
        """Test that container settings are properly configured."""
        settings = container.settings
        assert settings.log_level == "INFO"
        assert settings.max_concurrent_operations > 0
        assert settings.operation_timeout_ms > 0
        assert settings.cache_ttl_seconds > 0
    
    async def test_container_health_check(self, container: OmniMemoryContainer):
        """Test container health check functionality."""
        health = await container.health_check()
        
        assert health is not None
        assert "container" in health
        assert "timestamp" in health
        assert "services" in health
        assert "performance" in health
        assert health["container"] == "healthy"
    
    def test_service_provider_creation(self):
        """Test service provider creation and initialization."""
        from omnibase_core.enums.enum_log_level import EnumLogLevel
        
        # Create mock logger
        class MockLogger:
            def emit_log_event_sync(self, level, message, event_type="generic", **kwargs):
                pass
            
            async def emit_log_event_async(self, level, message, event_type="generic", **kwargs):
                pass
        
        logger = MockLogger()
        provider = OmniMemoryServiceProvider(logger)
        
        assert provider.provider_id is not None
        assert provider.logger is logger
        assert isinstance(provider.descriptors, dict)
        assert isinstance(provider.instances, dict)
    
    async def test_service_registration_and_resolution(self):
        """Test service registration and resolution functionality."""
        from omnibase_core.enums.enum_log_level import EnumLogLevel
        
        # Create mock logger
        class MockLogger:
            def emit_log_event_sync(self, level, message, event_type="generic", **kwargs):
                pass
            
            async def emit_log_event_async(self, level, message, event_type="generic", **kwargs):
                pass
        
        logger = MockLogger()
        provider = OmniMemoryServiceProvider(logger)
        
        # Register mock service
        registration_result = await provider.register_service(
            protocol_type=ProtocolMemoryStorage,
            service_class=MockMemoryStorageService,
            service_name="mock_storage",
            singleton=True,
        )
        
        assert registration_result.is_success
        assert "mock_storage" in provider.descriptors
        
        # Resolve service
        resolution_result = await provider.resolve_service(
            protocol_type=ProtocolMemoryStorage,
            service_name="mock_storage",
        )
        
        assert resolution_result.is_success
        service = resolution_result.value
        assert isinstance(service, MockMemoryStorageService)
        assert service.service_name == "mock_storage"
    
    def test_memory_record_validation(self, sample_memory_record: MemoryRecord):
        """Test memory record creation and validation."""
        assert sample_memory_record.memory_id is not None
        assert sample_memory_record.content == "This is a test memory record for ONEX validation"
        assert sample_memory_record.content_type == ContentType.TEXT
        assert sample_memory_record.priority == MemoryPriority.NORMAL
        assert sample_memory_record.source_agent == "test_agent"
        assert sample_memory_record.access_level == AccessLevel.INTERNAL
        assert "test" in sample_memory_record.tags
        assert "validation" in sample_memory_record.tags
        assert "onex" in sample_memory_record.tags
        assert sample_memory_record.created_at is not None
        assert sample_memory_record.updated_at is not None
    
    def test_memory_store_request_creation(self, sample_memory_record: MemoryRecord):
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
    
    def test_error_handling_creation(self):
        """Test ONEX error handling patterns."""
        # Test basic OmniMemoryError
        error = OmniMemoryError(
            error_code=OmniMemoryErrorCode.INVALID_INPUT,
            message="Test error message",
            context={"test_key": "test_value"},
        )
        
        assert error.omnimemory_error_code == OmniMemoryErrorCode.INVALID_INPUT
        assert error.message == "Test error message"
        assert error.context["test_key"] == "test_value"
        assert error.is_recoverable() is False  # Validation errors are not recoverable
        
        # Test ValidationError
        validation_error = ValidationError(
            message="Invalid field value",
            field_name="test_field",
            field_value="invalid_value",
        )
        
        assert validation_error.context["field_name"] == "test_field"
        assert validation_error.context["field_value"] == "invalid_value"
        assert "Review and correct the input" in validation_error.recovery_hint
    
    def test_error_categorization(self):
        """Test error categorization and metadata."""
        from omnimemory.protocols.error_models import get_error_category
        
        # Test validation error category
        validation_category = get_error_category(OmniMemoryErrorCode.INVALID_INPUT)
        assert validation_category is not None
        assert validation_category.recoverable is False
        assert validation_category.default_retry_count == 0
        
        # Test storage error category
        storage_category = get_error_category(OmniMemoryErrorCode.STORAGE_UNAVAILABLE)
        assert storage_category is not None
        assert storage_category.recoverable is True
        assert storage_category.default_retry_count > 0
    
    async def test_service_health_monitoring(self):
        """Test service health monitoring functionality."""
        from omnibase_core.enums.enum_log_level import EnumLogLevel
        
        # Create mock logger
        class MockLogger:
            def emit_log_event_sync(self, level, message, event_type="generic", **kwargs):
                pass
            
            async def emit_log_event_async(self, level, message, event_type="generic", **kwargs):
                pass
        
        logger = MockLogger()
        service = MockMemoryStorageService(
            service_name="test_service",
            logger=logger,
        )
        
        # Test health check
        health_result = await service.health_check()
        assert health_result.is_success
        
        health_data = health_result.value
        assert health_data["service_name"] == "test_service"
        assert health_data["status"] == "healthy"
        assert "uptime_seconds" in health_data
        assert "operation_count" in health_data
        
        # Test metrics collection
        metrics_result = await service.get_metrics()
        assert metrics_result.is_success
        
        metrics = metrics_result.value
        assert metrics["service_name"] == "test_service"
        assert "uptime_seconds" in metrics
        assert "storage_operations" in metrics
        assert "cache_hit_rate" in metrics
    
    async def test_configuration_management(self):
        """Test service configuration management."""
        from omnibase_core.enums.enum_log_level import EnumLogLevel
        
        # Create mock logger
        class MockLogger:
            def emit_log_event_sync(self, level, message, event_type="generic", **kwargs):
                pass
            
            async def emit_log_event_async(self, level, message, event_type="generic", **kwargs):
                pass
        
        logger = MockLogger()
        service = MockMemoryStorageService(
            service_name="test_service",
            logger=logger,
        )
        
        # Test valid configuration
        valid_config = {"cache_size": 1000, "timeout": 30}
        config_result = await service.configure(valid_config)
        assert config_result.is_success
        
        # Test invalid configuration
        invalid_config = {"invalid_key": "should_fail"}
        invalid_result = await service.configure(invalid_config)
        assert invalid_result.is_failure
    
    def test_monadic_patterns(self):
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
        error = SystemError(
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
    
    def test_contract_compliance(self):
        """Test that the implementation follows contract specifications."""
        # Verify contract.yaml can be loaded
        import yaml
        from pathlib import Path
        
        contract_path = Path("contract.yaml")
        assert contract_path.exists(), "contract.yaml must exist"
        
        with open(contract_path, 'r') as f:
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
    
    async def test_end_to_end_memory_operation(self, sample_memory_record: MemoryRecord):
        """Test end-to-end memory operation using mock services."""
        from omnibase_core.enums.enum_log_level import EnumLogLevel
        
        # Create mock logger
        class MockLogger:
            def emit_log_event_sync(self, level, message, event_type="generic", **kwargs):
                pass
            
            async def emit_log_event_async(self, level, message, event_type="generic", **kwargs):
                pass
        
        logger = MockLogger()
        provider = OmniMemoryServiceProvider(logger)
        
        # Register mock storage service
        await provider.register_service(
            protocol_type=ProtocolMemoryStorage,
            service_class=MockMemoryStorageService,
            service_name="storage_service",
        )
        
        # Resolve storage service
        resolution_result = await provider.resolve_service(
            protocol_type=ProtocolMemoryStorage,
            service_name="storage_service",
        )
        
        assert resolution_result.is_success
        storage_service = resolution_result.value
        
        # Create store request
        store_request = MemoryStoreRequest(
            memory=sample_memory_record,
            generate_embedding=True,
            index_immediately=True,
        )
        
        # Perform store operation
        store_result = await storage_service.store_memory(store_request)
        
        assert store_result.is_success
        response = store_result.value
        assert response.memory_id == sample_memory_record.memory_id
        assert response.storage_location == "/mock/storage/location"
        assert response.indexing_status == "completed"
        assert response.embedding_generated is True


if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v"])