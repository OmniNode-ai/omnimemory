"""Integration tests for Memory Storage Effect Node."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnimemory.nodes.node_memory_storage_effect import (
    EnumMemoryStorageOperationType,
    ModelMemoryStorageConfig,
    ModelMemoryStorageInput,
    ModelMemoryStorageOutput,
    NodeMemoryStorageEffect,
)


class MockONEXContainer:
    """Mock ONEX container for testing."""

    def __init__(self):
        self.services = {}

    def get_service(self, name: str):
        return self.services.get(name)

    def register_service(self, name: str, service):
        self.services[name] = service


class TestMemoryStorageEffectNode:
    """Test suite for Memory Storage Effect Node."""

    @pytest.fixture
    async def effect_node(self):
        """Create a memory storage effect node for testing."""
        container = MockONEXContainer()

        # Register test configuration
        config = ModelMemoryStorageConfig.for_environment("development")
        container.register_service("memory_storage_config", config)

        node = NodeMemoryStorageEffect(container)
        yield node

        # Cleanup
        if node.event_driven_service:
            await node.event_driven_service.shutdown()

    @pytest.mark.asyncio
    async def test_node_initialization(self, effect_node):
        """Test that the effect node initializes properly."""
        assert effect_node.node_type == "effect"
        assert effect_node.domain == "memory"
        assert isinstance(effect_node.config, ModelMemoryStorageConfig)
        assert effect_node.operation_count == 0
        assert effect_node.success_count == 0
        assert effect_node.error_count == 0

    @pytest.mark.asyncio
    async def test_health_check_operation(self, effect_node):
        """Test health check operation."""
        # Create health check request
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.HEALTH_CHECK,
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Execute health check
        result = await effect_node.process(request)

        # Verify response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.HEALTH_CHECK
        assert result.success is True
        assert result.correlation_id == request.correlation_id
        assert result.execution_time_ms > 0
        assert "health_status" in result.data
        assert result.health_status in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_get_stats_operation(self, effect_node):
        """Test get statistics operation."""
        # Create stats request
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.GET_STATS,
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Execute get stats
        result = await effect_node.process(request)

        # Verify response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.GET_STATS
        assert result.success is True
        assert result.system_stats is not None
        assert "operation_count" in result.system_stats
        assert "success_rate" in result.system_stats
        assert "circuit_breaker_states" in result.system_stats

    @pytest.mark.asyncio
    async def test_store_memory_operation_fallback(self, effect_node):
        """Test store memory operation using direct fallback (no event bus)."""
        # Create store request
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key="test_key_123",
            content={"test": "data", "number": 42},
            memory_type="persistent",
            metadata={"source": "test", "version": "1.0"},
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Execute store operation
        result = await effect_node.process(request)

        # Verify response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.STORE_MEMORY
        assert result.success is True
        assert result.memory_key == "test_key_123"
        assert result.memory_id is not None
        assert result.storage_backend == "persistent"

    @pytest.mark.asyncio
    async def test_retrieve_memory_operation_fallback(self, effect_node):
        """Test retrieve memory operation using direct fallback."""
        # Create retrieve request
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.RETRIEVE_MEMORY,
            memory_key="test_key_123",
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Execute retrieve operation
        result = await effect_node.process(request)

        # Verify response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.RETRIEVE_MEMORY
        assert result.success is True
        assert result.memory_key == "test_key_123"
        # Note: Direct fallback returns empty content since no real storage

    @pytest.mark.asyncio
    async def test_vector_search_operation_fallback(self, effect_node):
        """Test vector search operation using direct fallback."""
        # Create vector search request
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.VECTOR_SEARCH,
            query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            similarity_threshold=0.8,
            max_results=5,
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Execute vector search operation
        result = await effect_node.process(request)

        # Verify response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.VECTOR_SEARCH
        assert result.success is True
        assert result.search_results is not None
        assert isinstance(result.search_results, list)
        assert result.total_results is not None

    @pytest.mark.asyncio
    async def test_invalid_operation_type(self, effect_node):
        """Test handling of invalid operation type."""
        # This test uses a mock to simulate an invalid enum value
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.HEALTH_CHECK,  # Valid for creation
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Manually set invalid operation type (simulating deserialization error)
        request.operation_type = "INVALID_OPERATION"  # This should cause an error

        # Execute operation
        result = await effect_node.process(request)

        # Verify error response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.success is False
        assert result.error_message is not None
        assert result.error_code is not None

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, effect_node):
        """Test validation of missing required fields."""
        # Create store request without required fields
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.STORE_MEMORY,
            # Missing memory_key and content
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        # Execute operation
        result = await effect_node.process(request)

        # Verify validation error
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.success is False
        assert (
            "validation" in result.error_message.lower()
            or "required" in result.error_message.lower()
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, effect_node):
        """Test circuit breaker functionality."""
        # Check initial circuit breaker states
        assert all(
            cb["state"] == "closed" for cb in effect_node.circuit_breakers.values()
        )

        # Simulate failures to open circuit breaker
        for _ in range(effect_node.config.circuit_breaker_failure_threshold + 1):
            effect_node._record_failure("test failure")

        # Check that circuit breakers are now open
        assert any(
            cb["state"] == "open" for cb in effect_node.circuit_breakers.values()
        )

        # Test that operations fail with circuit breaker error
        request = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key="test_key",
            content="test_data",
            correlation_id=uuid4(),
            timestamp=time.time(),
        )

        result = await effect_node.process(request)
        assert result.success is False
        assert "circuit breaker" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_sanitization(self, effect_node):
        """Test error message sanitization."""
        # Enable error sanitization
        effect_node.config.enable_error_sanitization = True

        # Test error message with sensitive data
        error_with_secrets = "Connection failed: password=secret123 api_key=abc123"
        sanitized = effect_node._sanitize_error_message(error_with_secrets)

        assert "secret123" not in sanitized
        assert "abc123" not in sanitized
        assert "password=***" in sanitized
        assert "api_key=***" in sanitized

    @pytest.mark.asyncio
    async def test_correlation_id_validation(self, effect_node):
        """Test correlation ID validation."""
        # Test with valid UUID
        valid_uuid = uuid4()
        validated = effect_node._validate_correlation_id(valid_uuid)
        assert validated == valid_uuid

        # Test with None (should generate new UUID)
        validated = effect_node._validate_correlation_id(None)
        assert validated is not None
        assert isinstance(validated, type(uuid4()))

        # Test with string UUID
        uuid_str = str(uuid4())
        validated = effect_node._validate_correlation_id(uuid_str)
        assert str(validated) == uuid_str

    @pytest.mark.asyncio
    async def test_health_status_retrieval(self, effect_node):
        """Test health status retrieval."""
        health_status = await effect_node.get_health_status()

        assert health_status.status in ["HEALTHY", "DEGRADED", "UNHEALTHY"]
        assert health_status.details is not None
        assert "node_type" in health_status.details
        assert "domain" in health_status.details
        assert "operation_count" in health_status.details
        assert "success_rate" in health_status.details
        assert "circuit_breaker_states" in health_status.details

    @pytest.mark.asyncio
    async def test_configuration_environments(self):
        """Test configuration for different environments."""
        # Test development config
        dev_config = ModelMemoryStorageConfig.for_environment("development")
        assert dev_config.enable_error_sanitization is False
        assert dev_config.max_timeout_seconds == 60.0

        # Test production config
        prod_config = ModelMemoryStorageConfig.for_environment("production")
        assert prod_config.enable_error_sanitization is True
        assert prod_config.max_timeout_seconds == 10.0
        assert prod_config.max_concurrent_operations == 200

        # Test staging config
        staging_config = ModelMemoryStorageConfig.for_environment("staging")
        assert staging_config.max_timeout_seconds == 15.0
        assert staging_config.max_concurrent_operations == 100
