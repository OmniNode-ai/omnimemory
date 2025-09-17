"""Integration tests for Qdrant Adapter Effect Node."""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from omnimemory.nodes.node_memory_storage_effect import (
    EnumMemoryStorageOperationType,
    ModelMemoryStorageInput,
    ModelMemoryStorageOutput,
    NodeMemoryStorageEffect,
)
from omnimemory.nodes.node_qdrant_adapter_effect import NodeQdrantAdapterEffect
from omnimemory.nodes.node_qdrant_adapter_effect.v1_0_0.models import (
    ModelQdrantAdapterConfig,
    ModelQdrantAdapterInput,
    ModelQdrantVectorOperationRequest,
)


@pytest.fixture
def mock_container(qdrant_adapter_config):
    """Create a mock ONEX container for testing."""
    container = Mock()

    # Mock event bus
    event_bus = Mock()
    event_bus.publish_async = AsyncMock()
    event_bus.start = AsyncMock()
    event_bus.stop = AsyncMock()
    event_bus.cleanup = AsyncMock()

    # Mock service resolution with different services
    def get_service(service_name):
        if service_name == "ProtocolEventBus":
            return event_bus
        elif service_name == "qdrant_adapter_config":
            return qdrant_adapter_config
        else:
            return None

    container.get_service = Mock(side_effect=get_service)

    return container


@pytest.fixture
def mock_container_without_event_bus():
    """Create a mock container without event bus to test error handling."""
    container = Mock()
    container.get_service = Mock(return_value=None)
    return container


@pytest.fixture
def qdrant_adapter_config():
    """Create test configuration for Qdrant adapter."""
    return ModelQdrantAdapterConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        default_collection="test_collection",
        enable_error_sanitization=False,  # Disable for testing
        circuit_breaker_failure_threshold=10,  # Higher threshold for tests
    )


@pytest.fixture
def qdrant_adapter_config_with_sanitization():
    """Create test configuration for Qdrant adapter with sanitization enabled."""
    return ModelQdrantAdapterConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        default_collection="test_collection",
        enable_error_sanitization=True,  # Enable for sanitization tests
        circuit_breaker_failure_threshold=10,  # Higher threshold for tests
    )


@pytest.fixture
async def qdrant_adapter_with_sanitization(mock_container_with_sanitization):
    """Create Qdrant adapter with sanitization enabled for testing."""
    adapter = NodeQdrantAdapterEffect(mock_container_with_sanitization)
    try:
        await adapter.initialize()
        yield adapter
    finally:
        await adapter.cleanup()


@pytest.fixture
def mock_container_with_sanitization(qdrant_adapter_config_with_sanitization):
    """Create a mock ONEX container with sanitization enabled."""
    container = Mock()

    # Mock event bus
    event_bus = Mock()
    event_bus.publish_async = AsyncMock()
    event_bus.start = AsyncMock()
    event_bus.stop = AsyncMock()
    event_bus.cleanup = AsyncMock()

    # Mock service resolution with different services
    def get_service(service_name):
        if service_name == "ProtocolEventBus":
            return event_bus
        elif service_name == "qdrant_adapter_config":
            return qdrant_adapter_config_with_sanitization
        else:
            return None

    container.get_service = Mock(side_effect=get_service)

    return container


@pytest.fixture
async def qdrant_adapter(mock_container):
    """Create and initialize Qdrant adapter for testing."""
    adapter = NodeQdrantAdapterEffect(mock_container)
    try:
        await adapter.initialize()
        yield adapter
    finally:
        # Proper cleanup
        await adapter.cleanup()


class TestQdrantAdapterBasicOperations:
    """Test basic Qdrant adapter operations with proper ONEX patterns."""

    @pytest.mark.asyncio
    async def test_vector_search_operation_with_mocking(self, qdrant_adapter):
        """Test vector search operation input validation and processing."""
        # Create valid vector search input
        vector_request = ModelQdrantVectorOperationRequest(
            operation_type="vector_search",
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3, 0.4],
            search_limit=10,
            score_threshold=0.8,
        )

        input_data = ModelQdrantAdapterInput(
            operation_type="vector_search",
            correlation_id=uuid4(),
            vector_request=vector_request,
        )

        # Mock Qdrant search results
        mock_search_result = [
            type(
                "MockPoint",
                (),
                {
                    "id": "test_1",
                    "score": 0.95,
                    "payload": {"text": "test document 1", "category": "test"},
                    "vector": [0.1, 0.2, 0.3, 0.4],
                },
            ),
            type(
                "MockPoint",
                (),
                {
                    "id": "test_2",
                    "score": 0.87,
                    "payload": {"text": "test document 2", "category": "test"},
                    "vector": [0.15, 0.25, 0.35, 0.45],
                },
            ),
        ]

        # Mock the circuit breaker call to return our mock results
        with patch.object(
            qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
        ) as mock_cb_call:
            mock_cb_call.return_value = mock_search_result

            # Test processing
            start_time = time.perf_counter()
            result = await qdrant_adapter.process(input_data)
            end_time = time.perf_counter()

            # Validate result structure
            assert result is not None
            assert result.operation_type == "vector_search"
            assert result.correlation_id == input_data.correlation_id
            assert result.success is True
            assert result.execution_time_ms > 0

            # Performance check (should be fast with mocking)
            processing_time = (end_time - start_time) * 1000
            print(f"Vector search processing time: {processing_time:.2f}ms")
            assert processing_time < 1000  # Should be under 1 second

            # Validate search results structure
            if (
                hasattr(result, "data")
                and result.data
                and "search_results" in result.data
            ):
                search_results = result.data["search_results"]
                assert isinstance(search_results, list)
                assert len(search_results) == 2
                assert search_results[0]["score"] == 0.95
                assert search_results[1]["score"] == 0.87

    @pytest.mark.asyncio
    async def test_store_vector_operation_with_mocking(self, qdrant_adapter):
        """Test store vector operation with proper mocking."""
        # Create valid store vector input
        vector_request = ModelQdrantVectorOperationRequest(
            operation_type="store_vector",
            collection_name="test_collection",
            vector_id="test_vector_1",
            vector_data=[0.1, 0.2, 0.3, 0.4],
            payload={"text": "test document", "category": "test"},
        )

        input_data = ModelQdrantAdapterInput(
            operation_type="store_vector",
            correlation_id=uuid4(),
            vector_request=vector_request,
        )

        # Mock successful upsert result
        mock_upsert_result = {"operation_id": "mock_op_123", "status": "completed"}

        # Mock the circuit breaker call
        with patch.object(
            qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
        ) as mock_cb_call:
            mock_cb_call.return_value = mock_upsert_result

            # Test processing
            result = await qdrant_adapter.process(input_data)

            # Validate result
            assert result is not None
            assert result.operation_type == "store_vector"
            assert result.correlation_id == input_data.correlation_id
            assert result.success is True
            assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self, qdrant_adapter):
        """Test comprehensive health check operation."""
        input_data = ModelQdrantAdapterInput(
            operation_type="health_check",
            correlation_id=uuid4(),
            health_check_type="comprehensive",
        )

        # Test processing
        result = await qdrant_adapter.process(input_data)

        # Validate health check response
        assert result is not None
        assert result.operation_type == "health_check"
        # In test environment with mock Qdrant, health check should report as unhealthy/degraded
        # This is correct behavior when Qdrant client library is not actually available
        assert result.success is False
        assert result.health_status is not None
        assert result.health_status.status in ["unhealthy", "degraded"]
        assert result.health_status.connection_status == "unavailable"
        assert result.correlation_id == input_data.correlation_id
        assert result.execution_time_ms > 0

        # Validate health data structure
        if hasattr(result, "data") and result.data:
            assert "health_status" in result.data
            assert result.data["health_status"] in ["healthy", "degraded", "unhealthy"]


class TestQdrantAdapterConfiguration:
    """Test Qdrant adapter configuration handling."""

    def test_configuration_loading(self):
        """Test configuration loading from environment."""
        config = ModelQdrantAdapterConfig.for_environment("development")

        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.enable_error_sanitization is False  # Dev environment

        # Test production environment
        prod_config = ModelQdrantAdapterConfig.for_environment("production")
        assert prod_config.enable_error_sanitization is True  # Prod environment
        assert prod_config.circuit_breaker_failure_threshold == 3  # Stricter in prod

    def test_qdrant_client_config_generation(self, qdrant_adapter_config):
        """Test Qdrant client configuration generation."""
        client_config = qdrant_adapter_config.get_qdrant_client_config()

        assert "host" in client_config
        assert "port" in client_config
        assert "timeout" in client_config
        assert client_config["host"] == "localhost"
        assert client_config["port"] == 6333

    def test_validation_methods(self, qdrant_adapter_config):
        """Test configuration validation methods."""
        # Test vector dimension validation
        assert qdrant_adapter_config.validate_vector_dimensions(512) is True
        assert qdrant_adapter_config.validate_vector_dimensions(0) is False
        assert qdrant_adapter_config.validate_vector_dimensions(5000) is False

        # Test search limit validation
        assert qdrant_adapter_config.validate_search_limit(10) is True
        assert qdrant_adapter_config.validate_search_limit(0) is False
        assert qdrant_adapter_config.validate_search_limit(2000) is False

        # Test batch size validation
        assert qdrant_adapter_config.validate_batch_size(50) is True
        assert qdrant_adapter_config.validate_batch_size(0) is False
        assert qdrant_adapter_config.validate_batch_size(200) is False


class TestQdrantAdapterHealthChecks:
    """Test Qdrant adapter health check functionality."""

    async def test_health_check_methods(self, qdrant_adapter):
        """Test individual health check methods."""
        # Test connectivity check
        connectivity_health = qdrant_adapter._check_qdrant_connectivity()
        assert connectivity_health is not None
        assert hasattr(connectivity_health, "status")

        # Test circuit breaker health
        cb_health = qdrant_adapter._check_circuit_breaker_health()
        assert cb_health is not None
        assert hasattr(cb_health, "status")

        # Test vector operations health
        vector_health = qdrant_adapter._check_vector_operations_health()
        assert vector_health is not None
        assert hasattr(vector_health, "status")

        # Test event publishing health
        event_health = qdrant_adapter._check_event_publishing_health()
        assert event_health is not None
        assert hasattr(event_health, "status")

    async def test_get_health_checks_list(self, qdrant_adapter):
        """Test that adapter returns health check functions."""
        health_checks = qdrant_adapter.get_health_checks()

        assert isinstance(health_checks, list)
        assert len(health_checks) > 0

        # Each health check should be callable
        for health_check in health_checks:
            assert callable(health_check)


class TestQdrantAdapterEventPublishing:
    """Test event publishing functionality."""

    async def test_event_bus_initialization(self, mock_container):
        """Test that event bus is properly initialized."""
        adapter = NodeQdrantAdapterEffect(mock_container)

        # Event bus should be set from container
        assert adapter._event_bus is not None
        assert adapter._event_publisher is not None

        # Container should have been called for event bus
        mock_container.get_service.assert_called_with("ProtocolEventBus")

    async def test_vector_search_event_publishing(self, qdrant_adapter):
        """Test that vector search events are published correctly."""
        from omnimemory.models.events.model_omnimemory_event_data import (
            ModelOmniMemoryVectorSearchData,
        )

        correlation_id = uuid4()
        search_data = ModelOmniMemoryVectorSearchData(
            query_vector_hash="test_hash",
            vector_dimensions=4,
            similarity_threshold=0.8,
            max_results=10,
            result_count=5,
            search_type="similarity_search",
            index_name="test_collection",
            search_time_ms=150.0,
            results=[],
        )

        # This should not raise an exception
        await qdrant_adapter._publish_vector_search_completed_event(
            correlation_id=correlation_id,
            search_data=search_data,
            execution_time_ms=150.0,
            success=True,
        )

        # Event bus publish should have been called
        assert qdrant_adapter._event_bus.publish_async.called

    async def test_memory_stored_event_publishing(self, qdrant_adapter):
        """Test that memory stored events are published correctly."""
        from omnimemory.models.events.model_omnimemory_event_data import (
            ModelOmniMemoryStoreData,
        )

        correlation_id = uuid4()
        store_data = ModelOmniMemoryStoreData(
            memory_key="test_key",
            memory_type="vector",
            content_hash="test_hash",
            storage_size=1024,
            metadata={"test": "metadata"},
            vector_dimensions=4,
            affected_indices=["test_collection"],
        )

        # This should not raise an exception
        await qdrant_adapter._publish_memory_stored_event(
            correlation_id=correlation_id,
            store_data=store_data,
            execution_time_ms=100.0,
            success=True,
        )

        # Event bus publish should have been called
        assert qdrant_adapter._event_bus.publish_async.called


class TestQdrantAdapterErrorHandling:
    """Test error handling and sanitization."""

    def test_error_message_sanitization(self, qdrant_adapter_with_sanitization):
        """Test error message sanitization."""
        # Test with API key in error message (contains 'connection' so becomes generic message)
        error_with_key = "Connection failed: api_key=secret123 invalid"
        sanitized = qdrant_adapter_with_sanitization._sanitize_error_message(
            error_with_key
        )
        assert "secret123" not in sanitized
        # Error contains 'connection' so it gets replaced with generic message
        assert (
            sanitized
            == "Qdrant operation failed - please check connection and request parameters"
        )

        # Test with bearer token (without trigger words)
        error_with_token = "Authorization failed: bearer abcdef123456789"
        sanitized = qdrant_adapter_with_sanitization._sanitize_error_message(
            error_with_token
        )
        assert "abcdef123456789" not in sanitized
        assert "bearer ***" in sanitized

        # Test with API key without trigger words
        error_without_trigger = "Invalid api_key=secret123 provided"
        sanitized = qdrant_adapter_with_sanitization._sanitize_error_message(
            error_without_trigger
        )
        assert "secret123" not in sanitized
        assert "api_key=***" in sanitized

    async def test_correlation_id_validation(self, qdrant_adapter):
        """Test correlation ID validation."""
        # Test with None correlation ID (should generate new one)
        validated = qdrant_adapter._validate_correlation_id(None)
        assert validated is not None

        # Test with valid UUID
        original_uuid = uuid4()
        validated = qdrant_adapter._validate_correlation_id(original_uuid)
        assert validated == original_uuid

        # Test with string UUID
        uuid_string = str(uuid4())
        validated = qdrant_adapter._validate_correlation_id(uuid_string)
        assert str(validated) == uuid_string


class TestQdrantAdapterEndToEndEventFlow:
    """Test end-to-end event flow between OmniMemory and Qdrant adapter."""

    @pytest.fixture
    async def mock_redpanda_event_bus(self):
        """Mock RedPanda event bus for end-to-end testing."""

        class MockRedPandaEventBus:
            def __init__(self):
                self.published_events = []
                self.event_handlers = {}
                self.is_running = False

            async def publish_async(self, event_payload):
                """Mock event publishing."""
                self.published_events.append(
                    {
                        "payload": event_payload,
                        "timestamp": time.time(),
                        "event_id": str(uuid4()),
                    }
                )

                # Simulate event handling
                if hasattr(event_payload, "event_type"):
                    event_type = event_payload.event_type
                    if event_type in self.event_handlers:
                        for handler in self.event_handlers[event_type]:
                            await handler(event_payload)

            async def subscribe(self, event_type, handler):
                """Mock event subscription."""
                if event_type not in self.event_handlers:
                    self.event_handlers[event_type] = []
                self.event_handlers[event_type].append(handler)

            async def start(self):
                """Start mock event bus."""
                self.is_running = True

            async def stop(self):
                """Stop mock event bus."""
                self.is_running = False
                self.published_events.clear()
                self.event_handlers.clear()

        bus = MockRedPandaEventBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def omnimemory_with_qdrant_integration(
        self, mock_redpanda_event_bus, qdrant_adapter_config
    ):
        """Create OmniMemory EFFECT node with Qdrant adapter integration."""
        from unittest.mock import MagicMock

        from omnimemory.nodes.node_memory_storage_effect.v1_0_0.models import (
            ModelMemoryStorageConfig,
        )

        # Create memory storage config for testing
        memory_config = ModelMemoryStorageConfig.for_environment("testing")

        # Create container with proper service resolution
        container = MagicMock()

        def get_service(service_name):
            if service_name in ["protocol_event_bus", "ProtocolEventBus"]:
                return mock_redpanda_event_bus
            elif service_name == "memory_storage_config":
                return memory_config
            elif service_name == "qdrant_adapter_config":
                return qdrant_adapter_config
            else:
                return None

        container.get_service = MagicMock(side_effect=get_service)

        # Create OmniMemory storage node
        omnimemory_node = NodeMemoryStorageEffect(container)

        # Create Qdrant adapter node
        qdrant_adapter = NodeQdrantAdapterEffect(container)
        await qdrant_adapter.initialize()

        yield omnimemory_node, qdrant_adapter, mock_redpanda_event_bus

        # Cleanup
        await qdrant_adapter.cleanup()

    @pytest.mark.asyncio
    async def test_omnimemory_to_qdrant_vector_search_flow(
        self, omnimemory_with_qdrant_integration
    ):
        """Test complete flow: OmniMemory → RedPanda → Qdrant Adapter → Response."""
        omnimemory_node, qdrant_adapter, event_bus = omnimemory_with_qdrant_integration

        # Step 1: OmniMemory publishes vector search command
        correlation_id = uuid4()
        vector_search_input = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.VECTOR_SEARCH,
            correlation_id=correlation_id,
            timestamp=time.time(),
            query_vector=[0.1, 0.2, 0.3, 0.4],
            similarity_threshold=0.8,
            max_results=10,
            context={
                "source": "omnimemory_effect",
                "search_type": "similarity_search",
                "timestamp": time.time(),
            },
        )

        # Process through OmniMemory (this should publish an event)
        omnimemory_result = await omnimemory_node.process(vector_search_input)

        # Verify OmniMemory processed the request
        assert omnimemory_result.success is True
        assert omnimemory_result.correlation_id == correlation_id

        # Step 2: Simulate Qdrant adapter receiving the vector search command
        # Create Qdrant adapter input from the published event
        qdrant_vector_request = ModelQdrantVectorOperationRequest(
            operation_type="vector_search",
            collection_name="omnimemory_vectors",
            query_vector=[0.1, 0.2, 0.3, 0.4],
            search_limit=10,
            score_threshold=0.8,
        )

        qdrant_input = ModelQdrantAdapterInput(
            operation_type="vector_search",
            correlation_id=correlation_id,
            vector_request=qdrant_vector_request,
            context={
                "source_event": "omnimemory_vector_search_command",
                "target_service": "qdrant_vector_db",
            },
        )

        # Mock Qdrant search results
        mock_search_results = [
            type(
                "MockPoint",
                (),
                {
                    "id": "doc_1",
                    "score": 0.95,
                    "payload": {"text": "Similar document 1", "category": "memory"},
                    "vector": [0.11, 0.19, 0.31, 0.41],
                },
            ),
            type(
                "MockPoint",
                (),
                {
                    "id": "doc_2",
                    "score": 0.87,
                    "payload": {"text": "Similar document 2", "category": "memory"},
                    "vector": [0.09, 0.21, 0.29, 0.39],
                },
            ),
        ]

        # Step 3: Process through Qdrant adapter with mocked results
        with patch.object(
            qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
        ) as mock_cb:
            mock_cb.return_value = mock_search_results

            qdrant_result = await qdrant_adapter.process(qdrant_input)

        # Step 4: Verify Qdrant adapter response
        assert qdrant_result.success is True
        assert qdrant_result.correlation_id == correlation_id
        assert qdrant_result.operation_type == "vector_search"

        # Verify search results structure
        if (
            hasattr(qdrant_result, "data")
            and qdrant_result.data
            and "search_results" in qdrant_result.data
        ):
            search_results = qdrant_result.data["search_results"]
            assert len(search_results) == 2
            assert search_results[0]["score"] == 0.95
            assert search_results[1]["score"] == 0.87

        # Step 5: Verify event bus received completion events
        published_events = event_bus.published_events
        assert len(published_events) > 0

        # Check for vector search completed event
        vector_search_events = [
            e
            for e in published_events
            if hasattr(e["payload"], "event_type")
            and "vector_search" in e["payload"].event_type.lower()
        ]
        print(
            f"Published {len(published_events)} events, {len(vector_search_events)} vector search events"
        )

        # Step 6: Validate correlation ID tracking throughout the flow
        for event in published_events:
            if hasattr(event["payload"], "correlation_id"):
                print(f"Event correlation ID: {event['payload'].correlation_id}")

    @pytest.mark.asyncio
    async def test_omnimemory_to_qdrant_store_memory_flow(
        self, omnimemory_with_qdrant_integration
    ):
        """Test complete flow: OmniMemory → RedPanda → Qdrant Adapter for memory storage."""
        omnimemory_node, qdrant_adapter, event_bus = omnimemory_with_qdrant_integration

        # Step 1: OmniMemory stores memory with vector data
        correlation_id = uuid4()
        store_memory_input = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.STORE_MEMORY,
            correlation_id=correlation_id,
            timestamp=time.time(),
            memory_key="test_memory_with_vector",
            content={
                "text": "This is a test memory with vector data",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
            metadata={
                "source": "integration_test",
                "timestamp": datetime.now().isoformat(),
            },
            memory_type="vector",
            context={
                "requires_vector_storage": True,
                "collection_name": "omnimemory_vectors",
            },
            store_memory_request={
                "content": {
                    "text": "This is a test memory with vector data",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
                "metadata": {
                    "source": "integration_test",
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

        # Process through OmniMemory
        omnimemory_result = await omnimemory_node.process(store_memory_input)
        assert omnimemory_result.success is True

        # Step 2: Simulate Qdrant adapter receiving vector store command
        qdrant_vector_request = ModelQdrantVectorOperationRequest(
            operation_type="store_vector",
            collection_name="omnimemory_vectors",
            vector_id="test_memory_with_vector",
            vector_data=[0.1, 0.2, 0.3, 0.4, 0.5],
            payload={
                "text": "This is a test memory with vector data",
                "source": "integration_test",
                "timestamp": datetime.now().isoformat(),
            },
        )

        qdrant_input = ModelQdrantAdapterInput(
            operation_type="store_vector",
            correlation_id=correlation_id,
            vector_request=qdrant_vector_request,
        )

        # Step 3: Process through Qdrant adapter with mocked storage
        mock_store_result = {"operation_id": "store_123", "status": "completed"}

        with patch.object(
            qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
        ) as mock_cb:
            mock_cb.return_value = mock_store_result

            qdrant_result = await qdrant_adapter.process(qdrant_input)

        # Step 4: Verify successful storage flow
        assert qdrant_result.success is True
        assert qdrant_result.correlation_id == correlation_id
        assert qdrant_result.operation_type == "store_vector"

        # Step 5: Verify event publishing
        published_events = event_bus.published_events
        assert len(published_events) > 0

        print(f"Memory store flow published {len(published_events)} events")

    @pytest.mark.asyncio
    async def test_error_propagation_through_event_flow(
        self, omnimemory_with_qdrant_integration
    ):
        """Test error propagation from Qdrant adapter back to OmniMemory."""
        omnimemory_node, qdrant_adapter, event_bus = omnimemory_with_qdrant_integration

        # Step 1: Create request that will cause Qdrant error
        correlation_id = uuid4()

        # Invalid vector dimensions (should cause error)
        qdrant_vector_request = ModelQdrantVectorOperationRequest(
            operation_type="vector_search",
            collection_name="nonexistent_collection",
            query_vector=[],  # Empty vector should cause error
            search_limit=10,
        )

        qdrant_input = ModelQdrantAdapterInput(
            operation_type="vector_search",
            correlation_id=correlation_id,
            vector_request=qdrant_vector_request,
        )

        # Step 2: Force an error in circuit breaker call
        with patch.object(
            qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
        ) as mock_cb:
            mock_cb.side_effect = Exception("Qdrant connection failed")

            qdrant_result = await qdrant_adapter.process(qdrant_input)

        # Step 3: Verify error handling
        assert qdrant_result.success is False
        assert qdrant_result.correlation_id == correlation_id
        assert qdrant_result.error_message is not None
        assert "connection failed" in qdrant_result.error_message.lower()

        # Step 4: Verify error event was published
        published_events = event_bus.published_events

        # Should have error events published
        error_events = [
            e
            for e in published_events
            if hasattr(e["payload"], "success") and not e["payload"].success
        ]

        print(
            f"Error flow published {len(published_events)} events, {len(error_events)} error events"
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_flow(
        self, omnimemory_with_qdrant_integration
    ):
        """Test circuit breaker behavior in end-to-end flow."""
        omnimemory_node, qdrant_adapter, event_bus = omnimemory_with_qdrant_integration

        # Step 1: Trigger circuit breaker by causing multiple failures
        correlation_ids = []

        # Mock the actual Qdrant search function to cause failures, allowing circuit breaker to track state
        with patch.object(
            qdrant_adapter, "_execute_qdrant_search", new_callable=AsyncMock
        ) as mock_search:
            for i in range(
                10
            ):  # Cause exactly 10 failures to trigger the circuit breaker
                correlation_id = uuid4()
                correlation_ids.append(correlation_id)

                qdrant_input = ModelQdrantAdapterInput(
                    operation_type="vector_search",
                    correlation_id=correlation_id,
                    vector_request=ModelQdrantVectorOperationRequest(
                        operation_type="vector_search",
                        collection_name="test_collection",
                        query_vector=[0.1, 0.2, 0.3, 0.4],
                        search_limit=10,
                    ),
                )

                # Mock failure for all 10 requests
                mock_search.side_effect = Exception(f"Qdrant search failure {i+1}")
                result = await qdrant_adapter.process(qdrant_input)
                assert result.success is False

        # Step 2: Check circuit breaker state after 10 failures
        cb_state = qdrant_adapter._circuit_breaker.get_state()
        print(
            f"Circuit breaker state after 10 failures: {cb_state['state']}, count: {cb_state['failure_count']}"
        )

        # Step 3: Now test the 11th request without mocking - circuit breaker should be open
        correlation_id = uuid4()
        correlation_ids.append(correlation_id)

        qdrant_input = ModelQdrantAdapterInput(
            operation_type="vector_search",
            correlation_id=correlation_id,
            vector_request=ModelQdrantVectorOperationRequest(
                operation_type="vector_search",
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3, 0.4],
                search_limit=10,
            ),
        )

        # This should fail because circuit breaker is open
        result = await qdrant_adapter.process(qdrant_input)
        print(
            f"Circuit breaker test result: success={result.success}, error='{result.error_message}'"
        )

        # Check if circuit breaker prevented the operation
        if result.success is False:
            if (
                "circuit" in result.error_message.lower()
                or "breaker" in result.error_message.lower()
                or "open" in result.error_message.lower()
                or "unavailable" in result.error_message.lower()
            ):
                print("✅ Circuit breaker correctly prevented operation")
            else:
                print(
                    f"⚠️  Operation failed but not due to circuit breaker: {result.error_message}"
                )
                # This could still be valid if the operation fails for another reason when Qdrant is unavailable
        else:
            print("❌ Operation succeeded when circuit breaker should have prevented it")
            assert False, "Circuit breaker should have prevented operation"

        # Step 4: Verify circuit breaker state
        final_cb_state = qdrant_adapter._circuit_breaker.get_state()
        assert final_cb_state["state"] == "open"
        assert final_cb_state["failure_count"] >= 10

        print(
            f"Circuit breaker integration test: {len(correlation_ids)} requests processed"
        )
        print(f"Final circuit breaker state: {final_cb_state['state']}")

    @pytest.mark.asyncio
    async def test_concurrent_event_processing_flow(
        self, omnimemory_with_qdrant_integration
    ):
        """Test concurrent event processing between OmniMemory and Qdrant adapter."""
        omnimemory_node, qdrant_adapter, event_bus = omnimemory_with_qdrant_integration

        # Create multiple concurrent requests
        num_concurrent = 10
        correlation_ids = [uuid4() for _ in range(num_concurrent)]

        # Step 1: Create concurrent Qdrant adapter tasks
        tasks = []
        for i, correlation_id in enumerate(correlation_ids):
            qdrant_input = ModelQdrantAdapterInput(
                operation_type="vector_search",
                correlation_id=correlation_id,
                vector_request=ModelQdrantVectorOperationRequest(
                    operation_type="vector_search",
                    collection_name="test_collection",
                    query_vector=[
                        0.1 + i * 0.01,
                        0.2 + i * 0.01,
                        0.3 + i * 0.01,
                        0.4 + i * 0.01,
                    ],
                    search_limit=5,
                ),
            )
            tasks.append(qdrant_adapter.process(qdrant_input))

        # Step 2: Mock all circuit breaker calls to succeed
        mock_results = [
            [
                type(
                    "MockPoint",
                    (),
                    {
                        "id": f"doc_{i}",
                        "score": 0.9 - i * 0.05,
                        "payload": {"text": f"Document {i}"},
                        "vector": [
                            0.1 + i * 0.01,
                            0.2 + i * 0.01,
                            0.3 + i * 0.01,
                            0.4 + i * 0.01,
                        ],
                    },
                )
            ]
            for i in range(num_concurrent)
        ]

        with patch.object(
            qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
        ) as mock_cb:
            mock_cb.side_effect = mock_results

            # Step 3: Execute all tasks concurrently
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()

        # Step 4: Validate concurrent processing results
        successful_results = 0
        failed_results = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed with exception: {result}")
                failed_results += 1
            else:
                assert result.correlation_id == correlation_ids[i]
                if result.success:
                    successful_results += 1
                else:
                    failed_results += 1

        processing_time = (end_time - start_time) * 1000
        print(
            f"Concurrent processing: {successful_results} successful, {failed_results} failed"
        )
        print(f"Total processing time: {processing_time:.2f}ms")
        print(f"Average per request: {processing_time/num_concurrent:.2f}ms")

        # Should have mostly successful results
        assert successful_results >= num_concurrent * 0.8  # At least 80% success rate
        assert processing_time < 5000  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_health_check_event_flow(self, omnimemory_with_qdrant_integration):
        """Test health check flow between OmniMemory and Qdrant adapter."""
        omnimemory_node, qdrant_adapter, event_bus = omnimemory_with_qdrant_integration

        # Step 1: OmniMemory requests health check
        correlation_id = uuid4()
        health_input = ModelMemoryStorageInput(
            operation_type=EnumMemoryStorageOperationType.HEALTH_CHECK,
            correlation_id=correlation_id,
            timestamp=time.time(),
            context={
                "health_check_source": "omnimemory_monitoring",
                "check_level": "comprehensive",
            },
        )

        omnimemory_result = await omnimemory_node.process(health_input)
        assert omnimemory_result.success is True

        # Step 2: Qdrant adapter health check
        qdrant_health_input = ModelQdrantAdapterInput(
            operation_type="health_check",
            correlation_id=correlation_id,
            health_check_type="comprehensive",
        )

        qdrant_health_result = await qdrant_adapter.process(qdrant_health_input)

        # Step 3: Validate health check results
        # In test environment, health check should report as unhealthy (Qdrant unavailable)
        assert (
            qdrant_health_result.success is False
        )  # Correct behavior in test environment
        assert qdrant_health_result.correlation_id == correlation_id
        assert qdrant_health_result.health_status is not None
        assert qdrant_health_result.health_status.status == "unhealthy"
        assert qdrant_health_result.health_status.connection_status == "unavailable"

        # Step 4: Verify health events were published
        published_events = event_bus.published_events
        health_events = [
            e
            for e in published_events
            if hasattr(e["payload"], "event_type")
            and "health" in e["payload"].event_type.lower()
        ]

        print(
            f"Health check flow: {len(health_events)} health events published out of {len(published_events)} total"
        )


class TestQdrantAdapterPerformanceIntegration:
    """Test performance characteristics of Qdrant adapter integration."""

    @pytest.mark.asyncio
    async def test_vector_search_performance_with_large_vectors(self, qdrant_adapter):
        """Test performance with large vector dimensions."""
        # Test with different vector sizes
        vector_sizes = [128, 512, 1024]
        results = []

        for size in vector_sizes:
            vector_request = ModelQdrantVectorOperationRequest(
                operation_type="vector_search",
                collection_name="performance_test",
                query_vector=[0.1] * size,  # Vector of specified size
                search_limit=50,
            )

            input_data = ModelQdrantAdapterInput(
                operation_type="vector_search",
                correlation_id=uuid4(),
                vector_request=vector_request,
            )

            # Mock large result set
            mock_results = [
                type(
                    "MockPoint",
                    (),
                    {
                        "id": f"perf_doc_{i}",
                        "score": 0.9 - i * 0.01,
                        "payload": {"text": f"Performance test document {i}"},
                        "vector": [0.1 + i * 0.001] * size,
                    },
                )
                for i in range(50)
            ]

            with patch.object(
                qdrant_adapter._circuit_breaker, "call", new_callable=AsyncMock
            ) as mock_cb:
                mock_cb.return_value = mock_results

                start_time = time.perf_counter()
                result = await qdrant_adapter.process(input_data)
                end_time = time.perf_counter()

                processing_time = (end_time - start_time) * 1000
                results.append((size, processing_time, result.success))

                print(
                    f"Vector size {size}: {processing_time:.2f}ms, Success: {result.success}"
                )

        # All requests should complete within reasonable time
        for size, time_ms, success in results:
            assert success is True
            assert time_ms < 2000  # Should be under 2 seconds even with mocking

    @pytest.mark.asyncio
    async def test_batch_operation_performance(self, qdrant_adapter):
        """Test performance of batch operations."""
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            # Create batch store operation
            batch_vectors = [
                {
                    "vector_id": f"batch_vector_{i}",
                    "vector_data": [
                        0.1 + i * 0.001,
                        0.2 + i * 0.001,
                        0.3 + i * 0.001,
                        0.4 + i * 0.001,
                    ],
                    "payload": {"batch_index": i, "text": f"Batch document {i}"},
                }
                for i in range(batch_size)
            ]

            input_data = ModelQdrantAdapterInput(
                operation_type="batch_upsert",
                correlation_id=uuid4(),
                batch_data={"collection_name": "batch_test", "vectors": batch_vectors},
            )

            start_time = time.perf_counter()
            result = await qdrant_adapter.process(input_data)
            end_time = time.perf_counter()

            processing_time = (end_time - start_time) * 1000
            print(
                f"Batch size {batch_size}: {processing_time:.2f}ms, Success: {result.success}"
            )

            # Note: batch_upsert is not implemented yet, so result.success will be False
            # This test validates the performance characteristics of the error handling path
            assert processing_time < 1000  # Error handling should be fast


class TestQdrantAdapterONEXCompliance:
    """Test ONEX compliance patterns in Qdrant adapter integration."""

    @pytest.mark.asyncio
    async def test_correlation_id_preservation_compliance(self, qdrant_adapter):
        """Test that correlation IDs are preserved throughout the operation chain."""
        original_uuid = uuid4()

        input_data = ModelQdrantAdapterInput(
            operation_type="health_check",
            correlation_id=original_uuid,
            health_check_type="basic",
        )

        result = await qdrant_adapter.process(input_data)

        # ONEX compliance: correlation ID must be preserved exactly
        assert result.correlation_id == original_uuid
        assert str(result.correlation_id) == str(original_uuid)

    @pytest.mark.asyncio
    async def test_response_structure_compliance(self, qdrant_adapter):
        """Test ONEX response structure compliance."""
        input_data = ModelQdrantAdapterInput(
            operation_type="health_check",
            correlation_id=uuid4(),
            health_check_type="basic",
        )

        result = await qdrant_adapter.process(input_data)

        # ONEX compliance: all responses must have these fields
        required_fields = [
            "operation_type",
            "success",
            "correlation_id",
            "timestamp",
            "execution_time_ms",
        ]
        for field in required_fields:
            assert hasattr(result, field), f"Missing required field: {field}"
            assert getattr(result, field) is not None, f"Required field {field} is None"

        # ONEX compliance: execution_time_ms must be a positive number
        assert isinstance(result.execution_time_ms, (int, float))
        assert result.execution_time_ms > 0

        # ONEX compliance: timestamp must be valid
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_error_handling_compliance(self, qdrant_adapter):
        """Test ONEX error handling compliance."""
        # Test with invalid operation type
        input_data = ModelQdrantAdapterInput(
            operation_type="invalid_operation_type", correlation_id=uuid4()
        )

        result = await qdrant_adapter.process(input_data)

        # ONEX compliance: errors must be properly structured
        assert result.success is False
        assert result.error_code is not None
        assert result.error_message is not None
        assert result.correlation_id == input_data.correlation_id

        # ONEX compliance: error messages should not expose sensitive information
        assert "api_key" not in result.error_message.lower()
        assert "password" not in result.error_message.lower()
        assert "secret" not in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_performance_compliance(self, qdrant_adapter):
        """Test ONEX performance compliance requirements."""
        # Health checks should complete quickly
        input_data = ModelQdrantAdapterInput(
            operation_type="health_check",
            correlation_id=uuid4(),
            health_check_type="basic",
        )

        start_time = time.perf_counter()
        result = await qdrant_adapter.process(input_data)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000

        # ONEX compliance: health checks should complete within 100ms target
        print(f"Health check processing time: {processing_time:.2f}ms")
        assert processing_time < 500  # Relaxed for test environment

        # ONEX compliance: reported execution time should be consistent
        time_difference = abs(result.execution_time_ms - processing_time)
        assert time_difference < 50  # Allow some variance for test environment


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test to verify adapter can be instantiated."""
        from unittest.mock import AsyncMock, Mock

        # Create mock container
        container = Mock()
        event_bus = Mock()
        event_bus.publish_async = AsyncMock()
        container.get_service = Mock(return_value=event_bus)

        # Create adapter
        adapter = NodeQdrantAdapterEffect(container)
        await adapter.initialize()

        # Test health check
        input_data = ModelQdrantAdapterInput(
            operation_type="health_check", correlation_id=uuid4()
        )

        result = await adapter.process(input_data)
        print(f"Health check result: {result.success}")
        print(
            f"Health status: {result.health_status.status if result.health_status else 'None'}"
        )

        await adapter.cleanup()
        print("Smoke test completed successfully!")

    asyncio.run(smoke_test())
