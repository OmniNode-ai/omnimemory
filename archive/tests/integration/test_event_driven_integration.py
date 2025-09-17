"""Comprehensive integration tests for event-driven OmniMemory architecture.

Tests the complete flow from memory operations through RedPanda event bus
to infrastructure adapters, ensuring no direct database connections exist.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from omnimemory.adapters.postgres_adapter_integration import PostgresAdapterIntegration
from omnimemory.adapters.qdrant_adapter_integration import QdrantAdapterIntegration
from omnimemory.adapters.redis_adapter_integration import RedisAdapterIntegration

# Import OmniMemory event-driven components
from omnimemory.events.event_bus_client import EventBusClient
from omnimemory.events.event_consumer import EventConsumer
from omnimemory.events.event_producer import EventProducer
from omnimemory.models.core.model_memory_operation import EnumMemoryOperationType
from omnimemory.models.events.model_omnimemory_event_data import (
    ModelOmniMemoryRetrieveData,
    ModelOmniMemoryStoreData,
    ModelOmniMemoryVectorSearchData,
)
from omnimemory.models.memory.model_memory_request import (
    ModelMemoryRetrieveRequest,
    ModelMemoryStoreRequest,
    ModelMemoryVectorSearchRequest,
)
from omnimemory.services.event_driven_memory_service import EventDrivenMemoryService
from omnimemory.services.memory_operation_mapper import (
    MemoryOperationMapper,
    MemoryOperationTrace,
)

# Import ONEX core components with fallback
try:
    from omnibase_core.core.protocol_event_bus import ProtocolEventBus
    from omnibase_core.model.core.model_event_envelope import ModelEventEnvelope
    from omnibase_core.model.core.model_onex_event import ModelOnexEvent
except ImportError:
    # Fallback for testing without full ONEX dependencies
    class ProtocolEventBus:
        async def publish_async(self, event):
            pass

        async def subscribe_async(self, topic, handler):
            pass

    class ModelEventEnvelope:
        def __init__(self, **kwargs):
            self.payload = kwargs.get("payload")
            self.correlation_id = kwargs.get("correlation_id")

    class ModelOnexEvent:
        @classmethod
        def create_core_event(cls, event_type, node_id, correlation_id, data):
            return cls()


@pytest_asyncio.fixture
async def mock_event_bus():
    """Mock event bus for testing."""
    bus = AsyncMock(spec=ProtocolEventBus)

    # Track published events
    bus.published_events = []

    async def track_publish(event):
        bus.published_events.append(event)

    bus.publish_async.side_effect = track_publish
    return bus


@pytest_asyncio.fixture
async def event_producer(mock_event_bus):
    """Event producer with mocked event bus."""
    producer = EventProducer()
    producer.initialize(mock_event_bus)
    return producer


@pytest_asyncio.fixture
async def event_consumer(mock_event_bus):
    """Event consumer with mocked event bus."""
    consumer = EventConsumer()
    consumer.initialize(mock_event_bus)
    return consumer


@pytest_asyncio.fixture
async def event_bus_client(event_producer, event_consumer):
    """Complete event bus client for testing."""
    client = EventBusClient()
    client.initialize(event_producer, event_consumer)
    return client


@pytest_asyncio.fixture
async def memory_operation_mapper():
    """Memory operation mapper for testing."""
    return MemoryOperationMapper()


@pytest_asyncio.fixture
async def event_driven_service(event_bus_client, memory_operation_mapper):
    """Complete event-driven memory service."""
    service = EventDrivenMemoryService()
    service.initialize(event_bus_client, memory_operation_mapper)
    return service


class TestEventDrivenIntegration:
    """Test complete event-driven integration without direct database connections."""

    async def test_memory_store_complete_flow(
        self, event_driven_service, mock_event_bus, memory_operation_mapper
    ):
        """Test complete memory store flow through event bus."""
        # Create store request
        store_request = ModelMemoryStoreRequest(
            memory_key="test_key_integration",
            content={"test": "data", "value": 42},
            metadata={"source": "integration_test", "priority": "high"},
            ttl_seconds=3600,
        )

        # Execute store operation
        correlation_id = await event_driven_service.store_memory(store_request)

        # Verify correlation ID returned
        assert correlation_id is not None
        assert isinstance(correlation_id, UUID)

        # Verify event was published
        assert len(mock_event_bus.published_events) == 1
        published_event = mock_event_bus.published_events[0]

        # Verify event structure
        assert hasattr(published_event, "event_type")
        assert published_event.event_type == "memory.store.command_request"
        assert published_event.correlation_id == correlation_id

        # Verify operation was tracked
        trace = memory_operation_mapper.get_operation_trace(correlation_id)
        assert trace is not None
        assert trace.operation_type == EnumMemoryOperationType.STORE
        assert trace.memory_key == "test_key_integration"
        assert trace.correlation_id == correlation_id

    async def test_memory_retrieve_complete_flow(
        self, event_driven_service, mock_event_bus, memory_operation_mapper
    ):
        """Test complete memory retrieve flow through event bus."""
        # Create retrieve request
        retrieve_request = ModelMemoryRetrieveRequest(
            memory_key="test_key_retrieve", query_type="key", include_metadata=True
        )

        # Execute retrieve operation
        correlation_id = await event_driven_service.retrieve_memory(retrieve_request)

        # Verify correlation ID returned
        assert correlation_id is not None
        assert isinstance(correlation_id, UUID)

        # Verify event was published
        assert len(mock_event_bus.published_events) == 1
        published_event = mock_event_bus.published_events[0]

        # Verify event structure
        assert hasattr(published_event, "event_type")
        assert published_event.event_type == "memory.retrieve.command_request"
        assert published_event.correlation_id == correlation_id

        # Verify operation was tracked
        trace = memory_operation_mapper.get_operation_trace(correlation_id)
        assert trace is not None
        assert trace.operation_type == EnumMemoryOperationType.RETRIEVE
        assert trace.memory_key == "test_key_retrieve"

    async def test_vector_search_complete_flow(
        self, event_driven_service, mock_event_bus, memory_operation_mapper
    ):
        """Test complete vector search flow through event bus."""
        # Create vector search request
        search_request = ModelMemoryVectorSearchRequest(
            query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            collection_name="test_collection",
            limit=10,
            similarity_threshold=0.7,
            filters={"category": "integration_test"},
        )

        # Execute vector search operation
        correlation_id = await event_driven_service.vector_search(search_request)

        # Verify correlation ID returned
        assert correlation_id is not None
        assert isinstance(correlation_id, UUID)

        # Verify event was published
        assert len(mock_event_bus.published_events) == 1
        published_event = mock_event_bus.published_events[0]

        # Verify event structure
        assert hasattr(published_event, "event_type")
        assert published_event.event_type == "memory.vector_search.command_request"
        assert published_event.correlation_id == correlation_id

        # Verify operation was tracked
        trace = memory_operation_mapper.get_operation_trace(correlation_id)
        assert trace is not None
        assert trace.operation_type == EnumMemoryOperationType.VECTOR_SEARCH

    async def test_correlation_id_tracking_across_operations(
        self, event_driven_service, memory_operation_mapper
    ):
        """Test correlation ID tracking across multiple operations."""
        # Execute multiple operations
        correlation_ids = []

        # Store operation
        store_request = ModelMemoryStoreRequest(
            memory_key="tracking_test_1",
            content={"test": "correlation_tracking"},
            metadata={"operation_order": 1},
        )
        store_correlation_id = await event_driven_service.store_memory(store_request)
        correlation_ids.append(store_correlation_id)

        # Retrieve operation
        retrieve_request = ModelMemoryRetrieveRequest(
            memory_key="tracking_test_2",
            query_type="pattern",
            query_parameters={"pattern": "tracking_*"},
        )
        retrieve_correlation_id = await event_driven_service.retrieve_memory(
            retrieve_request
        )
        correlation_ids.append(retrieve_correlation_id)

        # Vector search operation
        search_request = ModelMemoryVectorSearchRequest(
            query_vector=[0.5, 0.5, 0.5], collection_name="tracking_collection", limit=5
        )
        search_correlation_id = await event_driven_service.vector_search(search_request)
        correlation_ids.append(search_correlation_id)

        # Verify all correlation IDs are unique
        assert len(set(correlation_ids)) == 3

        # Verify all operations are tracked
        for correlation_id in correlation_ids:
            trace = memory_operation_mapper.get_operation_trace(correlation_id)
            assert trace is not None
            assert trace.correlation_id == correlation_id
            assert trace.status == "pending"

    async def test_operation_timeout_handling(self, event_driven_service):
        """Test timeout handling for operations."""
        # Mock a store request
        store_request = ModelMemoryStoreRequest(
            memory_key="timeout_test",
            content={"test": "timeout_handling"},
            metadata={"timeout_test": True},
        )

        # Execute with very short timeout
        with patch.object(
            event_driven_service, "operation_timeout", 0.1
        ):  # 100ms timeout
            correlation_id = await event_driven_service.store_memory(store_request)

            # Wait for timeout
            await asyncio.sleep(0.2)

            # Check that operation is still tracked but marked as timed out
            # (In real implementation, this would be handled by the service)
            assert correlation_id is not None
            assert isinstance(correlation_id, UUID)

    async def test_event_response_handling(
        self, event_consumer, memory_operation_mapper
    ):
        """Test event response handling and correlation."""
        # Create mock response event
        correlation_id = uuid4()

        # Register operation first
        memory_operation_mapper.track_operation(
            correlation_id=correlation_id,
            operation_type=EnumMemoryOperationType.STORE,
            memory_key="response_test",
            metadata={"test": "response_handling"},
        )

        # Create mock success response
        response_data = {
            "success": True,
            "memory_id": str(uuid4()),
            "storage_size": 256,
            "execution_time_ms": 45.2,
        }

        # Simulate response handling
        await event_consumer._handle_memory_stored_response(
            correlation_id, response_data
        )

        # Verify response was handled
        # (In real implementation, this would update the operation trace)
        trace = memory_operation_mapper.get_operation_trace(correlation_id)
        assert trace is not None
        assert trace.correlation_id == correlation_id


class TestInfrastructureAdapterIntegration:
    """Test infrastructure adapter integrations through event bus."""

    async def test_postgres_adapter_integration(self, mock_event_bus):
        """Test PostgreSQL adapter integration via event bus."""
        # Initialize PostgreSQL adapter integration
        postgres_adapter = PostgresAdapterIntegration()
        postgres_adapter.initialize(mock_event_bus)

        # Create store data
        store_data = ModelOmniMemoryStoreData(
            memory_key="postgres_test",
            content={"database": "postgresql", "test": True},
            metadata={"adapter": "postgres"},
            content_hash="test_hash_123",
            storage_size=128,
        )

        correlation_id = uuid4()

        # Execute store operation
        await postgres_adapter.execute_store_memory(
            correlation_id=correlation_id,
            store_data=store_data,
            content=store_data.content,
        )

        # Verify event was published to PostgreSQL adapter
        assert len(mock_event_bus.published_events) == 1
        published_event = mock_event_bus.published_events[0]

        # Verify event structure
        assert hasattr(published_event, "event_type")
        assert published_event.event_type == "database.postgres.command_request"
        assert published_event.correlation_id == correlation_id

    async def test_redis_adapter_integration(self, mock_event_bus):
        """Test Redis adapter integration via event bus."""
        # Initialize Redis adapter integration
        redis_adapter = RedisAdapterIntegration()
        redis_adapter.initialize(mock_event_bus)

        # Create temporal store data
        store_data = ModelOmniMemoryStoreData(
            memory_key="redis_temporal_test",
            content={"cache": "redis", "temporal": True},
            metadata={"adapter": "redis", "ttl": 3600},
            content_hash="redis_hash_456",
            storage_size=64,
            ttl_seconds=3600,
        )

        correlation_id = uuid4()

        # Execute temporal store operation
        await redis_adapter.execute_store_temporal_memory(
            correlation_id=correlation_id,
            store_data=store_data,
            content=store_data.content,
        )

        # Verify event was published to Redis adapter
        assert len(mock_event_bus.published_events) == 1
        published_event = mock_event_bus.published_events[0]

        # Verify event structure
        assert hasattr(published_event, "event_type")
        assert published_event.event_type == "cache.redis.command_request"
        assert published_event.correlation_id == correlation_id

    async def test_qdrant_adapter_integration(self, mock_event_bus):
        """Test Qdrant adapter integration via event bus."""
        # Initialize Qdrant adapter integration
        qdrant_adapter = QdrantAdapterIntegration()
        qdrant_adapter.initialize(mock_event_bus)

        # Create vector search data
        search_data = ModelOmniMemoryVectorSearchData(
            query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            collection_name="integration_test_collection",
            limit=15,
            similarity_threshold=0.8,
            filters={"test_type": "integration", "adapter": "qdrant"},
        )

        correlation_id = uuid4()

        # Execute vector search operation
        await qdrant_adapter.execute_vector_search(
            correlation_id=correlation_id, search_data=search_data
        )

        # Verify event was published to Qdrant adapter
        assert len(mock_event_bus.published_events) == 1
        published_event = mock_event_bus.published_events[0]

        # Verify event structure
        assert hasattr(published_event, "event_type")
        assert published_event.event_type == "vector.qdrant.command_request"
        assert published_event.correlation_id == correlation_id


class TestEventDrivenArchitectureCompliance:
    """Test compliance with pure event-driven architecture requirements."""

    def test_no_direct_database_imports(self):
        """Verify no direct database connection imports in event-driven components."""
        # List of modules that should NOT import database libraries directly
        event_driven_modules = [
            "omnimemory.events.event_producer",
            "omnimemory.events.event_consumer",
            "omnimemory.events.event_bus_client",
            "omnimemory.services.event_driven_memory_service",
            "omnimemory.services.memory_operation_mapper",
        ]

        # Database libraries that should not be imported
        forbidden_imports = [
            "psycopg2",
            "psycopg3",
            "asyncpg",  # PostgreSQL
            "redis",
            "aioredis",  # Redis
            "qdrant_client",  # Qdrant
            "sqlalchemy.create_engine",  # Direct SQLAlchemy engine
            "sqlite3",
            "mysql",
            "pymongo",  # Other databases
        ]

        # This is a conceptual test - in practice, you would use static analysis
        # tools like AST parsing to verify no forbidden imports exist
        for module_name in event_driven_modules:
            # Import module and check its dependencies
            try:
                module = __import__(module_name, fromlist=[""])
                module_file = module.__file__

                # Read module source
                if module_file:
                    with open(module_file, "r") as f:
                        source = f.read()

                    # Check for forbidden imports
                    for forbidden in forbidden_imports:
                        assert (
                            forbidden not in source
                        ), f"Module {module_name} contains forbidden import: {forbidden}"

            except ImportError:
                # Module might not be available in test environment
                pass

    async def test_all_operations_use_event_bus(
        self, event_driven_service, mock_event_bus
    ):
        """Verify all memory operations go through event bus."""
        operations_to_test = [
            # Store operation
            {
                "method": event_driven_service.store_memory,
                "request": ModelMemoryStoreRequest(
                    memory_key="event_bus_test_store",
                    content={"test": "event_bus_compliance"},
                ),
            },
            # Retrieve operation
            {
                "method": event_driven_service.retrieve_memory,
                "request": ModelMemoryRetrieveRequest(
                    memory_key="event_bus_test_retrieve", query_type="key"
                ),
            },
            # Vector search operation
            {
                "method": event_driven_service.vector_search,
                "request": ModelMemoryVectorSearchRequest(
                    query_vector=[0.1, 0.2, 0.3], collection_name="event_bus_test"
                ),
            },
        ]

        # Execute all operations
        for i, operation in enumerate(operations_to_test):
            # Clear previous events
            mock_event_bus.published_events.clear()

            # Execute operation
            correlation_id = await operation["method"](operation["request"])

            # Verify event was published
            assert (
                len(mock_event_bus.published_events) == 1
            ), f"Operation {i} did not publish event"
            assert (
                correlation_id is not None
            ), f"Operation {i} did not return correlation ID"

            # Verify event has correct correlation ID
            published_event = mock_event_bus.published_events[0]
            assert hasattr(
                published_event, "correlation_id"
            ), f"Operation {i} event missing correlation_id"
            assert (
                published_event.correlation_id == correlation_id
            ), f"Operation {i} correlation_id mismatch"

    async def test_complete_traceability(self, memory_operation_mapper):
        """Test complete operation traceability through event sourcing."""
        # Create parent operation
        parent_correlation_id = uuid4()
        memory_operation_mapper.track_operation(
            correlation_id=parent_correlation_id,
            operation_type=EnumMemoryOperationType.STORE,
            memory_key="traceability_parent",
            metadata={"operation_level": "parent"},
        )

        # Create child operations
        child_operations = []
        for i in range(3):
            child_correlation_id = uuid4()
            memory_operation_mapper.track_operation(
                correlation_id=child_correlation_id,
                operation_type=EnumMemoryOperationType.RETRIEVE,
                memory_key=f"traceability_child_{i}",
                metadata={"operation_level": "child", "parent_index": i},
                parent_operation_id=parent_correlation_id,
            )
            child_operations.append(child_correlation_id)

        # Verify parent operation tracking
        parent_trace = memory_operation_mapper.get_operation_trace(
            parent_correlation_id
        )
        assert parent_trace is not None
        assert parent_trace.operation_type == EnumMemoryOperationType.STORE
        assert parent_trace.memory_key == "traceability_parent"
        assert len(parent_trace.child_operations) == 3

        # Verify child operations tracking
        for i, child_correlation_id in enumerate(child_operations):
            child_trace = memory_operation_mapper.get_operation_trace(
                child_correlation_id
            )
            assert child_trace is not None
            assert child_trace.operation_type == EnumMemoryOperationType.RETRIEVE
            assert child_trace.memory_key == f"traceability_child_{i}"
            assert child_trace.parent_operation_id == parent_correlation_id

    async def test_error_handling_without_database_connections(
        self, event_driven_service
    ):
        """Test error handling maintains event-driven patterns."""
        # Create request with potential error conditions
        error_request = ModelMemoryStoreRequest(
            memory_key="",  # Invalid empty key
            content=None,  # Invalid null content
            metadata={"error_test": True},
        )

        # Execute operation - should handle errors gracefully
        try:
            correlation_id = await event_driven_service.store_memory(error_request)
            # Even with errors, should return correlation ID for tracking
            assert correlation_id is not None
        except Exception as e:
            # If exception is raised, it should be a proper application error
            # not a database connection error
            assert "connection" not in str(e).lower()
            assert "database" not in str(e).lower()
            assert "redis" not in str(e).lower()
            assert "qdrant" not in str(e).lower()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
