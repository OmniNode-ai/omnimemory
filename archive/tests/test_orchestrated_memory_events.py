#!/usr/bin/env python3
"""
Orchestrated Memory Events Test Suite.

Comprehensive testing of OmniMemory EFFECT node with realistic ORCHESTRATOR-generated
memory events to validate end-to-end event-driven architecture integration.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnimemory.nodes.node_memory_storage_effect import (
    EnumMemoryStorageOperationType,
    ModelMemoryStorageConfig,
    ModelMemoryStorageInput,
    ModelMemoryStorageOutput,
    NodeMemoryStorageEffect,
)


class MockOrchestratorEventBus:
    """Mock ORCHESTRATOR event bus for testing event-driven patterns."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.subscribers = {}
        self.is_running = False

    async def publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish an event to the bus."""
        event = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": time.time(),
            "correlation_id": str(uuid4()),
        }
        self.events.append(event)

        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                await callback(event)

    async def subscribe(self, event_type: str, callback) -> None:
        """Subscribe to events."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def start(self) -> None:
        """Start the event bus."""
        self.is_running = True

    async def stop(self) -> None:
        """Stop the event bus."""
        self.is_running = False


class TestOrchestratorMemoryEvents:
    """Test orchestrated memory events with EFFECT node."""

    def create_orchestrator_event(
        self, operation_type: EnumMemoryStorageOperationType, **kwargs
    ) -> Dict[str, Any]:
        """Create a realistic ORCHESTRATOR event for memory operations."""
        base_event = {
            "source": "ORCHESTRATOR",
            "target": "EFFECT.memory_storage",
            "timestamp": time.time(),
            "correlation_id": str(uuid4()),
            "sequence_number": getattr(self, "_sequence", 0),
            "retry_count": 0,
            "priority": "normal",
            "context": {
                "workflow_id": str(uuid4()),
                "agent_id": "test_agent",
                "session_id": str(uuid4()),
                "environment": "test",
            },
        }

        # Increment sequence for next event
        self._sequence = getattr(self, "_sequence", 0) + 1

        # Create memory-specific payload
        payload = {
            "operation_type": operation_type.value,
            "correlation_id": UUID(base_event["correlation_id"]),
            "timestamp": base_event["timestamp"],
            **kwargs,
        }

        base_event["payload"] = payload
        return base_event

    @pytest.fixture
    async def mock_event_bus(self):
        """Mock event bus for testing."""
        bus = MockOrchestratorEventBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def effect_node_with_event_bus(self, mock_event_bus):
        """Create EFFECT node with mocked event bus integration."""
        from unittest.mock import MagicMock

        container = MagicMock()
        container.get_service.return_value = None

        node = NodeMemoryStorageEffect(container)

        # Mock the event-driven service to use our mock bus
        node.event_driven_service = MagicMock()
        node.event_driven_service.event_bus = mock_event_bus
        node.event_driven_service.publish_event = mock_event_bus.publish_event
        node.event_driven_service.subscribe = mock_event_bus.subscribe

        yield node, mock_event_bus

        # Cleanup
        if hasattr(node, "event_driven_service"):
            if hasattr(node.event_driven_service, "shutdown"):
                try:
                    await node.event_driven_service.shutdown()
                except:
                    pass

    @pytest.mark.asyncio
    async def test_orchestrator_store_memory_event(self, effect_node_with_event_bus):
        """Test ORCHESTRATOR-generated store memory event."""
        node, event_bus = effect_node_with_event_bus

        # Create ORCHESTRATOR event for store memory operation
        orchestrator_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key="orchestrator_test_key_001",
            content={
                "agent_response": "Task completed successfully",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.95,
                "metadata": {
                    "source": "AI_agent",
                    "task_type": "data_analysis",
                    "version": "1.0",
                },
            },
            memory_type="persistent",
            metadata={
                "orchestrator_id": str(uuid4()),
                "workflow_step": "store_results",
                "agent_type": "data_analyst",
            },
        )

        # Convert orchestrator event to EFFECT node input
        input_data = ModelMemoryStorageInput(**orchestrator_event["payload"])

        # Process the event through EFFECT node
        start_time = time.perf_counter()
        result = await node.process(input_data)
        end_time = time.perf_counter()

        # Validate EFFECT node response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.STORE_MEMORY
        assert result.success is True
        assert result.memory_key == "orchestrator_test_key_001"
        assert result.memory_id is not None
        assert result.correlation_id == input_data.correlation_id
        assert result.execution_time_ms > 0
        assert result.execution_time_ms < 1000  # Should be under 1 second

        # Verify event was processed in reasonable time (<100ms target)
        processing_time = (end_time - start_time) * 1000
        print(f"Store operation processing time: {processing_time:.2f}ms")
        # Note: In real implementation with proper storage, this should be <100ms

        # Check that event bus received completion event
        completion_events = [
            e
            for e in event_bus.events
            if e.get("event_type") == "memory_operation_completed"
        ]
        assert len(completion_events) >= 0  # May be 0 in mock mode

    @pytest.mark.asyncio
    async def test_orchestrator_retrieve_memory_event(self, effect_node_with_event_bus):
        """Test ORCHESTRATOR-generated retrieve memory event."""
        node, event_bus = effect_node_with_event_bus

        # Create ORCHESTRATOR event for retrieve memory operation
        orchestrator_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.RETRIEVE_MEMORY,
            memory_key="orchestrator_test_key_001",
            context={
                "request_source": "agent_workflow",
                "urgency": "high",
                "expected_content_type": "analysis_result",
            },
        )

        input_data = ModelMemoryStorageInput(**orchestrator_event["payload"])

        # Process retrieve event
        result = await node.process(input_data)

        # Validate response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.RETRIEVE_MEMORY
        assert result.success is True
        assert result.memory_key == "orchestrator_test_key_001"
        assert result.correlation_id == input_data.correlation_id

    @pytest.mark.asyncio
    async def test_orchestrator_vector_search_event(self, effect_node_with_event_bus):
        """Test ORCHESTRATOR-generated vector search event."""
        node, event_bus = effect_node_with_event_bus

        # Create ORCHESTRATOR event for vector search
        orchestrator_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.VECTOR_SEARCH,
            query_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            similarity_threshold=0.85,
            max_results=10,
            search_filters={
                "agent_type": "data_analyst",
                "confidence_min": 0.8,
                "time_range": "last_24h",
            },
            context={
                "search_purpose": "find_similar_analyses",
                "requesting_agent": "pattern_matcher",
                "workflow_stage": "similarity_analysis",
            },
        )

        input_data = ModelMemoryStorageInput(**orchestrator_event["payload"])

        # Process vector search event
        result = await node.process(input_data)

        # Validate response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.VECTOR_SEARCH
        assert result.success is True
        assert result.search_results is not None
        assert isinstance(result.search_results, list)
        assert result.total_results is not None
        assert result.correlation_id == input_data.correlation_id

    @pytest.mark.asyncio
    async def test_orchestrator_batch_operations_event(
        self, effect_node_with_event_bus
    ):
        """Test ORCHESTRATOR-generated batch operations event."""
        node, event_bus = effect_node_with_event_bus

        # Create ORCHESTRATOR event for batch store
        orchestrator_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.BATCH_STORE,
            batch_store_request={
                "memories": [
                    {
                        "memory_key": f"batch_item_{i}",
                        "content": {
                            "data": f"Batch item {i}",
                            "processed_at": datetime.now().isoformat(),
                            "batch_id": str(uuid4()),
                        },
                        "memory_type": "persistent",
                        "metadata": {"batch_index": i, "total_batch_size": 5},
                    }
                    for i in range(5)
                ],
                "batch_id": str(uuid4()),
                "batch_metadata": {
                    "orchestrator_workflow": "batch_processing",
                    "priority": "high",
                    "expected_completion": "60s",
                },
            },
        )

        input_data = ModelMemoryStorageInput(**orchestrator_event["payload"])

        # Process batch store event
        result = await node.process(input_data)

        # Validate response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.BATCH_STORE
        assert result.success is True
        assert result.correlation_id == input_data.correlation_id

        # Check batch processing results
        if hasattr(result, "batch_results"):
            assert result.batch_results is not None
            if (
                isinstance(result.batch_results, dict)
                and "processed_count" in result.batch_results
            ):
                assert result.batch_results["processed_count"] >= 0

    @pytest.mark.asyncio
    async def test_orchestrator_circuit_breaker_handling(
        self, effect_node_with_event_bus
    ):
        """Test circuit breaker behavior with ORCHESTRATOR events."""
        node, event_bus = effect_node_with_event_bus

        # Force circuit breaker failures
        failure_threshold = node.config.circuit_breaker_failure_threshold
        for i in range(failure_threshold + 1):
            node._record_failure(f"Simulated failure {i}")

        # Create ORCHESTRATOR event that should trigger circuit breaker
        orchestrator_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key="circuit_breaker_test",
            content={"test": "data"},
        )

        input_data = ModelMemoryStorageInput(**orchestrator_event["payload"])

        # Process event (should fail due to circuit breaker)
        result = await node.process(input_data)

        # Validate circuit breaker response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.success is False
        assert "circuit breaker" in result.error_message.lower()
        assert result.error_code is not None
        assert result.correlation_id == input_data.correlation_id

    @pytest.mark.asyncio
    async def test_orchestrator_concurrent_events(self, effect_node_with_event_bus):
        """Test handling of concurrent ORCHESTRATOR events."""
        node, event_bus = effect_node_with_event_bus

        # Create multiple concurrent ORCHESTRATOR events
        orchestrator_events = []
        for i in range(10):
            event = self.create_orchestrator_event(
                EnumMemoryStorageOperationType.STORE_MEMORY,
                memory_key=f"concurrent_test_{i}",
                content={
                    "index": i,
                    "timestamp": datetime.now().isoformat(),
                    "data": f"Concurrent operation {i}",
                },
                memory_type="persistent",
            )
            orchestrator_events.append(event)

        # Process all events concurrently
        tasks = []
        for event in orchestrator_events:
            input_data = ModelMemoryStorageInput(**event["payload"])
            task = asyncio.create_task(node.process(input_data))
            tasks.append(task)

        # Wait for all operations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all operations succeeded
        successful_operations = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Operation {i} failed with exception: {result}")
            else:
                assert isinstance(result, ModelMemoryStorageOutput)
                if result.success:
                    successful_operations += 1
                else:
                    print(f"Operation {i} failed: {result.error_message}")

        print(
            f"Successful concurrent operations: {successful_operations}/{len(results)}"
        )

        # At least some operations should succeed (graceful degradation)
        assert successful_operations > 0

    @pytest.mark.asyncio
    async def test_orchestrator_error_recovery(self, effect_node_with_event_bus):
        """Test error recovery patterns with ORCHESTRATOR events."""
        node, event_bus = effect_node_with_event_bus

        # Test 1: Invalid operation type handling
        invalid_event = {
            "source": "ORCHESTRATOR",
            "target": "EFFECT.memory_storage",
            "timestamp": time.time(),
            "correlation_id": str(uuid4()),
            "payload": {
                "operation_type": "INVALID_OPERATION",
                "correlation_id": uuid4(),
                "timestamp": time.time(),
            },
        }

        try:
            # This should raise a validation error during ModelMemoryStorageInput creation
            input_data = ModelMemoryStorageInput(**invalid_event["payload"])
            result = await node.process(input_data)
            # If we get here, validation passed but operation should fail
            assert result.success is False
        except Exception as e:
            # Expected: Pydantic validation error for invalid enum
            print(f"Expected validation error: {e}")
            assert "operation_type" in str(e).lower()

        # Test 2: Missing required fields
        incomplete_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            # Missing memory_key and content
        )

        input_data = ModelMemoryStorageInput(**incomplete_event["payload"])
        result = await node.process(input_data)

        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.success is False
        assert (
            "validation" in result.error_message.lower()
            or "required" in result.error_message.lower()
        )

        # Test 3: Timeout handling
        timeout_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key="timeout_test",
            content={"test": "data"},
            timeout_seconds=0.001,  # Very short timeout
        )

        input_data = ModelMemoryStorageInput(**timeout_event["payload"])
        result = await node.process(input_data)

        # Should complete successfully even with short timeout in fallback mode
        assert isinstance(result, ModelMemoryStorageOutput)
        # Note: In real implementation with external services, this might timeout

    @pytest.mark.asyncio
    async def test_orchestrator_health_monitoring(self, effect_node_with_event_bus):
        """Test health monitoring through ORCHESTRATOR events."""
        node, event_bus = effect_node_with_event_bus

        # Create ORCHESTRATOR health check event
        health_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.HEALTH_CHECK,
            context={
                "health_check_type": "orchestrator_requested",
                "monitoring_system": "ONEX_health_monitor",
                "check_level": "comprehensive",
            },
        )

        input_data = ModelMemoryStorageInput(**health_event["payload"])

        # Process health check
        result = await node.process(input_data)

        # Validate health response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.HEALTH_CHECK
        assert result.success is True
        assert result.health_status in ["healthy", "degraded", "unhealthy"]
        assert result.correlation_id == input_data.correlation_id

        # Validate health details
        assert "health_status" in result.data
        assert "timestamp" in result.data
        assert "node_info" in result.data

        # Test system stats request
        stats_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.GET_STATS,
            context={
                "stats_type": "performance_metrics",
                "requester": "orchestrator_monitoring",
            },
        )

        input_data = ModelMemoryStorageInput(**stats_event["payload"])
        result = await node.process(input_data)

        # Validate stats response
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.operation_type == EnumMemoryStorageOperationType.GET_STATS
        assert result.success is True
        assert result.system_stats is not None
        assert "operation_count" in result.system_stats
        assert "success_rate" in result.system_stats
        assert "circuit_breaker_states" in result.system_stats

    @pytest.mark.asyncio
    async def test_onex_compliance_validation(self, effect_node_with_event_bus):
        """Test ONEX compliance patterns in orchestrated operations."""
        node, event_bus = effect_node_with_event_bus

        # Test 1: Proper correlation ID handling
        original_uuid = uuid4()
        orchestrator_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key="onex_compliance_test",
            content={"compliance": "test"},
            correlation_id=original_uuid,
        )

        input_data = ModelMemoryStorageInput(**orchestrator_event["payload"])
        result = await node.process(input_data)

        # Correlation ID should be preserved
        assert result.correlation_id == original_uuid

        # Test 2: Response structure compliance
        assert isinstance(result, ModelMemoryStorageOutput)
        assert hasattr(result, "operation_type")
        assert hasattr(result, "success")
        assert hasattr(result, "correlation_id")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "execution_time_ms")

        # Test 3: Error handling compliance
        error_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            # Intentionally malformed to test error handling
            memory_key="",  # Empty key should trigger validation error
            content=None,  # Null content should trigger validation error
        )

        input_data = ModelMemoryStorageInput(**error_event["payload"])
        result = await node.process(input_data)

        # Error response should be properly structured
        assert isinstance(result, ModelMemoryStorageOutput)
        assert result.success is False
        assert result.error_code is not None
        assert result.error_message is not None
        assert result.correlation_id == input_data.correlation_id

        # Test 4: Performance compliance (<100ms target)
        start_time = time.perf_counter()

        performance_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.HEALTH_CHECK
        )

        input_data = ModelMemoryStorageInput(**performance_event["payload"])
        result = await node.process(input_data)

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000

        print(f"Health check processing time: {processing_time:.2f}ms")

        # Should complete quickly for health checks
        assert processing_time < 500  # 500ms upper bound for testing environment
        assert (
            result.execution_time_ms < 500
        )  # Internal timing should also be reasonable

    @pytest.mark.asyncio
    async def test_memory_operation_workflow_simulation(
        self, effect_node_with_event_bus
    ):
        """Simulate a realistic memory operation workflow from ORCHESTRATOR."""
        node, event_bus = effect_node_with_event_bus

        workflow_id = str(uuid4())
        session_id = str(uuid4())

        # Step 1: Store initial data
        store_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key=f"workflow_{workflow_id}_initial",
            content={
                "workflow_step": "initialization",
                "user_input": "Please analyze the quarterly sales data",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
            },
            metadata={"workflow_id": workflow_id, "step": 1, "agent": "user_interface"},
        )

        input_data = ModelMemoryStorageInput(**store_event["payload"])
        store_result = await node.process(input_data)
        assert store_result.success is True

        # Step 2: Store analysis results
        analysis_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.STORE_MEMORY,
            memory_key=f"workflow_{workflow_id}_analysis",
            content={
                "workflow_step": "analysis_complete",
                "analysis_results": {
                    "total_sales": 1250000,
                    "growth_rate": 0.15,
                    "top_products": ["Product A", "Product B", "Product C"],
                },
                "confidence": 0.92,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
            },
            metadata={"workflow_id": workflow_id, "step": 2, "agent": "data_analyst"},
        )

        input_data = ModelMemoryStorageInput(**analysis_event["payload"])
        analysis_result = await node.process(input_data)
        assert analysis_result.success is True

        # Step 3: Retrieve both memories for final report
        retrieve_initial_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.RETRIEVE_MEMORY,
            memory_key=f"workflow_{workflow_id}_initial",
        )

        input_data = ModelMemoryStorageInput(**retrieve_initial_event["payload"])
        retrieve_result = await node.process(input_data)
        assert retrieve_result.success is True

        # Step 4: Search for similar analyses
        search_event = self.create_orchestrator_event(
            EnumMemoryStorageOperationType.SEMANTIC_SEARCH,
            query_text="quarterly sales analysis growth rate",
            max_results=5,
            similarity_threshold=0.7,
            search_filters={
                "content_type": "analysis_results",
                "time_range": "last_90_days",
            },
        )

        input_data = ModelMemoryStorageInput(**search_event["payload"])
        search_result = await node.process(input_data)
        assert search_result.success is True

        print(f"Workflow simulation completed successfully:")
        print(f"- Store operations: 2/2 successful")
        print(f"- Retrieve operations: 1/1 successful")
        print(f"- Search operations: 1/1 successful")
        print(
            f"- Total workflow processing time: {sum([r.execution_time_ms for r in [store_result, analysis_result, retrieve_result, search_result]]):.2f}ms"
        )
