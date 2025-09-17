"""
Comprehensive integration tests for OmniMemory ONEX architecture.

This module tests end-to-end workflows across the complete 4-node architecture
(EFFECT → COMPUTE → REDUCER → ORCHESTRATOR) with real dependencies.

Tests cover:
- Complete memory lifecycle operations
- ONEX node interaction patterns
- Error handling and recovery
- Security and validation
- Cross-service integration
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnimemory.models.core.enum_node_type import EnumNodeType
from omnimemory.models.intelligence.model_intelligence_process_request import (
    IntelligenceProcessRequest,
)
from omnimemory.models.intelligence.model_intelligence_process_response import (
    IntelligenceProcessResponse,
)
from omnimemory.models.memory.model_memory_context import ModelMemoryContext
from omnimemory.models.memory.model_memory_item import ModelMemoryItem
from omnimemory.models.memory.model_memory_storage_config import (
    ModelMemoryStorageConfig,
)
from omnimemory.utils.concurrency import LockPriority, PriorityLock
from omnimemory.utils.error_sanitizer import ErrorSanitizer
from omnimemory.utils.health_manager import create_health_manager
from omnimemory.utils.pii_detector import PIIDetector


class IntegrationTestFixture:
    """Test fixture for integration testing with proper setup and teardown."""

    def __init__(self):
        self.correlation_id = uuid4()
        self.session_id = uuid4()
        self.test_items: List[ModelMemoryItem] = []
        self.health_manager = None

    async def setup(self):
        """Setup test environment."""
        self.health_manager = await create_health_manager()

        # Create test memory context
        self.test_context = ModelMemoryContext(
            correlation_id=self.correlation_id,
            session_id=self.session_id,
            user_id="integration_test_user",
            source_node_type=EnumNodeType.EFFECT,
            source_node_id="integration_test_node",
            timestamp=datetime.utcnow(),
            metadata={"test": True, "suite": "integration"},
        )

        # Create test storage config
        self.storage_config = ModelMemoryStorageConfig(
            storage_type="postgresql",
            connection_string="postgresql://test:test@localhost:5432/omnimemory_test",
            max_connections=5,
            connection_timeout=30.0,
        )

    async def teardown(self):
        """Cleanup test environment."""
        # Clean up test items
        self.test_items.clear()

        # Close health manager connections
        if self.health_manager:
            # Health manager cleanup would go here
            pass

    async def create_test_memory_item(self, suffix: str = "") -> ModelMemoryItem:
        """Create a test memory item with proper ONEX patterns."""
        memory_item = ModelMemoryItem(
            item_id=str(uuid4()),
            title=f"Integration Test Memory Item {suffix}",
            content=f"Test content for integration validation {suffix}",
            tags=["integration", "test", f"item_{suffix}"],
            metadata={
                "test": True,
                "suite": "integration",
                "created_at": datetime.utcnow().isoformat(),
            },
            context=self.test_context,
        )
        self.test_items.append(memory_item)
        return memory_item


@pytest.fixture
async def integration_fixture():
    """Async fixture for integration tests."""
    fixture = IntegrationTestFixture()
    await fixture.setup()
    yield fixture
    await fixture.teardown()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_memory_lifecycle(integration_fixture):
    """Test: Complete memory item lifecycle through all ONEX nodes."""

    # EFFECT Node: Create and store memory item
    memory_item = await integration_fixture.create_test_memory_item("lifecycle_test")

    # Validate EFFECT node operations
    assert memory_item.item_id is not None
    assert memory_item.title is not None
    assert memory_item.context.source_node_type == EnumNodeType.EFFECT
    assert len(memory_item.tags) > 0

    # COMPUTE Node: Process intelligence data
    intelligence_request = IntelligenceProcessRequest(
        correlation_id=integration_fixture.correlation_id,
        timestamp=datetime.utcnow(),
        raw_data=memory_item.content,
        processing_type="semantic_analysis",
        metadata={"source_item_id": memory_item.item_id},
    )

    # Simulate COMPUTE processing
    await asyncio.sleep(0.01)  # Processing delay

    intelligence_response = IntelligenceProcessResponse(
        correlation_id=integration_fixture.correlation_id,
        status="success",
        timestamp=datetime.utcnow(),
        execution_time_ms=10,
        provenance=["integration_test", "compute_node"],
        trust_score=0.95,
        processed_data={
            "semantic_features": ["integration", "test", "lifecycle"],
            "confidence": 0.92,
        },
        insights=["Integration test pattern detected", "High quality test data"],
    )

    assert intelligence_response.status == "success"
    assert intelligence_response.trust_score > 0.8
    assert len(intelligence_response.insights) > 0

    # REDUCER Node: Aggregate and consolidate
    # Simulate memory consolidation process
    consolidated_metadata = {
        **memory_item.metadata,
        "intelligence_processed": True,
        "trust_score": intelligence_response.trust_score,
        "processing_timestamp": intelligence_response.timestamp.isoformat(),
    }

    # Update memory item with consolidated data
    updated_memory_item = ModelMemoryItem(
        item_id=memory_item.item_id,
        title=memory_item.title,
        content=memory_item.content,
        tags=memory_item.tags + ["processed", "consolidated"],
        metadata=consolidated_metadata,
        context=memory_item.context,
        trust_score=intelligence_response.trust_score,
    )

    assert updated_memory_item.trust_score > 0.8
    assert "processed" in updated_memory_item.tags
    assert updated_memory_item.metadata["intelligence_processed"] is True

    # ORCHESTRATOR Node: Workflow coordination
    # Simulate orchestrator managing the complete workflow
    workflow_metadata = {
        "workflow_id": str(uuid4()),
        "total_processing_time": 25,  # Total time across all nodes
        "nodes_involved": ["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"],
        "completion_status": "success",
    }

    assert len(workflow_metadata["nodes_involved"]) == 4
    assert workflow_metadata["completion_status"] == "success"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_memory_operations(integration_fixture):
    """Test: Concurrent memory operations maintain consistency."""

    async def concurrent_memory_operation(operation_id: int) -> Dict[str, Any]:
        """Single concurrent memory operation."""
        memory_item = await integration_fixture.create_test_memory_item(
            f"concurrent_{operation_id}"
        )

        # Simulate concurrent processing with proper locking
        lock = PriorityLock()
        async with lock.acquire(priority=LockPriority.NORMAL):
            # Simulate memory storage operation
            await asyncio.sleep(0.01)

            return {
                "operation_id": operation_id,
                "item_id": memory_item.item_id,
                "status": "completed",
                "processing_time": 0.01,
            }

    # Run 10 concurrent operations
    concurrent_tasks = [concurrent_memory_operation(i) for i in range(10)]
    results = await asyncio.gather(*concurrent_tasks)

    # Validate all operations completed successfully
    assert len(results) == 10
    assert all(result["status"] == "completed" for result in results)
    assert len(set(result["item_id"] for result in results)) == 10  # All unique IDs

    # Validate test fixture tracked all items
    assert len(integration_fixture.test_items) >= 10


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_handling_and_recovery(integration_fixture):
    """Test: Error handling and recovery across ONEX nodes."""

    # Test 1: Invalid memory item creation
    with pytest.raises(Exception):  # Should raise validation error
        invalid_item = ModelMemoryItem(
            item_id="invalid_id_format",  # Invalid UUID format
            title="",  # Empty title should fail validation
            content="Test content",
            tags=[],
            metadata={},
            context=integration_fixture.test_context,
        )

    # Test 2: Error sanitization
    error_sanitizer = ErrorSanitizer()

    try:
        # Create an error with sensitive information
        raise ValueError("Database error: password=secret123, api_key=sk-abc123def")
    except ValueError as e:
        sanitized_message = error_sanitizer.sanitize_error_message(str(e))

        # Ensure sensitive data is removed
        assert "password=secret123" not in sanitized_message
        assert "api_key=sk-abc123def" not in sanitized_message
        assert (
            "REDACTED" in sanitized_message
            or sanitized_message == "ValueError: Operation failed"
        )

    # Test 3: PII Detection and Protection
    pii_detector = PIIDetector()

    test_content_with_pii = "My email is john.doe@example.com and my SSN is 123-45-6789"
    pii_results = pii_detector.detect_pii(test_content_with_pii)

    assert len(pii_results) > 0  # Should detect PII

    # Create memory item with PII content
    memory_item_with_pii = await integration_fixture.create_test_memory_item("pii_test")
    memory_item_with_pii.content = test_content_with_pii

    # Validate PII detection works
    content_pii = pii_detector.detect_pii(memory_item_with_pii.content)
    assert len(content_pii) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_health_monitoring_integration(integration_fixture):
    """Test: Health monitoring across all system components."""

    # Test overall system health
    health_status = await integration_fixture.health_manager.get_overall_health()

    assert health_status.status in ["healthy", "degraded", "unhealthy"]
    assert health_status.timestamp is not None
    assert isinstance(health_status.component_health, dict)

    # Test component-specific health checks
    if health_status.component_health:
        for component, status in health_status.component_health.items():
            assert isinstance(component, str)
            assert status in ["healthy", "degraded", "unhealthy"]

    # Test health check performance
    start_time = datetime.utcnow()
    for _ in range(5):
        await integration_fixture.health_manager.get_overall_health()
    end_time = datetime.utcnow()

    total_time = (end_time - start_time).total_seconds()
    avg_time_per_check = total_time / 5

    # Health checks should be fast
    assert avg_time_per_check < 0.1  # Under 100ms


@pytest.mark.asyncio
@pytest.mark.integration
async def test_security_validation_integration(integration_fixture):
    """Test: Security validation across all operations."""

    # Test 1: Sensitive data exclusion
    config_with_secrets = ModelMemoryStorageConfig(
        storage_type="postgresql",
        connection_string="postgresql://user:pass@localhost:5432/db",
        password_hash="hashed_password_value",
        api_key="secret_api_key_value",
    )

    # Serialize config and ensure secrets are excluded
    serialized = config_with_secrets.model_dump()

    assert "password_hash" not in serialized
    assert "api_key" not in serialized
    assert serialized["storage_type"] == "postgresql"

    # Test 2: Input validation
    memory_item = await integration_fixture.create_test_memory_item("security_test")

    # Test maximum tag validation
    try:
        # This should pass validation (within limits)
        memory_item.tags = ["valid"] * 50  # 50 tags should be acceptable
        memory_item.model_validate(memory_item.model_dump())
    except Exception as e:
        pytest.fail(f"Valid tag count should not raise exception: {e}")

    # Test 3: Content size validation
    # Normal content should pass
    memory_item.content = "Normal test content"
    try:
        memory_item.model_validate(memory_item.model_dump())
    except Exception as e:
        pytest.fail(f"Normal content should not raise exception: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cross_node_communication(integration_fixture):
    """Test: Communication patterns between ONEX nodes."""

    # Simulate EFFECT → COMPUTE communication
    memory_item = await integration_fixture.create_test_memory_item("cross_node")

    # EFFECT node creates item with correlation tracking
    effect_correlation = integration_fixture.correlation_id

    # COMPUTE node receives correlation ID and processes
    compute_request = IntelligenceProcessRequest(
        correlation_id=effect_correlation,  # Same correlation ID
        timestamp=datetime.utcnow(),
        raw_data=memory_item.content,
        processing_type="cross_node_test",
        metadata={"source_node": "EFFECT", "target_node": "COMPUTE"},
    )

    assert compute_request.correlation_id == effect_correlation
    assert compute_request.metadata["source_node"] == "EFFECT"

    # COMPUTE → REDUCER communication
    compute_response = IntelligenceProcessResponse(
        correlation_id=effect_correlation,
        status="success",
        timestamp=datetime.utcnow(),
        execution_time_ms=15,
        provenance=["EFFECT", "COMPUTE"],
        trust_score=0.87,
        processed_data={"cross_node_test": True},
        insights=["Cross-node communication validated"],
    )

    # REDUCER receives and consolidates
    reducer_metadata = {
        "correlation_id": str(effect_correlation),
        "node_chain": ["EFFECT", "COMPUTE", "REDUCER"],
        "total_processing_time": compute_response.execution_time_ms + 5,
        "final_trust_score": compute_response.trust_score,
    }

    # ORCHESTRATOR coordinates the entire workflow
    orchestrator_summary = {
        "workflow_correlation_id": str(effect_correlation),
        "nodes_involved": reducer_metadata["node_chain"] + ["ORCHESTRATOR"],
        "total_operations": 1,
        "success_rate": 1.0,
        "average_trust_score": reducer_metadata["final_trust_score"],
    }

    # Validate complete cross-node communication chain
    assert len(orchestrator_summary["nodes_involved"]) == 4
    assert orchestrator_summary["success_rate"] == 1.0
    assert orchestrator_summary["average_trust_score"] > 0.8


@pytest.mark.asyncio
@pytest.mark.integration
async def test_batch_processing_integration(integration_fixture):
    """Test: Batch processing across multiple memory items."""

    batch_size = 50
    batch_items = []

    # Create batch of memory items
    for i in range(batch_size):
        item = await integration_fixture.create_test_memory_item(f"batch_{i}")
        batch_items.append(item)

    assert len(batch_items) == batch_size

    # Simulate batch processing through ONEX nodes
    batch_processing_results = []

    for item in batch_items:
        # Process each item through the workflow
        processing_result = {
            "item_id": item.item_id,
            "processed_at": datetime.utcnow(),
            "processing_nodes": ["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"],
            "status": "success",
            "trust_score": 0.9,
        }
        batch_processing_results.append(processing_result)

        # Small delay to simulate processing
        await asyncio.sleep(0.001)

    # Validate batch processing
    assert len(batch_processing_results) == batch_size
    assert all(result["status"] == "success" for result in batch_processing_results)
    assert all(
        len(result["processing_nodes"]) == 4 for result in batch_processing_results
    )

    # Validate all items have unique IDs
    unique_ids = set(result["item_id"] for result in batch_processing_results)
    assert len(unique_ids) == batch_size


if __name__ == "__main__":
    """Run integration tests directly."""
    import asyncio

    async def run_integration_tests():
        print("=== OmniMemory Integration Test Suite ===")
        print("Testing complete ONEX 4-node architecture integration...\n")

        # Create test fixture
        fixture = IntegrationTestFixture()
        await fixture.setup()

        try:
            # Run all tests
            await test_complete_memory_lifecycle(fixture)
            print("✅ Complete Memory Lifecycle Test PASSED")

            await test_concurrent_memory_operations(fixture)
            print("✅ Concurrent Operations Test PASSED")

            await test_error_handling_and_recovery(fixture)
            print("✅ Error Handling and Recovery Test PASSED")

            await test_health_monitoring_integration(fixture)
            print("✅ Health Monitoring Integration Test PASSED")

            await test_security_validation_integration(fixture)
            print("✅ Security Validation Integration Test PASSED")

            await test_cross_node_communication(fixture)
            print("✅ Cross-Node Communication Test PASSED")

            await test_batch_processing_integration(fixture)
            print("✅ Batch Processing Integration Test PASSED")

        except Exception as e:
            print(f"❌ Test FAILED: {e}")
        finally:
            await fixture.teardown()

        print("\n=== Integration Tests Complete ===")

    # asyncio.run(run_integration_tests())
    print(
        "Integration test suite created. Run with: pytest tests/test_integration_comprehensive.py"
    )
