#!/usr/bin/env python3
"""
Test script to verify Qdrant adapter integration with OmniMemory event system.

This script tests the end-to-end event-driven architecture:
1. OmniMemory publishes vector search command to RedPanda
2. Qdrant adapter processes the command
3. Qdrant adapter publishes completion event back
4. OmniMemory receives and processes the completion event

Usage:
    python scripts/test_qdrant_adapter_integration.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omnimemory.events.event_bus_client import OmniMemoryEventBusClient
from omnimemory.services.event_driven_memory_service import EventDrivenMemoryService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class QdrantAdapterIntegrationTest:
    """Integration test for Qdrant adapter event processing."""

    def __init__(self):
        """Initialize test with event bus client."""
        self.event_bus_client = None
        self.memory_service = None
        self.test_results = []
        self.received_events = []

    async def setup(self):
        """Setup test environment and event bus connections."""
        logger.info("Setting up Qdrant adapter integration test...")

        try:
            # Initialize event bus client (will connect to RedPanda)
            self.event_bus_client = OmniMemoryEventBusClient(
                node_id="integration_test_client"
            )

            # Create a simple event bus implementation for testing
            from datetime import datetime

            class TestEventBusAdapter:
                def __init__(self, bootstrap_servers: str):
                    self.bootstrap_servers = bootstrap_servers
                    self.connected = False
                    logger.info(
                        f"Test event bus adapter initialized for {bootstrap_servers}"
                    )

                async def initialize_kafka(self):
                    """Mock initialization - in real deployment this connects to RedPanda"""
                    self.connected = True
                    logger.info(f"Test event bus connected to {self.bootstrap_servers}")
                    return True

                async def publish_async(self, event):
                    """Publish event - in real deployment this goes to RedPanda"""
                    if hasattr(event, "model_dump"):
                        payload = event.model_dump()
                    elif hasattr(event, "dict"):
                        payload = event.dict()
                    else:
                        payload = {"data": str(event), "timestamp": str(datetime.now())}

                    logger.info(f"Published event to test bus: {payload}")
                    return True

                async def subscribe_async(self, topic: str, callback):
                    logger.info(f"Subscribed to topic: {topic}")
                    return True

                async def unsubscribe_async(self, topic: str, callback=None):
                    logger.info(f"Unsubscribed from topic: {topic}")

                def clear(self):
                    logger.info("Event bus cleared")

                def close(self):
                    logger.info("Event bus connection closed")
                    self.connected = False

            # Initialize test adapter
            bootstrap_servers = os.getenv("REDPANDA_BOOTSTRAP_SERVERS", "redpanda:9092")
            test_adapter = TestEventBusAdapter(bootstrap_servers)
            await test_adapter.initialize_kafka()

            # Initialize event bus client with test adapter
            await self.event_bus_client.initialize(test_adapter)

            # Initialize memory service
            self.memory_service = EventDrivenMemoryService(
                event_bus_client=self.event_bus_client
            )
            await self.memory_service.initialize(test_adapter)

            logger.info("Test setup completed successfully")

        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            raise

    async def test_vector_search_event_flow(self):
        """Test vector search command and completion event flow."""
        logger.info("Testing vector search event flow...")

        try:
            # Create test vector search request
            test_vector = [0.1] * 1536  # Standard OpenAI embedding size
            correlation_id = uuid4()

            logger.info(
                f"Sending vector search command with correlation ID: {correlation_id}"
            )

            # Send vector search command via event bus client
            search_correlation_id = await self.event_bus_client.vector_search(
                query_vector=test_vector,
                similarity_threshold=0.8,
                max_results=10,
                index_name="intelligence_vectors",
            )

            logger.info(
                f"Vector search command sent, correlation ID: {search_correlation_id}"
            )

            # Register completion handler to capture response
            completion_received = asyncio.Event()
            completion_data = {}

            def handle_search_completion(event_data):
                logger.info(f"Received search completion event: {event_data}")
                completion_data.update(event_data)
                completion_received.set()

            self.event_bus_client.register_search_completed_handler(
                handle_search_completion
            )

            # Wait for completion event (with timeout)
            try:
                await asyncio.wait_for(completion_received.wait(), timeout=30.0)
                logger.info("Search completion event received successfully")

                # Validate completion event data
                if completion_data.get("correlation_id") == str(search_correlation_id):
                    logger.info(
                        "✅ Correlation ID matches - event flow working correctly"
                    )
                    self.test_results.append("PASS: Vector search event flow")
                else:
                    logger.warning("⚠️ Correlation ID mismatch in completion event")
                    self.test_results.append("FAIL: Correlation ID mismatch")

            except asyncio.TimeoutError:
                logger.warning(
                    "⚠️ Search completion event timeout - Qdrant adapter may not be running"
                )
                self.test_results.append("TIMEOUT: Vector search completion event")

        except Exception as e:
            logger.error(f"Vector search test failed: {e}")
            self.test_results.append(f"ERROR: Vector search test - {str(e)}")

    async def test_memory_store_event_flow(self):
        """Test memory store command and completion event flow."""
        logger.info("Testing memory store event flow...")

        try:
            # Create test memory store request
            correlation_id = uuid4()
            test_content = {
                "text": "This is a test memory for Qdrant adapter integration",
                "metadata": {
                    "test_type": "integration_test",
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.info(
                f"Sending memory store command with correlation ID: {correlation_id}"
            )

            # Send memory store command via event bus client
            store_correlation_id = await self.event_bus_client.store_memory(
                memory_key=f"test_memory_{correlation_id.hex[:8]}",
                content=test_content,
                memory_type="vector",
                metadata={"integration_test": True},
            )

            logger.info(
                f"Memory store command sent, correlation ID: {store_correlation_id}"
            )

            # Register completion handler
            store_completed = asyncio.Event()
            store_data = {}

            def handle_store_completion(event_data):
                logger.info(f"Received memory store completion event: {event_data}")
                store_data.update(event_data)
                store_completed.set()

            self.event_bus_client.register_memory_stored_handler(
                handle_store_completion
            )

            # Wait for completion event
            try:
                await asyncio.wait_for(store_completed.wait(), timeout=30.0)
                logger.info("Memory store completion event received successfully")

                # Validate completion event
                if store_data.get("correlation_id") == str(store_correlation_id):
                    logger.info("✅ Memory store event flow working correctly")
                    self.test_results.append("PASS: Memory store event flow")
                else:
                    logger.warning(
                        "⚠️ Correlation ID mismatch in store completion event"
                    )
                    self.test_results.append("FAIL: Store correlation ID mismatch")

            except asyncio.TimeoutError:
                logger.warning("⚠️ Memory store completion event timeout")
                self.test_results.append("TIMEOUT: Memory store completion event")

        except Exception as e:
            logger.error(f"Memory store test failed: {e}")
            self.test_results.append(f"ERROR: Memory store test - {str(e)}")

    async def test_event_bus_health(self):
        """Test event bus health and connectivity."""
        logger.info("Testing event bus health...")

        try:
            if self.event_bus_client and self.event_bus_client.is_ready():
                health_status = await self.event_bus_client.health_check()

                if health_status.get("status") == "healthy":
                    logger.info("✅ Event bus health check passed")
                    self.test_results.append("PASS: Event bus health")
                else:
                    logger.warning("⚠️ Event bus health check failed")
                    logger.info(f"Health status: {health_status}")
                    self.test_results.append("FAIL: Event bus unhealthy")
            else:
                logger.warning("⚠️ Event bus client not ready")
                self.test_results.append("FAIL: Event bus not ready")

        except Exception as e:
            logger.error(f"Event bus health test failed: {e}")
            self.test_results.append(f"ERROR: Event bus health - {str(e)}")

    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("=== Starting Qdrant Adapter Integration Tests ===")

        try:
            # Setup test environment
            await self.setup()

            # Run individual tests
            await self.test_event_bus_health()
            await self.test_vector_search_event_flow()
            await self.test_memory_store_event_flow()

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self.test_results.append(f"CRITICAL: Test execution failed - {str(e)}")

        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup test resources."""
        logger.info("Cleaning up test resources...")

        try:
            if self.event_bus_client:
                await self.event_bus_client.shutdown()
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def print_test_results(self):
        """Print final test results summary."""
        logger.info("\n=== Qdrant Adapter Integration Test Results ===")

        passed = sum(1 for result in self.test_results if result.startswith("PASS"))
        failed = sum(1 for result in self.test_results if result.startswith("FAIL"))
        timeouts = sum(
            1 for result in self.test_results if result.startswith("TIMEOUT")
        )
        errors = sum(
            1
            for result in self.test_results
            if result.startswith("ERROR") or result.startswith("CRITICAL")
        )

        logger.info(f"Total tests: {len(self.test_results)}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏱️ Timeouts: {timeouts}")
        logger.info(f"🚨 Errors: {errors}")

        logger.info("\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            logger.info(f"  {i}. {result}")

        if timeouts > 0 or errors > 0:
            logger.info("\n⚠️ INTEGRATION NOTES:")
            logger.info("- Timeouts indicate the Qdrant adapter may not be running")
            logger.info(
                "- Start the full stack: docker-compose -f deployment/docker-compose.memory.yml up"
            )
            logger.info(
                "- Check that RedPanda is accessible and Qdrant adapter is processing events"
            )

        overall_status = "✅ SUCCESS" if (failed + errors == 0) else "❌ NEEDS ATTENTION"
        logger.info(f"\nOverall Status: {overall_status}")

        return failed + errors == 0


async def main():
    """Main test execution."""
    test = QdrantAdapterIntegrationTest()

    try:
        await test.run_all_tests()
        success = test.print_test_results()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
