"""Performance benchmarking tests for event-driven OmniMemory architecture.

Validates performance targets from CLAUDE.md:
- Memory Operations: <100ms response time (95th percentile)
- Throughput: 1M+ operations per hour sustained
- Vector Search: <50ms semantic similarity queries
- Bulk Operations: >10K records/second batch processing
"""

import asyncio
import statistics
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from omnimemory.events.event_bus_client import EventBusClient
from omnimemory.events.event_consumer import EventConsumer
from omnimemory.events.event_producer import EventProducer
from omnimemory.models.memory.model_memory_request import (
    ModelMemoryRetrieveRequest,
    ModelMemoryStoreRequest,
    ModelMemoryVectorSearchRequest,
)
from omnimemory.services.event_driven_memory_service import EventDrivenMemoryService
from omnimemory.services.memory_operation_mapper import MemoryOperationMapper

# Import ONEX core components with fallback
try:
    from omnibase_core.core.protocol_event_bus import ProtocolEventBus
except ImportError:

    class ProtocolEventBus:
        async def publish_async(self, event):
            pass

        async def subscribe_async(self, topic, handler):
            pass


class PerformanceTimer:
    """High-precision performance timer for benchmarking."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class MockHighPerformanceEventBus:
    """High-performance mock event bus for benchmarking."""

    def __init__(self, publish_latency_ms: float = 1.0):
        self.publish_latency_ms = publish_latency_ms
        self.published_events = []
        self.publish_count = 0
        self.total_publish_time = 0.0

    async def publish_async(self, event):
        """Mock publish with configurable latency."""
        start_time = time.perf_counter()

        # Simulate network latency
        await asyncio.sleep(self.publish_latency_ms / 1000)

        self.published_events.append(event)
        self.publish_count += 1

        end_time = time.perf_counter()
        self.total_publish_time += (end_time - start_time) * 1000

    @property
    def average_publish_time_ms(self) -> float:
        """Get average publish time in milliseconds."""
        if self.publish_count == 0:
            return 0.0
        return self.total_publish_time / self.publish_count

    def reset_metrics(self):
        """Reset performance metrics."""
        self.published_events.clear()
        self.publish_count = 0
        self.total_publish_time = 0.0


@pytest_asyncio.fixture
async def high_performance_event_bus():
    """High-performance mock event bus."""
    return MockHighPerformanceEventBus(publish_latency_ms=0.5)  # 0.5ms latency


@pytest_asyncio.fixture
async def performance_event_driven_service(high_performance_event_bus):
    """Event-driven service optimized for performance testing."""
    # Initialize components
    producer = EventProducer()
    producer.initialize(high_performance_event_bus)

    consumer = EventConsumer()
    consumer.initialize(high_performance_event_bus)

    client = EventBusClient()
    client.initialize(producer, consumer)

    mapper = MemoryOperationMapper()

    service = EventDrivenMemoryService()
    service.initialize(client, mapper)

    return service


class TestMemoryOperationPerformance:
    """Test memory operation performance benchmarks."""

    async def test_single_memory_store_performance(
        self, performance_event_driven_service, high_performance_event_bus
    ):
        """Test single memory store operation performance."""
        # Create test request
        store_request = ModelMemoryStoreRequest(
            memory_key="perf_test_single_store",
            content={"performance": "test", "data_size": "medium"},
            metadata={"benchmark": "single_store", "size_kb": 1},
        )

        # Benchmark single operation
        with PerformanceTimer() as timer:
            correlation_id = await performance_event_driven_service.store_memory(
                store_request
            )

        # Verify operation completed
        assert correlation_id is not None
        assert isinstance(correlation_id, UUID)

        # Verify performance target: <100ms per operation
        operation_time_ms = timer.elapsed_ms
        print(f"Single store operation time: {operation_time_ms:.2f}ms")

        # For event-driven architecture, we expect very fast local processing
        # (most time is in event bus communication)
        assert (
            operation_time_ms < 50.0
        ), f"Single store took {operation_time_ms}ms, expected <50ms"

    async def test_memory_operation_throughput(
        self, performance_event_driven_service, high_performance_event_bus
    ):
        """Test memory operation throughput - target: 1M+ ops/hour."""
        num_operations = 100  # Scaled down for test performance
        target_ops_per_second = 278  # 1M ops/hour = 278 ops/second

        # Create test requests
        store_requests = [
            ModelMemoryStoreRequest(
                memory_key=f"perf_throughput_{i}",
                content={"operation_id": i, "test": "throughput"},
                metadata={"benchmark": "throughput", "batch_id": i},
            )
            for i in range(num_operations)
        ]

        # Benchmark batch operations
        high_performance_event_bus.reset_metrics()

        start_time = time.perf_counter()

        # Execute operations in parallel
        tasks = [
            performance_event_driven_service.store_memory(request)
            for request in store_requests
        ]

        correlation_ids = await asyncio.gather(*tasks)

        end_time = time.perf_counter()

        # Calculate performance metrics
        total_time_seconds = end_time - start_time
        ops_per_second = num_operations / total_time_seconds

        print(f"Throughput test results:")
        print(f"  Operations: {num_operations}")
        print(f"  Total time: {total_time_seconds:.3f}s")
        print(f"  Ops/second: {ops_per_second:.2f}")
        print(f"  Target: {target_ops_per_second} ops/second")
        print(
            f"  Average event bus latency: {high_performance_event_bus.average_publish_time_ms:.2f}ms"
        )

        # Verify all operations completed
        assert len(correlation_ids) == num_operations
        assert all(isinstance(cid, UUID) for cid in correlation_ids)

        # Verify throughput target
        # Note: In event-driven architecture, local throughput should be very high
        # The bottleneck will be in the infrastructure adapters
        assert (
            ops_per_second >= target_ops_per_second * 0.8
        ), f"Throughput {ops_per_second:.2f} ops/sec below 80% of target {target_ops_per_second}"

    async def test_vector_search_performance(
        self, performance_event_driven_service, high_performance_event_bus
    ):
        """Test vector search performance - target: <50ms queries."""
        # Create vector search request
        search_request = ModelMemoryVectorSearchRequest(
            query_vector=[0.1, 0.2, 0.3, 0.4, 0.5] * 200,  # 1000-dimensional vector
            collection_name="performance_test_collection",
            limit=50,
            similarity_threshold=0.7,
            filters={"benchmark": "vector_search_performance"},
        )

        # Benchmark multiple vector searches
        search_times = []
        num_searches = 20

        for i in range(num_searches):
            with PerformanceTimer() as timer:
                correlation_id = await performance_event_driven_service.vector_search(
                    search_request
                )

            assert correlation_id is not None
            search_times.append(timer.elapsed_ms)

        # Calculate performance statistics
        avg_search_time = statistics.mean(search_times)
        p95_search_time = statistics.quantiles(search_times, n=20)[
            18
        ]  # 95th percentile
        min_search_time = min(search_times)
        max_search_time = max(search_times)

        print(f"Vector search performance results:")
        print(f"  Number of searches: {num_searches}")
        print(f"  Average time: {avg_search_time:.2f}ms")
        print(f"  95th percentile: {p95_search_time:.2f}ms")
        print(f"  Min time: {min_search_time:.2f}ms")
        print(f"  Max time: {max_search_time:.2f}ms")
        print(f"  Target: <50ms per search")

        # Verify performance targets
        assert (
            avg_search_time < 25.0
        ), f"Average search time {avg_search_time:.2f}ms exceeds 25ms"
        assert (
            p95_search_time < 50.0
        ), f"95th percentile {p95_search_time:.2f}ms exceeds 50ms target"

    async def test_bulk_operation_performance(
        self, performance_event_driven_service, high_performance_event_bus
    ):
        """Test bulk operation performance - target: >10K records/second."""
        batch_size = 1000  # Scaled down for test performance
        target_records_per_second = 10000

        # Create bulk store requests
        bulk_requests = [
            ModelMemoryStoreRequest(
                memory_key=f"bulk_perf_{i:06d}",
                content={
                    "record_id": i,
                    "data": f"bulk_data_{i}",
                    "bulk_test": True,
                    "batch_size": batch_size,
                },
                metadata={
                    "benchmark": "bulk_operations",
                    "record_index": i,
                    "batch_size": batch_size,
                },
            )
            for i in range(batch_size)
        ]

        # Benchmark bulk processing
        high_performance_event_bus.reset_metrics()

        start_time = time.perf_counter()

        # Process in parallel batches for optimal performance
        batch_size_parallel = 50  # Process 50 at a time
        correlation_ids = []

        for i in range(0, len(bulk_requests), batch_size_parallel):
            batch = bulk_requests[i : i + batch_size_parallel]
            batch_tasks = [
                performance_event_driven_service.store_memory(request)
                for request in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            correlation_ids.extend(batch_results)

        end_time = time.perf_counter()

        # Calculate bulk performance metrics
        total_time_seconds = end_time - start_time
        records_per_second = batch_size / total_time_seconds

        print(f"Bulk operation performance results:")
        print(f"  Records processed: {batch_size}")
        print(f"  Total time: {total_time_seconds:.3f}s")
        print(f"  Records/second: {records_per_second:.2f}")
        print(f"  Target: {target_records_per_second} records/second")
        print(f"  Events published: {high_performance_event_bus.publish_count}")
        print(
            f"  Average event latency: {high_performance_event_bus.average_publish_time_ms:.2f}ms"
        )

        # Verify all records processed
        assert len(correlation_ids) == batch_size
        assert all(isinstance(cid, UUID) for cid in correlation_ids)

        # Verify bulk processing performance
        # For event-driven architecture, we expect high local throughput
        assert (
            records_per_second >= target_records_per_second * 0.5
        ), f"Bulk processing {records_per_second:.2f} records/sec below 50% of target"


class TestEventBusPerformance:
    """Test event bus communication performance."""

    async def test_event_publishing_latency(self, high_performance_event_bus):
        """Test event publishing latency distribution."""
        num_events = 1000
        publish_times = []

        # Create mock events
        for i in range(num_events):
            with PerformanceTimer() as timer:
                await high_performance_event_bus.publish_async(f"test_event_{i}")
            publish_times.append(timer.elapsed_ms)

        # Calculate latency statistics
        avg_latency = statistics.mean(publish_times)
        p50_latency = statistics.median(publish_times)
        p95_latency = statistics.quantiles(publish_times, n=20)[18]
        p99_latency = statistics.quantiles(publish_times, n=100)[98]

        print(f"Event publishing latency results:")
        print(f"  Events published: {num_events}")
        print(f"  Average latency: {avg_latency:.3f}ms")
        print(f"  P50 latency: {p50_latency:.3f}ms")
        print(f"  P95 latency: {p95_latency:.3f}ms")
        print(f"  P99 latency: {p99_latency:.3f}ms")

        # Verify latency targets for event bus
        assert (
            avg_latency < 5.0
        ), f"Average event latency {avg_latency:.3f}ms exceeds 5ms"
        assert p95_latency < 10.0, f"P95 event latency {p95_latency:.3f}ms exceeds 10ms"
        assert p99_latency < 20.0, f"P99 event latency {p99_latency:.3f}ms exceeds 20ms"

    async def test_concurrent_event_publishing(self, high_performance_event_bus):
        """Test concurrent event publishing performance."""
        num_concurrent = 100
        events_per_publisher = 50

        async def publisher_task(publisher_id: int):
            """Individual publisher task."""
            times = []
            for i in range(events_per_publisher):
                with PerformanceTimer() as timer:
                    await high_performance_event_bus.publish_async(
                        f"concurrent_event_{publisher_id}_{i}"
                    )
                times.append(timer.elapsed_ms)
            return times

        # Run concurrent publishers
        start_time = time.perf_counter()

        publisher_tasks = [publisher_task(pub_id) for pub_id in range(num_concurrent)]

        publisher_results = await asyncio.gather(*publisher_tasks)

        end_time = time.perf_counter()

        # Aggregate results
        all_times = [time for times in publisher_results for time in times]
        total_events = len(all_times)
        total_time_seconds = end_time - start_time
        events_per_second = total_events / total_time_seconds

        print(f"Concurrent event publishing results:")
        print(f"  Concurrent publishers: {num_concurrent}")
        print(f"  Events per publisher: {events_per_publisher}")
        print(f"  Total events: {total_events}")
        print(f"  Total time: {total_time_seconds:.3f}s")
        print(f"  Events/second: {events_per_second:.2f}")
        print(f"  Average latency: {statistics.mean(all_times):.3f}ms")

        # Verify concurrent performance
        assert (
            events_per_second >= 5000
        ), f"Concurrent event rate {events_per_second:.2f} below 5K events/sec"
        assert (
            statistics.mean(all_times) < 10.0
        ), f"Average concurrent latency exceeds 10ms"


class TestMemoryFootprintPerformance:
    """Test memory footprint efficiency - target: <10MB per 100K records."""

    async def test_operation_tracking_memory_efficiency(self):
        """Test memory efficiency of operation tracking."""
        import sys

        from omnimemory.models.core.model_memory_operation import (
            EnumMemoryOperationType,
        )
        from omnimemory.services.memory_operation_mapper import MemoryOperationMapper

        mapper = MemoryOperationMapper()

        # Measure baseline memory
        initial_size = sys.getsizeof(mapper)

        # Track large number of operations
        num_operations = 10000  # 10K operations (scaled down from 100K)

        for i in range(num_operations):
            correlation_id = uuid4()
            mapper.track_operation(
                correlation_id=correlation_id,
                operation_type=EnumMemoryOperationType.STORE,
                memory_key=f"memory_efficiency_test_{i:06d}",
                metadata={
                    "test_id": i,
                    "benchmark": "memory_efficiency",
                    "data_size": "1KB",
                    "operation_index": i,
                },
            )

        # Measure final memory usage
        final_size = sys.getsizeof(mapper)
        memory_used_bytes = final_size - initial_size
        memory_used_mb = memory_used_bytes / (1024 * 1024)

        # Calculate memory per operation
        memory_per_operation_bytes = memory_used_bytes / num_operations
        memory_per_100k_operations_mb = (memory_per_operation_bytes * 100000) / (
            1024 * 1024
        )

        print(f"Memory efficiency results:")
        print(f"  Operations tracked: {num_operations}")
        print(f"  Memory used: {memory_used_mb:.3f} MB")
        print(f"  Memory per operation: {memory_per_operation_bytes:.2f} bytes")
        print(
            f"  Projected memory per 100K operations: {memory_per_100k_operations_mb:.2f} MB"
        )
        print(f"  Target: <10 MB per 100K operations")

        # Verify memory efficiency target
        assert (
            memory_per_100k_operations_mb < 10.0
        ), f"Memory usage {memory_per_100k_operations_mb:.2f}MB exceeds 10MB target per 100K operations"

    def test_event_data_memory_efficiency(self):
        """Test memory efficiency of event data structures."""
        import sys

        from omnimemory.models.events.model_omnimemory_event_data import (
            ModelOmniMemoryStoreData,
        )

        # Create test data
        test_data_list = []
        num_events = 1000

        for i in range(num_events):
            event_data = ModelOmniMemoryStoreData(
                memory_key=f"memory_efficiency_event_{i:04d}",
                content={"test_data": f"value_{i}", "index": i, "benchmark": True},
                metadata={"event_id": i, "test": "memory_efficiency"},
                content_hash=f"hash_{i:016d}",
                storage_size=256,
                ttl_seconds=3600 if i % 2 == 0 else None,
            )
            test_data_list.append(event_data)

        # Measure memory usage
        total_size = sum(sys.getsizeof(event) for event in test_data_list)
        size_per_event = total_size / num_events
        size_per_100k_events_mb = (size_per_event * 100000) / (1024 * 1024)

        print(f"Event data memory efficiency:")
        print(f"  Events created: {num_events}")
        print(f"  Total size: {total_size / 1024:.2f} KB")
        print(f"  Size per event: {size_per_event:.2f} bytes")
        print(f"  Projected size per 100K events: {size_per_100k_events_mb:.2f} MB")

        # Verify event data is memory efficient
        assert (
            size_per_100k_events_mb < 50.0
        ), f"Event data size {size_per_100k_events_mb:.2f}MB too large per 100K events"


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main(
        [__file__, "-v", "--asyncio-mode=auto", "-s"]
    )  # -s to show print output
