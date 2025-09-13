"""
Comprehensive demonstration of advanced architecture improvements for OmniMemory.

This example shows how to use:
- Resource management with circuit breakers
- Concurrency improvements with semaphores and locks
- Migration progress tracking
- Observability with correlation ID tracking
- Health checking with comprehensive dependency monitoring
"""

import asyncio
import time
from datetime import datetime
from typing import List

from omnimemory.utils import (
    # Resource management
    resource_manager,
    CircuitBreakerConfig,
    with_circuit_breaker,
    with_timeout,

    # Observability
    correlation_context,
    trace_operation,
    OperationType,

    # Concurrency
    get_priority_lock,
    get_fair_semaphore,
    register_connection_pool,
    ConnectionPoolConfig,
    LockPriority,
    with_priority_lock,
    with_fair_semaphore,

    # Health management
    health_manager,
    HealthCheckConfig,
    DependencyType,
    create_postgresql_health_check,
    create_redis_health_check,
)

from omnimemory.models.foundation import (
    MigrationProgressTracker,
    MigrationStatus,
    MigrationPriority,
)

import structlog

logger = structlog.get_logger(__name__)

class AdvancedArchitectureDemo:
    """
    Comprehensive demonstration of advanced architecture features.
    """

    def __init__(self):
        self.migration_tracker: MigrationProgressTracker = None
        self.demo_files = [
            f"/fake/path/file_{i:03d}.txt" for i in range(50)
        ]

    async def demo_resource_management(self):
        """Demonstrate resource management with circuit breakers."""
        print("\n=== Resource Management Demo ===")

        async with correlation_context(
            operation="resource_management_demo",
            user_id="demo_user"
        ):
            # Configure circuit breaker for external service
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=5,
                success_threshold=2,
                timeout=2.0
            )

            # Simulate external service calls with circuit breaker
            async def unreliable_service():
                """Simulate an unreliable external service."""
                import random
                if random.random() < 0.7:  # 70% failure rate
                    raise Exception("Service temporarily unavailable")
                return {"status": "success", "data": "service_response"}

            print("Testing circuit breaker with unreliable service...")

            for i in range(10):
                try:
                    result = await with_circuit_breaker(
                        "external_service",
                        unreliable_service,
                        config
                    )
                    print(f"Call {i+1}: ✅ {result}")
                except Exception as e:
                    print(f"Call {i+1}: ❌ {e}")

                await asyncio.sleep(0.5)

            # Show circuit breaker statistics
            stats = resource_manager.get_circuit_breaker_stats()
            print(f"\nCircuit breaker stats: {stats}")

    async def demo_concurrency_improvements(self):
        """Demonstrate concurrency improvements with locks and semaphores."""
        print("\n=== Concurrency Improvements Demo ===")

        async with correlation_context(operation="concurrency_demo"):
            # Demo priority locks
            print("Testing priority locks...")

            async def worker(worker_id: int, priority: LockPriority, work_duration: float):
                async with trace_operation(
                    f"worker_{worker_id}",
                    OperationType.EXTERNAL_API,
                    worker_id=worker_id,
                    priority=priority.name
                ):
                    async with with_priority_lock(
                        "shared_resource",
                        priority=priority,
                        timeout=10.0
                    ):
                        print(f"Worker {worker_id} (priority: {priority.name}) acquired lock")
                        await asyncio.sleep(work_duration)
                        print(f"Worker {worker_id} releasing lock")

            # Start workers with different priorities
            tasks = [
                asyncio.create_task(worker(1, LockPriority.LOW, 2.0)),
                asyncio.create_task(worker(2, LockPriority.HIGH, 1.0)),
                asyncio.create_task(worker(3, LockPriority.NORMAL, 1.5)),
                asyncio.create_task(worker(4, LockPriority.CRITICAL, 0.5)),
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Demo fair semaphores
            print("\nTesting fair semaphores...")

            async def semaphore_worker(worker_id: int):
                async with with_fair_semaphore("limited_resource", permits=2, timeout=5.0):
                    print(f"Semaphore worker {worker_id} acquired permit")
                    await asyncio.sleep(1.0)
                    print(f"Semaphore worker {worker_id} releasing permit")

            # Start multiple workers competing for limited permits
            semaphore_tasks = [
                asyncio.create_task(semaphore_worker(i))
                for i in range(5)
            ]

            await asyncio.gather(*semaphore_tasks, return_exceptions=True)

    async def demo_migration_progress_tracking(self):
        """Demonstrate migration progress tracking."""
        print("\n=== Migration Progress Tracking Demo ===")

        async with correlation_context(operation="migration_demo"):
            # Create migration tracker
            self.migration_tracker = MigrationProgressTracker(
                name="Legacy Tool Migration Demo",
                priority=MigrationPriority.HIGH,
                configuration={
                    "batch_size": 10,
                    "parallel_workers": 3,
                    "retry_attempts": 3
                }
            )

            # Add files to track
            for file_path in self.demo_files:
                self.migration_tracker.add_file(
                    file_path,
                    file_size=1024 * (len(file_path) % 10 + 1),  # Simulate varying file sizes
                    file_type="intelligence_tool",
                    complexity="medium"
                )

            print(f"Created migration tracker for {len(self.demo_files)} files")

            # Simulate batch processing
            batch_size = 10
            batch_count = 0

            for i in range(0, len(self.demo_files), batch_size):
                batch_count += 1
                batch_files = self.demo_files[i:i + batch_size]
                batch_id = f"batch_{batch_count:03d}"

                print(f"\nProcessing {batch_id} with {len(batch_files)} files...")

                # Start batch
                self.migration_tracker.start_batch(batch_id, len(batch_files))

                # Process files in batch
                for file_path in batch_files:
                    # Start file processing
                    self.migration_tracker.start_file_processing(file_path, batch_id)

                    # Simulate processing time
                    await asyncio.sleep(0.1)

                    # Simulate success/failure (90% success rate)
                    import random
                    success = random.random() > 0.1

                    if success:
                        self.migration_tracker.complete_file_processing(file_path)
                    else:
                        self.migration_tracker.complete_file_processing(
                            file_path,
                            success=False,
                            error_message="Processing failed - invalid format"
                        )

                # Complete batch
                self.migration_tracker.complete_batch(batch_id)

                # Show progress
                summary = self.migration_tracker.get_progress_summary()
                print(f"Progress: {summary['completion_percentage']:.1f}% "
                      f"({summary['file_counts']['processed']}/{summary['file_counts']['total']} files)")

            # Final summary
            print(f"\n=== Migration Summary ===")
            final_summary = self.migration_tracker.get_progress_summary()
            print(f"Status: {final_summary['status']}")
            print(f"Completion: {final_summary['completion_percentage']:.1f}%")
            print(f"Success Rate: {final_summary['success_rate']:.1f}%")
            print(f"Files: {final_summary['file_counts']}")
            print(f"Processing Rate: {final_summary['processing_rates']['files_per_second']:.2f} files/sec")
            print(f"Elapsed Time: {final_summary['elapsed_time']}")
            if final_summary['estimated_completion']:
                print(f"Estimated Completion: {final_summary['estimated_completion']}")

    async def demo_observability_tracking(self):
        """Demonstrate observability with correlation ID tracking."""
        print("\n=== Observability Tracking Demo ===")

        # Main operation with correlation context
        async with correlation_context(
            correlation_id="demo-12345",
            operation="observability_demo",
            user_id="demo_user",
            session_type="demonstration"
        ) as ctx:
            print(f"Started operation with correlation ID: {ctx.correlation_id}")

            # Nested operations inherit correlation context
            async with trace_operation(
                "data_processing",
                OperationType.MEMORY_STORE,
                data_size=1000,
                processing_type="batch"
            ) as trace_id:
                print(f"Data processing trace ID: {trace_id}")

                # Simulate some work
                await asyncio.sleep(0.5)

                # Another nested operation
                async with trace_operation(
                    "validation",
                    OperationType.INTELLIGENCE_PROCESS,
                    validation_rules=["format", "integrity", "schema"]
                ):
                    await asyncio.sleep(0.2)
                    print("Validation completed")

                print("Data processing completed")

            # Operation with performance tracking
            async with trace_operation(
                "performance_critical_operation",
                OperationType.MEMORY_RETRIEVE,
                trace_performance=True,
                cache_enabled=True
            ):
                # Simulate memory-intensive operation
                data = list(range(10000))
                processed = [x * 2 for x in data]
                await asyncio.sleep(0.3)
                print(f"Processed {len(processed)} items")

    async def demo_health_check_system(self):
        """Demonstrate comprehensive health check system."""
        print("\n=== Health Check System Demo ===")

        async with correlation_context(operation="health_check_demo"):
            # Register mock health checks
            await self._register_demo_health_checks()

            # Perform comprehensive health check
            print("Performing comprehensive health check...")
            health_response = await health_manager.get_comprehensive_health()

            print(f"Overall Status: {health_response.status}")
            print(f"Check Latency: {health_response.latency_ms:.2f}ms")
            print(f"System Uptime: {health_response.uptime_seconds}s")

            print(f"\nResource Usage:")
            resources = health_response.resource_usage
            print(f"  CPU: {resources.cpu_usage_percent:.1f}%")
            print(f"  Memory: {resources.memory_usage_mb:.1f}MB ({resources.memory_usage_percent:.1f}%)")
            print(f"  Disk: {resources.disk_usage_percent:.1f}%")

            print(f"\nDependency Status:")
            for dep in health_response.dependencies:
                status_emoji = "✅" if dep.status == "healthy" else "⚠️" if dep.status == "degraded" else "❌"
                print(f"  {status_emoji} {dep.name}: {dep.status} ({dep.latency_ms:.1f}ms)")
                if dep.error_message:
                    print(f"    Error: {dep.error_message}")

            # Show circuit breaker stats
            cb_stats = health_manager.get_circuit_breaker_stats()
            if cb_stats:
                print(f"\nCircuit Breaker Stats:")
                for name, stats in cb_stats.items():
                    print(f"  {name}: {stats['state']} "
                          f"(calls: {stats['total_calls']}, failures: {stats['failure_count']})")

    async def _register_demo_health_checks(self):
        """Register demo health checks for the demonstration."""

        # Mock PostgreSQL health check
        async def mock_postgresql_check():
            from omnimemory.utils.health_manager import HealthCheckConfig, HealthCheckResult, HealthStatus, DependencyType

            config = HealthCheckConfig(
                name="postgresql",
                dependency_type=DependencyType.DATABASE,
                critical=True
            )

            # Simulate connection check
            await asyncio.sleep(0.05)  # Simulate network latency

            return HealthCheckResult(
                config=config,
                status=HealthStatus.HEALTHY,
                latency_ms=0.0,
                metadata={"connection_pool": "healthy", "active_connections": 5}
            )

        # Mock Redis health check
        async def mock_redis_check():
            from omnimemory.utils.health_manager import HealthCheckConfig, HealthCheckResult, HealthStatus, DependencyType

            config = HealthCheckConfig(
                name="redis",
                dependency_type=DependencyType.CACHE,
                critical=True
            )

            await asyncio.sleep(0.03)

            return HealthCheckResult(
                config=config,
                status=HealthStatus.HEALTHY,
                latency_ms=0.0,
                metadata={"memory_usage": "45%", "connected_clients": 12}
            )

        # Mock Pinecone health check (simulated as degraded)
        async def mock_pinecone_check():
            from omnimemory.utils.health_manager import HealthCheckConfig, HealthCheckResult, HealthStatus, DependencyType

            config = HealthCheckConfig(
                name="pinecone",
                dependency_type=DependencyType.VECTOR_DB,
                critical=False  # Non-critical for demo
            )

            await asyncio.sleep(0.1)

            return HealthCheckResult(
                config=config,
                status=HealthStatus.DEGRADED,
                latency_ms=0.0,
                error_message="High latency detected",
                metadata={"index_status": "ready", "vector_count": 50000}
            )

        # Register health checks
        health_manager.register_health_check(
            HealthCheckConfig(
                name="postgresql",
                dependency_type=DependencyType.DATABASE,
                critical=True,
                timeout=5.0
            ),
            mock_postgresql_check
        )

        health_manager.register_health_check(
            HealthCheckConfig(
                name="redis",
                dependency_type=DependencyType.CACHE,
                critical=True,
                timeout=3.0
            ),
            mock_redis_check
        )

        health_manager.register_health_check(
            HealthCheckConfig(
                name="pinecone",
                dependency_type=DependencyType.VECTOR_DB,
                critical=False,
                timeout=10.0
            ),
            mock_pinecone_check
        )

    async def run_full_demo(self):
        """Run the complete architecture demonstration."""
        print("🚀 Advanced Architecture Improvements Demo")
        print("=" * 50)

        start_time = time.time()

        try:
            # Run all demonstrations
            await self.demo_resource_management()
            await self.demo_concurrency_improvements()
            await self.demo_migration_progress_tracking()
            await self.demo_observability_tracking()
            await self.demo_health_check_system()

        except Exception as e:
            logger.error(
                "demo_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            print(f"\n❌ Demo failed: {e}")
            raise

        finally:
            total_time = time.time() - start_time
            print(f"\n✅ Demo completed in {total_time:.2f} seconds")
            print("=" * 50)

async def main():
    """Main entry point for the demonstration."""
    demo = AdvancedArchitectureDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    # Configure logging for demo
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    asyncio.run(main())