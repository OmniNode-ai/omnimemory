# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for AdapterRuntimeTickMemory.

Tests the tick-based lifecycle detection adapter that wraps HandlerMemoryTick.

Test Categories:
    - Initialization: Adapter setup, initialization state
    - Not Initialized: RuntimeError on process_tick before initialize()
    - Delegation: process_tick() delegates to the underlying handler
    - Health Check: health_check() aggregates handler health
    - Describe: describe() returns correct adapter metadata
    - Shutdown: shutdown() is idempotent and resets state

Related Tickets:
    - OMN-1603: Add adapter implementations for memory lifecycle orchestrator

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_adapter_runtime_tick_memory.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

from omnimemory.nodes.memory_lifecycle_orchestrator.adapters import (
    AdapterRuntimeTickMemory,
    ModelRuntimeTickAdapterHealth,
    ModelRuntimeTickAdapterMetadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def container() -> ModelONEXContainer:
    """Provide an ONEX container for adapter initialization."""
    return ModelONEXContainer()


@pytest.fixture
def fixed_now() -> datetime:
    """Provide a fixed timestamp for deterministic testing."""
    return datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def tick_id() -> UUID:
    """Provide a fixed tick ID for testing."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def correlation_id() -> UUID:
    """Provide a fixed correlation ID for testing."""
    return UUID("87654321-4321-8765-4321-876543218765")


@pytest.fixture
def scheduler_id() -> str:
    """Provide a scheduler ID for testing."""
    return "test-scheduler-001"


@pytest.fixture
def runtime_tick(
    fixed_now: datetime,
    tick_id: UUID,
    correlation_id: UUID,
    scheduler_id: str,
) -> ModelRuntimeTick:
    """Provide a runtime tick for testing."""
    return ModelRuntimeTick(
        now=fixed_now,
        tick_id=tick_id,
        sequence_number=1,
        scheduled_at=fixed_now,
        correlation_id=correlation_id,
        scheduler_id=scheduler_id,
        tick_interval_ms=1000,
    )


@pytest.fixture
def tick_envelope(
    runtime_tick: ModelRuntimeTick,
    correlation_id: UUID,
) -> ModelEventEnvelope[ModelRuntimeTick]:
    """Provide a tick event envelope for testing."""
    return ModelEventEnvelope(
        payload=runtime_tick,
        correlation_id=correlation_id,
        envelope_id=uuid4(),
    )


@pytest.fixture
def adapter(container: ModelONEXContainer) -> AdapterRuntimeTickMemory:
    """Provide an uninitialized adapter for testing."""
    return AdapterRuntimeTickMemory(container)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterInitialization:
    """Verify adapter initialization behavior."""

    def test_not_initialized_before_initialize(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """Adapter reports not initialized before initialize() is called."""
        assert adapter.initialized is False

    @pytest.mark.asyncio
    async def test_initialized_after_initialize(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """Adapter reports initialized after initialize() is called."""
        await adapter.initialize()
        assert adapter.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_batch_size(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """Adapter initializes successfully with custom batch size."""
        await adapter.initialize(batch_size=50)
        assert adapter.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_without_projection_reader(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """Adapter initializes without a projection reader (no-op mode)."""
        await adapter.initialize(projection_reader=None)
        assert adapter.initialized is True


# ---------------------------------------------------------------------------
# Not Initialized Guard Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNotInitializedGuard:
    """Verify that process_tick() raises RuntimeError before initialize()."""

    @pytest.mark.asyncio
    async def test_process_tick_raises_if_not_initialized(
        self,
        adapter: AdapterRuntimeTickMemory,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """process_tick() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.process_tick(tick_envelope)


# ---------------------------------------------------------------------------
# Delegation Tests (no projection reader = empty results)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTickDelegation:
    """Verify that process_tick() delegates to the underlying handler."""

    @pytest.mark.asyncio
    async def test_process_tick_returns_empty_events_without_reader(
        self,
        adapter: AdapterRuntimeTickMemory,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """process_tick() returns empty events when no projection reader."""
        await adapter.initialize(projection_reader=None)
        output = await adapter.process_tick(tick_envelope)
        # Without a projection reader, no lifecycle events are emitted
        assert output.events == ()

    @pytest.mark.asyncio
    async def test_process_tick_records_metrics(
        self,
        adapter: AdapterRuntimeTickMemory,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """process_tick() returns output with metrics keys."""
        await adapter.initialize(projection_reader=None)
        output = await adapter.process_tick(tick_envelope)
        assert "expired_count" in output.metrics
        assert "archive_initiated_count" in output.metrics
        assert output.metrics["expired_count"] == 0.0
        assert output.metrics["archive_initiated_count"] == 0.0


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterHealthCheck:
    """Verify health_check() aggregates handler health correctly."""

    @pytest.mark.asyncio
    async def test_health_before_initialization(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """health_check() reflects uninitialized state."""
        health = await adapter.health_check()
        assert isinstance(health, ModelRuntimeTickAdapterHealth)
        assert health.initialized is False
        assert health.handler_health.initialized is False

    @pytest.mark.asyncio
    async def test_health_after_initialization(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """health_check() reflects initialized state."""
        await adapter.initialize()
        health = await adapter.health_check()
        assert health.initialized is True
        assert health.handler_health.initialized is True

    @pytest.mark.asyncio
    async def test_health_reports_no_projection_reader(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """health_check() reports no projection reader when none configured."""
        await adapter.initialize(projection_reader=None)
        health = await adapter.health_check()
        assert health.handler_health.projection_reader_available is False


# ---------------------------------------------------------------------------
# Describe Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDescribe:
    """Verify describe() returns correct adapter metadata."""

    @pytest.mark.asyncio
    async def test_describe_returns_correct_name(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """describe() returns the correct adapter class name."""
        await adapter.initialize()
        metadata = await adapter.describe()
        assert isinstance(metadata, ModelRuntimeTickAdapterMetadata)
        assert metadata.name == "AdapterRuntimeTickMemory"

    @pytest.mark.asyncio
    async def test_describe_includes_handler_metadata(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """describe() includes metadata from the underlying handler."""
        await adapter.initialize()
        metadata = await adapter.describe()
        assert metadata.handler_metadata.name == "HandlerMemoryTick"

    @pytest.mark.asyncio
    async def test_describe_reflects_initialization_state(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """describe() reflects initialization state."""
        # Before initialization
        metadata = await adapter.describe()
        assert metadata.initialized is False

        # After initialization
        await adapter.initialize()
        metadata = await adapter.describe()
        assert metadata.initialized is True


# ---------------------------------------------------------------------------
# Shutdown Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterShutdown:
    """Verify shutdown() is idempotent and resets state."""

    @pytest.mark.asyncio
    async def test_shutdown_resets_initialized(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """shutdown() resets initialized to False."""
        await adapter.initialize()
        assert adapter.initialized is True
        await adapter.shutdown()
        assert adapter.initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """shutdown() can be called multiple times without error."""
        await adapter.initialize()
        await adapter.shutdown()
        await adapter.shutdown()  # Should not raise
        assert adapter.initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_before_initialize_is_safe(
        self, adapter: AdapterRuntimeTickMemory
    ) -> None:
        """shutdown() on uninitialized adapter is a no-op."""
        await adapter.shutdown()  # Should not raise
        assert adapter.initialized is False
