# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerMemoryTick.

Tests the RuntimeTick handler that evaluates memory entities for TTL
expiration and archive eligibility. Tests cover the handler's behavior
with mocked projection readers and various memory lifecycle scenarios.

Test Categories:
    - Initialization: Handler setup and configuration
    - No Candidates: Empty projection reader scenarios
    - Expiration Detection: Finding expired memories correctly
    - Archive Detection: Finding archive candidates correctly
    - Deduplication: Same memory not re-emitted
    - Batch Limiting: Respecting batch_size configuration

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_handler_memory_tick.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_tick import (
    HandlerMemoryTick,
    ModelMemoryArchiveInitiated,
    ModelMemoryExpiredEvent,
    ModelMemoryLifecycleProjection,
    ModelMemoryTickResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def container() -> ModelONEXContainer:
    """Provide an ONEX container for handler initialization.

    Returns:
        A ModelONEXContainer instance for dependency injection.
    """
    return ModelONEXContainer()


@pytest.fixture
def fixed_now() -> datetime:
    """Provide a fixed timestamp for deterministic testing.

    Returns:
        A fixed datetime in UTC timezone.
    """
    return datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def tick_id() -> UUID:
    """Provide a fixed tick ID for testing.

    Returns:
        A deterministic UUID for the tick.
    """
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def correlation_id() -> UUID:
    """Provide a fixed correlation ID for testing.

    Returns:
        A deterministic UUID for correlation.
    """
    return UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@pytest.fixture
def runtime_tick(fixed_now: datetime, tick_id: UUID) -> ModelRuntimeTick:
    """Create a runtime tick event for testing.

    Args:
        fixed_now: Fixed timestamp for the tick.
        tick_id: Tick identifier.

    Returns:
        Configured ModelRuntimeTick instance.
    """
    return ModelRuntimeTick(
        now=fixed_now,
        tick_id=tick_id,
        sequence_number=1,
        scheduled_at=fixed_now,
        correlation_id=uuid4(),
        scheduler_id="test-scheduler-001",
        tick_interval_ms=1000,
    )


@pytest.fixture
def tick_envelope(
    runtime_tick: ModelRuntimeTick,
    fixed_now: datetime,
    correlation_id: UUID,
) -> ModelEventEnvelope[ModelRuntimeTick]:
    """Create an event envelope containing a runtime tick.

    Args:
        runtime_tick: The tick event payload.
        fixed_now: Fixed timestamp.
        correlation_id: Correlation ID for tracing.

    Returns:
        Event envelope wrapping the runtime tick.
    """
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        envelope_timestamp=fixed_now,
        correlation_id=correlation_id,
        payload=runtime_tick,
        source_service="test-runtime",
        event_type="RuntimeTick",
    )


def create_projection(
    entity_id: UUID | None = None,
    lifecycle_state: str = "active",
    expires_at: datetime | None = None,
    expired_at: datetime | None = None,
    archived_at: datetime | None = None,
    lifecycle_revision: int = 1,
    expiration_emitted_at: datetime | None = None,
    archive_initiated_at: datetime | None = None,
) -> ModelMemoryLifecycleProjection:
    """Create a memory lifecycle projection for testing.

    Args:
        entity_id: Memory entity ID (auto-generated if not provided).
        lifecycle_state: Current lifecycle state.
        expires_at: TTL deadline.
        expired_at: When memory transitioned to EXPIRED.
        archived_at: When memory was archived.
        lifecycle_revision: Revision for optimistic locking.
        expiration_emitted_at: When expiration event was emitted.
        archive_initiated_at: When archive initiation was emitted.

    Returns:
        Configured ModelMemoryLifecycleProjection instance.
    """
    return ModelMemoryLifecycleProjection(
        entity_id=entity_id or uuid4(),
        lifecycle_state=lifecycle_state,
        expires_at=expires_at,
        expired_at=expired_at,
        archived_at=archived_at,
        lifecycle_revision=lifecycle_revision,
        expiration_emitted_at=expiration_emitted_at,
        archive_initiated_at=archive_initiated_at,
    )


class MockProjectionReader:
    """Mock implementation of ProtocolModelMemoryLifecycleProjectionReader."""

    def __init__(
        self,
        expired_candidates: list[ModelMemoryLifecycleProjection] | None = None,
        archive_candidates: list[ModelMemoryLifecycleProjection] | None = None,
    ) -> None:
        """Initialize mock reader with test data.

        Args:
            expired_candidates: List of projections to return for expired queries.
            archive_candidates: List of projections to return for archive queries.
        """
        self._expired_candidates = expired_candidates or []
        self._archive_candidates = archive_candidates or []
        self.get_expired_candidates_calls: list[dict] = []
        self.get_archive_candidates_calls: list[dict] = []

    async def get_expired_candidates(
        self,
        now: datetime,
        domain: str,
        correlation_id: UUID,
        limit: int = 100,
    ) -> list[ModelMemoryLifecycleProjection]:
        """Return mock expired candidates.

        Args:
            now: Current time for deadline comparison.
            domain: Domain scope for the query.
            correlation_id: Correlation ID for tracing.
            limit: Maximum results to return.

        Returns:
            Configured list of expired candidate projections.
        """
        self.get_expired_candidates_calls.append(
            {
                "now": now,
                "domain": domain,
                "correlation_id": correlation_id,
                "limit": limit,
            }
        )
        return self._expired_candidates[:limit]

    async def get_archive_candidates(
        self,
        now: datetime,
        domain: str,
        correlation_id: UUID,
        limit: int = 100,
    ) -> list[ModelMemoryLifecycleProjection]:
        """Return mock archive candidates.

        Args:
            now: Current time for consistent evaluation.
            domain: Domain scope for the query.
            correlation_id: Correlation ID for tracing.
            limit: Maximum results to return.

        Returns:
            Configured list of archive candidate projections.
        """
        self.get_archive_candidates_calls.append(
            {
                "now": now,
                "domain": domain,
                "correlation_id": correlation_id,
                "limit": limit,
            }
        )
        return self._archive_candidates[:limit]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHandlerMemoryTickInitialization:
    """Tests for HandlerMemoryTick initialization."""

    def test_handler_creates_with_container(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler can be created with container.

        Given: An ONEX container
        When: Creating HandlerMemoryTick
        Then: Handler is created successfully but not initialized
        """
        handler = HandlerMemoryTick(container)
        assert handler is not None
        assert handler._projection_reader is None
        assert handler.initialized is False

    @pytest.mark.asyncio
    async def test_handler_initializes_without_projection_reader(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler can be initialized without projection reader.

        Given: No projection reader provided
        When: Initializing HandlerMemoryTick
        Then: Handler is initialized in stub mode
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize()

        assert handler.initialized is True
        assert handler._projection_reader is None

    @pytest.mark.asyncio
    async def test_handler_initializes_with_projection_reader(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler can be initialized with projection reader.

        Given: A mock projection reader
        When: Initializing HandlerMemoryTick
        Then: Handler is initialized with the reader attached
        """
        reader = MockProjectionReader()
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        assert handler.initialized is True
        assert handler._projection_reader is reader

    @pytest.mark.asyncio
    async def test_handler_initializes_with_custom_batch_size(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler can be initialized with custom batch size.

        Given: A custom batch_size value
        When: Initializing HandlerMemoryTick
        Then: Handler uses the custom batch size
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize(batch_size=50)

        assert handler._batch_size == 50

    @pytest.mark.asyncio
    async def test_handler_default_batch_size(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler uses default batch size of 100.

        Given: No batch_size provided
        When: Initializing HandlerMemoryTick
        Then: Handler uses default batch size of 100
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize()

        assert handler._batch_size == 100

    def test_handler_properties(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler exposes correct properties.

        Given: A HandlerMemoryTick instance
        When: Accessing handler properties
        Then: Properties return expected values
        """
        handler = HandlerMemoryTick(container)

        assert handler.handler_id == "handler-memory-tick"
        assert handler.category == EnumMessageCategory.COMMAND
        assert handler.message_types == {"ModelRuntimeTick"}
        assert handler.node_kind == EnumNodeKind.ORCHESTRATOR

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test health_check returns correct status before initialization.

        Given: A handler that has not been initialized
        When: Calling health_check
        Then: Returns typed health model with initialized=False and no circuit breaker state
        """
        handler = HandlerMemoryTick(container)

        health = await handler.health_check()

        assert health.initialized is False
        assert health.circuit_breaker_state is None
        assert health.projection_reader_available is False
        assert health.batch_size == 100

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test health_check returns correct status after initialization.

        Given: A handler that has been initialized
        When: Calling health_check
        Then: Returns typed health model with initialized=True and circuit breaker state
        """
        reader = MockProjectionReader()
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader, batch_size=50)

        health = await handler.health_check()

        assert health.initialized is True
        assert health.circuit_breaker_state == "closed"
        assert health.projection_reader_available is True
        assert health.batch_size == 50

    @pytest.mark.asyncio
    async def test_describe_returns_handler_metadata(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test describe returns handler metadata.

        Given: A HandlerMemoryTick instance
        When: Calling describe
        Then: Returns typed metadata model with comprehensive handler information
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize()

        metadata = await handler.describe()

        assert metadata.name == "HandlerMemoryTick"
        assert metadata.description  # Non-empty description
        assert "memory_expiration" in metadata.capabilities
        assert "archive_initiation" in metadata.capabilities
        assert metadata.initialized is True
        assert "ModelRuntimeTick" in metadata.message_types


# =============================================================================
# No Candidates Tests
# =============================================================================


class TestNoCandidates:
    """Tests for scenarios with no lifecycle candidates."""

    @pytest.mark.asyncio
    async def test_tick_with_no_projection_reader(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """Test tick returns empty result when no projection reader.

        Given: Handler with no projection reader (stub mode)
        When: Processing a runtime tick
        Then: Result contains zero events and success metrics
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize()

        output = await handler.handle(tick_envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.events == ()
        assert output.metrics["expired_count"] == 0.0
        assert output.metrics["archive_initiated_count"] == 0.0

    @pytest.mark.asyncio
    async def test_tick_with_empty_projection(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """Test tick returns empty result when projection has no candidates.

        Given: Handler with empty projection reader
        When: Processing a runtime tick
        Then: Result contains zero events
        """
        reader = MockProjectionReader(
            expired_candidates=[],
            archive_candidates=[],
        )
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert output.events == ()
        assert output.metrics["expired_count"] == 0.0
        assert output.metrics["archive_initiated_count"] == 0.0

    @pytest.mark.asyncio
    async def test_projection_reader_called_correctly(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
        correlation_id: UUID,
    ) -> None:
        """Test projection reader methods are called with correct parameters.

        Given: Handler with mock projection reader
        When: Processing a runtime tick
        Then: Projection reader methods are called with correct args
        """
        reader = MockProjectionReader()
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader, batch_size=50)

        await handler.handle(tick_envelope)

        # Verify expired candidates call
        assert len(reader.get_expired_candidates_calls) == 1
        call = reader.get_expired_candidates_calls[0]
        assert call["now"] == fixed_now
        assert call["domain"] == "memory"
        assert call["correlation_id"] == correlation_id
        assert call["limit"] == 50

        # Verify archive candidates call
        assert len(reader.get_archive_candidates_calls) == 1
        call = reader.get_archive_candidates_calls[0]
        assert call["now"] == fixed_now
        assert call["domain"] == "memory"
        assert call["limit"] == 50


# =============================================================================
# Expiration Detection Tests
# =============================================================================


class TestExpirationDetection:
    """Tests for memory expiration detection."""

    @pytest.mark.asyncio
    async def test_finds_expired_memory(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler finds and emits event for expired memory.

        Given: A projection with one expired memory
        When: Processing a runtime tick
        Then: One ModelMemoryExpiredEvent is emitted
        """
        expired_memory = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now - timedelta(hours=1),  # Expired 1 hour ago
            lifecycle_revision=3,
        )
        reader = MockProjectionReader(expired_candidates=[expired_memory])
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 1
        event = output.events[0]
        assert isinstance(event, ModelMemoryExpiredEvent)
        assert event.entity_id == expired_memory.entity_id
        assert event.memory_id == expired_memory.entity_id
        assert event.expires_at == expired_memory.expires_at
        assert event.lifecycle_revision == 3

    @pytest.mark.asyncio
    async def test_finds_multiple_expired_memories(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler finds and emits events for multiple expired memories.

        Given: A projection with multiple expired memories
        When: Processing a runtime tick
        Then: Multiple ModelMemoryExpiredEvent events are emitted
        """
        expired_memories = [
            create_projection(
                lifecycle_state="active",
                expires_at=fixed_now - timedelta(hours=i),
            )
            for i in range(1, 4)
        ]
        reader = MockProjectionReader(expired_candidates=expired_memories)
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 3
        assert all(isinstance(e, ModelMemoryExpiredEvent) for e in output.events)
        assert output.metrics["expired_count"] == 3.0

    @pytest.mark.asyncio
    async def test_skips_memory_not_needing_expiration(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler skips memories that don't need expiration event.

        Given: A projection with memory that already has expiration emitted
        When: Processing a runtime tick
        Then: No event is emitted for that memory
        """
        # Memory that already had expiration emitted
        already_emitted = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now - timedelta(hours=1),
            expiration_emitted_at=fixed_now - timedelta(minutes=30),
        )
        reader = MockProjectionReader(expired_candidates=[already_emitted])
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        # Event should be filtered out by needs_expiration_event() check
        assert len(output.events) == 0

    @pytest.mark.asyncio
    async def test_expiration_event_contains_correct_causation_id(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        tick_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test expired event has tick_id as causation_id.

        Given: A runtime tick with specific tick_id
        When: Processing a tick with expired memory
        Then: Emitted event has tick_id as causation_id
        """
        expired_memory = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now - timedelta(hours=1),
        )
        reader = MockProjectionReader(expired_candidates=[expired_memory])
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 1
        event = output.events[0]
        assert isinstance(event, ModelMemoryExpiredEvent)
        assert event.causation_id == tick_id


# =============================================================================
# Archive Detection Tests
# =============================================================================


class TestArchiveDetection:
    """Tests for archive candidate detection."""

    @pytest.mark.asyncio
    async def test_finds_archive_candidate(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler finds and emits event for archive candidate.

        Given: A projection with one expired memory pending archive
        When: Processing a runtime tick
        Then: One ModelMemoryArchiveInitiated event is emitted
        """
        archive_candidate = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=2),
            lifecycle_revision=5,
        )
        reader = MockProjectionReader(archive_candidates=[archive_candidate])
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 1
        event = output.events[0]
        assert isinstance(event, ModelMemoryArchiveInitiated)
        assert event.entity_id == archive_candidate.entity_id
        assert event.memory_id == archive_candidate.entity_id
        assert event.expired_at == archive_candidate.expired_at
        assert event.lifecycle_revision == 5

    @pytest.mark.asyncio
    async def test_finds_multiple_archive_candidates(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler finds and emits events for multiple archive candidates.

        Given: A projection with multiple expired memories pending archive
        When: Processing a runtime tick
        Then: Multiple ModelMemoryArchiveInitiated events are emitted
        """
        archive_candidates = [
            create_projection(
                lifecycle_state="expired",
                expired_at=fixed_now - timedelta(hours=i),
            )
            for i in range(1, 4)
        ]
        reader = MockProjectionReader(archive_candidates=archive_candidates)
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 3
        assert all(isinstance(e, ModelMemoryArchiveInitiated) for e in output.events)
        assert output.metrics["archive_initiated_count"] == 3.0

    @pytest.mark.asyncio
    async def test_skips_already_initiated_archive(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler skips memories that already have archive initiated.

        Given: A projection with memory that already has archive initiated
        When: Processing a runtime tick
        Then: No event is emitted for that memory
        """
        already_initiated = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=2),
            archive_initiated_at=fixed_now - timedelta(hours=1),
        )
        reader = MockProjectionReader(archive_candidates=[already_initiated])
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        # Event should be filtered out by needs_archive_event() check
        assert len(output.events) == 0

    @pytest.mark.asyncio
    async def test_skips_already_archived_memory(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler skips memories that are already archived.

        Given: A projection with memory that is already archived
        When: Processing a runtime tick
        Then: No event is emitted for that memory
        """
        already_archived = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=2),
            archived_at=fixed_now - timedelta(hours=1),
        )
        reader = MockProjectionReader(archive_candidates=[already_archived])
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 0


# =============================================================================
# Combined Scenarios Tests
# =============================================================================


class TestCombinedScenarios:
    """Tests for combined expiration and archive scenarios."""

    @pytest.mark.asyncio
    async def test_finds_both_expired_and_archive_candidates(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler processes both expirations and archives in one tick.

        Given: Projections with both expired and archive candidate memories
        When: Processing a runtime tick
        Then: Events for both categories are emitted
        """
        expired_memory = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now - timedelta(hours=1),
        )
        archive_candidate = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=2),
        )
        reader = MockProjectionReader(
            expired_candidates=[expired_memory],
            archive_candidates=[archive_candidate],
        )
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader)

        output = await handler.handle(tick_envelope)

        assert len(output.events) == 2
        assert output.metrics["expired_count"] == 1.0
        assert output.metrics["archive_initiated_count"] == 1.0

        # Check event types
        event_types = [type(e) for e in output.events]
        assert ModelMemoryExpiredEvent in event_types
        assert ModelMemoryArchiveInitiated in event_types


# =============================================================================
# Batch Limiting Tests
# =============================================================================


class TestBatchLimiting:
    """Tests for batch size limiting behavior."""

    @pytest.mark.asyncio
    async def test_respects_batch_size_for_expirations(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        fixed_now: datetime,
    ) -> None:
        """Test handler respects batch_size when processing expirations.

        Given: More expired memories than batch_size
        When: Processing a runtime tick
        Then: Only batch_size events are emitted
        """
        # Create 10 expired memories
        expired_memories = [
            create_projection(
                lifecycle_state="active",
                expires_at=fixed_now - timedelta(hours=i),
            )
            for i in range(1, 11)
        ]
        reader = MockProjectionReader(expired_candidates=expired_memories)
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader, batch_size=5)

        output = await handler.handle(tick_envelope)

        # Should only emit batch_size (5) events
        assert len(output.events) == 5
        assert output.metrics["expired_count"] == 5.0

    @pytest.mark.asyncio
    async def test_batch_size_passed_to_reader(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """Test batch_size is passed to projection reader.

        Given: Handler with custom batch_size
        When: Processing a runtime tick
        Then: Projection reader receives the batch_size as limit
        """
        reader = MockProjectionReader()
        handler = HandlerMemoryTick(container)
        await handler.initialize(projection_reader=reader, batch_size=25)

        await handler.handle(tick_envelope)

        # Both reader calls should have limit=25
        assert reader.get_expired_candidates_calls[0]["limit"] == 25
        assert reader.get_archive_candidates_calls[0]["limit"] == 25


# =============================================================================
# Handler Output Tests
# =============================================================================


class TestHandlerOutput:
    """Tests for handler output structure."""

    @pytest.mark.asyncio
    async def test_output_structure(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
        correlation_id: UUID,
    ) -> None:
        """Test handler returns correctly structured output.

        Given: A handler processing a tick
        When: Tick processing completes
        Then: Output has correct structure and metadata
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize()

        output = await handler.handle(tick_envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-memory-tick"
        assert output.correlation_id == correlation_id
        assert output.input_envelope_id == tick_envelope.envelope_id
        assert output.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_output_metrics_present(
        self,
        container: ModelONEXContainer,
        tick_envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> None:
        """Test handler output includes required metrics.

        Given: A handler processing a tick
        When: Tick processing completes
        Then: Output metrics include all expected keys
        """
        handler = HandlerMemoryTick(container)
        await handler.initialize()

        output = await handler.handle(tick_envelope)

        assert "expired_count" in output.metrics
        assert "archive_initiated_count" in output.metrics
        assert "batch_size" in output.metrics
        assert output.metrics["batch_size"] == 100.0  # Default batch size


# =============================================================================
# Model Tests
# =============================================================================


class TestModelMemoryLifecycleProjection:
    """Tests for ModelMemoryLifecycleProjection model."""

    def test_needs_expiration_event_returns_true(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_expiration_event returns True for valid candidate.

        Given: Active memory with expired TTL and no emission marker
        When: Checking needs_expiration_event
        Then: Returns True
        """
        projection = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now - timedelta(hours=1),
            expiration_emitted_at=None,
        )

        assert projection.needs_expiration_event(fixed_now) is True

    def test_needs_expiration_event_false_when_already_emitted(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_expiration_event returns False when already emitted.

        Given: Active memory with expiration already emitted
        When: Checking needs_expiration_event
        Then: Returns False
        """
        projection = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now - timedelta(hours=1),
            expiration_emitted_at=fixed_now - timedelta(minutes=30),
        )

        assert projection.needs_expiration_event(fixed_now) is False

    def test_needs_expiration_event_false_when_not_expired(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_expiration_event returns False when not yet expired.

        Given: Active memory with future expiration
        When: Checking needs_expiration_event
        Then: Returns False
        """
        projection = create_projection(
            lifecycle_state="active",
            expires_at=fixed_now + timedelta(hours=1),  # Future
        )

        assert projection.needs_expiration_event(fixed_now) is False

    def test_needs_expiration_event_false_when_not_active(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_expiration_event returns False when not active.

        Given: Expired memory (wrong state)
        When: Checking needs_expiration_event
        Then: Returns False
        """
        projection = create_projection(
            lifecycle_state="expired",
            expires_at=fixed_now - timedelta(hours=1),
        )

        assert projection.needs_expiration_event(fixed_now) is False

    def test_needs_archive_event_returns_true(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_archive_event returns True for valid candidate.

        Given: Expired memory with no archive markers
        When: Checking needs_archive_event
        Then: Returns True
        """
        projection = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=1),
            archived_at=None,
            archive_initiated_at=None,
        )

        assert projection.needs_archive_event(fixed_now) is True

    def test_needs_archive_event_false_when_already_initiated(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_archive_event returns False when already initiated.

        Given: Expired memory with archive already initiated
        When: Checking needs_archive_event
        Then: Returns False
        """
        projection = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=2),
            archive_initiated_at=fixed_now - timedelta(hours=1),
        )

        assert projection.needs_archive_event(fixed_now) is False

    def test_needs_archive_event_false_when_already_archived(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_archive_event returns False when already archived.

        Given: Expired memory that is already archived
        When: Checking needs_archive_event
        Then: Returns False
        """
        projection = create_projection(
            lifecycle_state="expired",
            expired_at=fixed_now - timedelta(hours=2),
            archived_at=fixed_now - timedelta(hours=1),
        )

        assert projection.needs_archive_event(fixed_now) is False

    def test_needs_archive_event_false_when_not_expired(
        self,
        fixed_now: datetime,
    ) -> None:
        """Test needs_archive_event returns False when not expired state.

        Given: Active memory (wrong state for archive)
        When: Checking needs_archive_event
        Then: Returns False
        """
        projection = create_projection(
            lifecycle_state="active",
        )

        assert projection.needs_archive_event(fixed_now) is False


class TestModelMemoryTickResult:
    """Tests for ModelMemoryTickResult model."""

    def test_result_model_creation(self, fixed_now: datetime) -> None:
        """Test ModelMemoryTickResult can be created with valid data.

        Given: Valid tick result data
        When: Creating ModelMemoryTickResult
        Then: Model is created successfully
        """
        tick_id = uuid4()
        result = ModelMemoryTickResult(
            expired_count=5,
            archive_initiated_count=3,
            tick_id=tick_id,
            sequence_number=42,
            evaluated_at=fixed_now,
        )

        assert result.expired_count == 5
        assert result.archive_initiated_count == 3
        assert result.tick_id == tick_id
        assert result.sequence_number == 42
        assert result.evaluated_at == fixed_now

    def test_result_model_is_frozen(self, fixed_now: datetime) -> None:
        """Test ModelMemoryTickResult is immutable.

        Given: A ModelMemoryTickResult instance
        When: Attempting to modify a field
        Then: Raises an error
        """
        result = ModelMemoryTickResult(
            expired_count=5,
            archive_initiated_count=3,
            tick_id=uuid4(),
            sequence_number=1,
            evaluated_at=fixed_now,
        )

        with pytest.raises(Exception):  # Pydantic ValidationError for frozen models
            result.expired_count = 10  # type: ignore[misc]
