"""Integration tests for memory_lifecycle_orchestrator node.

Tests the lifecycle state machine transitions, adapters, and handlers
for memory lifecycle management. These are component integration tests
that do not require real database connections.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from omnimemory.enums.enum_memory_lifecycle_state import EnumMemoryLifecycleState
from omnimemory.nodes.memory_lifecycle_orchestrator.adapters.adapter_postgres_deactivate_memory import (
    AdapterPostgresDeactivateMemory,
    ModelMemoryExpireRequest,
)
from omnimemory.nodes.memory_lifecycle_orchestrator.adapters.adapter_runtime_tick_memory import (
    AdapterRuntimeTickMemory,
)
from omnimemory.nodes.memory_lifecycle_orchestrator.handlers.handler_filesystem_archive import (
    HandlerFileSystemArchive,
    ModelArchiveRecord,
    ModelArchiveRequest,
)
from omnimemory.nodes.memory_lifecycle_orchestrator.validators.validator_lifecycle_transition import (
    VALID_TRANSITIONS,
    apply_transition,
    validate_transition,
)


class TestLifecycleStateTransitions:
    """Test the lifecycle state machine transitions."""

    def test_valid_transitions_from_active(self) -> None:
        """ACTIVE can transition to EXPIRED or stay ACTIVE (promotion)."""
        assert validate_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.EXPIRED,
        )
        assert validate_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.ACTIVE,
        )
        # Invalid: can't skip to ARCHIVED
        assert not validate_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.ARCHIVED,
        )

    def test_valid_transitions_from_expired(self) -> None:
        """EXPIRED can transition to ARCHIVED or ACTIVE (promotion)."""
        assert validate_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ARCHIVED,
        )
        assert validate_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ACTIVE,
        )
        # Invalid: can't go directly to DELETED
        assert not validate_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.DELETED,
        )

    def test_valid_transitions_from_archived(self) -> None:
        """ARCHIVED can transition to DELETED or ACTIVE (resurrection)."""
        assert validate_transition(
            EnumMemoryLifecycleState.ARCHIVED,
            EnumMemoryLifecycleState.DELETED,
        )
        assert validate_transition(
            EnumMemoryLifecycleState.ARCHIVED,
            EnumMemoryLifecycleState.ACTIVE,
        )

    def test_deleted_is_terminal(self) -> None:
        """DELETED is a terminal state with no valid transitions."""
        for target in EnumMemoryLifecycleState:
            assert not validate_transition(
                EnumMemoryLifecycleState.DELETED,
                target,
            )

    def test_valid_transitions_dict_completeness(self) -> None:
        """VALID_TRANSITIONS dict covers all states."""
        for state in EnumMemoryLifecycleState:
            assert state in VALID_TRANSITIONS, f"Missing state: {state}"

    def test_apply_transition_increments_revision(self) -> None:
        """Every successful transition increments lifecycle_revision."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = apply_transition(
            current_state=EnumMemoryLifecycleState.ACTIVE,
            target_state=EnumMemoryLifecycleState.EXPIRED,
            current_revision=5,
            now=now,
        )
        assert result.success
        assert result.new_revision == 6

    def test_apply_transition_sets_correct_timestamp_field(self) -> None:
        """Each transition type sets the appropriate timestamp field."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)

        # ACTIVE -> EXPIRED sets expired_at
        result = apply_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.EXPIRED,
            current_revision=0,
            now=now,
        )
        assert result.timestamp_field == "expired_at"

        # EXPIRED -> ARCHIVED sets archived_at
        result = apply_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ARCHIVED,
            current_revision=1,
            now=now,
        )
        assert result.timestamp_field == "archived_at"

        # ARCHIVED -> DELETED sets deleted_at
        result = apply_transition(
            EnumMemoryLifecycleState.ARCHIVED,
            EnumMemoryLifecycleState.DELETED,
            current_revision=2,
            now=now,
        )
        assert result.timestamp_field == "deleted_at"

        # Promotion sets last_promoted_at
        result = apply_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ACTIVE,
            current_revision=3,
            now=now,
            is_promotion=True,
        )
        assert result.timestamp_field == "last_promoted_at"

    def test_invalid_transition_returns_error(self) -> None:
        """Invalid transitions return success=False with error message."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = apply_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.DELETED,  # Invalid: can't skip states
            current_revision=0,
            now=now,
        )
        assert not result.success
        assert result.error is not None
        assert "Invalid transition" in result.error
        # Revision should NOT change on failed transition
        assert result.new_revision == 0

    def test_apply_transition_preserves_from_and_to_states(self) -> None:
        """Transition result includes both from and to states."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = apply_transition(
            current_state=EnumMemoryLifecycleState.ACTIVE,
            target_state=EnumMemoryLifecycleState.EXPIRED,
            current_revision=0,
            now=now,
        )
        assert result.from_state == EnumMemoryLifecycleState.ACTIVE
        assert result.to_state == EnumMemoryLifecycleState.EXPIRED


class TestAdapterRuntimeTickMemory:
    """Test the tick adapter."""

    @pytest.mark.asyncio
    async def test_process_tick_returns_result(self) -> None:
        """Tick processing returns a properly structured result."""
        adapter = AdapterRuntimeTickMemory()
        tick_id = uuid4()
        correlation_id = uuid4()
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)

        result = await adapter.process_tick(
            tick_id=tick_id,
            correlation_id=correlation_id,
            now=now,
        )

        assert result.tick_id == tick_id
        assert result.correlation_id == correlation_id
        assert result.processed_at == now
        # Currently returns empty lists (placeholder implementation)
        assert result.memories_expired == 0
        assert result.memories_pending_archive == 0

    def test_handler_id_is_set(self) -> None:
        """Adapter has a proper handler_id for tracing."""
        adapter = AdapterRuntimeTickMemory()
        assert adapter.handler_id == "adapter-runtime-tick-memory"

    @pytest.mark.asyncio
    async def test_process_tick_handles_errors_gracefully(self) -> None:
        """Tick processing captures errors in result."""
        adapter = AdapterRuntimeTickMemory()
        tick_id = uuid4()
        correlation_id = uuid4()
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)

        # Current placeholder implementation doesn't raise errors,
        # but verify error list is initialized empty
        result = await adapter.process_tick(
            tick_id=tick_id,
            correlation_id=correlation_id,
            now=now,
        )
        assert isinstance(result.errors, list)
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_process_tick_with_different_timestamps(self) -> None:
        """Tick processing uses the passed timestamp, not system time."""
        adapter = AdapterRuntimeTickMemory()

        # Use a specific historical timestamp
        historical_now = datetime(2020, 6, 15, 8, 30, 0, tzinfo=timezone.utc)

        result = await adapter.process_tick(
            tick_id=uuid4(),
            correlation_id=uuid4(),
            now=historical_now,
        )

        # Result should use the passed timestamp exactly
        assert result.processed_at == historical_now
        assert result.processed_at.year == 2020


class TestAdapterPostgresDeactivateMemory:
    """Test the deactivate/expire adapter."""

    @pytest.mark.asyncio
    async def test_expire_memories_processes_request(self) -> None:
        """Expire request is processed and returns result."""
        adapter = AdapterPostgresDeactivateMemory()
        memory_ids = [uuid4(), uuid4()]
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        correlation_id = uuid4()

        request = ModelMemoryExpireRequest(
            memory_ids=memory_ids,
            now=now,
            correlation_id=correlation_id,
            reason="ttl_exceeded",
        )

        result = await adapter.expire_memories(request)

        assert result.correlation_id == correlation_id
        assert result.requested_count == 2
        # Placeholder returns success for all (assumes ACTIVE state)
        assert result.expired_count == 2
        assert len(result.transitions) == 2

    def test_handler_id_is_set(self) -> None:
        """Adapter has a proper handler_id for tracing."""
        adapter = AdapterPostgresDeactivateMemory()
        assert adapter.handler_id == "adapter-postgres-deactivate-memory"

    @pytest.mark.asyncio
    async def test_expire_single_memory(self) -> None:
        """Expiring a single memory returns correct result."""
        adapter = AdapterPostgresDeactivateMemory()
        memory_id = uuid4()
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        correlation_id = uuid4()

        request = ModelMemoryExpireRequest(
            memory_ids=[memory_id],
            now=now,
            correlation_id=correlation_id,
            reason="manual_expiration",
        )

        result = await adapter.expire_memories(request)

        assert result.requested_count == 1
        assert result.expired_count == 1
        assert result.failed_count == 0
        assert memory_id in result.expired_ids

    @pytest.mark.asyncio
    async def test_expire_empty_list_returns_zero_counts(self) -> None:
        """Expiring empty list returns zero counts."""
        adapter = AdapterPostgresDeactivateMemory()
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        correlation_id = uuid4()

        request = ModelMemoryExpireRequest(
            memory_ids=[],
            now=now,
            correlation_id=correlation_id,
            reason="ttl_exceeded",
        )

        result = await adapter.expire_memories(request)

        assert result.requested_count == 0
        assert result.expired_count == 0
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_expire_transitions_contain_correct_state_changes(self) -> None:
        """Transitions in result show ACTIVE -> EXPIRED."""
        adapter = AdapterPostgresDeactivateMemory()
        memory_id = uuid4()
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        correlation_id = uuid4()

        request = ModelMemoryExpireRequest(
            memory_ids=[memory_id],
            now=now,
            correlation_id=correlation_id,
            reason="ttl_exceeded",
        )

        result = await adapter.expire_memories(request)

        assert len(result.transitions) == 1
        transition = result.transitions[0]
        assert transition.from_state == EnumMemoryLifecycleState.ACTIVE
        assert transition.to_state == EnumMemoryLifecycleState.EXPIRED
        assert transition.timestamp_field == "expired_at"


class TestHandlerFileSystemArchive:
    """Test the archive handler."""

    @pytest.mark.asyncio
    async def test_archive_creates_jsonl_gzip_file(self) -> None:
        """Archive creates properly formatted JSONL+gzip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            correlation_id = uuid4()

            records = [
                ModelArchiveRecord(
                    memory_id=uuid4(),
                    lifecycle_revision=3,
                    lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                    archived_at=now,
                    correlation_id=correlation_id,
                    content="Test memory content",
                    metadata={"key": "value"},
                ),
            ]

            request = ModelArchiveRequest(
                records=records,
                archive_root_path=tmpdir,
                now=now,
                correlation_id=correlation_id,
            )

            result = await handler.archive_memories(request)

            assert result.archived_count == 1
            assert "2026-01-22" in result.archive_path
            assert result.archive_size_bytes > 0

            # Verify file exists and is gzipped
            archive_path = Path(result.archive_path)
            assert archive_path.exists()
            assert archive_path.suffix == ".gz"

    @pytest.mark.asyncio
    async def test_archive_includes_lifecycle_revision(self) -> None:
        """Archived records include lifecycle_revision for consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            memory_id = uuid4()
            correlation_id = uuid4()

            record = ModelArchiveRecord(
                memory_id=memory_id,
                lifecycle_revision=42,  # Specific revision
                lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                archived_at=now,
                correlation_id=correlation_id,
                content="Important memory",
            )

            request = ModelArchiveRequest(
                records=[record],
                archive_root_path=tmpdir,
                now=now,
                correlation_id=correlation_id,
            )

            await handler.archive_memories(request)

            # Read back and verify revision
            read_records = await handler.read_archive(
                Path(tmpdir) / "2026-01-22" / "memory_archive.jsonl.gz"
            )
            assert len(read_records) == 1
            assert read_records[0].lifecycle_revision == 42
            assert read_records[0].memory_id == memory_id

    @pytest.mark.asyncio
    async def test_archive_appends_to_existing_file(self) -> None:
        """Multiple archives on same day append to same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            correlation_id = uuid4()

            # First archive
            request1 = ModelArchiveRequest(
                records=[
                    ModelArchiveRecord(
                        memory_id=uuid4(),
                        lifecycle_revision=1,
                        lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                        archived_at=now,
                        correlation_id=correlation_id,
                        content="First memory",
                    ),
                ],
                archive_root_path=tmpdir,
                now=now,
                correlation_id=correlation_id,
            )
            await handler.archive_memories(request1)

            # Second archive (same day)
            request2 = ModelArchiveRequest(
                records=[
                    ModelArchiveRecord(
                        memory_id=uuid4(),
                        lifecycle_revision=2,
                        lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                        archived_at=now,
                        correlation_id=correlation_id,
                        content="Second memory",
                    ),
                ],
                archive_root_path=tmpdir,
                now=now,
                correlation_id=correlation_id,
            )
            await handler.archive_memories(request2)

            # Read back - should have both records
            read_records = await handler.read_archive(
                Path(tmpdir) / "2026-01-22" / "memory_archive.jsonl.gz"
            )
            assert len(read_records) == 2

    def test_handler_id_is_set(self) -> None:
        """Handler has a proper handler_id for tracing."""
        handler = HandlerFileSystemArchive()
        assert handler.handler_id == "handler-filesystem-archive"

    @pytest.mark.asyncio
    async def test_archive_empty_records_returns_error(self) -> None:
        """Archiving empty records list returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            correlation_id = uuid4()

            request = ModelArchiveRequest(
                records=[],
                archive_root_path=tmpdir,
                now=now,
                correlation_id=correlation_id,
            )

            result = await handler.archive_memories(request)

            assert result.archived_count == 0
            assert len(result.errors) > 0
            assert "No records to archive" in result.errors[0]

    @pytest.mark.asyncio
    async def test_archive_partitions_by_date(self) -> None:
        """Archives on different dates go to different directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            correlation_id = uuid4()

            # Day 1
            day1 = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            request1 = ModelArchiveRequest(
                records=[
                    ModelArchiveRecord(
                        memory_id=uuid4(),
                        lifecycle_revision=1,
                        lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                        archived_at=day1,
                        correlation_id=correlation_id,
                        content="Day 1 memory",
                    ),
                ],
                archive_root_path=tmpdir,
                now=day1,
                correlation_id=correlation_id,
            )
            result1 = await handler.archive_memories(request1)

            # Day 2
            day2 = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
            request2 = ModelArchiveRequest(
                records=[
                    ModelArchiveRecord(
                        memory_id=uuid4(),
                        lifecycle_revision=1,
                        lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                        archived_at=day2,
                        correlation_id=correlation_id,
                        content="Day 2 memory",
                    ),
                ],
                archive_root_path=tmpdir,
                now=day2,
                correlation_id=correlation_id,
            )
            result2 = await handler.archive_memories(request2)

            assert "2026-01-22" in result1.archive_path
            assert "2026-01-23" in result2.archive_path

            # Verify separate files
            assert Path(tmpdir, "2026-01-22", "memory_archive.jsonl.gz").exists()
            assert Path(tmpdir, "2026-01-23", "memory_archive.jsonl.gz").exists()

    @pytest.mark.asyncio
    async def test_read_archive_returns_empty_for_nonexistent(self) -> None:
        """Reading nonexistent archive returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)

            records = await handler.read_archive(
                Path(tmpdir) / "nonexistent" / "archive.jsonl.gz"
            )

            assert records == []

    @pytest.mark.asyncio
    async def test_archive_preserves_metadata(self) -> None:
        """Archived records preserve all metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            created = datetime(2026, 1, 20, 8, 0, 0, tzinfo=timezone.utc)
            expires = datetime(2026, 1, 21, 8, 0, 0, tzinfo=timezone.utc)
            expired = datetime(2026, 1, 21, 10, 0, 0, tzinfo=timezone.utc)
            memory_id = uuid4()
            correlation_id = uuid4()

            record = ModelArchiveRecord(
                memory_id=memory_id,
                lifecycle_revision=5,
                lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                archived_at=now,
                correlation_id=correlation_id,
                content="Memory with full metadata",
                metadata={"source": "test", "category": "important"},
                created_at=created,
                expires_at=expires,
                expired_at=expired,
            )

            request = ModelArchiveRequest(
                records=[record],
                archive_root_path=tmpdir,
                now=now,
                correlation_id=correlation_id,
            )

            await handler.archive_memories(request)

            # Read back and verify all fields
            read_records = await handler.read_archive(
                Path(tmpdir) / "2026-01-22" / "memory_archive.jsonl.gz"
            )
            assert len(read_records) == 1
            r = read_records[0]
            assert r.memory_id == memory_id
            assert r.lifecycle_revision == 5
            assert r.content == "Memory with full metadata"
            assert r.metadata == {"source": "test", "category": "important"}
            assert r.created_at == created
            assert r.expires_at == expires
            assert r.expired_at == expired


class TestFullLifecycleFlow:
    """Test complete lifecycle flow: ACTIVE -> EXPIRED -> ARCHIVED."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_timestamps(self) -> None:
        """Memory goes through full lifecycle with proper timestamps."""
        # Use envelope timestamps throughout (never datetime.now())
        t0 = datetime(2026, 1, 22, 10, 0, 0, tzinfo=timezone.utc)  # Creation
        t1 = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)  # Expiration
        t2 = datetime(2026, 1, 22, 14, 0, 0, tzinfo=timezone.utc)  # Archive

        # Acknowledge creation timestamp for documentation
        _ = t0

        revision = 0

        # Step 1: ACTIVE -> EXPIRED
        result1 = apply_transition(
            current_state=EnumMemoryLifecycleState.ACTIVE,
            target_state=EnumMemoryLifecycleState.EXPIRED,
            current_revision=revision,
            now=t1,
        )
        assert result1.success
        assert result1.timestamp_field == "expired_at"
        revision = result1.new_revision
        assert revision == 1

        # Step 2: EXPIRED -> ARCHIVED
        result2 = apply_transition(
            current_state=EnumMemoryLifecycleState.EXPIRED,
            target_state=EnumMemoryLifecycleState.ARCHIVED,
            current_revision=revision,
            now=t2,
        )
        assert result2.success
        assert result2.timestamp_field == "archived_at"
        revision = result2.new_revision
        assert revision == 2

    @pytest.mark.asyncio
    async def test_full_lifecycle_to_deletion(self) -> None:
        """Memory goes through complete lifecycle to DELETED."""
        t1 = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 22, 14, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 22, 16, 0, 0, tzinfo=timezone.utc)

        revision = 0

        # ACTIVE -> EXPIRED
        result1 = apply_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.EXPIRED,
            revision,
            t1,
        )
        assert result1.success
        revision = result1.new_revision

        # EXPIRED -> ARCHIVED
        result2 = apply_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ARCHIVED,
            revision,
            t2,
        )
        assert result2.success
        revision = result2.new_revision

        # ARCHIVED -> DELETED
        result3 = apply_transition(
            EnumMemoryLifecycleState.ARCHIVED,
            EnumMemoryLifecycleState.DELETED,
            revision,
            t3,
        )
        assert result3.success
        assert result3.timestamp_field == "deleted_at"
        revision = result3.new_revision
        assert revision == 3

    @pytest.mark.asyncio
    async def test_promotion_resets_to_active(self) -> None:
        """Promotion from any non-terminal state returns to ACTIVE."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)

        # Can promote from EXPIRED
        result = apply_transition(
            current_state=EnumMemoryLifecycleState.EXPIRED,
            target_state=EnumMemoryLifecycleState.ACTIVE,
            current_revision=5,
            now=now,
            is_promotion=True,
        )
        assert result.success
        assert result.to_state == EnumMemoryLifecycleState.ACTIVE
        assert result.timestamp_field == "last_promoted_at"

        # Can promote from ARCHIVED (resurrection)
        result = apply_transition(
            current_state=EnumMemoryLifecycleState.ARCHIVED,
            target_state=EnumMemoryLifecycleState.ACTIVE,
            current_revision=10,
            now=now,
            is_promotion=True,
        )
        assert result.success
        assert result.to_state == EnumMemoryLifecycleState.ACTIVE

    @pytest.mark.asyncio
    async def test_cannot_promote_from_deleted(self) -> None:
        """Cannot promote (resurrect) from DELETED state."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)

        result = apply_transition(
            current_state=EnumMemoryLifecycleState.DELETED,
            target_state=EnumMemoryLifecycleState.ACTIVE,
            current_revision=5,
            now=now,
            is_promotion=True,
        )
        assert not result.success
        assert "Invalid transition" in result.error

    @pytest.mark.asyncio
    async def test_revision_tracks_all_transitions(self) -> None:
        """Revision number correctly tracks all state changes."""
        now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)

        # Start with revision 0
        revision = 0

        # Transition 1: ACTIVE -> EXPIRED
        result = apply_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.EXPIRED,
            revision,
            now,
        )
        assert result.new_revision == 1
        revision = result.new_revision

        # Transition 2: EXPIRED -> ACTIVE (promotion)
        result = apply_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ACTIVE,
            revision,
            now,
            is_promotion=True,
        )
        assert result.new_revision == 2
        revision = result.new_revision

        # Transition 3: ACTIVE -> EXPIRED again
        result = apply_transition(
            EnumMemoryLifecycleState.ACTIVE,
            EnumMemoryLifecycleState.EXPIRED,
            revision,
            now,
        )
        assert result.new_revision == 3
        revision = result.new_revision

        # Transition 4: EXPIRED -> ARCHIVED
        result = apply_transition(
            EnumMemoryLifecycleState.EXPIRED,
            EnumMemoryLifecycleState.ARCHIVED,
            revision,
            now,
        )
        assert result.new_revision == 4

    @pytest.mark.asyncio
    async def test_integrated_expire_and_archive_flow(self) -> None:
        """Integration test combining expire adapter and archive handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Expire memories using adapter
            expire_adapter = AdapterPostgresDeactivateMemory()
            memory_id = uuid4()
            expire_time = datetime(2026, 1, 22, 12, 0, 0, tzinfo=timezone.utc)
            correlation_id = uuid4()

            expire_request = ModelMemoryExpireRequest(
                memory_ids=[memory_id],
                now=expire_time,
                correlation_id=correlation_id,
                reason="ttl_exceeded",
            )

            expire_result = await expire_adapter.expire_memories(expire_request)
            assert expire_result.expired_count == 1

            # Step 2: Archive the expired memory
            archive_handler = HandlerFileSystemArchive(archive_root_path=tmpdir)
            archive_time = datetime(2026, 1, 22, 14, 0, 0, tzinfo=timezone.utc)

            archive_record = ModelArchiveRecord(
                memory_id=memory_id,
                lifecycle_revision=expire_result.transitions[0].new_revision,
                lifecycle_state=EnumMemoryLifecycleState.EXPIRED,
                archived_at=archive_time,
                correlation_id=correlation_id,
                content="Expired memory content",
            )

            archive_request = ModelArchiveRequest(
                records=[archive_record],
                archive_root_path=tmpdir,
                now=archive_time,
                correlation_id=correlation_id,
            )

            archive_result = await archive_handler.archive_memories(archive_request)
            assert archive_result.archived_count == 1

            # Step 3: Verify archived record
            read_records = await archive_handler.read_archive(
                archive_result.archive_path
            )
            assert len(read_records) == 1
            assert read_records[0].memory_id == memory_id
            assert read_records[0].lifecycle_revision == 1
