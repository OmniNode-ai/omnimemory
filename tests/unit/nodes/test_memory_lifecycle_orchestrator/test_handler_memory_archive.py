# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerMemoryArchive.

Tests the memory archive handler that performs EXPIRED -> ARCHIVED state
transitions with filesystem archival. Tests cover archive format, path
generation, conflict detection, and state validation.

Test Categories:
    - Initialization: Handler setup and configuration
    - Archive Path Generation: Date-based directory structure
    - Archive Format: JSONL gzip compression
    - Conflict Detection: Revision mismatch handling
    - State Validation: Only EXPIRED memories can be archived
    - Atomic Write: File write mechanics (via _write_archive_atomic)

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_handler_memory_archive.py -v
"""

from __future__ import annotations

import gzip
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from omnibase_core.models.infrastructure.model_value import ModelValue
from omnibase_core.models.metadata.model_generic_metadata import ModelGenericMetadata
from pydantic import ValidationError

from omnimemory.nodes.memory_lifecycle_orchestrator.handlers import (
    HandlerMemoryArchive,
    ModelArchiveMemoryCommand,
    ModelArchiveRecord,
    ModelMemoryArchiveResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fixed_now() -> datetime:
    """Provide a fixed timestamp for deterministic testing.

    Returns:
        A fixed datetime in UTC timezone.
    """
    return datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def memory_id() -> UUID:
    """Provide a fixed memory ID for testing.

    Returns:
        A deterministic UUID for the memory.
    """
    return UUID("12345678-abcd-1234-abcd-567812345678")


@pytest.fixture
def archive_base_path(tmp_path: Path) -> Path:
    """Provide a temporary archive base path.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to temporary archive directory.
    """
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


@pytest.fixture
def handler(archive_base_path: Path) -> HandlerMemoryArchive:
    """Create handler without database pool for testing.

    Args:
        archive_base_path: Base path for archive storage.

    Returns:
        HandlerMemoryArchive instance for testing.
    """
    return HandlerMemoryArchive(
        db_pool=None,
        archive_base_path=archive_base_path,
    )


@pytest.fixture
def archive_command(memory_id: UUID) -> ModelArchiveMemoryCommand:
    """Create an archive command for testing.

    Args:
        memory_id: The memory entity ID.

    Returns:
        Configured ModelArchiveMemoryCommand instance.
    """
    return ModelArchiveMemoryCommand(
        memory_id=memory_id,
        expected_revision=5,
    )


@pytest.fixture
def sample_archive_record(
    memory_id: UUID,
    fixed_now: datetime,
) -> ModelArchiveRecord:
    """Create a sample archive record for testing.

    Args:
        memory_id: The memory entity ID.
        fixed_now: Fixed timestamp.

    Returns:
        Configured ModelArchiveRecord instance.
    """
    from datetime import timedelta

    return ModelArchiveRecord(
        memory_id=memory_id,
        content="Test memory content for archival",
        content_type="text/plain",
        created_at=fixed_now - timedelta(days=30),  # Created 30 days ago
        expired_at=fixed_now - timedelta(days=1),  # Expired 1 day ago
        archived_at=fixed_now,
        lifecycle_revision=5,
        metadata=ModelGenericMetadata(
            tags=["important", "archive"],
            custom_fields={"source": ModelValue.from_string("test")},
        ),
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHandlerMemoryArchiveInitialization:
    """Tests for HandlerMemoryArchive initialization."""

    def test_handler_creates_without_db_pool(
        self,
        archive_base_path: Path,
    ) -> None:
        """Test handler can be created without database pool.

        Given: No db_pool provided
        When: Creating HandlerMemoryArchive
        Then: Handler is created successfully
        """
        handler = HandlerMemoryArchive(
            db_pool=None,
            archive_base_path=archive_base_path,
        )
        assert handler is not None
        assert handler._db_pool is None

    def test_handler_default_archive_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handler uses temp directory-based default path.

        Given: No archive_base_path provided and no env var set
        When: Creating HandlerMemoryArchive
        Then: Handler uses temp directory-based path
        """
        # Ensure env var is not set
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_PATH", raising=False)

        handler = HandlerMemoryArchive()
        expected_path = Path(tempfile.gettempdir()) / "omnimemory" / "archives"
        assert handler.archive_base_path == expected_path

    def test_handler_archive_path_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handler reads archive path from environment variable.

        Given: OMNIMEMORY_ARCHIVE_PATH environment variable is set
        When: Creating HandlerMemoryArchive without explicit path
        Then: Handler uses path from environment variable
        """
        env_path = "/custom/env/archive/path"
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_PATH", env_path)

        handler = HandlerMemoryArchive()
        assert handler.archive_base_path == Path(env_path)

    def test_handler_explicit_path_overrides_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        archive_base_path: Path,
    ) -> None:
        """Test explicit path parameter overrides environment variable.

        Given: Both env var and explicit path provided
        When: Creating HandlerMemoryArchive
        Then: Explicit path takes precedence over env var
        """
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_PATH", "/ignored/env/path")

        handler = HandlerMemoryArchive(archive_base_path=archive_base_path)
        assert handler.archive_base_path == archive_base_path

    def test_handler_custom_archive_path(
        self,
        archive_base_path: Path,
    ) -> None:
        """Test handler uses custom archive path.

        Given: Custom archive_base_path
        When: Creating HandlerMemoryArchive
        Then: Handler uses the custom path
        """
        handler = HandlerMemoryArchive(archive_base_path=archive_base_path)
        assert handler.archive_base_path == archive_base_path

    def test_archive_base_path_property(
        self,
        archive_base_path: Path,
    ) -> None:
        """Test archive_base_path property returns correct path.

        Given: Handler with custom archive path
        When: Accessing archive_base_path property
        Then: Returns the configured path
        """
        handler = HandlerMemoryArchive(archive_base_path=archive_base_path)
        assert handler.archive_base_path == archive_base_path


# =============================================================================
# Archive Path Generation Tests
# =============================================================================


class TestArchivePathGeneration:
    """Tests for date-based archive path generation."""

    def test_get_archive_path_format(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
        archive_base_path: Path,
    ) -> None:
        """Test archive path follows date-based directory structure.

        Given: Memory ID and archive timestamp
        When: Generating archive path
        Then: Path follows {base}/{year}/{month}/{day}/{id}.jsonl.gz pattern
        """
        path = handler._get_archive_path(memory_id, fixed_now)

        expected = archive_base_path / "2026" / "01" / "25" / f"{memory_id}.jsonl.gz"
        assert path == expected

    def test_archive_path_month_padding(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        archive_base_path: Path,
    ) -> None:
        """Test month is zero-padded in path.

        Given: Archive date with single-digit month
        When: Generating archive path
        Then: Month is zero-padded (e.g., '01' not '1')
        """
        jan_date = datetime(2026, 1, 15, tzinfo=timezone.utc)
        path = handler._get_archive_path(memory_id, jan_date)

        assert "/01/" in str(path)

    def test_archive_path_day_padding(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        archive_base_path: Path,
    ) -> None:
        """Test day is zero-padded in path.

        Given: Archive date with single-digit day
        When: Generating archive path
        Then: Day is zero-padded (e.g., '05' not '5')
        """
        fifth_date = datetime(2026, 3, 5, tzinfo=timezone.utc)
        path = handler._get_archive_path(memory_id, fifth_date)

        assert "/05/" in str(path)

    def test_archive_path_file_extension(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test archive path has .jsonl.gz extension.

        Given: Memory ID
        When: Generating archive path
        Then: Path ends with .jsonl.gz
        """
        path = handler._get_archive_path(memory_id, fixed_now)

        assert path.suffix == ".gz"
        assert path.stem.endswith(".jsonl")

    def test_archive_path_different_dates(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        archive_base_path: Path,
    ) -> None:
        """Test archive paths differ for different dates.

        Given: Same memory ID with different dates
        When: Generating archive paths
        Then: Paths are different
        """
        date1 = datetime(2026, 1, 25, tzinfo=timezone.utc)
        date2 = datetime(2026, 6, 15, tzinfo=timezone.utc)

        path1 = handler._get_archive_path(memory_id, date1)
        path2 = handler._get_archive_path(memory_id, date2)

        assert path1 != path2
        assert "2026/01/25" in str(path1)
        assert "2026/06/15" in str(path2)


# =============================================================================
# Archive Format Tests
# =============================================================================


class TestArchiveFormat:
    """Tests for archive serialization and compression format."""

    def test_serialize_produces_gzip_bytes(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test serialization produces gzip-compressed bytes.

        Given: An archive record
        When: Serializing for archive
        Then: Output is valid gzip-compressed data
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test content",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        compressed = handler._serialize_for_archive_sync(record)

        # Verify it's valid gzip by decompressing
        decompressed = gzip.decompress(compressed)
        assert isinstance(decompressed, bytes)

    def test_serialize_produces_jsonl_format(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test serialization produces JSONL format (JSON + newline).

        Given: An archive record
        When: Serializing for archive
        Then: Decompressed output is valid JSON followed by newline
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test content",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        compressed = handler._serialize_for_archive_sync(record)
        decompressed = gzip.decompress(compressed).decode("utf-8")

        # JSONL format: JSON followed by newline
        assert decompressed.endswith("\n")

        # Remove trailing newline and parse JSON
        json_str = decompressed.rstrip("\n")
        parsed = json.loads(json_str)

        assert parsed["memory_id"] == str(memory_id)
        assert parsed["content"] == "Test content"
        assert parsed["content_type"] == "text/plain"

    def test_serialize_preserves_metadata(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test serialization preserves metadata in archive.

        Given: An archive record with ModelGenericMetadata
        When: Serializing and deserializing
        Then: Metadata structure is preserved correctly
        """
        metadata = ModelGenericMetadata(
            tags=["a", "b"],
            custom_fields={
                "source": ModelValue.from_string("test"),
                "priority": ModelValue.from_integer(1),
            },
        )
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Content with metadata",
            content_type="application/json",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=3,
            metadata=metadata,
        )

        compressed = handler._serialize_for_archive_sync(record)
        decompressed = gzip.decompress(compressed).decode("utf-8")
        parsed = json.loads(decompressed.rstrip("\n"))

        # Verify metadata structure is preserved after serialization
        assert parsed["metadata"] is not None
        assert parsed["metadata"]["tags"] == ["a", "b"]
        assert parsed["metadata"]["custom_fields"]["source"]["raw_value"] == "test"
        assert parsed["metadata"]["custom_fields"]["priority"]["raw_value"] == 1

    def test_serialize_includes_archive_version(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test serialization includes archive_version field.

        Given: An archive record
        When: Serializing
        Then: Output includes archive_version for future migrations
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        compressed = handler._serialize_for_archive_sync(record)
        decompressed = gzip.decompress(compressed).decode("utf-8")
        parsed = json.loads(decompressed.rstrip("\n"))

        assert "archive_version" in parsed
        assert parsed["archive_version"] == "1.0"


# =============================================================================
# Atomic Write Tests
# =============================================================================


class TestAtomicWrite:
    """Tests for atomic file write functionality."""

    @pytest.mark.asyncio
    async def test_write_creates_file(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test atomic write creates file at specified path.

        Given: Compressed data and target path
        When: Calling _write_archive_atomic
        Then: File is created at target path
        """
        target_path = archive_base_path / "2026" / "01" / "25" / "test.jsonl.gz"
        test_data = b"compressed test data"

        bytes_written = await handler._write_archive_atomic(target_path, test_data)

        assert target_path.exists()
        assert bytes_written == len(test_data)

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test atomic write creates parent directories if needed.

        Given: Target path with non-existent parent directories
        When: Calling _write_archive_atomic
        Then: Parent directories are created
        """
        deep_path = archive_base_path / "2030" / "12" / "31" / "deep.jsonl.gz"
        test_data = b"test"

        await handler._write_archive_atomic(deep_path, test_data)

        assert deep_path.parent.exists()
        assert deep_path.parent.is_dir()

    @pytest.mark.asyncio
    async def test_write_file_contents_correct(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test atomic write produces file with correct contents.

        Given: Specific data to write
        When: Writing and reading back
        Then: Contents match original data
        """
        target_path = archive_base_path / "content_test.jsonl.gz"
        test_data = b"exact content to verify"

        await handler._write_archive_atomic(target_path, test_data)

        with open(target_path, "rb") as f:
            written_content = f.read()

        assert written_content == test_data

    @pytest.mark.asyncio
    async def test_write_returns_bytes_count(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test atomic write returns correct byte count.

        Given: Data of known size
        When: Calling _write_archive_atomic
        Then: Returns correct number of bytes written
        """
        target_path = archive_base_path / "size_test.jsonl.gz"
        test_data = b"x" * 1024  # 1KB

        bytes_written = await handler._write_archive_atomic(target_path, test_data)

        assert bytes_written == 1024

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test atomic write overwrites existing file.

        Given: Existing file at target path
        When: Writing new data
        Then: File contains new data, not old
        """
        target_path = archive_base_path / "overwrite_test.jsonl.gz"

        # Write initial file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(b"old content")

        # Overwrite with atomic write
        new_data = b"new content"
        await handler._write_archive_atomic(target_path, new_data)

        with open(target_path, "rb") as f:
            content = f.read()

        assert content == new_data


# =============================================================================
# Command Model Tests
# =============================================================================


class TestArchiveCommandModel:
    """Tests for ModelArchiveMemoryCommand validation."""

    def test_command_requires_memory_id(self) -> None:
        """Test command requires memory_id field.

        Given: No memory_id provided
        When: Creating ModelArchiveMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelArchiveMemoryCommand(
                expected_revision=1,
            )  # type: ignore[call-arg]

    def test_command_requires_expected_revision(self, memory_id: UUID) -> None:
        """Test command requires expected_revision field.

        Given: No expected_revision provided
        When: Creating ModelArchiveMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelArchiveMemoryCommand(
                memory_id=memory_id,
            )  # type: ignore[call-arg]

    def test_command_rejects_negative_revision(self, memory_id: UUID) -> None:
        """Test command rejects negative expected_revision.

        Given: negative expected_revision
        When: Creating ModelArchiveMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelArchiveMemoryCommand(
                memory_id=memory_id,
                expected_revision=-1,
            )

    def test_command_archive_path_optional(self, memory_id: UUID) -> None:
        """Test archive_path is optional.

        Given: No archive_path provided
        When: Creating ModelArchiveMemoryCommand
        Then: Command is created with archive_path=None
        """
        command = ModelArchiveMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
        )
        assert command.archive_path is None

    def test_command_with_archive_path(
        self,
        memory_id: UUID,
        archive_base_path: Path,
    ) -> None:
        """Test command accepts custom archive_path.

        Given: Custom archive_path
        When: Creating ModelArchiveMemoryCommand
        Then: Command uses the custom path
        """
        custom_path = archive_base_path / "custom" / "path.jsonl.gz"
        command = ModelArchiveMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
            archive_path=custom_path,
        )
        assert command.archive_path == custom_path

    def test_command_is_frozen(self, memory_id: UUID) -> None:
        """Test command model is immutable.

        Given: A ModelArchiveMemoryCommand instance
        When: Attempting to modify a field
        Then: Error is raised
        """
        command = ModelArchiveMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
        )

        with pytest.raises(ValidationError):
            command.expected_revision = 5  # type: ignore[misc]


# =============================================================================
# Result Model Tests
# =============================================================================


class TestArchiveResultModel:
    """Tests for ModelMemoryArchiveResult validation."""

    def test_result_success_state(
        self,
        memory_id: UUID,
        fixed_now: datetime,
        archive_base_path: Path,
    ) -> None:
        """Test result model for successful archive.

        Given: Successful archive data
        When: Creating ModelMemoryArchiveResult
        Then: Model represents success correctly
        """
        archive_path = archive_base_path / "test.jsonl.gz"
        result = ModelMemoryArchiveResult(
            memory_id=memory_id,
            success=True,
            archived_at=fixed_now,
            archive_path=archive_path,
            bytes_written=1024,
            conflict=False,
        )

        assert result.success is True
        assert result.archived_at == fixed_now
        assert result.archive_path == archive_path
        assert result.bytes_written == 1024
        assert result.conflict is False
        assert result.error_message is None

    def test_result_conflict_state(self, memory_id: UUID) -> None:
        """Test result model for conflict scenario.

        Given: Conflict archive data
        When: Creating ModelMemoryArchiveResult
        Then: Model represents conflict correctly
        """
        result = ModelMemoryArchiveResult(
            memory_id=memory_id,
            success=False,
            conflict=True,
            error_message="Revision conflict: expected 5, memory was modified",
        )

        assert result.success is False
        assert result.conflict is True
        assert result.error_message is not None
        assert "Revision conflict" in result.error_message
        assert result.archived_at is None
        assert result.archive_path is None
        assert result.bytes_written == 0

    def test_result_failure_state(self, memory_id: UUID) -> None:
        """Test result model for failure (invalid state).

        Given: Invalid state failure data
        When: Creating ModelMemoryArchiveResult
        Then: Model represents failure correctly
        """
        result = ModelMemoryArchiveResult(
            memory_id=memory_id,
            success=False,
            conflict=False,
            error_message="Cannot archive memory in state active",
        )

        assert result.success is False
        assert result.conflict is False
        assert result.error_message is not None
        assert "Cannot archive" in result.error_message

    def test_result_bytes_written_default(self, memory_id: UUID) -> None:
        """Test bytes_written defaults to 0.

        Given: No bytes_written provided
        When: Creating ModelMemoryArchiveResult
        Then: bytes_written defaults to 0
        """
        result = ModelMemoryArchiveResult(
            memory_id=memory_id,
            success=False,
            error_message="Error",
        )

        assert result.bytes_written == 0

    def test_result_bytes_written_non_negative(self, memory_id: UUID) -> None:
        """Test bytes_written rejects negative values.

        Given: Negative bytes_written
        When: Creating ModelMemoryArchiveResult
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelMemoryArchiveResult(
                memory_id=memory_id,
                success=True,
                bytes_written=-100,
            )


# =============================================================================
# Archive Record Model Tests
# =============================================================================


class TestArchiveRecordModel:
    """Tests for ModelArchiveRecord validation."""

    def test_record_creation(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test archive record can be created with valid data.

        Given: Valid archive record data
        When: Creating ModelArchiveRecord
        Then: Record is created successfully
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test memory content",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=5,
        )

        assert record.memory_id == memory_id
        assert record.content == "Test memory content"
        assert record.content_type == "text/plain"
        assert record.lifecycle_revision == 5
        assert record.archive_version == "1.0"

    def test_record_default_archive_version(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test archive_version defaults to '1.0'.

        Given: No archive_version provided
        When: Creating ModelArchiveRecord
        Then: archive_version defaults to '1.0'
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        assert record.archive_version == "1.0"

    def test_record_metadata_optional(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test metadata field is optional.

        Given: No metadata provided
        When: Creating ModelArchiveRecord
        Then: metadata is None
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        assert record.metadata is None

    def test_record_with_metadata(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test record accepts ModelGenericMetadata.

        Given: ModelGenericMetadata instance
        When: Creating ModelArchiveRecord
        Then: metadata is stored correctly
        """
        metadata = ModelGenericMetadata(
            tags=["important"],
            custom_fields={
                "source": ModelValue.from_string("agent"),
                "priority": ModelValue.from_integer(1),
            },
        )
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
            metadata=metadata,
        )

        assert record.metadata == metadata
        assert record.metadata is not None
        assert record.metadata.tags == ["important"]
        assert record.metadata.custom_fields is not None
        assert record.metadata.custom_fields["source"].to_python_value() == "agent"
        assert record.metadata.custom_fields["priority"].to_python_value() == 1

    def test_record_is_frozen(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test archive record model is immutable.

        Given: A ModelArchiveRecord instance
        When: Attempting to modify a field
        Then: Error is raised
        """
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="Test",
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        with pytest.raises(ValidationError):
            record.content = "Modified"  # type: ignore[misc]

    def test_record_revision_non_negative(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test lifecycle_revision rejects negative values.

        Given: Negative lifecycle_revision
        When: Creating ModelArchiveRecord
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelArchiveRecord(
                memory_id=memory_id,
                content="Test",
                content_type="text/plain",
                created_at=fixed_now,
                expired_at=fixed_now,
                archived_at=fixed_now,
                lifecycle_revision=-1,
            )


# =============================================================================
# Handler Error Handling Tests
# =============================================================================


class TestHandlerErrorHandling:
    """Tests for handler error handling behavior."""

    @pytest.mark.asyncio
    async def test_handle_without_db_pool_raises_error(
        self,
        archive_base_path: Path,
        archive_command: ModelArchiveMemoryCommand,
    ) -> None:
        """Test handler raises RuntimeError without db_pool.

        Given: Handler without db_pool configured
        When: Calling handle()
        Then: RuntimeError is raised
        """
        handler = HandlerMemoryArchive(
            db_pool=None,
            archive_base_path=archive_base_path,
        )

        # The handler raises RuntimeError when trying to read memory
        with pytest.raises(RuntimeError, match="Database pool not configured"):
            await handler.handle(archive_command)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_archive_path_with_uuid_hyphens(
        self,
        handler: HandlerMemoryArchive,
        fixed_now: datetime,
    ) -> None:
        """Test archive path preserves UUID format with hyphens.

        Given: Standard UUID with hyphens
        When: Generating archive path
        Then: Path contains UUID with hyphens
        """
        memory_id = UUID("12345678-1234-5678-1234-567812345678")
        path = handler._get_archive_path(memory_id, fixed_now)

        assert "12345678-1234-5678-1234-567812345678" in str(path)

    def test_large_content_serialization(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test serialization handles large content.

        Given: Archive record with large content (1MB)
        When: Serializing
        Then: Compression produces smaller output
        """
        large_content = "x" * (1024 * 1024)  # 1MB of text
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content=large_content,
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        compressed = handler._serialize_for_archive_sync(record)

        # Compression should reduce size significantly
        original_size = len(large_content.encode("utf-8"))
        assert len(compressed) < original_size / 5  # At least 5x compression

    def test_unicode_content_serialization(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test serialization handles unicode content correctly.

        Given: Archive record with unicode content
        When: Serializing and deserializing
        Then: Unicode is preserved correctly
        """
        unicode_content = "Hello World! Emojis allowed in content for test purposes."
        record = ModelArchiveRecord(
            memory_id=memory_id,
            content=unicode_content,
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        compressed = handler._serialize_for_archive_sync(record)
        decompressed = gzip.decompress(compressed).decode("utf-8")
        parsed = json.loads(decompressed.rstrip("\n"))

        assert parsed["content"] == unicode_content

    @pytest.mark.asyncio
    async def test_write_empty_data(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test writing empty data creates empty file.

        Given: Empty bytes data
        When: Writing to archive
        Then: File is created with 0 bytes
        """
        target_path = archive_base_path / "empty.jsonl.gz"
        empty_data = b""

        bytes_written = await handler._write_archive_atomic(target_path, empty_data)

        assert target_path.exists()
        assert bytes_written == 0
        assert target_path.stat().st_size == 0

    def test_archive_path_year_2099(
        self,
        handler: HandlerMemoryArchive,
        memory_id: UUID,
        archive_base_path: Path,
    ) -> None:
        """Test archive path handles far future dates.

        Given: Date in year 2099
        When: Generating archive path
        Then: Path is generated correctly
        """
        future_date = datetime(2099, 12, 31, tzinfo=timezone.utc)
        path = handler._get_archive_path(memory_id, future_date)

        assert "2099/12/31" in str(path)
