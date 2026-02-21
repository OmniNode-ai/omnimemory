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
from omnibase_core.container import ModelONEXContainer
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
def container() -> ModelONEXContainer:
    """Provide an ONEX container for dependency injection.

    Returns:
        ModelONEXContainer instance.
    """
    return ModelONEXContainer()


@pytest.fixture
async def handler(
    container: ModelONEXContainer,
    archive_base_path: Path,
) -> HandlerMemoryArchive:
    """Create an initialized handler without database pool for testing.

    Args:
        container: ONEX dependency injection container.
        archive_base_path: Base path for archive storage.

    Returns:
        Initialized HandlerMemoryArchive instance for testing.
    """
    handler = HandlerMemoryArchive(container)
    await handler.initialize(
        db_pool=None,
        archive_base_path=archive_base_path,
    )
    return handler


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

    def test_handler_creates_with_container(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test handler can be created with only container.

        Given: ModelONEXContainer
        When: Creating HandlerMemoryArchive
        Then: Handler is created successfully but not initialized
        """
        handler = HandlerMemoryArchive(container)
        assert handler is not None
        assert handler._db_pool is None
        assert handler.initialized is False

    @pytest.mark.asyncio
    async def test_handler_initializes_without_db_pool(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test handler can be initialized without database pool.

        Given: No db_pool provided to initialize
        When: Calling initialize()
        Then: Handler is initialized successfully
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )
        assert handler.initialized is True
        assert handler._db_pool is None

    @pytest.mark.asyncio
    async def test_handler_default_archive_path(
        self,
        container: ModelONEXContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test handler uses temp directory-based default path.

        Given: No archive_base_path provided and no env var set
        When: Calling initialize() without archive_base_path
        Then: Handler uses temp directory-based path
        """
        # Ensure env var is not set
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_PATH", raising=False)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(db_pool=None)
        expected_path = Path(tempfile.gettempdir()) / "omnimemory" / "archives"
        assert handler.archive_base_path == expected_path

    @pytest.mark.asyncio
    async def test_handler_archive_path_from_env_var(
        self,
        container: ModelONEXContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test handler reads archive path from environment variable.

        Given: OMNIMEMORY_ARCHIVE_PATH environment variable is set
        When: Calling initialize() without explicit path
        Then: Handler uses path from environment variable
        """
        env_path = "/custom/env/archive/path"
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_PATH", env_path)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(db_pool=None)
        assert handler.archive_base_path == Path(env_path)

    @pytest.mark.asyncio
    async def test_handler_explicit_path_overrides_env_var(
        self,
        container: ModelONEXContainer,
        monkeypatch: pytest.MonkeyPatch,
        archive_base_path: Path,
    ) -> None:
        """Test explicit path parameter overrides environment variable.

        Given: Both env var and explicit path provided
        When: Calling initialize()
        Then: Explicit path takes precedence over env var
        """
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_PATH", "/ignored/env/path")

        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )
        assert handler.archive_base_path == archive_base_path

    @pytest.mark.asyncio
    async def test_handler_custom_archive_path(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test handler uses custom archive path.

        Given: Custom archive_base_path
        When: Calling initialize()
        Then: Handler uses the custom path
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )
        assert handler.archive_base_path == archive_base_path

    @pytest.mark.asyncio
    async def test_archive_base_path_property(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test archive_base_path property returns correct path.

        Given: Handler initialized with custom archive path
        When: Accessing archive_base_path property
        Then: Returns the configured path
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )
        assert handler.archive_base_path == archive_base_path

    def test_archive_base_path_none_before_init(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test archive_base_path is None before initialization.

        Given: Handler created but not initialized
        When: Accessing archive_base_path property
        Then: Returns None
        """
        handler = HandlerMemoryArchive(container)
        assert handler.archive_base_path is None

    @pytest.mark.asyncio
    async def test_initialized_property(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test initialized property reflects initialization state.

        Given: Handler in various states
        When: Checking initialized property
        Then: Returns correct boolean value
        """
        handler = HandlerMemoryArchive(container)
        assert handler.initialized is False

        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )
        assert handler.initialized is True

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test health_check returns status information.

        Given: Initialized handler
        When: Calling health_check()
        Then: Returns typed health model with status information
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )

        health = await handler.health_check()

        assert health.initialized is True
        assert health.db_pool_available is False
        assert health.archive_base_path == str(archive_base_path)
        assert health.circuit_breaker_state is not None

    @pytest.mark.asyncio
    async def test_describe(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test describe returns handler metadata.

        Given: Initialized handler
        When: Calling describe()
        Then: Returns typed metadata model with handler information
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
        )

        metadata = await handler.describe()

        assert metadata.name == "HandlerMemoryArchive"
        assert metadata.description  # Non-empty description
        assert metadata.capabilities
        assert "archive_expired_memory" in metadata.capabilities


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
    async def test_handle_without_initialization_raises_error(
        self,
        container: ModelONEXContainer,
        archive_command: ModelArchiveMemoryCommand,
    ) -> None:
        """Test handler raises RuntimeError when not initialized.

        Given: Handler not initialized
        When: Calling handle()
        Then: RuntimeError is raised
        """
        handler = HandlerMemoryArchive(container)

        with pytest.raises(RuntimeError, match="Handler not initialized"):
            await handler.handle(archive_command)

    @pytest.mark.asyncio
    async def test_handle_without_db_pool_raises_error(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        archive_command: ModelArchiveMemoryCommand,
    ) -> None:
        """Test handler raises RuntimeError without db_pool.

        Given: Handler initialized without db_pool
        When: Calling handle()
        Then: RuntimeError is raised when trying to read memory
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize(
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


# =============================================================================
# Path Validation Security Tests
# =============================================================================


class TestPathValidationSecurity:
    """Tests for archive path validation and directory traversal prevention.

    These tests verify that custom archive paths are validated to prevent
    directory traversal attacks and arbitrary file writes outside the
    configured archive base directory.

    Security Note:
        These tests are critical for ensuring the handler rejects malicious
        paths that could be used to overwrite system files or access
        sensitive directories.
    """

    def test_validate_path_within_base_directory(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test valid path within base directory is accepted.

        Given: Path under the archive base directory
        When: Validating the path
        Then: Validation returns None (success)
        """
        valid_path = archive_base_path / "2026" / "01" / "25" / "test.jsonl.gz"
        error = handler._validate_archive_path(valid_path)

        assert error is None

    def test_validate_path_rejects_parent_traversal(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test path with .. traversal is rejected.

        Given: Path with .. components escaping base directory
        When: Validating the path
        Then: Validation returns error message
        """
        # Attempt to escape via .. traversal
        malicious_path = archive_base_path / ".." / ".." / "etc" / "passwd"
        error = handler._validate_archive_path(malicious_path)

        assert error is not None
        assert "outside allowed directory" in error

    def test_validate_path_rejects_absolute_outside_base(
        self,
        handler: HandlerMemoryArchive,
        tmp_path: Path,
    ) -> None:
        """Test absolute path outside base directory is rejected.

        Given: Absolute path not under archive base
        When: Validating the path
        Then: Validation returns error message
        """
        # Use a different tmp directory that's outside our archive base
        different_temp = tmp_path.parent / "other_location"
        different_temp.mkdir(exist_ok=True)
        malicious_path = different_temp / "malicious" / "archive.jsonl.gz"
        error = handler._validate_archive_path(malicious_path)

        assert error is not None
        assert "outside allowed directory" in error

    def test_validate_path_allows_nested_subdirectories(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test deeply nested path within base is accepted.

        Given: Path with many nested subdirectories under base
        When: Validating the path
        Then: Validation returns None (success)
        """
        nested_path = (
            archive_base_path / "a" / "b" / "c" / "d" / "e" / "f" / "archive.jsonl.gz"
        )
        error = handler._validate_archive_path(nested_path)

        assert error is None

    def test_validate_path_rejects_sibling_directory(
        self,
        handler: HandlerMemoryArchive,
        tmp_path: Path,
    ) -> None:
        """Test path in sibling directory is rejected.

        Given: Path in a sibling directory of the archive base
        When: Validating the path
        Then: Validation returns error message
        """
        # Create a sibling directory to the archive base
        sibling_dir = tmp_path / "sibling"
        sibling_dir.mkdir()

        malicious_path = sibling_dir / "stolen.jsonl.gz"
        error = handler._validate_archive_path(malicious_path)

        assert error is not None
        assert "outside allowed directory" in error

    def test_validate_path_handles_relative_paths(
        self,
        handler: HandlerMemoryArchive,
        archive_base_path: Path,
    ) -> None:
        """Test relative path is resolved and validated correctly.

        Given: Relative path components under base directory
        When: Validating the path
        Then: Path is resolved and validated correctly
        """
        # Path with . components (current directory references)
        relative_path = archive_base_path / "2026" / "." / "01" / "test.jsonl.gz"
        error = handler._validate_archive_path(relative_path)

        assert error is None

    def test_validate_path_error_on_uninitialized_handler(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test validation fails gracefully when handler not initialized.

        Given: Handler not initialized (no archive base path)
        When: Validating a path
        Then: Returns error message about missing initialization
        """
        handler = HandlerMemoryArchive(container)
        # Note: Not calling initialize()

        test_path = Path("/some/path.jsonl.gz")
        error = handler._validate_archive_path(test_path)

        assert error is not None
        assert "not initialized" in error


# =============================================================================
# Explicit Guard Tests (No Assert Statements)
# =============================================================================


class TestExplicitGuards:
    """Tests for explicit guard patterns replacing assert statements.

    These tests verify that the handler uses explicit if/raise guards
    instead of assert statements for critical path validation. Assert
    statements can be disabled with python -O flag, making them unsuitable
    for security-critical checks.
    """

    @pytest.mark.asyncio
    async def test_circuit_breaker_guard_raises_on_bug(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test circuit breaker guard raises RuntimeError if not set.

        Given: Handler with _db_circuit_breaker artificially set to None
        When: Calling handle() after initialization
        Then: RuntimeError is raised indicating a bug

        Note: This tests the guard pattern, not normal operation. In normal
        operation, initialize() always sets the circuit breaker.
        """
        handler = HandlerMemoryArchive(container)
        await handler.initialize()

        # Artificially corrupt internal state to test guard
        handler._db_circuit_breaker = None
        handler._initialized = True

        command = ModelArchiveMemoryCommand(
            memory_id=UUID("12345678-abcd-1234-abcd-567812345678"),
            expected_revision=1,
        )

        with pytest.raises(RuntimeError, match="Circuit breaker not initialized"):
            await handler.handle(command)

    def test_archive_base_path_guard_raises_on_bug(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test archive base path guard raises RuntimeError if not set.

        Given: Handler with _archive_base_path artificially set to None
        When: Calling _get_archive_path()
        Then: RuntimeError is raised indicating a bug

        Note: This tests the guard pattern, not normal operation.
        """
        handler = HandlerMemoryArchive(container)
        # Don't initialize, manually set initialized flag
        handler._initialized = True
        handler._archive_base_path = None

        memory_id = UUID("12345678-abcd-1234-abcd-567812345678")
        now = datetime(2026, 1, 25, tzinfo=timezone.utc)

        with pytest.raises(RuntimeError, match="Archive base path not initialized"):
            handler._get_archive_path(memory_id, now)


# =============================================================================
# Compression Level Configuration Tests
# =============================================================================


class TestCompressionLevelConfiguration:
    """Tests for configurable gzip compression level.

    Covers all three resolution sources:
    1. Constructor ``compression_level`` argument (highest priority)
    2. ``OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL`` environment variable
    3. Built-in default (level 6)

    Also covers range validation and the public property.

    Related ticket: OMN-1544
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_compression_level_is_six(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test handler uses compression level 6 when nothing is configured.

        Given: No compression_level argument and no env var set
        When: Calling initialize()
        Then: compression_level property returns 6
        """
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", raising=False)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(db_pool=None, archive_base_path=archive_base_path)

        assert handler.compression_level == 6

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constructor_argument_sets_level(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test explicit constructor argument is used as compression level.

        Given: compression_level=1 passed to initialize()
        When: Accessing compression_level property
        Then: Returns 1
        """
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", raising=False)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=1,
        )

        assert handler.compression_level == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constructor_argument_max_level(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test compression level 9 (maximum) is accepted.

        Given: compression_level=9 passed to initialize()
        When: Accessing compression_level property
        Then: Returns 9
        """
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", raising=False)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=9,
        )

        assert handler.compression_level == 9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_env_var_sets_compression_level(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL env var is respected.

        Given: OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL=3 set in environment
        When: Calling initialize() without explicit compression_level
        Then: compression_level property returns 3
        """
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", "3")

        handler = HandlerMemoryArchive(container)
        await handler.initialize(db_pool=None, archive_base_path=archive_base_path)

        assert handler.compression_level == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constructor_argument_overrides_env_var(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test explicit constructor argument takes precedence over env var.

        Given: OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL=9 set in environment
          and compression_level=1 passed to initialize()
        When: Accessing compression_level property
        Then: Returns 1 (constructor wins)
        """
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", "9")

        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=1,
        )

        assert handler.compression_level == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_level_zero_raises_value_error(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test compression_level=0 raises ValueError with clear message.

        Given: compression_level=0 passed to initialize()
        When: Calling initialize()
        Then: ValueError is raised mentioning valid range 1-9
        """
        handler = HandlerMemoryArchive(container)

        with pytest.raises(
            ValueError, match="compression_level must be between 1 and 9"
        ):
            await handler.initialize(
                db_pool=None,
                archive_base_path=archive_base_path,
                compression_level=0,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_level_ten_raises_value_error(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test compression_level=10 raises ValueError with clear message.

        Given: compression_level=10 passed to initialize()
        When: Calling initialize()
        Then: ValueError is raised mentioning valid range 1-9
        """
        handler = HandlerMemoryArchive(container)

        with pytest.raises(
            ValueError, match="compression_level must be between 1 and 9"
        ):
            await handler.initialize(
                db_pool=None,
                archive_base_path=archive_base_path,
                compression_level=10,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_level_negative_raises_value_error(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
    ) -> None:
        """Test negative compression_level raises ValueError.

        Given: compression_level=-1 passed to initialize()
        When: Calling initialize()
        Then: ValueError is raised
        """
        handler = HandlerMemoryArchive(container)

        with pytest.raises(
            ValueError, match="compression_level must be between 1 and 9"
        ):
            await handler.initialize(
                db_pool=None,
                archive_base_path=archive_base_path,
                compression_level=-1,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_env_var_value_raises_value_error(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test out-of-range env var raises ValueError.

        Given: OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL=0 set in environment
        When: Calling initialize() without explicit compression_level
        Then: ValueError is raised mentioning valid range 1-9
        """
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", "0")

        handler = HandlerMemoryArchive(container)

        with pytest.raises(
            ValueError, match="compression_level must be between 1 and 9"
        ):
            await handler.initialize(db_pool=None, archive_base_path=archive_base_path)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_integer_env_var_raises_value_error(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test non-integer env var raises ValueError with clear message.

        Given: OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL="fast" set in environment
        When: Calling initialize() without explicit compression_level
        Then: ValueError is raised mentioning the invalid value
        """
        monkeypatch.setenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", "fast")

        handler = HandlerMemoryArchive(container)

        with pytest.raises(
            ValueError,
            match="OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL must be an integer",
        ):
            await handler.initialize(db_pool=None, archive_base_path=archive_base_path)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compression_level_reflected_in_describe(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test describe() reports the configured compression level.

        Given: Handler initialized with compression_level=2
        When: Calling describe()
        Then: ModelMemoryArchiveMetadata.compression_level is 2
        """
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", raising=False)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=2,
        )

        metadata = await handler.describe()

        assert metadata.compression_level == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compression_level_affects_serialize_output(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        memory_id: UUID,
        fixed_now: datetime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that compression level is actually used during serialization.

        Given: Two handlers with different compression levels (1 and 9)
        When: Serializing the same record
        Then: Both produce valid, decompressible gzip data with the same content
        """
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", raising=False)

        record = ModelArchiveRecord(
            memory_id=memory_id,
            content="x" * 10000,  # Repeated content compresses well
            content_type="text/plain",
            created_at=fixed_now,
            expired_at=fixed_now,
            archived_at=fixed_now,
            lifecycle_revision=1,
        )

        handler_fast = HandlerMemoryArchive(container)
        await handler_fast.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=1,
        )

        handler_best = HandlerMemoryArchive(container)
        await handler_best.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=9,
        )

        compressed_fast = handler_fast._serialize_for_archive_sync(record)
        compressed_best = handler_best._serialize_for_archive_sync(record)

        # Both outputs must be valid gzip and decompress to the same content
        decompressed_fast = gzip.decompress(compressed_fast)
        decompressed_best = gzip.decompress(compressed_best)
        assert decompressed_fast == decompressed_best

        # Level 9 should produce equal or smaller output than level 1
        # for highly compressible data (repeated characters)
        assert len(compressed_best) <= len(compressed_fast)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compression_level_resets_on_shutdown(
        self,
        container: ModelONEXContainer,
        archive_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test compression level resets to default after shutdown.

        Given: Handler initialized with compression_level=9
        When: Calling shutdown()
        Then: _compression_level resets to built-in default (6)
        """
        monkeypatch.delenv("OMNIMEMORY_ARCHIVE_COMPRESSION_LEVEL", raising=False)

        handler = HandlerMemoryArchive(container)
        await handler.initialize(
            db_pool=None,
            archive_base_path=archive_base_path,
            compression_level=9,
        )
        assert handler.compression_level == 9

        await handler.shutdown()

        # After shutdown, compression_level property resets to default
        assert handler.compression_level == handler._DEFAULT_COMPRESSION_LEVEL
