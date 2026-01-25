# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for archiving memories to cold storage.

This module provides the HandlerMemoryArchive handler that moves EXPIRED
memories to filesystem archive with atomic writes and gzip compression.

Archive Format:
    - Format: JSONL (JSON Lines) with gzip compression (.jsonl.gz)
    - Partitioning: Date-based directory structure ({base}/{year}/{month}/{day}/)
    - Naming: {memory_id}.jsonl.gz

Atomic Write Pattern:
    The handler uses atomic writes to prevent partial/corrupt archives:
    1. Write compressed data to temporary file
    2. fsync to ensure durability
    3. Atomic rename to final path

    Note: Atomic write mechanics will be provided by OMN-1524 (infra primitive).
    Until then, this handler uses a local implementation.

Optimistic Locking:
    Uses expected_revision to prevent double-archive race conditions:
    1. Read memory with current revision
    2. Archive to filesystem
    3. Update DB state only if revision unchanged
    4. Return conflict=True if revision mismatch

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration
    - OMN-1524: Atomic write primitive (pending)

Example::

    from omnimemory.nodes.memory_lifecycle_orchestrator.handlers import (
        HandlerMemoryArchive,
        ModelArchiveMemoryCommand,
    )
    from pathlib import Path
    from uuid import UUID

    # Option 1: Use OMNIMEMORY_ARCHIVE_PATH environment variable (recommended)
    # export OMNIMEMORY_ARCHIVE_PATH=/var/omnimemory/archives
    handler = HandlerMemoryArchive(db_pool=pool)

    # Option 2: Explicit path (useful for testing)
    handler = HandlerMemoryArchive(
        db_pool=pool,
        archive_base_path=Path("/custom/archive/path"),
    )

    command = ModelArchiveMemoryCommand(
        memory_id=UUID("abc12345-..."),
        expected_revision=5,
        archive_path=Path("/var/omnimemory/archives/2026/01/25/abc12345.jsonl.gz"),
    )

    result = await handler.handle(command)
    if result.success:
        print(f"Archived to {result.archive_path} ({result.bytes_written} bytes)")
    elif result.conflict:
        print("Revision conflict - memory was modified")
    else:
        print(f"Archive failed: {result.error_message}")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1453.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from omnibase_core.models.metadata.model_generic_metadata import ModelGenericMetadata
from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums import EnumLifecycleState

if TYPE_CHECKING:
    from asyncpg import Pool
    from asyncpg.exceptions import InterfaceError, InternalClientError, PostgresError
else:
    try:
        from asyncpg.exceptions import (
            InterfaceError,
            InternalClientError,
            PostgresError,
        )
    except ImportError:
        # Fallback if asyncpg not installed - use base Exception
        # This allows the module to be imported even without asyncpg
        PostgresError = Exception  # type: ignore[misc,assignment]
        InterfaceError = Exception  # type: ignore[misc,assignment]
        InternalClientError = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerMemoryArchive",
    "ModelArchiveMemoryCommand",
    "ModelArchiveRecord",
    "ModelMemoryArchiveResult",
    "ProtocolOrphanedArchiveTracker",
]


@runtime_checkable
class ProtocolOrphanedArchiveTracker(Protocol):
    """Protocol for tracking orphaned archive files.

    An orphaned archive file occurs when the archive is successfully written
    to disk but the database state update fails (e.g., due to revision conflict
    or database error). These files exist on disk but are not tracked in the
    database, requiring periodic cleanup.

    Implementations of this protocol can:
    - Log orphaned files for later cleanup
    - Store in a dedicated cleanup queue
    - Send alerts for immediate investigation
    - Track metrics on orphan frequency

    Example::

        class FileOrphanTracker:
            async def track_orphan(
                self,
                memory_id: UUID,
                archive_path: Path,
                reason: str,
            ) -> None:
                with open("/var/log/orphans.jsonl", "a") as f:
                    f.write(json.dumps({
                        "memory_id": str(memory_id),
                        "archive_path": str(archive_path),
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }) + "\\n")

    .. versionadded:: 0.1.0
        Added for orphan tracking in OMN-1453.
    """

    async def track_orphan(
        self,
        memory_id: UUID,
        archive_path: Path,
        reason: str,
    ) -> None:
        """Track an orphaned archive file for later cleanup.

        Called when an archive file is written but the database state
        cannot be updated, leaving the file orphaned.

        Args:
            memory_id: UUID of the memory that was archived.
            archive_path: Filesystem path where the orphaned archive exists.
            reason: Description of why the file was orphaned (e.g.,
                "revision_conflict_during_state_update" or
                "database_error_during_state_update").
        """
        ...


class ModelArchiveMemoryCommand(BaseModel):  # omnimemory-model-exempt: handler command
    """Command to archive a memory to cold storage.

    This command initiates the archival process for a specific memory entity.
    The expected_revision field enables optimistic locking to prevent race
    conditions during concurrent archive attempts.

    Attributes:
        memory_id: UUID of the memory entity to archive.
        expected_revision: Expected lifecycle revision for optimistic lock.
            If the actual revision differs, the archive operation fails
            with conflict=True to prevent double-archive.
        archive_path: Target filesystem path for the archive file.
            If not provided, the handler generates a path using date-based
            partitioning.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    memory_id: UUID = Field(
        ...,
        description="ID of memory to archive",
    )
    expected_revision: int = Field(
        ...,
        ge=0,
        description="Expected revision for optimistic lock",
    )
    archive_path: Path | None = Field(
        default=None,
        description="Optional target archive file path (auto-generated if not provided)",
    )


class ModelMemoryArchiveResult(BaseModel):  # omnimemory-model-exempt: handler result
    """Result of an archive operation.

    Contains detailed information about the archive attempt, including
    success status, file location, and any error details.

    Attributes:
        memory_id: UUID of the archived memory.
        success: Whether the archive operation completed successfully.
        archived_at: Timestamp when the archive was created.
        archive_path: Filesystem path where the archive was written.
        bytes_written: Number of compressed bytes written to the archive.
        conflict: True if a revision conflict prevented archival.
        orphaned: True if an archive file was written but database state
            update failed, leaving an orphaned file on disk.
        error_message: Human-readable error description if failed.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    memory_id: UUID = Field(
        ...,
        description="ID of the archived memory",
    )
    success: bool = Field(
        ...,
        description="Whether the archive operation succeeded",
    )
    archived_at: datetime | None = Field(
        default=None,
        description="Timestamp of successful archive",
    )
    archive_path: Path | None = Field(
        default=None,
        description="Path to the archive file",
    )
    bytes_written: int = Field(
        default=0,
        ge=0,
        description="Number of compressed bytes written",
    )
    conflict: bool = Field(
        default=False,
        description="True if revision conflict prevented archival",
    )
    orphaned: bool = Field(
        default=False,
        description="True if archive file was written but DB state update failed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if archive failed",
    )


class ModelArchiveRecord(BaseModel):  # omnimemory-model-exempt: archive record format
    """Record format for archived memory.

    This model defines the schema for archived memory records. Each archive
    file contains one JSONL record (one JSON object per line), compressed
    with gzip.

    The archive_version field enables future schema migrations while
    maintaining backwards compatibility with existing archives.

    Attributes:
        memory_id: UUID of the archived memory.
        content: The memory content (text, structured data, etc.).
        content_type: MIME type or content classification.
        created_at: When the memory was originally created.
        expired_at: When the memory transitioned to EXPIRED state.
        archived_at: When the memory was archived to cold storage.
        lifecycle_revision: The revision number at time of archival.
        archive_version: Schema version for archive format migrations.
        metadata: Optional additional metadata from the memory.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    memory_id: UUID = Field(
        ...,
        description="UUID of the archived memory",
    )
    content: str = Field(
        ...,
        description="The memory content",
    )
    content_type: str = Field(
        ...,
        description="MIME type or content classification",
    )
    created_at: datetime = Field(
        ...,
        description="When the memory was originally created",
    )
    expired_at: datetime = Field(
        ...,
        description="When the memory transitioned to EXPIRED state",
    )
    archived_at: datetime = Field(
        ...,
        description="When the memory was archived to cold storage",
    )
    lifecycle_revision: int = Field(
        ...,
        ge=0,
        description="The revision number at time of archival",
    )
    archive_version: str = Field(
        default="1.0",
        description="Schema version for archive format migrations",
    )
    metadata: ModelGenericMetadata | None = Field(
        default=None,
        description="Optional additional metadata from the memory",
    )


class ModelMemoryRow(BaseModel):  # omnimemory-model-exempt: handler internal
    """Internal model for memory row data from database.

    Used internally by the handler to represent memory data fetched
    from the database before archival.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    id: UUID
    content: str
    content_type: str
    created_at: datetime
    expired_at: datetime | None
    lifecycle_state: EnumLifecycleState
    lifecycle_revision: int
    metadata: ModelGenericMetadata | None = None


class HandlerMemoryArchive:
    """Handler for archiving memories to cold storage.

    This handler performs the complete archival workflow:

    1. **Read memory content** from database with optimistic lock check
    2. **Validate state** - only EXPIRED memories can be archived
    3. **Serialize** to archive format (JSONL)
    4. **Compress** with gzip (domain decision - compression format owned here)
    5. **Write atomically** (temp file + rename) - uses infra primitive
    6. **Update database** state: EXPIRED -> ARCHIVED

    Archive Directory Structure::

        {archive_base_path}/
            2026/
                01/
                    25/
                        {memory_id_1}.jsonl.gz
                        {memory_id_2}.jsonl.gz
                    26/
                        {memory_id_3}.jsonl.gz

    Thread Safety:
        This handler is stateless and safe for concurrent use. Each archive
        operation is independent and uses optimistic locking to handle races.

    Attributes:
        archive_base_path: Base directory for archive storage.

    Note:
        Atomic write mechanics will be provided by OMN-1524 (infra primitive).
        The current implementation uses a local atomic write pattern.
    """

    # Gzip compression level for archive files (valid range: 1-9).
    #
    # Level 6 is the gzip default and provides an optimal balance between
    # compression ratio and CPU time for archive storage workloads:
    #   - Levels 1-3: Faster compression but lower ratio (~20-30% savings)
    #   - Level 6: Balanced - good ratio (~60-70% savings) with moderate CPU
    #   - Levels 7-9: Higher ratio (~70-75% savings) but significantly slower
    #
    # For cold storage archives where read latency is acceptable and storage
    # cost matters, level 6 optimizes throughput while maintaining substantial
    # space savings. Higher levels provide diminishing returns for JSON/JSONL
    # content which already compresses well.
    _ARCHIVE_COMPRESSION_LEVEL: int = 6

    def __init__(
        self,
        db_pool: Pool | None = None,
        archive_base_path: Path | None = None,
        orphan_tracker: ProtocolOrphanedArchiveTracker | None = None,
    ) -> None:
        """Initialize the archive handler.

        Args:
            db_pool: PostgreSQL connection pool for database operations.
                If None, database operations will raise RuntimeError.
            archive_base_path: Base directory for archive storage.
                If None, reads from OMNIMEMORY_ARCHIVE_PATH environment variable.
                If env var not set, falls back to {tempdir}/omnimemory/archives.
                Directories are created on-demand during archive operations.
            orphan_tracker: Optional tracker for orphaned archive files.
                If provided, will be called when an archive file is written
                but the database state update fails. This enables monitoring
                and cleanup of orphaned files.
        """
        self._db_pool = db_pool
        self._orphan_tracker = orphan_tracker

        if archive_base_path is not None:
            self._archive_base_path = archive_base_path
        else:
            env_path = os.environ.get("OMNIMEMORY_ARCHIVE_PATH")
            if env_path:
                self._archive_base_path = Path(env_path)
                logger.debug(
                    "Using archive path from OMNIMEMORY_ARCHIVE_PATH: %s",
                    self._archive_base_path,
                )
            else:
                self._archive_base_path = (
                    Path(tempfile.gettempdir()) / "omnimemory" / "archives"
                )
                logger.info(
                    "OMNIMEMORY_ARCHIVE_PATH not set, using default archive path: %s",
                    self._archive_base_path,
                )

    @property
    def archive_base_path(self) -> Path:
        """Get the base path for archives."""
        return self._archive_base_path

    async def handle(
        self,
        command: ModelArchiveMemoryCommand,
    ) -> ModelMemoryArchiveResult:
        """Handle an archive command.

        Performs the complete archival workflow with optimistic locking
        to prevent race conditions.

        Args:
            command: Archive command with memory ID and expected revision.

        Returns:
            Result indicating success, conflict, or failure with details.

        Raises:
            RuntimeError: If database pool is not configured.
        """
        now = datetime.now(timezone.utc)

        # Step 1: Read memory content (with optimistic lock check)
        try:
            memory = await self._read_memory(
                command.memory_id,
                command.expected_revision,
            )
        except ValueError as e:
            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                error_message=str(e),
            )

        if memory is None:
            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                conflict=True,
                error_message=(
                    f"Revision conflict: expected {command.expected_revision}, "
                    "memory was modified or not found"
                ),
            )

        # Step 2: Validate state - only EXPIRED memories can be archived
        if memory.lifecycle_state != EnumLifecycleState.EXPIRED:
            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                error_message=(
                    f"Cannot archive memory in state {memory.lifecycle_state.value}. "
                    "Only EXPIRED memories can be archived."
                ),
            )

        # Step 3: Build archive record
        record = ModelArchiveRecord(
            memory_id=memory.id,
            content=memory.content,
            content_type=memory.content_type,
            created_at=memory.created_at,
            expired_at=memory.expired_at or now,  # Fallback if not set
            archived_at=now,
            lifecycle_revision=memory.lifecycle_revision,
            metadata=memory.metadata,
        )

        # Step 4: Serialize and compress
        compressed_bytes = self._serialize_for_archive(record)

        # Step 5: Determine archive path
        archive_path = command.archive_path or self._get_archive_path(
            command.memory_id,
            now,
        )

        # Step 6: Write atomically
        try:
            bytes_written = await self._write_archive_atomic(
                archive_path,
                compressed_bytes,
            )
        except OSError as e:
            logger.error(
                "Failed to write archive for memory %s: %s",
                command.memory_id,
                e,
            )
            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                error_message=f"Archive write failed: {e}",
            )

        # Step 7: Update database state
        try:
            updated = await self._mark_archived(
                command.memory_id,
                command.expected_revision,
                now,
                archive_path,
            )
            if not updated:
                # Revision conflict during state update
                # Note: Archive file was written but state not updated
                # This is a known edge case - the file exists but memory
                # may be re-archived. Idempotent archive format handles this.
                logger.warning(
                    "Revision conflict during state update for memory %s. "
                    "Archive file written to %s but state not updated.",
                    command.memory_id,
                    archive_path,
                )

                # Track orphaned file if tracker configured
                if self._orphan_tracker is not None:
                    await self._orphan_tracker.track_orphan(
                        memory_id=command.memory_id,
                        archive_path=archive_path,
                        reason="revision_conflict_during_state_update",
                    )

                return ModelMemoryArchiveResult(
                    memory_id=command.memory_id,
                    success=False,
                    conflict=True,
                    orphaned=True,
                    archive_path=archive_path,
                    bytes_written=bytes_written,
                    error_message=(
                        "Revision conflict during state update. "
                        "Archive file written but state not updated."
                    ),
                )
        except PostgresError as e:
            logger.error(
                "Database error updating state for memory %s: %s",
                command.memory_id,
                e,
            )

            # Track orphaned file if tracker configured
            if self._orphan_tracker is not None:
                await self._orphan_tracker.track_orphan(
                    memory_id=command.memory_id,
                    archive_path=archive_path,
                    reason="database_error_during_state_update",
                )

            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                orphaned=True,
                archive_path=archive_path,
                bytes_written=bytes_written,
                error_message=f"Database error during state update: {e}",
            )
        except (InterfaceError, InternalClientError) as e:
            # Handle asyncpg client-side errors not covered by PostgresError:
            # - InterfaceError: Pool closing, connection already acquired, etc.
            # - InternalClientError: Protocol errors, schema cache issues, etc.
            logger.error(
                "Client error updating state for memory %s: %s",
                command.memory_id,
                e,
            )

            # Track orphaned file if tracker configured
            if self._orphan_tracker is not None:
                await self._orphan_tracker.track_orphan(
                    memory_id=command.memory_id,
                    archive_path=archive_path,
                    reason="client_error_during_state_update",
                )

            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                orphaned=True,
                archive_path=archive_path,
                bytes_written=bytes_written,
                error_message=f"Client error during state update: {e}",
            )
        except TimeoutError as e:
            # Handle pool acquisition timeout (pool exhaustion)
            logger.error(
                "Pool timeout updating state for memory %s: %s",
                command.memory_id,
                e,
            )

            # Track orphaned file if tracker configured
            if self._orphan_tracker is not None:
                await self._orphan_tracker.track_orphan(
                    memory_id=command.memory_id,
                    archive_path=archive_path,
                    reason="pool_timeout_during_state_update",
                )

            return ModelMemoryArchiveResult(
                memory_id=command.memory_id,
                success=False,
                orphaned=True,
                archive_path=archive_path,
                bytes_written=bytes_written,
                error_message=f"Pool timeout during state update: {e}",
            )

        logger.info(
            "Successfully archived memory %s to %s (%d bytes)",
            command.memory_id,
            archive_path,
            bytes_written,
        )

        return ModelMemoryArchiveResult(
            memory_id=command.memory_id,
            success=True,
            archived_at=now,
            archive_path=archive_path,
            bytes_written=bytes_written,
        )

    def _get_archive_path(self, memory_id: UUID, archived_at: datetime) -> Path:
        """Generate archive path with date-based partitioning.

        Creates a hierarchical directory structure based on the archive date
        to enable efficient browsing and cleanup of old archives.

        Pattern: {base}/{year}/{month:02d}/{day:02d}/{memory_id}.jsonl.gz

        Args:
            memory_id: UUID of the memory being archived.
            archived_at: Timestamp of the archive operation.

        Returns:
            Path to the archive file.

        Example:
            >>> handler._get_archive_path(
            ...     UUID("abc12345-..."),
            ...     datetime(2026, 1, 25, 10, 30, 0),
            ... )
            Path("/var/omnimemory/archives/2026/01/25/abc12345-....jsonl.gz")
        """
        return (
            self._archive_base_path
            / str(archived_at.year)
            / f"{archived_at.month:02d}"
            / f"{archived_at.day:02d}"
            / f"{memory_id}.jsonl.gz"
        )

    def _serialize_for_archive(self, record: ModelArchiveRecord) -> bytes:
        """Serialize record to compressed JSONL bytes.

        The domain owns the format decision (gzip + JSONL).
        Infrastructure will own atomic write mechanics (OMN-1524).

        Compression is applied here because:
        1. Archive format is a domain decision
        2. Compression ratio for JSON is significant (typically 5-10x)
        3. Keeping compression in domain allows format-specific optimization

        Args:
            record: The archive record to serialize.

        Returns:
            Gzip-compressed JSONL bytes.
        """
        jsonl_line = record.model_dump_json() + "\n"
        return gzip.compress(
            jsonl_line.encode("utf-8"),
            compresslevel=self._ARCHIVE_COMPRESSION_LEVEL,
        )

    def _write_archive_sync(
        self,
        archive_path: Path,
        compressed_bytes: bytes,
    ) -> int:
        """Synchronous atomic write - runs in thread pool.

        Uses the temp file + rename pattern to ensure atomic writes:
        1. Create parent directories if needed
        2. Write to temporary file in same directory
        3. fsync to ensure durability
        4. Atomic rename to final path

        This method performs blocking I/O and should be called via
        asyncio.to_thread() from async contexts.

        Note: This will be replaced by omnibase_infra.write_atomic_bytes()
        when OMN-1524 is implemented.

        Args:
            archive_path: Target path for the archive file.
            compressed_bytes: Compressed archive data to write.

        Returns:
            Number of bytes written.

        Raises:
            OSError: If directory creation or file write fails.
        """
        # Ensure parent directory exists
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file, then atomic rename
        # Using same directory ensures rename is atomic (same filesystem)
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=f"{archive_path.stem}.",
            dir=archive_path.parent,
        )

        try:
            # Write compressed data
            os.write(fd, compressed_bytes)
            # Ensure data is flushed to disk
            os.fsync(fd)
            os.close(fd)
            fd = -1  # Mark as closed

            # Atomic rename
            Path(temp_path).rename(archive_path)

            return len(compressed_bytes)

        except Exception:
            # Cleanup temp file on failure
            if fd >= 0:
                os.close(fd)
            temp_path_obj = Path(temp_path)
            if temp_path_obj.exists():
                temp_path_obj.unlink()
            raise

    async def _write_archive_atomic(
        self,
        archive_path: Path,
        compressed_bytes: bytes,
    ) -> int:
        """Write archive file atomically using thread pool.

        Delegates to _write_archive_sync via asyncio.to_thread() to avoid
        blocking the event loop during file I/O operations.

        Args:
            archive_path: Target path for the archive file.
            compressed_bytes: Compressed archive data to write.

        Returns:
            Number of bytes written.

        Raises:
            OSError: If directory creation or file write fails.
        """
        return await asyncio.to_thread(
            self._write_archive_sync,
            archive_path,
            compressed_bytes,
        )

    async def _read_memory(
        self,
        memory_id: UUID,
        expected_revision: int,
    ) -> ModelMemoryRow | None:
        """Read memory from database with optimistic lock check.

        Fetches the memory entity and validates that its revision matches
        the expected revision. Returns None if the revision doesn't match,
        indicating a concurrent modification.

        Args:
            memory_id: UUID of the memory to read.
            expected_revision: Expected lifecycle_revision value.

        Returns:
            Memory row if found and revision matches, None otherwise.

        Raises:
            RuntimeError: If database pool is not configured.
            ValueError: If memory is not found.
        """
        if self._db_pool is None:
            raise RuntimeError(
                "Database pool not configured. "
                "Initialize handler with db_pool parameter."
            )

        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id,
                    content,
                    content_type,
                    created_at,
                    expired_at,
                    lifecycle_state,
                    lifecycle_revision,
                    metadata
                FROM memories
                WHERE id = $1
                """,
                memory_id,
            )

            if row is None:
                raise ValueError(f"Memory {memory_id} not found")

            # Check revision matches
            if row["lifecycle_revision"] != expected_revision:
                logger.debug(
                    "Revision mismatch for memory %s: expected %d, got %d",
                    memory_id,
                    expected_revision,
                    row["lifecycle_revision"],
                )
                return None

            # Convert raw metadata dict to ModelGenericMetadata if present
            raw_metadata = row["metadata"]
            metadata: ModelGenericMetadata | None = None
            if raw_metadata is not None:
                metadata = ModelGenericMetadata.model_validate(raw_metadata)

            return ModelMemoryRow(
                id=row["id"],
                content=row["content"],
                content_type=row["content_type"],
                created_at=row["created_at"],
                expired_at=row["expired_at"],
                lifecycle_state=EnumLifecycleState(row["lifecycle_state"]),
                lifecycle_revision=row["lifecycle_revision"],
                metadata=metadata,
            )

    async def _mark_archived(
        self,
        memory_id: UUID,
        expected_revision: int,
        archived_at: datetime,
        archive_path: Path,
    ) -> bool:
        """Update memory state to ARCHIVED with optimistic locking.

        Performs an atomic update that only succeeds if the current
        revision matches the expected revision. Increments the revision
        on successful update.

        Args:
            memory_id: UUID of the memory to update.
            expected_revision: Expected current revision (optimistic lock).
            archived_at: Timestamp of the archive operation.
            archive_path: Path where the archive was written.

        Returns:
            True if update succeeded, False if revision conflict.

        Raises:
            RuntimeError: If database pool is not configured.
        """
        if self._db_pool is None:
            raise RuntimeError(
                "Database pool not configured. "
                "Initialize handler with db_pool parameter."
            )

        async with self._db_pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE memories
                SET
                    lifecycle_state = $1,
                    lifecycle_revision = lifecycle_revision + 1,
                    archived_at = $2,
                    archive_path = $3,
                    updated_at = $2
                WHERE id = $4
                  AND lifecycle_revision = $5
                  AND lifecycle_state = $6
                """,
                EnumLifecycleState.ARCHIVED.value,
                archived_at,
                str(archive_path),
                memory_id,
                expected_revision,
                EnumLifecycleState.EXPIRED.value,
            )

            # Check if update affected any rows
            rows_affected = int(result.split()[-1])
            return rows_affected > 0
