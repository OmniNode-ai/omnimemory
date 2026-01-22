# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for archiving memories to cold storage.

Uses HandlerFileSystem with DIRECT strategy per handler_reuse_matrix.md.
Archive format: JSONL + gzip, partitioned by date.

Example::

    import asyncio
    from datetime import datetime
    from uuid import uuid4

    from omnimemory.nodes.memory_lifecycle_orchestrator.handlers import (
        HandlerFileSystemArchive,
        ModelArchiveRecord,
        ModelArchiveRequest,
    )
    from omnimemory.enums import EnumMemoryLifecycleState

    async def example():
        handler = HandlerFileSystemArchive()
        now = datetime.now()

        record = ModelArchiveRecord(
            memory_id=uuid4(),
            lifecycle_revision=5,
            lifecycle_state=EnumMemoryLifecycleState.ARCHIVED,
            archived_at=now,
            correlation_id=uuid4(),
            content="Important memory content",
            metadata={"source": "agent_001"},
        )

        request = ModelArchiveRequest(
            records=[record],
            now=now,
            correlation_id=uuid4(),
        )

        result = await handler.archive_memories(request)
        print(f"Archived {result.archived_count} records to {result.archive_path}")

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1392.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums import (
    EnumMemoryLifecycleState,  # noqa: TC001 - needed at runtime for Pydantic
)

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerFileSystemArchive",
    "ModelArchiveRecord",
    "ModelArchiveRequest",
    "ModelArchiveResult",
]


class ModelArchiveRecord(BaseModel):
    """A single archived memory record.

    Contains the complete memory snapshot at archive time, including
    the lifecycle_revision to prevent "archive wrote wrong version" issues.

    Attributes:
        memory_id: Unique memory identifier.
        lifecycle_revision: Revision number at time of archive.
        lifecycle_state: State at time of archive (typically ARCHIVED).
        archived_at: Timestamp when archive occurred.
        correlation_id: Correlation ID for distributed tracing.
        content: Memory content string.
        metadata: Memory metadata snapshot.
        created_at: Original memory creation time.
        expires_at: Configured expiration time.
        expired_at: Actual expiration time if expired.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    memory_id: UUID = Field(description="Unique memory identifier")
    lifecycle_revision: int = Field(
        ge=0,
        description="Revision at time of archive",
    )
    lifecycle_state: EnumMemoryLifecycleState = Field(
        description="State at time of archive",
    )
    archived_at: datetime = Field(description="Archive timestamp")
    correlation_id: UUID = Field(description="Correlation ID for tracing")

    # Memory payload snapshot
    content: str = Field(description="Memory content")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Memory metadata snapshot",
    )

    # Provenance
    created_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    expired_at: datetime | None = Field(default=None)


class ModelArchiveRequest(BaseModel):
    """Request to archive memories.

    Attributes:
        records: List of records to archive.
        archive_root_path: Root directory for archives.
        now: Authoritative timestamp from envelope.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    records: list[ModelArchiveRecord] = Field(description="Records to archive")
    archive_root_path: str = Field(
        default="var/omnimemory/archive/",
        description="Root directory for archives",
    )
    now: datetime = Field(description="Authoritative timestamp from envelope")
    correlation_id: UUID = Field(description="Correlation ID for tracing")


class ModelArchiveResult(BaseModel):
    """Result of archive operation.

    Attributes:
        correlation_id: Correlation ID for tracing.
        archived_count: Number of records successfully archived.
        archive_path: Path to the archive file.
        archive_size_bytes: Size of the archive file in bytes.
        errors: List of error messages if any occurred.
    """

    model_config = ConfigDict(from_attributes=True)

    correlation_id: UUID = Field(description="Correlation ID for tracing")
    archived_count: int = Field(default=0, description="Number of records archived")
    archive_path: str = Field(description="Path to archive file")
    archive_size_bytes: int = Field(default=0, description="Size of archive file")
    errors: list[str] = Field(default_factory=list)


class HandlerFileSystemArchive:
    """Handler for archiving memories to cold storage.

    Archive format:
    - JSONL (one JSON object per line)
    - Gzip compressed
    - Partitioned by date: {archive_root}/{YYYY-MM-DD}/memory_archive.jsonl.gz

    Each record includes lifecycle_revision to prevent
    "archive wrote wrong version" problems.

    IMPORTANT: Uses passed `now` timestamp for all operations.

    Attributes:
        handler_id: Unique identifier for this handler.

    Example::

        handler = HandlerFileSystemArchive()
        result = await handler.archive_memories(request)
    """

    def __init__(self, archive_root_path: str = "var/omnimemory/archive/") -> None:
        """Initialize the archive handler.

        Args:
            archive_root_path: Root directory for archives.
        """
        self._handler_id = "handler-filesystem-archive"
        self._archive_root = Path(archive_root_path)

    @property
    def handler_id(self) -> str:
        """Return handler identifier for tracing."""
        return self._handler_id

    async def archive_memories(
        self,
        request: ModelArchiveRequest,
    ) -> ModelArchiveResult:
        """Archive memories to JSONL+gzip file.

        Creates/appends to daily archive file:
        {archive_root}/{YYYY-MM-DD}/memory_archive.jsonl.gz

        Args:
            request: Archive request with records and configuration.

        Returns:
            ModelArchiveResult with archive path and stats.
        """
        if not request.records:
            return ModelArchiveResult(
                correlation_id=request.correlation_id,
                archived_count=0,
                archive_path="",
                errors=["No records to archive"],
            )

        # Determine archive path based on date
        date_str = request.now.strftime("%Y-%m-%d")
        archive_dir = self._archive_root / date_str
        archive_path = archive_dir / "memory_archive.jsonl.gz"

        result = ModelArchiveResult(
            correlation_id=request.correlation_id,
            archived_count=0,
            archive_path=str(archive_path),
        )

        try:
            # Ensure directory exists
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Write records in append mode
            await self._write_records(archive_path, request.records)

            result.archived_count = len(request.records)
            result.archive_size_bytes = (
                archive_path.stat().st_size if archive_path.exists() else 0
            )

            logger.info(
                "Archived %d memories to %s (%d bytes)",
                result.archived_count,
                archive_path,
                result.archive_size_bytes,
            )

        except OSError as e:
            error_msg = f"Archive write error: {e!s}"
            result.errors.append(error_msg)
            logger.exception("Failed to archive memories: %s", error_msg)

        return result

    async def _write_records(
        self,
        archive_path: Path,
        records: list[ModelArchiveRecord],
    ) -> None:
        """Write records to JSONL+gzip file.

        Appends to existing file if present. Note that gzip doesn't support
        true append mode, so we read existing content and rewrite.

        Args:
            archive_path: Path to the archive file.
            records: List of records to write.
        """
        # Read existing content if file exists
        existing_lines: list[str] = []
        if archive_path.exists():
            with gzip.open(archive_path, "rt", encoding="utf-8") as f:
                existing_lines = f.readlines()

        # Append new records
        new_lines = [record.model_dump_json() + "\n" for record in records]

        # Write all content (gzip doesn't support true append)
        with gzip.open(archive_path, "wt", encoding="utf-8") as f:
            f.writelines(existing_lines)
            f.writelines(new_lines)

    async def read_archive(
        self,
        archive_path: str | Path,
    ) -> list[ModelArchiveRecord]:
        """Read archived records from a JSONL+gzip file.

        Useful for restore/inspection operations.

        Args:
            archive_path: Path to the archive file.

        Returns:
            List of archived records.
        """
        path = Path(archive_path)
        if not path.exists():
            return []

        records: list[ModelArchiveRecord] = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    records.append(ModelArchiveRecord.model_validate(data))

        logger.debug("Read %d records from archive %s", len(records), path)
        return records

    async def list_archives(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Path]:
        """List available archive files within a date range.

        Args:
            start_date: Start of date range (inclusive). None for no lower bound.
            end_date: End of date range (inclusive). None for no upper bound.

        Returns:
            List of archive file paths sorted by date.
        """
        if not self._archive_root.exists():
            return []

        archives: list[tuple[datetime, Path]] = []

        for date_dir in self._archive_root.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

            # Check date range
            if start_date and dir_date.date() < start_date.date():
                continue
            if end_date and dir_date.date() > end_date.date():
                continue

            archive_file = date_dir / "memory_archive.jsonl.gz"
            if archive_file.exists():
                archives.append((dir_date, archive_file))

        # Sort by date
        archives.sort(key=lambda x: x[0])
        return [path for _, path in archives]
