"""
Migration progress tracking model for OmniMemory ONEX architecture.

This module provides models for tracking migration progress across the system:
- Progress tracking with detailed metrics
- Status monitoring and error tracking
- Estimated completion time calculation
- Batch processing support
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

# Maximum reasonable length for an exception class name
MAX_EXCEPTION_NAME_LENGTH = 100

from ...enums import (
    EnumFileProcessingStatus,
    EnumMigrationPriority,
    EnumMigrationStatus,
    EnumPriorityLevel,
)
from .model_progress_summary import (
    ModelProgressPerformanceMetrics,
    ProgressSummaryResponse,
)
from .model_typed_collections import ModelConfiguration, ModelMetadata


class BatchProcessingMetrics(BaseModel):
    """Metrics for batch processing operations."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    batch_id: str = Field(description="Unique batch identifier")
    batch_size: int = Field(description="Number of items in batch")
    processed_count: int = Field(default=0, description="Number of items processed")
    failed_count: int = Field(default=0, description="Number of items failed")
    start_time: datetime | None = Field(default=None, description="Batch start time")
    end_time: datetime | None = Field(default=None, description="Batch end time")
    error_messages: list[str] = Field(
        default_factory=list, description="Error messages"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        """Calculate success rate for the batch."""
        if self.processed_count == 0:
            return 0.0
        return (self.processed_count - self.failed_count) / self.processed_count

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> timedelta | None:
        """Calculate batch processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class FileProcessingInfo(BaseModel):
    """Information about individual file processing."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    file_path: str = Field(description="Path to the file being processed")
    file_size: int | None = Field(default=None, description="File size in bytes")
    status: EnumFileProcessingStatus = Field(default=EnumFileProcessingStatus.PENDING)
    start_time: datetime | None = Field(
        default=None, description="Processing start time"
    )
    end_time: datetime | None = Field(default=None, description="Processing end time")
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    batch_id: str | None = Field(default=None, description="Associated batch ID")
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional file metadata"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processing_duration(self) -> timedelta | None:
        """Calculate file processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class MigrationProgressMetrics(BaseModel):
    """Comprehensive metrics for migration progress tracking."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    total_files: int = Field(description="Total number of files to process")
    processed_files: int = Field(default=0, description="Number of files processed")
    failed_files: int = Field(default=0, description="Number of files failed")
    skipped_files: int = Field(default=0, description="Number of files skipped")

    total_size_bytes: int | None = Field(
        default=None, description="Total size of all files"
    )
    processed_size_bytes: int = Field(default=0, description="Size of processed files")

    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Migration start time",
    )
    last_update_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time",
    )
    estimated_completion: datetime | None = Field(
        default=None, description="Estimated completion time"
    )

    files_per_second: float = Field(
        default=0.0, description="Processing rate in files per second"
    )
    bytes_per_second: float = Field(
        default=0.0, description="Processing rate in bytes per second"
    )

    current_batch: str | None = Field(
        default=None, description="Current batch being processed"
    )
    batch_metrics: list[BatchProcessingMetrics] = Field(
        default_factory=list, description="Batch processing metrics"
    )

    # Performance optimization: Cache expensive calculations (using PrivateAttr for underscore names)
    _cached_completion_percentage: float | None = PrivateAttr(default=None)
    _cached_success_rate: float | None = PrivateAttr(default=None)
    _cache_invalidated_at: datetime | None = PrivateAttr(default=None)
    _cache_ttl_seconds: int = PrivateAttr(default=60)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage with caching for performance."""
        # Check cache validity
        if self._is_cache_valid() and self._cached_completion_percentage is not None:
            return self._cached_completion_percentage

        # Calculate and cache
        if self.total_files == 0:
            result = 0.0
        else:
            result = (self.processed_files / self.total_files) * 100

        self._cached_completion_percentage = result
        return result

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate with caching for performance."""
        # Check cache validity
        if self._is_cache_valid() and self._cached_success_rate is not None:
            return self._cached_success_rate

        # Calculate and cache
        if self.processed_files == 0:
            result = 0.0
        else:
            successful_files = self.processed_files - self.failed_files
            result = (successful_files / self.processed_files) * 100

        self._cached_success_rate = result
        return result

    @computed_field  # type: ignore[prop-decorator]
    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed processing time."""
        return self.last_update_time - self.start_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remaining_files(self) -> int:
        """Calculate number of remaining files."""
        return self.total_files - self.processed_files

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per file in milliseconds."""
        if self.processed_files == 0:
            return 0.0
        elapsed_ms = self.elapsed_time.total_seconds() * 1000
        return elapsed_ms / self.processed_files

    def update_processing_rates(self) -> None:
        """Update processing rates based on current progress."""
        elapsed_seconds = self.elapsed_time.total_seconds()

        if elapsed_seconds > 0:
            self.files_per_second = self.processed_files / elapsed_seconds
            self.bytes_per_second = self.processed_size_bytes / elapsed_seconds

    def estimate_completion_time(self) -> datetime | None:
        """Estimate completion time based on current processing rate."""
        if self.files_per_second <= 0 or self.remaining_files <= 0:
            return None

        remaining_seconds = self.remaining_files / self.files_per_second
        self.estimated_completion = self.last_update_time + timedelta(
            seconds=remaining_seconds
        )
        return self.estimated_completion

    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid."""
        if self._cache_invalidated_at is None:
            return False

        cache_age = (
            datetime.now(timezone.utc) - self._cache_invalidated_at
        ).total_seconds()
        return cache_age < self._cache_ttl_seconds

    def invalidate_cache(self) -> None:
        """Manually invalidate the metrics cache."""
        self._cached_completion_percentage = None
        self._cached_success_rate = None
        self._cache_invalidated_at = datetime.now(timezone.utc)


class MigrationProgressTracker(BaseModel):
    """
    Comprehensive migration progress tracker for OmniMemory.

    Tracks migration progress across multiple dimensions:
    - File-level processing status
    - Batch-level metrics
    - Overall migration progress
    - Error tracking and recovery
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    migration_id: UUID = Field(
        default_factory=uuid4, description="Unique migration identifier"
    )
    name: str = Field(description="Migration name or description")
    status: EnumMigrationStatus = Field(
        default=EnumMigrationStatus.PENDING, description="Current migration status"
    )
    priority: EnumMigrationPriority = Field(
        default=EnumMigrationPriority.NORMAL, description="Migration priority"
    )

    metrics: MigrationProgressMetrics = Field(description="Progress metrics")
    files: list[FileProcessingInfo] = Field(
        default_factory=list, description="File processing information"
    )

    error_summary: dict[str, int] = Field(
        default_factory=dict, description="Error count by type"
    )
    recovery_attempts: int = Field(default=0, description="Number of recovery attempts")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )

    configuration: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Migration configuration"
    )
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional metadata"
    )

    def add_file(
        self, file_path: str, file_size: int | None = None, **metadata: Any
    ) -> FileProcessingInfo:
        """Add a file to be tracked for processing."""
        from .model_typed_collections import ModelKeyValuePair

        # Convert dict metadata to ModelMetadata
        metadata_obj = ModelMetadata()
        if metadata:
            metadata_obj.pairs = [
                ModelKeyValuePair(key=str(k), value=str(v)) for k, v in metadata.items()
            ]

        file_info = FileProcessingInfo(
            file_path=file_path, file_size=file_size, metadata=metadata_obj
        )
        self.files.append(file_info)
        self.metrics.total_files = len(self.files)

        if file_size is not None:
            if self.metrics.total_size_bytes is None:
                self.metrics.total_size_bytes = 0
            self.metrics.total_size_bytes += file_size

        self._update_timestamp()
        return file_info

    def start_file_processing(
        self, file_path: str, batch_id: str | None = None
    ) -> bool:
        """Mark a file as started processing."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.status = EnumFileProcessingStatus.PROCESSING
            file_info.start_time = datetime.now(timezone.utc)
            file_info.batch_id = batch_id
            self._update_timestamp()
            return True
        return False

    def complete_file_processing(
        self, file_path: str, success: bool = True, error_message: str | None = None
    ) -> None:
        """Mark a file as completed processing."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.end_time = datetime.now(timezone.utc)

            if success:
                file_info.status = EnumFileProcessingStatus.COMPLETED
                self.metrics.processed_files += 1
                if file_info.file_size is not None:
                    self.metrics.processed_size_bytes += file_info.file_size
            else:
                file_info.status = EnumFileProcessingStatus.FAILED
                file_info.error_message = error_message
                self.metrics.failed_files += 1

                # Track error types - extract from message pattern "ErrorType: message"
                if error_message:
                    error_type = self._extract_error_type(error_message)
                    self.error_summary[error_type] = (
                        self.error_summary.get(error_type, 0) + 1
                    )

            self._update_progress_metrics()
            self._update_timestamp()

    def skip_file_processing(self, file_path: str, reason: str) -> None:
        """Mark a file as skipped."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.status = EnumFileProcessingStatus.SKIPPED
            file_info.error_message = f"Skipped: {reason}"
            self.metrics.skipped_files += 1
            self._update_timestamp()

    def start_batch(self, batch_id: str, batch_size: int) -> BatchProcessingMetrics:
        """Start a new batch processing."""
        batch_metrics = BatchProcessingMetrics(
            batch_id=batch_id,
            batch_size=batch_size,
            start_time=datetime.now(timezone.utc),
        )
        self.metrics.batch_metrics.append(batch_metrics)
        self.metrics.current_batch = batch_id
        self._update_timestamp()
        return batch_metrics

    def complete_batch(self, batch_id: str) -> None:
        """Complete batch processing."""
        batch_metrics = self._find_batch(batch_id)
        if batch_metrics:
            batch_metrics.end_time = datetime.now(timezone.utc)
            if self.metrics.current_batch == batch_id:
                self.metrics.current_batch = None
            self._update_timestamp()

    def get_progress_summary(self) -> ProgressSummaryResponse:
        """Get a comprehensive progress summary."""
        # Convert MigrationPriority to EnumPriorityLevel (both use same string values)
        priority_level = EnumPriorityLevel(self.priority.value)
        return ProgressSummaryResponse(
            migration_id=str(self.migration_id),
            name=self.name,
            status=self.status,
            priority=priority_level,
            completion_percentage=self.metrics.completion_percentage,
            success_rate=self.metrics.success_rate,
            elapsed_time=str(self.metrics.elapsed_time),
            estimated_completion=self.metrics.estimated_completion,
            total_items=self.metrics.total_files,
            processed_items=self.metrics.processed_files,
            successful_items=self.metrics.processed_files - self.metrics.failed_files,
            failed_items=self.metrics.failed_files,
            current_batch_id=self.metrics.current_batch,
            active_workers=len(
                [b for b in self.metrics.batch_metrics if b.end_time is None]
            ),
            recent_errors=(
                [
                    f"{error_type}: {count} occurrences"
                    for error_type, count in list(self.error_summary.items())[-5:]
                ]
                if self.error_summary
                else []
            ),
            performance_metrics=ModelProgressPerformanceMetrics(
                files_per_second=self.metrics.files_per_second,
                bytes_per_second=self.metrics.bytes_per_second,
                average_processing_time=self.metrics.average_processing_time_ms,
            ),
        )

    def _find_file(self, file_path: str) -> FileProcessingInfo | None:
        """Find file info by path."""
        return next((f for f in self.files if f.file_path == file_path), None)

    def _extract_error_type(self, error_message: str) -> str:
        """Extract error type from error message string.

        Handles various error message formats:
        - Simple: "ValueError: message"
        - Module-qualified: "requests.exceptions.HTTPError: message"
        - Chained: "ValueError: IOError: message" (extracts first error type)
        - Custom exceptions without standard suffix

        Args:
            error_message: Error message string to parse

        Returns:
            Extracted error type name, or "UnknownError" if not parseable
        """
        if not error_message or ": " not in error_message:
            return "UnknownError"

        potential_type = error_message.split(": ", 1)[0].strip()

        if not potential_type:
            return "UnknownError"

        # Handle module-qualified names (e.g., "requests.exceptions.HTTPError")
        # Extract only the final class name
        if "." in potential_type:
            parts = potential_type.split(".")
            # Validate each part is non-empty and looks like a Python identifier
            # (empty parts indicate invalid paths like "module..name")
            if all(part and part.isidentifier() for part in parts):
                # Use the last part (the actual class name)
                potential_type = parts[-1]
            else:
                return "UnknownError"

        # Validate it looks like an exception class name:
        # - Starts with uppercase letter
        # - Is a valid Python identifier
        # - Not too long (avoid false matches on long sentences)
        if (
            potential_type
            and potential_type[0].isupper()
            and potential_type.isidentifier()
            and len(potential_type) <= MAX_EXCEPTION_NAME_LENGTH
        ):
            return potential_type

        return "UnknownError"

    def _find_batch(self, batch_id: str) -> BatchProcessingMetrics | None:
        """Find batch metrics by ID."""
        return next(
            (b for b in self.metrics.batch_metrics if b.batch_id == batch_id), None
        )

    def _update_progress_metrics(self) -> None:
        """Update progress metrics and estimates with cache invalidation."""
        # Invalidate cache since metrics are changing
        self.metrics.invalidate_cache()

        self.metrics.last_update_time = datetime.now(timezone.utc)
        self.metrics.update_processing_rates()
        self.metrics.estimate_completion_time()

    def _update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def retry_failed_files(self, max_retries: int = 3) -> list[FileProcessingInfo]:
        """Get list of failed files that can be retried."""
        retryable_files = []
        for file_info in self.files:
            if (
                file_info.status == EnumFileProcessingStatus.FAILED
                and file_info.retry_count < max_retries
            ):
                file_info.retry_count += 1
                file_info.status = EnumFileProcessingStatus.PENDING
                retryable_files.append(file_info)

        if retryable_files:
            self.recovery_attempts += 1
            self._update_timestamp()

        return retryable_files
