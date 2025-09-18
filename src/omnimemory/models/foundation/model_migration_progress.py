"""
Migration progress tracking model for OmniMemory ONEX architecture.

This module provides models for tracking migration progress across the system:
- Progress tracking with detailed metrics
- Status monitoring and error tracking
- Estimated completion time calculation
- Batch processing support
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from omnimemory.enums import FileProcessingStatus, MigrationPriority, MigrationStatus

from ...utils.error_sanitizer import ErrorSanitizer, SanitizationLevel
from .model_progress_summary import ProgressSummaryResponse
from .model_typed_collections import ModelConfiguration, ModelMetadata

# Initialize error sanitizer for secure logging
_error_sanitizer = ErrorSanitizer(
    default_level=SanitizationLevel.STANDARD, enable_stack_trace_filter=True
)


class BatchProcessingMetrics(BaseModel):
    """Metrics for batch processing operations."""

    batch_id: str = Field(description="Unique batch identifier")
    batch_size: int = Field(description="Number of items in batch")
    processed_count: int = Field(default=0, description="Number of items processed")
    failed_count: int = Field(default=0, description="Number of items failed")
    start_time: Optional[datetime] = Field(default=None, description="Batch start time")
    end_time: Optional[datetime] = Field(default=None, description="Batch end time")
    error_messages: List[str] = Field(
        default_factory=list, description="Error messages"
    )

    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate for the batch."""
        if self.processed_count == 0:
            return 0.0
        return (self.processed_count - self.failed_count) / self.processed_count

    @computed_field
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate batch processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class FileProcessingInfo(BaseModel):
    """Information about individual file processing."""

    file_path: str = Field(description="Path to the file being processed")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    status: FileProcessingStatus = Field(default=FileProcessingStatus.PENDING)
    start_time: Optional[datetime] = Field(
        default=None, description="Processing start time"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="Processing end time"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    batch_id: Optional[str] = Field(default=None, description="Associated batch ID")
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional file metadata"
    )

    @computed_field
    @property
    def processing_duration(self) -> Optional[timedelta]:
        """Calculate file processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class MigrationProgressMetrics(BaseModel):
    """Comprehensive metrics for migration progress tracking."""

    total_files: int = Field(description="Total number of files to process")
    processed_files: int = Field(default=0, description="Number of files processed")
    failed_files: int = Field(default=0, description="Number of files failed")
    skipped_files: int = Field(default=0, description="Number of files skipped")

    total_size_bytes: Optional[int] = Field(
        default=None, description="Total size of all files"
    )
    processed_size_bytes: int = Field(default=0, description="Size of processed files")

    start_time: datetime = Field(
        default_factory=datetime.now, description="Migration start time"
    )
    last_update_time: datetime = Field(
        default_factory=datetime.now, description="Last update time"
    )
    estimated_completion: Optional[datetime] = Field(
        default=None, description="Estimated completion time"
    )

    files_per_second: float = Field(
        default=0.0, description="Processing rate in files per second"
    )
    bytes_per_second: float = Field(
        default=0.0, description="Processing rate in bytes per second"
    )

    current_batch: Optional[str] = Field(
        default=None, description="Current batch being processed"
    )
    batch_metrics: List[BatchProcessingMetrics] = Field(
        default_factory=list, description="Batch processing metrics"
    )

    # Performance optimization: Cache expensive calculations
    _cached_completion_percentage: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Cached completion percentage to avoid recalculation",
    )
    _cached_success_rate: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Cached success rate to avoid recalculation",
    )
    _cache_invalidated_at: Optional[datetime] = Field(
        default=None,
        exclude=True,
        description="Timestamp when cache was last invalidated",
    )
    _cache_ttl_seconds: int = Field(
        default=60,  # 1 minute cache TTL
        exclude=True,
        description="Cache time-to-live in seconds for metrics",
    )

    @computed_field
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

    @computed_field
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

    @computed_field
    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed processing time."""
        return self.last_update_time - self.start_time

    @computed_field
    @property
    def remaining_files(self) -> int:
        """Calculate number of remaining files."""
        return self.total_files - self.processed_files

    def update_processing_rates(self):
        """Update processing rates based on current progress."""
        elapsed_seconds = self.elapsed_time.total_seconds()

        if elapsed_seconds > 0:
            self.files_per_second = self.processed_files / elapsed_seconds
            self.bytes_per_second = self.processed_size_bytes / elapsed_seconds

    def estimate_completion_time(self) -> Optional[datetime]:
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

        cache_age = (datetime.now() - self._cache_invalidated_at).total_seconds()
        return cache_age < self._cache_ttl_seconds

    def invalidate_cache(self) -> None:
        """Manually invalidate the metrics cache."""
        self._cached_completion_percentage = None
        self._cached_success_rate = None
        self._cache_invalidated_at = datetime.now()


class MigrationProgressTracker(BaseModel):
    """
    Comprehensive migration progress tracker for OmniMemory.

    Tracks migration progress across multiple dimensions:
    - File-level processing status
    - Batch-level metrics
    - Overall migration progress
    - Error tracking and recovery
    """

    migration_id: UUID = Field(
        default_factory=uuid4, description="Unique migration identifier"
    )
    name: str = Field(description="Migration name or description")
    status: MigrationStatus = Field(
        default=MigrationStatus.PENDING, description="Current migration status"
    )
    priority: MigrationPriority = Field(
        default=MigrationPriority.NORMAL, description="Migration priority"
    )

    metrics: MigrationProgressMetrics = Field(description="Progress metrics")
    files: List[FileProcessingInfo] = Field(
        default_factory=list, description="File processing information"
    )

    error_summary: Dict[str, int] = Field(
        default_factory=dict, description="Error count by type"
    )
    recovery_attempts: int = Field(default=0, description="Number of recovery attempts")

    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    configuration: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Migration configuration"
    )
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional metadata"
    )

    def add_file(
        self, file_path: str, file_size: Optional[int] = None, **metadata
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

        if file_size:
            if self.metrics.total_size_bytes is None:
                self.metrics.total_size_bytes = 0
            self.metrics.total_size_bytes += file_size

        self._update_timestamp()
        return file_info

    def start_file_processing(
        self, file_path: str, batch_id: Optional[str] = None
    ) -> bool:
        """Mark a file as started processing."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.status = FileProcessingStatus.PROCESSING
            file_info.start_time = datetime.now()
            file_info.batch_id = batch_id
            self._update_timestamp()
            return True
        return False

    def complete_file_processing(
        self, file_path: str, success: bool = True, error_message: Optional[str] = None
    ):
        """Mark a file as completed processing."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.end_time = datetime.now()

            if success:
                file_info.status = FileProcessingStatus.COMPLETED
                self.metrics.processed_files += 1
                if file_info.file_size:
                    self.metrics.processed_size_bytes += file_info.file_size
            else:
                file_info.status = FileProcessingStatus.FAILED
                file_info.error_message = error_message
                self.metrics.failed_files += 1

                # Track error types
                if error_message:
                    error_type = type(Exception(error_message)).__name__
                    self.error_summary[error_type] = (
                        self.error_summary.get(error_type, 0) + 1
                    )

            self._update_progress_metrics()
            self._update_timestamp()

    def skip_file_processing(self, file_path: str, reason: str):
        """Mark a file as skipped."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.status = FileProcessingStatus.SKIPPED
            file_info.error_message = f"Skipped: {reason}"
            self.metrics.skipped_files += 1
            self._update_timestamp()

    def start_batch(self, batch_id: str, batch_size: int) -> BatchProcessingMetrics:
        """Start a new batch processing."""
        batch_metrics = BatchProcessingMetrics(
            batch_id=batch_id, batch_size=batch_size, start_time=datetime.now()
        )
        self.metrics.batch_metrics.append(batch_metrics)
        self.metrics.current_batch = batch_id
        self._update_timestamp()
        return batch_metrics

    def complete_batch(self, batch_id: str):
        """Complete batch processing."""
        batch_metrics = self._find_batch(batch_id)
        if batch_metrics:
            batch_metrics.end_time = datetime.now()
            if self.metrics.current_batch == batch_id:
                self.metrics.current_batch = None
            self._update_timestamp()

    def get_progress_summary(self) -> ProgressSummaryResponse:
        """Get a comprehensive progress summary."""
        return ProgressSummaryResponse(
            migration_id=str(self.migration_id),
            name=self.name,
            status=self.status,
            priority=self.priority,
            completion_percentage=self.metrics.completion_percentage,
            success_rate=self.metrics.success_rate,
            elapsed_time=str(self.metrics.elapsed_time),
            estimated_completion=self.metrics.estimated_completion,
            total_items=self.metrics.total_files,
            processed_items=self.metrics.processed_files,
            successful_items=self.metrics.processed_files - self.metrics.failed_files,
            failed_items=self.metrics.failed_files,
            current_batch_id=getattr(self.metrics, "current_batch", None),
            active_workers=len(
                [b for b in self.metrics.batch_metrics if b.end_time is None]
            ),
            recent_errors=(
                [
                    _error_sanitizer.sanitize_error_message(
                        str(e), level=SanitizationLevel.STRICT
                    )
                    for e in self.error_summary[-5:]
                ]
                if self.error_summary
                else []
            ),
            performance_metrics={
                "files_per_second": self.metrics.files_per_second,
                "bytes_per_second": self.metrics.bytes_per_second,
                "average_processing_time": getattr(
                    self.metrics, "average_processing_time_ms", 0.0
                ),
            },
        )

    def _find_file(self, file_path: str) -> Optional[FileProcessingInfo]:
        """Find file info by path."""
        return next((f for f in self.files if f.file_path == file_path), None)

    def _find_batch(self, batch_id: str) -> Optional[BatchProcessingMetrics]:
        """Find batch metrics by ID."""
        return next(
            (b for b in self.metrics.batch_metrics if b.batch_id == batch_id), None
        )

    def _update_progress_metrics(self):
        """Update progress metrics and estimates with cache invalidation."""
        # Invalidate cache since metrics are changing
        self.metrics.invalidate_cache()

        self.metrics.last_update_time = datetime.now()
        self.metrics.update_processing_rates()
        self.metrics.estimate_completion_time()

    def _update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()

    def retry_failed_files(self, max_retries: int = 3) -> List[FileProcessingInfo]:
        """Get list of failed files that can be retried."""
        retryable_files = []
        for file_info in self.files:
            if (
                file_info.status == FileProcessingStatus.FAILED
                and file_info.retry_count < max_retries
            ):
                file_info.retry_count += 1
                file_info.status = FileProcessingStatus.PENDING
                retryable_files.append(file_info)

        if retryable_files:
            self.recovery_attempts += 1
            self._update_timestamp()

        return retryable_files
