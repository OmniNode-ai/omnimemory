"""
Migration progress metrics model for OmniMemory ONEX architecture.

This module provides comprehensive metrics for migration progress tracking.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from .model_batch_processing_metrics import ModelBatchProcessingMetrics


class ModelMigrationProgressMetrics(BaseModel):
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

    current_batch: UUID | None = Field(
        default=None, description="Current batch being processed"
    )
    batch_metrics: List[ModelBatchProcessingMetrics] = Field(
        default_factory=list, description="Batch processing metrics"
    )

    # Performance optimization: Cache expensive calculations
    cached_completion_percentage: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Cached completion percentage to avoid recalculation",
    )
    cached_success_rate: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Cached success rate to avoid recalculation",
    )
    cache_invalidated_at: Optional[datetime] = Field(
        default=None,
        exclude=True,
        description="Timestamp when cache was last invalidated",
    )
    cache_ttl_seconds: int = Field(
        default=60,  # 1 minute cache TTL
        exclude=True,
        description="Cache time-to-live in seconds for metrics",
    )

    @computed_field
    def completion_percentage(self) -> float:
        """Calculate completion percentage with caching for performance."""
        # Check cache validity
        if self._is_cache_valid() and self.cached_completion_percentage is not None:
            return self.cached_completion_percentage

        # Calculate and cache
        if self.total_files == 0:
            result = 0.0
        else:
            result = (self.processed_files / self.total_files) * 100

        self.cached_completion_percentage = result
        return result

    @computed_field
    def success_rate(self) -> float:
        """Calculate overall success rate with caching for performance."""
        # Check cache validity
        if self._is_cache_valid() and self.cached_success_rate is not None:
            return self.cached_success_rate

        # Calculate and cache
        if self.processed_files == 0:
            result = 0.0
        else:
            successful_files = self.processed_files - self.failed_files
            result = (successful_files / self.processed_files) * 100

        self.cached_success_rate = result
        return result

    @computed_field
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed processing time."""
        return self.last_update_time - self.start_time

    @computed_field
    def remaining_files(self) -> int:
        """Calculate number of remaining files."""
        return self.total_files - self.processed_files

    def update_processing_rates(self) -> None:
        """Update processing rates based on current progress."""
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()

        if elapsed_seconds > 0:
            self.files_per_second = self.processed_files / elapsed_seconds
            self.bytes_per_second = self.processed_size_bytes / elapsed_seconds

    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on current processing rate."""
        remaining_files_count = self.total_files - self.processed_files
        if self.files_per_second <= 0 or remaining_files_count <= 0:
            return None

        remaining_seconds = remaining_files_count / self.files_per_second
        self.estimated_completion = self.last_update_time + timedelta(
            seconds=remaining_seconds
        )
        return self.estimated_completion

    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid."""
        if self.cache_invalidated_at is None:
            return False

        cache_age = (datetime.now() - self.cache_invalidated_at).total_seconds()
        return cache_age < self.cache_ttl_seconds

    def invalidate_cache(self) -> None:
        """Manually invalidate the metrics cache."""
        self.cached_completion_percentage = None
        self.cached_success_rate = None
        self.cache_invalidated_at = datetime.now()
