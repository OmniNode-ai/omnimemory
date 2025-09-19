"""
Migration progress tracker model for OmniMemory ONEX architecture.

This module provides comprehensive migration progress tracking functionality.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ...enums import (
    FileProcessingStatus,
    MigrationPriority,
    MigrationStatus,
    PriorityLevel,
)
from ...utils.error_sanitizer import ErrorSanitizer, SanitizationLevel
from .model_batch_processing_metrics import ModelBatchProcessingMetrics
from .model_configuration import ModelConfiguration
from .model_file_info import ModelFileInfo
from .model_file_processing_info import ModelFileProcessingInfo
from .model_metadata import ModelMetadata
from .model_migration_progress_metrics import ModelMigrationProgressMetrics
from .model_progress_summary import ModelProgressSummaryResponse

# Initialize error sanitizer for secure logging
_error_sanitizer = ErrorSanitizer(level=SanitizationLevel.STANDARD)


class ModelMigrationProgressTracker(BaseModel):
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

    metrics: ModelMigrationProgressMetrics = Field(description="Progress metrics")
    files: List[ModelFileProcessingInfo] = Field(
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
        self,
        file_path: Union[str, Path],
        file_size: Optional[int] = None,
        **metadata: Union[str, int, float, bool],
    ) -> ModelFileProcessingInfo:
        """Add a file to be tracked for processing with strongly typed file information."""
        from .model_key_value_pair import ModelKeyValuePair

        # Convert dict metadata to ModelMetadata
        metadata_obj = ModelMetadata()
        if metadata:
            metadata_obj.pairs = [
                ModelKeyValuePair(key=str(k), value=str(v)) for k, v in metadata.items()
            ]

        # Create strongly typed file info
        file_info_obj = ModelFileInfo(
            path=Path(file_path) if isinstance(file_path, str) else file_path,
            size_bytes=file_size,
        )

        file_processing_info = ModelFileProcessingInfo(
            file_info=file_info_obj, metadata=metadata_obj
        )
        self.files.append(file_processing_info)
        self.metrics.total_files = len(self.files)

        if file_size:
            if self.metrics.total_size_bytes is None:
                self.metrics.total_size_bytes = 0
            self.metrics.total_size_bytes += file_size

        self._update_timestamp()
        return file_processing_info

    def start_file_processing(
        self, file_path: Union[str, Path], batch_id: UUID | None = None
    ) -> bool:
        """Mark a file as started processing."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.status = FileProcessingStatus.IN_PROGRESS
            file_info.start_time = datetime.now()
            file_info.batch_id = batch_id
            self._update_timestamp()
            return True
        return False

    def complete_file_processing(
        self,
        file_path: Union[str, Path],
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark a file as completed processing."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.end_time = datetime.now()

            if success:
                file_info.status = FileProcessingStatus.COMPLETED
                self.metrics.processed_files += 1
                if file_info.file_info.size_bytes:
                    self.metrics.processed_size_bytes += file_info.file_info.size_bytes
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

    def skip_file_processing(self, file_path: Union[str, Path], reason: str) -> None:
        """Mark a file as skipped."""
        file_info = self._find_file(file_path)
        if file_info:
            file_info.status = FileProcessingStatus.SKIPPED
            file_info.error_message = f"Skipped: {reason}"
            self.metrics.skipped_files += 1
            self._update_timestamp()

    def start_batch(
        self, batch_id: UUID, batch_size: int
    ) -> ModelBatchProcessingMetrics:
        """Start a new batch processing."""
        batch_metrics = ModelBatchProcessingMetrics(
            batch_id=batch_id, batch_size=batch_size, start_time=datetime.now()
        )
        self.metrics.batch_metrics.append(batch_metrics)
        self.metrics.current_batch = batch_id
        self._update_timestamp()
        return batch_metrics

    def complete_batch(self, batch_id: UUID) -> None:
        """Complete batch processing."""
        batch_metrics = self._find_batch(batch_id)
        if batch_metrics:
            batch_metrics.end_time = datetime.now()
            if self.metrics.current_batch == batch_id:
                self.metrics.current_batch = None
            self._update_timestamp()

    def _convert_priority(self, migration_priority: MigrationPriority) -> PriorityLevel:
        """Convert MigrationPriority to PriorityLevel for progress summary."""
        mapping = {
            MigrationPriority.LOW: PriorityLevel.LOW,
            MigrationPriority.NORMAL: PriorityLevel.MEDIUM,
            MigrationPriority.MEDIUM: PriorityLevel.MEDIUM,
            MigrationPriority.HIGH: PriorityLevel.HIGH,
            MigrationPriority.CRITICAL: PriorityLevel.CRITICAL,
            MigrationPriority.IMMEDIATE: PriorityLevel.CRITICAL,
        }
        return mapping.get(migration_priority, PriorityLevel.MEDIUM)

    def get_progress_summary(self) -> ModelProgressSummaryResponse:
        """Get a comprehensive progress summary."""
        return ModelProgressSummaryResponse(
            migration_id=self.migration_id,
            name=self.name,
            status=self.status,
            priority=self._convert_priority(self.priority),
            completion_percentage=(
                (self.metrics.processed_files / self.metrics.total_files * 100.0)
                if self.metrics.total_files > 0
                else 0.0
            ),
            success_rate=(
                (
                    (self.metrics.processed_files - self.metrics.failed_files)
                    / self.metrics.processed_files
                    * 100.0
                )
                if self.metrics.processed_files > 0
                else 0.0
            ),
            elapsed_time=str(datetime.now() - self.metrics.start_time),
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
                    _error_sanitizer.sanitize_error(
                        Exception(f"{error_type}: {count} occurrences")
                    )
                    for error_type, count in list(self.error_summary.items())[-5:]
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

    def _find_file(
        self, file_path: Union[str, Path]
    ) -> Optional[ModelFileProcessingInfo]:
        """Find file info by path."""
        search_path = Path(file_path) if isinstance(file_path, str) else file_path
        return next((f for f in self.files if f.file_info.path == search_path), None)

    def _find_batch(self, batch_id: UUID) -> Optional[ModelBatchProcessingMetrics]:
        """Find batch metrics by ID."""
        return next(
            (b for b in self.metrics.batch_metrics if b.batch_id == batch_id), None
        )

    def _update_progress_metrics(self) -> None:
        """Update progress metrics and estimates with cache invalidation."""
        # Invalidate cache since metrics are changing
        self.metrics.invalidate_cache()

        self.metrics.last_update_time = datetime.now()
        self.metrics.update_processing_rates()
        self.metrics.estimate_completion_time()

    def _update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()

    def retry_failed_files(self, max_retries: int = 3) -> List[ModelFileProcessingInfo]:
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
