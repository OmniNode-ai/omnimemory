"""
Batch processing metrics model for OmniMemory ONEX architecture.

This module provides models for tracking batch processing operations.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, computed_field


class ModelBatchProcessingMetrics(BaseModel):
    """Metrics for batch processing operations."""

    batch_id: UUID = Field(description="Unique batch identifier")
    batch_size: int = Field(description="Number of items in batch")
    processed_count: int = Field(default=0, description="Number of items processed")
    failed_count: int = Field(default=0, description="Number of items failed")
    start_time: Optional[datetime] = Field(default=None, description="Batch start time")
    end_time: Optional[datetime] = Field(default=None, description="Batch end time")
    error_messages: List[str] = Field(
        default_factory=list, description="Error messages"
    )

    @computed_field
    def success_rate(self) -> float:
        """Calculate success rate for the batch."""
        if self.processed_count == 0:
            return 0.0
        return (self.processed_count - self.failed_count) / self.processed_count

    @computed_field
    def duration(self) -> Optional[timedelta]:
        """Calculate batch processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
