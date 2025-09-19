"""
File processing information model for OmniMemory ONEX architecture.

This module provides models for tracking individual file processing status.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from ...enums import FileProcessingStatus
from .model_file_info import ModelFileInfo
from .model_metadata import ModelMetadata


class ModelFileProcessingInfo(BaseModel):
    """Information about individual file processing with strongly typed file info."""

    file_info: ModelFileInfo = Field(description="Strongly typed file information")
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
    retry_count: int = Field(
        default=0, ge=0, le=10, description="Number of retry attempts (0-10)"
    )
    batch_id: UUID | None = Field(default=None, description="Associated batch ID")
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional processing metadata"
    )

    @computed_field
    def processing_duration(self) -> Optional[timedelta]:
        """Calculate file processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
