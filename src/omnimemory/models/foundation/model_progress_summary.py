# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ONEX-compliant typed models for migration progress summaries.

in progress reporting, ensuring type safety and validation.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums import EnumMigrationStatus, EnumPriorityLevel


class ModelProgressPerformanceMetrics(BaseModel):
    """Typed model for progress performance metrics."""

    model_config = ConfigDict(extra="forbid")

    files_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="File processing rate in files per second",
    )
    bytes_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Data processing rate in bytes per second",
    )
    average_processing_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Average processing time per file in milliseconds",
    )


class ProgressSummaryResponse(BaseModel):
    """Strongly typed progress summary response."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    migration_id: str = Field(description="Unique identifier for the migration")

    name: str = Field(description="Human-readable name of the migration")

    status: EnumMigrationStatus = Field(description="Current migration status")

    priority: EnumPriorityLevel = Field(description="Migration priority level")

    completion_percentage: float = Field(
        description="Percentage of completion (0.0-100.0)"
    )

    success_rate: float = Field(description="Success rate of processed items (0.0-1.0)")

    elapsed_time: str = Field(description="Time elapsed since migration started")

    estimated_completion: datetime | None = Field(
        default=None, description="Estimated completion time"
    )

    total_items: int = Field(description="Total number of items to migrate")

    processed_items: int = Field(description="Number of items processed")

    successful_items: int = Field(description="Number of successfully processed items")

    failed_items: int = Field(description="Number of failed items")

    current_batch_id: str | None = Field(
        default=None, description="Current batch being processed"
    )

    active_workers: int = Field(description="Number of active worker processes")

    recent_errors: list[str] = Field(
        default_factory=list, description="Recent error messages"
    )

    performance_metrics: ModelProgressPerformanceMetrics = Field(
        default_factory=ModelProgressPerformanceMetrics,
        description="Performance metrics (files/s, bytes/s, avg time)",
    )
