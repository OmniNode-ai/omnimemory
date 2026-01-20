"""
ONEX-compliant typed models for migration progress summaries.

This module provides strongly typed replacements for Dict[str, Any] patterns
in progress reporting, ensuring type safety and validation.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums import EnumMigrationStatus, EnumPriorityLevel


class ProgressSummaryResponse(BaseModel):
    """Strongly typed progress summary response."""

    model_config = ConfigDict(frozen=False)

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

    performance_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics for the migration (e.g., latency_ms, throughput)",
    )
