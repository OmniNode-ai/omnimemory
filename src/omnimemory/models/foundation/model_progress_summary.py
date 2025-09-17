"""
ONEX-compliant typed models for migration progress summaries.

This module provides strongly typed replacements for Dict[str, Any] patterns
in progress reporting, ensuring type safety and validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from ...enums.enum_migration_status import MigrationStatus
from ...enums.enum_priority_level import EnumPriorityLevel


class ProgressSummaryResponse(BaseModel):
    """Strongly typed progress summary response."""

    migration_id: str = Field(description="Unique identifier for the migration")

    name: str = Field(description="Human-readable name of the migration")

    status: MigrationStatus = Field(description="Current migration status")

    priority: EnumPriorityLevel = Field(description="Migration priority level")

    completion_percentage: float = Field(
        description="Percentage of completion (0.0-100.0)"
    )

    success_rate: float = Field(description="Success rate of processed items (0.0-1.0)")

    elapsed_time: str = Field(description="Time elapsed since migration started")

    estimated_completion: Optional[datetime] = Field(
        default=None, description="Estimated completion time"
    )

    total_items: int = Field(description="Total number of items to migrate")

    processed_items: int = Field(description="Number of items processed")

    successful_items: int = Field(description="Number of successfully processed items")

    failed_items: int = Field(description="Number of failed items")

    current_batch_id: Optional[str] = Field(
        default=None, description="Current batch being processed"
    )

    active_workers: int = Field(description="Number of active worker processes")

    recent_errors: List[str] = Field(
        default_factory=list, description="Recent error messages"
    )

    performance_metrics: dict = Field(
        default_factory=dict, description="Performance metrics for the migration"
    )
