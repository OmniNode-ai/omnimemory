"""
Semaphore metrics model following ONEX standards.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ModelSemaphoreMetrics(BaseModel):
    """Strongly typed semaphore performance metrics."""

    name: str = Field(description="Name of the semaphore")

    max_value: int = Field(description="Maximum value of the semaphore")

    current_value: int = Field(description="Current value of the semaphore")

    waiting_count: int = Field(description="Number of tasks waiting for the semaphore")

    total_acquisitions: int = Field(
        default=0, description="Total number of semaphore acquisitions"
    )

    total_releases: int = Field(
        default=0, description="Total number of semaphore releases"
    )

    average_hold_time_ms: Optional[float] = Field(
        default=None, description="Average time semaphore is held"
    )

    max_hold_time_ms: Optional[float] = Field(
        default=None, description="Maximum time semaphore was held"
    )

    acquisition_timeouts: int = Field(
        default=0, description="Number of acquisition timeouts"
    )

    fairness_violations: int = Field(
        default=0, description="Number of fairness violations detected"
    )
