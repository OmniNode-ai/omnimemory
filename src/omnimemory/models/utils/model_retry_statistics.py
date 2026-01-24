"""
Retry statistics model for OmniMemory ONEX architecture.

This module contains the model for collecting retry operation statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelRetryStatistics",
]


class ModelRetryStatistics(BaseModel):
    """Statistics about retry operations."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    total_operations: int = Field(
        default=0, description="Total number of operations attempted"
    )
    successful_operations: int = Field(
        default=0, description="Number of successful operations"
    )
    failed_operations: int = Field(
        default=0, description="Number of permanently failed operations"
    )
    total_retries: int = Field(default=0, description="Total number of retry attempts")
    average_attempts: float = Field(
        default=0.0, description="Average number of attempts per operation"
    )
    common_exceptions: dict[str, int] = Field(
        default_factory=dict, description="Count of common exceptions encountered"
    )
