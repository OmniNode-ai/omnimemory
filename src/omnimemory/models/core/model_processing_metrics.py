# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Processing metrics model for operation timing and performance tracking.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..foundation.model_typed_collections import ModelMetadata

# Percentage calculation constants
PERCENTAGE_TOTAL = 100.0


class ModelProcessingMetrics(BaseModel):
    """Processing metrics for tracking operation timing and performance."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    # Core timing metrics
    processing_time_ms: float = Field(
        description="Total processing time in milliseconds"
    )
    start_time: datetime = Field(description="When processing started")
    end_time: datetime = Field(description="When processing completed")

    # Performance breakdowns
    validation_time_ms: float = Field(
        default=0.0, description="Time spent on input validation in milliseconds"
    )
    computation_time_ms: float = Field(
        default=0.0, description="Time spent on core computation in milliseconds"
    )
    storage_time_ms: float = Field(
        default=0.0, description="Time spent on storage operations in milliseconds"
    )
    serialization_time_ms: float = Field(
        default=0.0, description="Time spent on serialization in milliseconds"
    )

    # Resource metrics
    memory_usage_bytes: int = Field(
        default=0, description="Peak memory usage during processing in bytes"
    )
    cpu_usage_percent: float = Field(
        default=0.0, description="CPU usage percentage during processing"
    )

    # Quality metrics
    retry_count: int = Field(default=0, description="Number of retries performed")
    cache_hit: bool = Field(
        default=False, description="Whether operation result was served from cache"
    )

    # Additional performance metadata
    performance_metadata: ModelMetadata = Field(
        default_factory=ModelMetadata,
        description="Additional performance-related metadata",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def efficiency_score(self) -> float:
        """
        Calculate efficiency score based on processing metrics.

        Returns:
            Float between 0.0 and 1.0 indicating processing efficiency
        """
        # Base efficiency starts at 1.0
        efficiency = 1.0

        # Penalize retries
        if self.retry_count > 0:
            efficiency *= max(0.1, 1.0 - (self.retry_count * 0.2))

        # Reward cache hits
        if self.cache_hit:
            efficiency *= 1.1  # 10% bonus for cache hits

        # Cap at 1.0
        return min(1.0, efficiency)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def breakdown_percentages(self) -> dict[str, float]:
        """
        Calculate percentage breakdown of processing time.

        Returns:
            Dictionary with percentage breakdown of processing stages.
            Percentages are normalized to never exceed 100% total.
        """
        total_accounted = (
            self.validation_time_ms
            + self.computation_time_ms
            + self.storage_time_ms
            + self.serialization_time_ms
        )

        if total_accounted == 0:
            return {
                "validation": 0.0,
                "computation": 0.0,
                "storage": 0.0,
                "serialization": 0.0,
                "other": PERCENTAGE_TOTAL,
            }

        # Use total processing time as the denominator when available
        total = (
            self.processing_time_ms if self.processing_time_ms > 0 else total_accounted
        )

        # Calculate raw percentages
        validation_pct = (self.validation_time_ms / total) * 100
        computation_pct = (self.computation_time_ms / total) * 100
        storage_pct = (self.storage_time_ms / total) * 100
        serialization_pct = (self.serialization_time_ms / total) * 100

        known_total = validation_pct + computation_pct + storage_pct + serialization_pct

        # Handle two scenarios:
        # 1. Overlap (components exceed total): Normalize proportionally to 100%
        # 2. No overlap (components < total): "other" captures untracked time
        if known_total > PERCENTAGE_TOTAL:
            # Normalize proportionally - components overlap due to parallel execution
            # or measurement imprecision, so scale them to fit within 100%
            scale = PERCENTAGE_TOTAL / known_total
            return {
                "validation": validation_pct * scale,
                "computation": computation_pct * scale,
                "storage": storage_pct * scale,
                "serialization": serialization_pct * scale,
                "other": 0.0,  # No untracked time when overlap exists
            }
        else:
            # "other" captures untracked overhead time
            return {
                "validation": validation_pct,
                "computation": computation_pct,
                "storage": storage_pct,
                "serialization": serialization_pct,
                "other": PERCENTAGE_TOTAL - known_total,
            }
