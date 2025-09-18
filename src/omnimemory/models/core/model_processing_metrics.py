"""
Processing metrics model for operation timing and performance tracking.
"""

from datetime import datetime
from typing import Dict

from pydantic import BaseModel, Field, computed_field, field_validator

from ..foundation.model_typed_collections import ModelMetadata


class ModelProcessingMetrics(BaseModel):
    """Processing metrics for tracking operation timing and performance."""

    # Core timing metrics
    processing_time_ms: float = Field(
        ge=0.0,
        description="Total processing time in milliseconds (for performance monitoring)",
    )
    start_time: datetime = Field(description="When processing started")
    end_time: datetime = Field(description="When processing completed")

    # Performance breakdowns
    validation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on input validation in milliseconds",
    )
    computation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on core computation in milliseconds",
    )
    storage_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on storage operations in milliseconds",
    )
    serialization_time_ms: float = Field(
        default=0.0, ge=0.0, description="Time spent on serialization in milliseconds"
    )

    # Resource metrics
    memory_usage_bytes: int = Field(
        default=0,
        ge=0,
        description="Peak memory usage during processing in bytes (for monitoring)",
    )
    cpu_usage_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during processing (0-100%)",
    )

    # Quality metrics
    retry_count: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of retries performed (0-10 max for performance)",
    )
    cache_hit: bool = Field(
        default=False, description="Whether operation result was served from cache"
    )

    # Additional performance metadata
    performance_metadata: ModelMetadata = Field(
        default_factory=ModelMetadata,
        description="Additional performance-related metadata",
    )

    @computed_field
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

    @computed_field
    def breakdown_percentages(self) -> Dict[str, float]:
        """
        Calculate percentage breakdown of processing time.

        Returns:
            Dictionary with percentage breakdown of processing stages
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
                "other": 100.0,
            }

        # Calculate percentages
        validation_pct = (self.validation_time_ms / total_accounted) * 100
        computation_pct = (self.computation_time_ms / total_accounted) * 100
        storage_pct = (self.storage_time_ms / total_accounted) * 100
        serialization_pct = (self.serialization_time_ms / total_accounted) * 100

        # Account for any untracked time
        other_pct = max(
            0.0,
            100.0
            - (validation_pct + computation_pct + storage_pct + serialization_pct),
        )

        return {
            "validation": validation_pct,
            "computation": computation_pct,
            "storage": storage_pct,
            "serialization": serialization_pct,
            "other": other_pct,
        }

    @field_validator("performance_metadata")
    @classmethod
    def validate_performance_metadata(cls, v: ModelMetadata) -> ModelMetadata:
        """Validate performance metadata for security and size limits."""
        if hasattr(v, "metadata") and isinstance(v.metadata, dict):
            # Limit metadata size to prevent memory issues
            if len(v.metadata) > 50:
                # Keep only first 50 entries
                limited_metadata = dict(list(v.metadata.items())[:50])
                v.metadata = limited_metadata
        return v
