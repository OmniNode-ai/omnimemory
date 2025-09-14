"""
Processing metrics model for operation timing and performance tracking.
"""

from datetime import datetime
from typing import Dict, Any

from pydantic import BaseModel, Field, computed_field


class ModelProcessingMetrics(BaseModel):
    """Processing metrics for tracking operation timing and performance."""

    # Core timing metrics
    processing_time_ms: float = Field(
        description="Total processing time in milliseconds"
    )
    start_time: datetime = Field(
        description="When processing started"
    )
    end_time: datetime = Field(
        description="When processing completed"
    )

    # Performance breakdowns
    validation_time_ms: float = Field(
        default=0.0,
        description="Time spent on input validation in milliseconds"
    )
    computation_time_ms: float = Field(
        default=0.0,
        description="Time spent on core computation in milliseconds"
    )
    storage_time_ms: float = Field(
        default=0.0,
        description="Time spent on storage operations in milliseconds"
    )
    serialization_time_ms: float = Field(
        default=0.0,
        description="Time spent on serialization in milliseconds"
    )

    # Resource metrics
    memory_usage_bytes: int = Field(
        default=0,
        description="Peak memory usage during processing in bytes"
    )
    cpu_usage_percent: float = Field(
        default=0.0,
        description="CPU usage percentage during processing"
    )

    # Quality metrics
    retry_count: int = Field(
        default=0,
        description="Number of retries performed"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether operation result was served from cache"
    )

    # Additional performance metadata
    performance_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional performance-related metadata"
    )

    @computed_field
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

    @computed_field
    @property
    def breakdown_percentages(self) -> Dict[str, float]:
        """
        Calculate percentage breakdown of processing time.
        
        Returns:
            Dictionary with percentage breakdown of processing stages
        """
        total_accounted = (
            self.validation_time_ms +
            self.computation_time_ms +
            self.storage_time_ms +
            self.serialization_time_ms
        )
        
        if total_accounted == 0:
            return {
                "validation": 0.0,
                "computation": 0.0,
                "storage": 0.0,
                "serialization": 0.0,
                "other": 100.0
            }
        
        # Calculate percentages
        validation_pct = (self.validation_time_ms / total_accounted) * 100
        computation_pct = (self.computation_time_ms / total_accounted) * 100
        storage_pct = (self.storage_time_ms / total_accounted) * 100
        serialization_pct = (self.serialization_time_ms / total_accounted) * 100
        
        # Account for any untracked time
        other_pct = max(0.0, 100.0 - (validation_pct + computation_pct + storage_pct + serialization_pct))
        
        return {
            "validation": validation_pct,
            "computation": computation_pct,
            "storage": storage_pct,
            "serialization": serialization_pct,
            "other": other_pct
        }