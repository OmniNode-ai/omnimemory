"""
Memory metadata model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from .enum_memory_operation_type import EnumMemoryOperationType


class ModelMemoryMetadata(BaseModel):
    """Metadata for memory operations following ONEX standards."""

    # Operation identification
    operation_type: EnumMemoryOperationType = Field(
        description="Type of memory operation being performed",
    )
    operation_version: str = Field(
        default="1.0",
        description="Version of the operation schema",
    )

    # Performance tracking
    execution_time_ms: int | None = Field(
        default=None,
        description="Execution time in milliseconds",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
    )

    # Resource utilization
    memory_usage_mb: float | None = Field(
        default=None,
        description="Memory usage in megabytes",
    )
    cpu_usage_percent: float | None = Field(
        default=None,
        description="CPU usage percentage",
    )

    # Quality metrics
    success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Success rate for this type of operation",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the operation result",
    )

    # Audit information
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the metadata was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When the metadata was last updated",
    )

    # Additional context
    notes: str | None = Field(
        default=None,
        description="Additional notes or context",
    )
    error_details: str | None = Field(
        default=None,
        description="Error details if operation failed",
    )