"""
Memory metadata model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from ...enums.enum_memory_operation_type import EnumMemoryOperationType
from ..foundation.model_semver import ModelSemVer
from ..foundation.model_success_metrics import ModelSuccessRate, ModelConfidenceScore
from ..foundation.model_notes import ModelNotesCollection
from ..foundation.model_error_details import ModelErrorDetails


class ModelMemoryMetadata(BaseModel):
    """Metadata for memory operations following ONEX standards."""

    # Operation identification
    operation_type: EnumMemoryOperationType = Field(
        description="Type of memory operation being performed",
    )
    operation_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer.from_string("1.0.0"),
        description="Version of the operation schema following semantic versioning",
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
    success_rate: ModelSuccessRate | None = Field(
        default=None,
        description="Success rate metrics for this type of operation",
    )
    confidence_score: ModelConfidenceScore | None = Field(
        default=None,
        description="Confidence score metrics for the operation result",
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
    notes: ModelNotesCollection | None = Field(
        default=None,
        description="Additional notes or context as a structured collection",
    )
    last_error: ModelErrorDetails | None = Field(
        default=None,
        description="Full error details if operation failed",
    )