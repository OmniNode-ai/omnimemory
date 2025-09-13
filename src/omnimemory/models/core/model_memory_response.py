"""
Memory response model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .enum_memory_operation_status import EnumMemoryOperationStatus
from .model_memory_metadata import ModelMemoryMetadata


class ModelMemoryResponse(BaseModel):
    """Base memory response model following ONEX standards."""

    # Response identification
    request_id: UUID = Field(
        description="Identifier of the original request",
    )
    response_id: UUID = Field(
        description="Unique identifier for this response",
    )

    # Response status
    status: EnumMemoryOperationStatus = Field(
        description="Status of the memory operation",
    )
    success: bool = Field(
        description="Whether the operation was successful",
    )

    # Response data
    data: dict[str, str] = Field(
        default_factory=dict,
        description="Response data with string values for type safety",
    )
    results: dict[str, str] = Field(
        default_factory=dict,
        description="Operation results with string values for type safety",
    )

    # Error information
    error_code: str | None = Field(
        default=None,
        description="Error code if operation failed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )

    # Response metadata
    metadata: ModelMemoryMetadata = Field(
        description="Metadata for the response",
    )

    # Timing information
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the response was created",
    )
    processed_at: datetime | None = Field(
        default=None,
        description="When the request was processed",
    )

    # Provenance tracking
    provenance: list[str] = Field(
        default_factory=list,
        description="Provenance chain for traceability",
    )

    # Quality indicators
    trust_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trust score for the response",
    )