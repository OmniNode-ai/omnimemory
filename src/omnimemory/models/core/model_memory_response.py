"""
Memory response model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums.enum_operation_status import EnumOperationStatus
from .model_memory_metadata import ModelMemoryMetadata
from .model_processing_metrics import ModelProcessingMetrics
from .model_operation_metadata import ModelOperationMetadata
from .model_provenance import ModelProvenanceChain
from ..foundation.model_memory_data import ModelMemoryResponseData
from ..foundation.model_error_details import ModelErrorDetails
from ..foundation.model_trust_score import ModelTrustScore


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
    status: EnumOperationStatus = Field(
        description="Status of the memory operation",
    )
    success: bool = Field(
        description="Whether the operation was successful",
    )

    # Response data
    data: ModelMemoryResponseData | None = Field(
        default=None,
        description="Structured response data following ONEX standards",
    )

    # Error information - replaced individual error fields with comprehensive error model
    error: ModelErrorDetails | None = Field(
        default=None,
        description="Comprehensive error details if operation failed",
    )

    # Processing metrics - new model for timing and performance tracking
    processing_metrics: ModelProcessingMetrics | None = Field(
        default=None,
        description="Processing timing and performance metrics",
    )

    # Operation metadata - new model for operation-specific information
    operation_metadata: ModelOperationMetadata = Field(
        description="Operation-specific metadata and context",
    )

    # Response metadata
    metadata: ModelMemoryMetadata = Field(
        description="Memory-specific metadata for the response",
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

    # Provenance tracking - using structured model instead of list[str]
    provenance: ModelProvenanceChain | None = Field(
        default=None,
        description="Comprehensive provenance chain for traceability following ONEX standards",
    )

    # Quality indicators
    trust_score: ModelTrustScore | None = Field(
        default=None,
        description="Trust score metrics for the response",
    )