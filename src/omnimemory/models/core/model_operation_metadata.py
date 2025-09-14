"""
Operation metadata model for tracking operation-specific information.
"""

from typing import Dict, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..foundation.model_typed_collections import (
    ModelConfiguration,
    ModelMetadata,
)


class ModelOperationMetadata(BaseModel):
    """Operation metadata for tracking operation-specific information."""

    # Operation identification
    operation_type: str = Field(
        description="Type of operation performed (e.g., 'memory_store', 'semantic_search')"
    )
    operation_version: str = Field(
        default="1.0.0",
        description="Version of the operation implementation"
    )

    # Request context
    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Correlation ID for tracing related operations"
    )
    session_id: Optional[UUID] = Field(
        default=None,
        description="Session ID for multi-operation sessions"
    )
    user_id: Optional[UUID] = Field(
        default=None,
        description="User identifier who initiated the operation"
    )

    # Source information
    source_component: str = Field(
        description="Component that initiated the operation"
    )
    source_version: Optional[str] = Field(
        default=None,
        description="Version of the source component"
    )

    # Configuration
    operation_config: ModelConfiguration = Field(
        default_factory=ModelConfiguration,
        description="Configuration parameters used for the operation"
    )

    # Quality and compliance
    compliance_level: str = Field(
        default="standard",
        description="ONEX compliance level (standard, strict, audit)"
    )
    quality_gates_passed: bool = Field(
        default=True,
        description="Whether all quality gates were passed"
    )

    # Environment context
    environment: str = Field(
        default="production",
        description="Environment where operation was executed"
    )
    node_id: Optional[UUID] = Field(
        default=None,
        description="ONEX node identifier that processed the operation"
    )

    # Feature flags and experiments
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags active during operation"
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="A/B test or experiment identifier"
    )

    # Additional custom metadata
    custom_metadata: ModelMetadata = Field(
        default_factory=ModelMetadata,
        description="Additional operation-specific metadata"
    )

    # Tags for categorization
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for operation categorization and filtering"
    )