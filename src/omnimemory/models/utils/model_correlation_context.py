"""
Correlation context Pydantic model for OmniMemory ONEX architecture.

This module contains the ModelCorrelationContext model for correlation tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.model_typed_collections import ModelMetadata
from .model_structured_log_entry import TraceLevel

__all__ = [
    "ModelCorrelationContext",
]


class ModelCorrelationContext(BaseModel):
    """Context information for correlation tracking."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for correlating related operations",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request identifier for HTTP/API requests",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user identifier for audit trails",
    )
    operation: str | None = Field(
        default=None,
        description="Name of the operation being performed",
    )
    parent_correlation_id: str | None = Field(
        default=None,
        description="Parent correlation ID for hierarchical tracing",
    )
    trace_level: TraceLevel = Field(
        default=TraceLevel.INFO,
        description="Trace level for logging verbosity",
    )
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata,
        description="Additional metadata for the correlation context",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the context was created",
    )
