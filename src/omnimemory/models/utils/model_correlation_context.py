"""
Correlation context Pydantic model for OmniMemory ONEX architecture.

This module contains the ModelCorrelationContext model for correlation tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..foundation.model_typed_collections import ModelMetadata
from .model_structured_log_entry import TraceLevel

__all__ = [
    "ModelCorrelationContext",
]


class ModelCorrelationContext(BaseModel):
    """Context information for correlation tracking."""

    correlation_id: str = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    request_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    operation: str | None = Field(default=None)
    parent_correlation_id: str | None = Field(default=None)
    trace_level: TraceLevel = Field(default=TraceLevel.INFO)
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
