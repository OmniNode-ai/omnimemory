"""
ModelPIIDetectionResult Pydantic model for OmniMemory ONEX architecture.

This module contains the model for PII detection scan results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .model_pii_match import (
    ModelPIIMatch,  # noqa: TC001 - Pydantic needs runtime access
)
from .model_pii_type import PIIType  # noqa: TC001 - Pydantic needs runtime access

__all__ = [
    "ModelPIIDetectionResult",
]


class ModelPIIDetectionResult(BaseModel):
    """Result of PII detection scan."""

    model_config = ConfigDict(extra="forbid")

    has_pii: bool = Field(description="Whether any PII was detected")
    matches: list[ModelPIIMatch] = Field(
        default_factory=list, description="List of PII matches found"
    )
    sanitized_content: str = Field(description="Content with PII masked/removed")
    pii_types_detected: set[PIIType] = Field(
        default_factory=set, description="Types of PII found"
    )
    scan_duration_ms: float = Field(
        ge=0, description="Time taken for the scan in milliseconds"
    )
