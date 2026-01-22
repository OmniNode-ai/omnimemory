"""
ModelPIIDetectionResult Pydantic model for OmniMemory ONEX architecture.

This module contains the model for PII detection scan results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .model_pii_match import ModelPIIMatch
    from .model_pii_type import PIIType

__all__ = [
    "ModelPIIDetectionResult",
]


class ModelPIIDetectionResult(BaseModel):
    """Result of PII detection scan."""

    has_pii: bool = Field(description="Whether any PII was detected")
    matches: list[ModelPIIMatch] = Field(
        default_factory=list, description="List of PII matches found"
    )
    sanitized_content: str = Field(description="Content with PII masked/removed")
    pii_types_detected: set[PIIType] = Field(
        default_factory=set, description="Types of PII found"
    )
    scan_duration_ms: float = Field(
        description="Time taken for the scan in milliseconds"
    )
