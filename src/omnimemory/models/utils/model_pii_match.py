"""
ModelPIIMatch Pydantic model for OmniMemory ONEX architecture.

This module contains the model for representing a detected PII match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .model_pii_type import PIIType

__all__ = [
    "ModelPIIMatch",
]


class ModelPIIMatch(BaseModel):
    """A detected PII match in content."""

    pii_type: PIIType = Field(description="Type of PII detected")
    value: str = Field(description="The detected PII value (may be masked)")
    start_index: int = Field(description="Start position in the content")
    end_index: int = Field(description="End position in the content")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    masked_value: str = Field(description="Masked version of the detected value")
