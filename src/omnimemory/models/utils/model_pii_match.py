# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
ModelPIIMatch Pydantic model for OmniMemory ONEX architecture.

This module contains the model for representing a detected PII match.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .model_pii_type import PIIType  # noqa: TC001 - Pydantic needs runtime access

__all__ = [
    "ModelPIIMatch",
]


class ModelPIIMatch(BaseModel):
    """A detected PII match in content."""

    model_config = ConfigDict(extra="forbid")

    pii_type: PIIType = Field(description="Type of PII detected")
    value: str = Field(description="The detected PII value (may be masked)")
    start_index: int = Field(description="Start position in the content")
    end_index: int = Field(description="End position in the content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    masked_value: str = Field(description="Masked version of the detected value")
