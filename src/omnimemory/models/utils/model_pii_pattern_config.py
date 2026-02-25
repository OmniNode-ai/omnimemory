# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
ModelPIIPatternConfig Pydantic model for OmniMemory ONEX architecture.

This module contains the configuration model for PII pattern matching.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelPIIPatternConfig",
]


class ModelPIIPatternConfig(BaseModel):
    """Strongly typed PII pattern configuration replacing Dict[str, Any]."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="Regex pattern for PII detection")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Base confidence score for matches"
    )
    mask_template: str = Field(description="Template for masking detected values")
