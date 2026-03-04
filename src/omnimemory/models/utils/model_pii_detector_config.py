# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ModelPIIDetectorConfig Pydantic model for OmniMemory ONEX architecture."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelPIIDetectorConfig",
]


class ModelPIIDetectorConfig(BaseModel):
    """Configuration for PII detection with extracted magic numbers."""

    model_config = ConfigDict(extra="forbid")

    # Confidence thresholds
    high_confidence: float = Field(
        default=0.98, ge=0.0, le=1.0, description="High confidence threshold"
    )
    medium_high_confidence: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Medium-high confidence threshold"
    )
    medium_confidence: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Medium confidence threshold"
    )
    reduced_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Reduced confidence for complex patterns",
    )
    low_confidence: float = Field(
        default=0.60, ge=0.0, le=1.0, description="Low confidence threshold"
    )

    # Pattern matching limits
    max_text_length: int = Field(
        default=50000, ge=1000, description="Maximum text length to analyze"
    )
    max_matches_per_type: int = Field(
        default=100, ge=1, description="Maximum matches per PII type"
    )

    # Context analysis settings
    enable_context_analysis: bool = Field(
        default=True, description="Enable context-aware detection"
    )
    context_window_size: int = Field(
        default=50, ge=10, le=200, description="Context analysis window size"
    )
