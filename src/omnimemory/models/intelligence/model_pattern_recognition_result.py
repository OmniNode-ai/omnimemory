"""
Pattern recognition result model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelPatternRecognitionResult(BaseModel):
    """Pattern recognition result following ONEX standards."""

    # Result identification
    result_id: UUID = Field(
        description="Unique identifier for the pattern recognition result",
    )
    pattern_id: UUID = Field(
        description="Identifier for the recognized pattern",
    )

    # Pattern information
    pattern_name: str = Field(
        description="Human-readable name for the pattern",
    )
    pattern_type: str = Field(
        description="Type or category of the pattern",
    )
    pattern_description: str = Field(
        description="Description of the recognized pattern",
    )

    # Recognition details
    matched_content: str = Field(
        description="Content that matched the pattern",
    )
    match_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of the pattern match",
    )
    match_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the pattern recognition",
    )

    # Pattern characteristics
    pattern_frequency: int = Field(
        description="Frequency of this pattern in the dataset",
    )
    pattern_significance: float = Field(
        ge=0.0,
        le=1.0,
        description="Statistical significance of the pattern",
    )
    pattern_uniqueness: float = Field(
        ge=0.0,
        le=1.0,
        description="Uniqueness score of the pattern",
    )

    # Context information
    context_window: str = Field(
        description="Contextual window around the matched pattern",
    )
    related_patterns: list[str] = Field(
        default_factory=list,
        description="IDs of related patterns",
    )

    # Quality metrics
    precision_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Precision of the pattern recognition",
    )
    recall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall of the pattern recognition",
    )
    f1_score: float = Field(
        ge=0.0,
        le=1.0,
        description="F1 score of the pattern recognition",
    )

    # Processing information
    algorithm_used: str = Field(
        description="Algorithm used for pattern recognition",
    )
    model_version: str = Field(
        description="Version of the pattern recognition model",
    )
    processing_time_ms: int = Field(
        description="Time taken for pattern recognition",
    )

    # Temporal information
    recognized_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the pattern was recognized",
    )

    # Metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the pattern",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations for the pattern",
    )
