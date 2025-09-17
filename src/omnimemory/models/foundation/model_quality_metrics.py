"""
Combined quality metrics model following ONEX standards.
"""

from pydantic import BaseModel, Field, field_validator

from .model_confidence_score import ModelConfidenceScore
from .model_success_rate import ModelSuccessRate


class ModelQualityMetrics(BaseModel):
    """Combined quality metrics following ONEX standards."""

    success_rate: ModelSuccessRate = Field(
        description="Success rate metrics",
    )
    confidence_score: ModelConfidenceScore = Field(
        description="Confidence score metrics",
    )
    reliability_index: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined reliability index based on success rate and confidence",
    )
    quality_grade: str = Field(
        description="Overall quality grade (A+, A, B+, B, C+, C, D, F)",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving quality metrics",
    )

    @field_validator("quality_grade")
    @classmethod
    def validate_quality_grade(cls, v: str) -> str:
        """Validate quality grade format."""
        valid_grades = {"A+", "A", "B+", "B", "C+", "C", "D", "F"}
        if v not in valid_grades:
            raise ValueError(f"Quality grade must be one of {valid_grades}")
        return v

    @property
    def is_high_quality(self) -> bool:
        """Check if metrics indicate high quality."""
        return self.quality_grade in {"A+", "A", "B+"} and self.reliability_index >= 0.8
