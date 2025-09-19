"""
Quality metrics model following ONEX standards.
"""

from pydantic import BaseModel, Field, field_validator

from ...enums.foundation.enum_quality_grade import EnumQualityGrade
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
    quality_grade: EnumQualityGrade = Field(
        description="Overall quality grade",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving quality metrics",
    )

    @field_validator("quality_grade")
    @classmethod
    def validate_quality_grade(cls, v: EnumQualityGrade) -> EnumQualityGrade:
        """Validate quality grade format."""
        # Enum validation is handled automatically by Pydantic
        return v

    @property
    def is_high_quality(self) -> bool:
        """Check if metrics indicate high quality."""
        return (
            self.quality_grade
            in {EnumQualityGrade.A_PLUS, EnumQualityGrade.A, EnumQualityGrade.B_PLUS}
            and self.reliability_index >= 0.8
        )
