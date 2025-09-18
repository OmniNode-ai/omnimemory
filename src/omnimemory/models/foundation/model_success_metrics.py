"""
Success metrics models following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ModelSuccessRate(BaseModel):
    """Success rate metric following ONEX standards."""

    rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Success rate as a decimal between 0.0 and 1.0",
    )
    total_operations: int = Field(
        ge=0,
        description="Total number of operations measured",
    )
    successful_operations: int = Field(
        ge=0,
        description="Number of successful operations",
    )
    calculation_window_start: datetime = Field(
        description="Start time of the calculation window",
    )
    calculation_window_end: datetime = Field(
        description="End time of the calculation window",
    )
    measurement_type: str = Field(
        description="Type of operation measured (e.g., 'memory_storage', 'retrieval')",
    )

    @field_validator("successful_operations")
    @classmethod
    def validate_successful_operations(cls, v: int, info: ValidationInfo) -> int:
        """Validate successful operations doesn't exceed total."""
        if hasattr(info, "data") and "total_operations" in info.data:
            total = info.data["total_operations"]
            if v > total:
                raise ValueError("Successful operations cannot exceed total operations")
        return v

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.rate

    @property
    def failed_operations(self) -> int:
        """Calculate number of failed operations."""
        return self.total_operations - self.successful_operations

    def to_percentage(self) -> float:
        """Convert rate to percentage."""
        return self.rate * 100.0


class ModelConfidenceScore(BaseModel):
    """Confidence score metric following ONEX standards."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score as a decimal between 0.0 and 1.0",
    )
    measurement_basis: str = Field(
        description="Basis for confidence measurement "
        "(e.g., 'data_quality', 'algorithm_certainty')",
    )
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Factors that contributed to this confidence score",
    )
    reliability_indicators: dict[str, float] = Field(
        default_factory=dict,
        description="Individual reliability indicators and their values",
    )
    sample_size: int | None = Field(
        default=None,
        ge=0,
        description="Sample size used for confidence calculation",
    )
    calculation_method: str = Field(
        description="Method used to calculate confidence "
        "(e.g., 'statistical', 'heuristic', 'ml_based')",
    )
    measured_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the confidence score was calculated",
    )

    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.score >= 0.9:
            return "Very High"
        elif self.score >= 0.75:
            return "High"
        elif self.score >= 0.5:
            return "Medium"
        elif self.score >= 0.25:
            return "Low"
        else:
            return "Very Low"

    def to_percentage(self) -> float:
        """Convert score to percentage."""
        return self.score * 100.0

    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if confidence score meets reliability threshold."""
        return self.score >= threshold


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
