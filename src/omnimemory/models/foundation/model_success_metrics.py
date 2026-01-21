"""
Success metrics models following ONEX standards.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Confidence level thresholds for human-readable categorization
CONFIDENCE_VERY_HIGH_THRESHOLD = 0.9
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.5
CONFIDENCE_LOW_THRESHOLD = 0.25

# Reliability threshold for high quality determination
HIGH_QUALITY_RELIABILITY_THRESHOLD = 0.8


class ModelSuccessRate(BaseModel):
    """Success rate metric following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid")

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
        if hasattr(info, "data") and "total_operations" in getattr(info, "data", {}):
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


class ModelConfidenceInterval(BaseModel):
    """Confidence interval for statistical predictions following ONEX standards."""

    model_config = ConfigDict(extra="forbid")

    lower_bound: float = Field(description="Lower bound of the confidence interval")
    upper_bound: float = Field(description="Upper bound of the confidence interval")
    confidence_level: float = Field(
        ge=0.0,
        le=1.0,
        default=0.95,
        description="Confidence level (e.g., 0.95 for 95% confidence)",
    )
    point_estimate: float | None = Field(
        default=None,
        description="Point estimate (optional, typically the mean or median)",
    )

    @field_validator("upper_bound")
    @classmethod
    def validate_bounds(cls, v: float, info: ValidationInfo) -> float:
        """Validate that upper bound is greater than or equal to lower bound."""
        if hasattr(info, "data") and "lower_bound" in info.data:
            lower = info.data["lower_bound"]
            if v < lower:
                raise ValueError("Upper bound must be >= lower bound")
        return v

    @property
    def interval_width(self) -> float:
        """Calculate the width of the confidence interval."""
        return self.upper_bound - self.lower_bound

    @property
    def midpoint(self) -> float:
        """Calculate the midpoint of the confidence interval."""
        return (self.lower_bound + self.upper_bound) / 2


class ModelConfidenceScore(BaseModel):
    """Confidence score metric following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score as a decimal between 0.0 and 1.0",
    )
    measurement_basis: str = Field(
        description="Basis for confidence measurement (e.g., 'data_quality')",
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
        description="Confidence calculation method (e.g., 'statistical')",
    )
    measured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the confidence score was calculated",
    )

    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.score >= CONFIDENCE_VERY_HIGH_THRESHOLD:
            return "Very High"
        elif self.score >= CONFIDENCE_HIGH_THRESHOLD:
            return "High"
        elif self.score >= CONFIDENCE_MEDIUM_THRESHOLD:
            return "Medium"
        elif self.score >= CONFIDENCE_LOW_THRESHOLD:
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

    model_config = ConfigDict(frozen=False, extra="forbid")

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
        return (
            self.quality_grade in {"A+", "A", "B+"}
            and self.reliability_index >= HIGH_QUALITY_RELIABILITY_THRESHOLD
        )
