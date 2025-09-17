"""
Confidence score metric model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelConfidenceScore(BaseModel):
    """Confidence score metric following ONEX standards."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score as a decimal between 0.0 and 1.0",
    )
    measurement_basis: str = Field(
        description="Basis for confidence measurement (e.g., 'data_quality', 'algorithm_certainty')",
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
        description="Method used to calculate confidence (e.g., 'statistical', 'heuristic', 'ml_based')",
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
