"""
Intelligence analysis model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums.enum_intelligence_operation_type import EnumIntelligenceOperationType


class ModelIntelligenceAnalysis(BaseModel):
    """Intelligence analysis result following ONEX standards."""

    # Analysis identification
    analysis_id: UUID = Field(
        description="Unique identifier for the analysis",
    )
    operation_type: EnumIntelligenceOperationType = Field(
        description="Type of intelligence operation performed",
    )

    # Input information
    input_content: str = Field(
        description="Content that was analyzed",
    )
    input_type: str = Field(
        description="Type of input content (text, document, etc.)",
    )

    # Analysis results
    results: dict[str, str] = Field(
        default_factory=dict,
        description="Analysis results with string values for type safety",
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Key insights derived from the analysis",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations based on the analysis",
    )

    # Confidence and quality metrics
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis results",
    )
    accuracy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated accuracy of the analysis",
    )
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Completeness of the analysis",
    )

    # Processing information
    processing_time_ms: int = Field(
        description="Time taken to perform the analysis",
    )
    model_version: str = Field(
        description="Version of the analysis model used",
    )
    algorithm_used: str = Field(
        description="Algorithm or method used for analysis",
    )

    # Metadata and context
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the analysis",
    )
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Additional context for the analysis",
    )

    # Temporal information
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was performed",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the analysis results expire",
    )

    # Quality assurance
    validated: bool = Field(
        default=False,
        description="Whether the analysis has been validated",
    )
    validation_score: float | None = Field(
        default=None,
        description="Validation score if validated",
    )

    # Usage tracking
    access_count: int = Field(
        default=0,
        description="Number of times this analysis has been accessed",
    )
    last_accessed_at: datetime | None = Field(
        default=None,
        description="When the analysis was last accessed",
    )