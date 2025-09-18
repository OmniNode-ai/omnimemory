"""
Semantic analysis result model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelSemanticAnalysisResult(BaseModel):
    """Semantic analysis result following ONEX standards."""

    # Result identification
    result_id: UUID = Field(
        description="Unique identifier for the semantic analysis result",
    )
    analysis_type: str = Field(
        description="Type of semantic analysis performed",
    )

    # Input information
    analyzed_content: str = Field(
        description="Content that was semantically analyzed",
    )
    content_language: str = Field(
        default="en",
        description="Language of the analyzed content",
    )

    # Semantic features
    semantic_vector: list[float] = Field(
        default_factory=list,
        description="Semantic vector representation of the content",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Key concepts identified in the content",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities found in the content",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topics associated with the content",
    )

    # Semantic relationships
    concept_relationships: dict[str, str] = Field(
        default_factory=dict,
        description="Relationships between concepts",
    )
    similarity_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Similarity scores to reference concepts",
    )

    # Sentiment and emotion
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment score (-1 negative, +1 positive)",
    )
    emotion_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Emotion scores for different emotions",
    )

    # Complexity and readability
    complexity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Complexity score of the content",
    )
    readability_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Readability score of the content",
    )

    # Quality metrics
    coherence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Coherence score of the content",
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score to the domain",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the semantic analysis",
    )

    # Processing information
    model_name: str = Field(
        description="Name of the semantic model used",
    )
    model_version: str = Field(
        description="Version of the semantic model",
    )
    processing_time_ms: int = Field(
        description="Time taken for semantic analysis",
    )

    # Temporal information
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the semantic analysis was performed",
    )

    # Context and metadata
    domain_context: str | None = Field(
        default=None,
        description="Domain context for the analysis",
    )
    analysis_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Parameters used for the analysis",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the analysis",
    )
