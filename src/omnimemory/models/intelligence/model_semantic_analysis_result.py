# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Semantic analysis result model following ONEX standards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.foundation.model_semver import ModelSemVer  # noqa: TC001

from .model_semantic_entity_list import ModelSemanticEntityList  # noqa: TC001


class ModelSemanticAnalysisResult(BaseModel):
    """Semantic analysis result following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid")

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
    content_language: str | None = Field(
        default=None,
        description="Language of the analyzed content (None if not detected)",
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
    entity_list: ModelSemanticEntityList | None = Field(
        default=None,
        description="Full entity list with types, spans, and confidence scores. "
        "Populated when analysis_type includes entity extraction.",
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
    sentiment_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Sentiment score (-1 negative, +1 positive). None if not computed.",
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
    coherence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Coherence score of the content. None if not computed.",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance score to the domain. None if not computed.",
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
    model_version: ModelSemVer = Field(
        description="Semantic version of the semantic model",
    )
    processing_time_ms: int = Field(
        ge=0,
        description="Time taken for semantic analysis in milliseconds",
    )

    # Temporal information
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
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
