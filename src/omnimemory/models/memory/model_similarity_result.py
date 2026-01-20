"""
Similarity result model for vector comparison operations following ONEX standards.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSimilarityResult(BaseModel):
    """Result of a similarity comparison between two vectors."""

    model_config = ConfigDict(extra="forbid")

    # Metric information
    metric: Literal["cosine", "euclidean"] = Field(
        description="The distance metric used for comparison",
    )

    # Scoring information
    similarity: float | None = Field(
        default=None,
        description="Cosine similarity (1.0=identical, 0.0=orthogonal). None for L2.",
    )
    distance: float = Field(
        description="Distance between vectors (cosine: 1-similarity, euclidean: L2)",
    )

    # Match determination
    is_match: bool | None = Field(
        default=None,
        description="Whether vectors match within threshold (None if not provided)",
    )
    threshold: float | None = Field(
        default=None,
        description="The threshold used for is_match determination",
    )

    # Vector information
    dimensions: int = Field(
        ge=1,
        description="Number of dimensions in the compared vectors",
    )

    # Diagnostic information
    notes: str | None = Field(
        default=None,
        description="Optional debug/diagnostic notes",
    )
