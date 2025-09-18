"""
Memory search result model following ONEX standards.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from .model_memory_item import ModelMemoryItem


class ModelMemorySearchResult(BaseModel):
    """Search result model for memory queries following ONEX standards."""

    # Result identification
    result_id: UUID = Field(
        description="Unique identifier for this search result",
    )
    query_id: UUID = Field(
        description="Identifier of the query that produced this result",
    )

    # Result content
    memory_item: ModelMemoryItem = Field(
        description="The memory item that matched the query",
    )

    # Scoring information
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score for this result",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the match",
    )
    combined_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined score used for ranking",
    )

    # Match information
    match_type: str = Field(
        description="Type of match (exact, partial, semantic, etc.)",
    )
    matched_fields: list[str] = Field(
        default_factory=list,
        description="Fields that matched the query",
    )
    highlighted_content: str | None = Field(
        default=None,
        description="Content with search terms highlighted",
    )

    # Ranking information
    rank: int = Field(
        description="Position of this result in the result set",
    )
    total_results: int = Field(
        description="Total number of results for the query",
    )

    # Processing metadata
    processing_time_ms: float = Field(
        description="Time taken to process this result",
    )
    storage_source: str = Field(
        description="Storage system that provided this result",
    )

    # Quality indicators
    match_quality: str = Field(
        description="Quality of the match (high, medium, low)",
    )
    explanation: str | None = Field(
        default=None,
        description="Explanation of why this item matched",
    )
