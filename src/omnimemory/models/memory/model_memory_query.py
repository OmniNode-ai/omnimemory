"""
Memory query model following ONEX standards.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from ...enums import EnumMemoryStorageType


class ModelMemoryQuery(BaseModel):
    """Query model for memory search and retrieval following ONEX standards."""

    # Query identification
    query_id: UUID = Field(
        description="Unique identifier for the query",
    )

    # Query content
    query_text: str = Field(
        description="The main query text or search terms",
    )
    query_type: str = Field(
        description="Type of query (semantic, keyword, structured, etc.)",
    )

    # Filters
    item_types: list[str] = Field(
        default_factory=list,
        description="Filter by specific item types",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Filter by tags",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Filter by keywords",
    )

    # Storage targeting
    storage_types: list[EnumMemoryStorageType] = Field(
        default_factory=list,
        description="Target specific storage types for the query",
    )
    storage_locations: list[str] = Field(
        default_factory=list,
        description="Target specific storage locations",
    )

    # Result parameters
    limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of results to return",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip (for pagination)",
    )

    # Scoring parameters
    min_relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for results",
    )
    min_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for results",
    )
    boost_recent: bool = Field(
        default=False,
        description="Whether to boost more recent items in scoring",
    )
    boost_popular: bool = Field(
        default=False,
        description="Whether to boost more frequently accessed items",
    )

    # Sorting
    sort_by: str = Field(
        default="relevance",
        description="Field to sort results by",
    )
    sort_order: str = Field(
        default="desc",
        description="Sort order (asc or desc)",
    )

    # Options
    include_metadata: bool = Field(
        default=True,
        description="Whether to include item metadata in results",
    )
    include_content: bool = Field(
        default=True,
        description="Whether to include full content in results",
    )
    highlight_matches: bool = Field(
        default=False,
        description="Whether to highlight search matches in content",
    )

    # Advanced options
    semantic_search: bool = Field(
        default=True,
        description="Whether to use semantic search capabilities",
    )
    fuzzy_matching: bool = Field(
        default=False,
        description="Whether to use fuzzy matching for terms",
    )
    expand_query: bool = Field(
        default=False,
        description="Whether to expand query with related terms",
    )
