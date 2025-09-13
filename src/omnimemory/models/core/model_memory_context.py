"""
Memory context model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .enum_node_type import EnumNodeType


class ModelMemoryContext(BaseModel):
    """Context information for memory operations following ONEX standards."""

    correlation_id: UUID = Field(
        description="Unique correlation identifier for tracing operations across nodes",
    )
    session_id: UUID | None = Field(
        default=None,
        description="Session identifier for grouping related operations",
    )
    user_id: str | None = Field(
        default=None,
        description="User identifier for authorization and personalization",
    )

    # ONEX node information
    source_node_type: EnumNodeType = Field(
        description="Type of ONEX node initiating the operation",
    )
    source_node_id: str = Field(
        description="Identifier of the source node",
    )

    # Operation metadata
    operation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the operation was initiated",
    )
    timeout_ms: int = Field(
        default=30000,
        description="Timeout for the operation in milliseconds",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Operation priority (1=lowest, 10=highest)",
    )

    # Context tags and metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering operations",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the operation",
    )

    # Trust and validation
    trust_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trust score for the operation source",
    )
    validation_required: bool = Field(
        default=False,
        description="Whether the operation requires additional validation",
    )