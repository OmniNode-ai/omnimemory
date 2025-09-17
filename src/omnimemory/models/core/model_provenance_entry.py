"""
Provenance entry model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelProvenanceEntry(BaseModel):
    """Single provenance entry following ONEX standards."""

    # Operation identification
    operation_id: UUID = Field(
        description="Unique identifier for the operation that created this provenance entry",
    )
    operation_type: str = Field(
        description="Type of operation (store, retrieve, update, delete, migrate, etc.)",
    )

    # Source identification
    source_component: str = Field(
        description="Component that performed the operation (memory_manager, intelligence_engine, etc.)",
    )
    source_version: str | None = Field(
        default=None,
        description="Version of the source component that performed the operation",
    )

    # Actor identification
    actor_type: str = Field(
        description="Type of actor that initiated the operation (user, system, agent, migration)",
    )
    actor_id: str | None = Field(
        default=None,
        description="Identifier of the actor (user ID, system name, agent name)",
    )

    # Temporal information
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this provenance entry was created",
    )

    # Operation context
    operation_context: dict[str, str] = Field(
        default_factory=dict,
        description="Additional context about the operation",
    )

    # Data transformation
    input_hash: str | None = Field(
        default=None,
        description="Hash of input data for integrity verification",
    )
    output_hash: str | None = Field(
        default=None,
        description="Hash of output data for integrity verification",
    )
    transformation_description: str | None = Field(
        default=None,
        description="Description of how data was transformed",
    )
