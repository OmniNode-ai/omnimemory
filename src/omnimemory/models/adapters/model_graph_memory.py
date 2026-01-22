# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph Memory adapter models.

This module contains Pydantic models for the Graph Memory adapter,
following ONEX naming conventions (all models prefixed with "Model").

Models:
    - ModelMemoryConnection: Represents a connection between two memories
    - ModelRelatedMemory: A memory found through relationship traversal
    - ModelRelatedMemoryResult: Result of a find_related operation
    - ModelConnectionsResult: Result of a get_connections operation
    - ModelGraphMemoryHealth: Health status information for the adapter
    - ModelGraphMemoryConfig: Configuration for the Graph Memory adapter

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

import re
from typing import Literal

from omnibase_core.types.type_json import JsonType
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ModelConnectionsResult",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    "PropertyValue",
]

# Type alias for graph property values (semantic naming only).
# Delegates to omnibase_core's JsonType which uses PEP 695 recursive type
# definition for Pydantic 2.x compatibility. Matches ModelGraphDatabaseNode.properties.
type PropertyValue = JsonType


class ModelMemoryConnection(BaseModel):
    """Represents a connection (relationship) between two memories.

    Attributes:
        source_id: The source memory ID.
        target_id: The target memory ID.
        relationship_type: The type of relationship (e.g., "related_to", "caused_by").
        weight: Strength of the connection (0.0-1.0). Defaults to 1.0.
        is_outgoing: True if this is an outgoing edge from source, False if incoming.
        created_at: ISO timestamp when the connection was created.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    source_id: str = Field(
        ...,
        description="Source memory ID",
    )
    target_id: str = Field(
        ...,
        description="Target memory ID",
    )
    relationship_type: str = Field(
        ...,
        description="Type of relationship between memories",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strength of connection (0.0-1.0)",
    )
    is_outgoing: bool = Field(
        default=True,
        description="True if outgoing from source, False if incoming",
    )
    created_at: str | None = Field(
        default=None,
        description="ISO timestamp when connection was created",
    )


class ModelRelatedMemory(BaseModel):
    """A memory found through relationship traversal.

    Attributes:
        memory_id: The related memory's identifier.
        score: Relevance score based on path weight and distance (0.0-1.0).
        path: Path endpoints as [start_memory_id, related_memory_id]. Does not
            include intermediate nodes; use 'depth' to determine hop count.
        depth: Number of hops from the starting memory.
        labels: Graph labels on the memory node.
        properties: Additional properties from the graph node.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    memory_id: str = Field(
        ...,
        description="The related memory's identifier",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Relevance score based on graph distance. Calculated as 1/(depth+1). "
            "Minimum depth is 1 (starting node excluded), "
            "so range is 0.5 (depth=1) to ~0.09 (depth=10). "
            "Score never reaches 1.0 since depth is always >= 1. "
            "Depth-to-score mapping: "
            "depth=1 -> 0.50, "
            "depth=2 -> 0.33, "
            "depth=3 -> 0.25, "
            "depth=4 -> 0.20, "
            "depth=5 -> 0.17, "
            "depth=6 -> 0.14, "
            "depth=7 -> 0.125, "
            "depth=8 -> 0.11, "
            "depth=9 -> 0.10, "
            "depth=10 -> 0.09."
        ),
    )
    path: list[str] = Field(
        default_factory=list,
        description=(
            "Path endpoints: [start_memory_id, related_memory_id]. "
            "Note: Intermediate nodes are not included in this list. "
            "Use 'depth' field to determine the number of hops."
        ),
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Number of hops from starting memory",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Graph labels on the memory node",
    )
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict,
        description="Additional node properties",
    )


class ModelRelatedMemoryResult(BaseModel):
    """Result of a find_related operation.

    Attributes:
        status: Operation status (success, error, not_found, no_results).
        memories: List of related memories ordered by relevance score.
        total_count: Total number of related memories returned (after filtering).
        candidates_found: Number of candidates found before min_score filtering.
            Useful for understanding how many results were filtered out.
        max_depth_reached: The maximum traversal depth that was reached.
        execution_time_ms: Time taken to execute the query in milliseconds.
        error_message: Error details if status is "error".
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: Literal["success", "error", "not_found", "no_results"] = Field(
        ...,
        description="Operation status",
    )
    memories: list[ModelRelatedMemory] = Field(
        default_factory=list,
        description="Related memories ordered by relevance",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of related memories returned (after filtering)",
    )
    candidates_found: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of candidates found before min_score filtering. "
            "Compare with total_count to see how many were filtered out."
        ),
    )
    max_depth_reached: int = Field(
        default=0,
        ge=0,
        description="Maximum traversal depth reached",
    )
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if status is error",
    )


class ModelConnectionsResult(BaseModel):
    """Result of a get_connections operation.

    Attributes:
        status: Operation status (success, error, not_found, no_results).
        connections: List of connections for the memory.
        total_count: Total number of connections found.
        error_message: Error details if status is "error".
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: Literal["success", "error", "not_found", "no_results"] = Field(
        ...,
        description="Operation status",
    )
    connections: list[ModelMemoryConnection] = Field(
        default_factory=list,
        description="Connections for the memory",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of connections",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if status is error",
    )


class ModelGraphMemoryHealth(BaseModel):
    """Health status information for the graph memory adapter.

    Attributes:
        is_healthy: Overall health status.
        initialized: Whether the adapter has been initialized.
        handler_healthy: Health status from the underlying graph handler.
        error_message: Error details if unhealthy.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    is_healthy: bool = Field(
        ...,
        description="Overall health status",
    )
    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )
    handler_healthy: bool | None = Field(
        default=None,
        description="Health status from underlying graph handler (None if not checked)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if unhealthy",
    )


class ModelGraphMemoryConfig(BaseModel):
    """Configuration for the Graph Memory adapter.

    Attributes:
        max_depth: Maximum allowed traversal depth. Bounded to prevent
            expensive deep traversals. Defaults to 5.
        default_depth: Default traversal depth if not specified. Defaults to 2.
        default_limit: Default result limit. Defaults to 100.
        max_limit: Maximum allowed result limit. Defaults to 1000.
        bidirectional: Whether to traverse relationships in both directions.
            Defaults to True.
        memory_node_label: Graph label for memory nodes. Defaults to "Memory".
        timeout_seconds: Query timeout in seconds. Defaults to 30.0.
        score_filter_multiplier: Multiplier for query limit when min_score
            filtering is used. Higher values fetch more candidates but
            increase query cost. Defaults to 3.0. Range: 1.0-10.0.
        ensure_indexes: Whether to create indexes on memory_id during
            initialization. Defaults to True. Index creation is idempotent.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    max_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum allowed traversal depth",
    )
    default_depth: int = Field(
        default=2,
        ge=1,
        description="Default traversal depth",
    )
    default_limit: int = Field(
        default=100,
        ge=1,
        description="Default result limit",
    )
    max_limit: int = Field(
        default=1000,
        ge=1,
        description="Maximum allowed result limit",
    )
    bidirectional: bool = Field(
        default=True,
        description="Traverse relationships in both directions",
    )
    memory_node_label: str = Field(
        default="Memory",
        description="Graph label for memory nodes",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Query timeout in seconds",
    )
    score_filter_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description=(
            "Multiplier for query limit when min_score filtering is used. "
            "Higher values fetch more candidates but increase query cost. "
            "If min_score is high (>0.5), consider increasing this value."
        ),
    )
    ensure_indexes: bool = Field(
        default=True,
        description=(
            "Whether to create indexes on memory_id during initialization. "
            "Index creation is idempotent (safe to run multiple times). "
            "Set to False to skip index creation if managing indexes manually."
        ),
    )

    @field_validator("memory_node_label")
    @classmethod
    def validate_node_label(cls, v: str) -> str:
        """Validate memory_node_label is a valid Cypher label.

        Cypher labels must start with a letter or underscore and contain
        only letters, numbers, and underscores. This validation prevents
        potential injection issues when the label is used in queries.

        Args:
            v: The memory_node_label value to validate.

        Returns:
            The validated label if valid.

        Raises:
            ValueError: If the label does not match the required pattern.
        """
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v):
            msg = (
                f"memory_node_label '{v}' is not a valid Cypher label. "
                "Must start with a letter or underscore, and contain only "
                "letters, numbers, and underscores."
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_bounds(self) -> ModelGraphMemoryConfig:
        """Ensure default values do not exceed their maximums."""
        if self.default_depth > self.max_depth:
            msg = (
                f"default_depth ({self.default_depth}) "
                f"must be <= max_depth ({self.max_depth})"
            )
            raise ValueError(msg)
        if self.default_limit > self.max_limit:
            msg = (
                f"default_limit ({self.default_limit}) "
                f"must be <= max_limit ({self.max_limit})"
            )
            raise ValueError(msg)
        return self


# Backward compatibility alias - deprecated, use ModelGraphMemoryConfig
AdapterGraphMemoryConfig = ModelGraphMemoryConfig
