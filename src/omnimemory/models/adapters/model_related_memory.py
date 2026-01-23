# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Related memory model for Graph Memory adapter.

This module contains the ModelRelatedMemory Pydantic model representing
a memory found through relationship traversal in the graph.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

from omnibase_core.types.type_json import JsonType
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelRelatedMemory",
    "PropertyValue",
]

# Type alias for graph property values (semantic naming only).
# Delegates to omnibase_core's JsonType which uses PEP 695 recursive type
# definition for Pydantic 2.x compatibility. Matches ModelGraphDatabaseNode.properties.
type PropertyValue = JsonType


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
        default=1,
        ge=1,
        description="Number of hops from starting memory (minimum 1, starting node excluded)",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Graph labels on the memory node",
    )
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict,
        description="Additional node properties",
    )
