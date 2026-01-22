# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory connection model for Graph Memory adapter.

This module contains the ModelMemoryConnection Pydantic model representing
a connection (relationship) between two memories in the graph.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelMemoryConnection",
]


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
