# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A relationship between two snapshots held by the mock graph handler (OMN-21623)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelGraphRelationship"]


class ModelGraphRelationship(BaseModel):
    """Represents a relationship between two snapshots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(..., description="The source snapshot ID")
    target_id: str = Field(..., description="The target snapshot ID")
    relationship_type: str = Field(
        ...,
        description='The type of relationship (e.g., "related_to", "caused_by")',
    )
    weight: float = Field(
        default=1.0,
        description="Weight or strength of the relationship (0.0-1.0)",
    )
