# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Related memory result model for Graph Memory adapter.

This module contains the ModelRelatedMemoryResult Pydantic model representing
the result of a find_related operation in the graph.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Runtime import required for Pydantic schema building (not TYPE_CHECKING)
from omnimemory.models.adapters.model_related_memory import (
    ModelRelatedMemory,  # noqa: TC001
)

__all__ = [
    "ModelRelatedMemoryResult",
]


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
