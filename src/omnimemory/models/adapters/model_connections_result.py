# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Connections result model for Graph Memory adapter.

This module contains the ModelConnectionsResult Pydantic model representing
the result of a get_connections operation in the graph.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Runtime import required for Pydantic schema building (not TYPE_CHECKING)
from omnimemory.models.adapters.model_memory_connection import (
    ModelMemoryConnection,  # noqa: TC001
)

__all__ = [
    "ModelConnectionsResult",
]


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
