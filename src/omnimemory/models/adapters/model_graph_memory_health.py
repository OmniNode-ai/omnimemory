# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Graph memory health model for Graph Memory adapter.

This module contains the ModelGraphMemoryHealth Pydantic model representing
the health status information for the graph memory adapter.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelGraphMemoryHealth",
]


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
