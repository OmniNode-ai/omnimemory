# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event model specification for handler routing.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelEventModel"]


class ModelEventModel(BaseModel):
    """Event model specification for handler routing.

    Supports both full format (name + module) and shorthand (just name string).

    Attributes:
        name: Event model class name (e.g., "ModelNodeIntrospectionEvent").
        module: Fully qualified module path containing the event model (optional).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str = Field(..., description="Event model class name")
    module: str | None = Field(
        default=None, description="Module path containing the event model"
    )
