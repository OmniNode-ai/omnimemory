# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler specification for handler routing.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelHandlerSpec"]


class ModelHandlerSpec(BaseModel):
    """Handler specification for handler routing.

    Attributes:
        name: Handler class name (e.g., "HandlerNodeIntrospected").
        module: Fully qualified module path containing the handler.
        method: Optional method name if handler has multiple entry points.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str = Field(..., description="Handler class name")
    module: str = Field(..., description="Module path containing the handler")
    method: str | None = Field(default=None, description="Optional handler method name")
