# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Single handler routing entry for declarative dispatch.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.contracts.model_event_model import ModelEventModel  # noqa: TC001
from omnimemory.models.contracts.model_handler_spec import (
    ModelHandlerSpec,  # noqa: TC001
)

__all__ = ["ModelHandlerRoutingEntry"]


class ModelHandlerRoutingEntry(BaseModel):
    """Single handler routing entry mapping event to handler.

    This follows the omnibase_infra contract.yaml format for handler routing.
    Supports both full format and shorthand formats for event_model.

    Attributes:
        event_model: Event model specification (name + module) or just name string.
        handler: Handler specification (name + module).
        operation: Operation name for operation-based routing.
        action: Action name for action-based routing.
        description: Human-readable description of this routing entry.
        output_events: List of event types this handler may emit.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    event_model: ModelEventModel | str | None = Field(
        default=None,
        description="Event model specification for event-based routing (string or dict)",
    )
    handler: ModelHandlerSpec = Field(
        ...,
        description="Handler specification",
    )
    operation: str | None = Field(
        default=None,
        description="Operation name for operation-based routing",
    )
    action: str | None = Field(
        default=None,
        description="Action name for action-based routing",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description",
    )
    output_events: list[str] = Field(
        default_factory=list,
        description="Event types this handler may emit",
    )
