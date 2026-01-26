# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler routing configuration for declarative dispatch.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.contracts.model_handler_routing_entry import (  # noqa: TC001
    ModelHandlerRoutingEntry,
)
from omnimemory.models.contracts.model_handler_spec import (
    ModelHandlerSpec,  # noqa: TC001
)

__all__ = ["ModelHandlerRouting"]


class ModelHandlerRouting(BaseModel):
    """Handler routing configuration for declarative dispatch.

    This model represents the handler_routing section of ONEX contracts,
    following the omnibase_infra standard for declarative handler dispatch.

    Attributes:
        routing_strategy: Strategy for matching events/operations to handlers.
        execution_mode: How handlers execute (sequential, parallel, single).
        handlers: List of routing entries mapping events/operations to handlers.
        default_handler: Optional fallback handler for unmatched events.
        router_handler: Optional main routing handler for multi-backend patterns.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    routing_strategy: Literal[
        "payload_type_match",
        "event_type_match",
        "operation_match",
        "operation_type_match",
        "action_type_match",
    ] = Field(
        default="payload_type_match",
        description="Strategy for matching events to handlers",
    )
    execution_mode: Literal["sequential", "parallel", "single"] = Field(
        default="sequential",
        description="Handler execution mode",
    )
    handlers: list[ModelHandlerRoutingEntry] = Field(
        default_factory=list,
        description="Handler routing entries",
    )
    default_handler: ModelHandlerSpec | None = Field(
        default=None,
        description="Default handler for unmatched events",
    )
    router_handler: ModelHandlerSpec | None = Field(
        default=None,
        description="Main routing handler for multi-backend patterns",
    )
    partial_failure_handling: bool = Field(
        default=True,
        description="Whether to handle partial failures in parallel execution",
    )
