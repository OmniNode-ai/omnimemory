# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for handler routing support using omnibase_core subcontract.

Temporary workaround until OMN-1588 adds handler_routing to base contracts.
"""

from __future__ import annotations

from omnibase_core.models.contracts.subcontracts import (
    ModelHandlerRoutingSubcontract,  # noqa: TC002 - Pydantic needs runtime access
)
from pydantic import Field

__all__ = ["MixinHandlerRouting"]


class MixinHandlerRouting:
    """Mixin adding handler_routing field using omnibase_core's subcontract."""

    handler_routing: ModelHandlerRoutingSubcontract | None = Field(
        default=None,
        description="Handler routing configuration using ONEX standard format",
    )
