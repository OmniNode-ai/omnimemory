# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for handler routing support in extended contracts.

Provides the handler_routing field extension for ONEX contracts.
Temporary workaround until OMN-1588 adds native support to omnibase_core.

Note: This mixin provides only the field definition. Each extended class
must define its own model_config with extra="ignore" to properly override
the base class configuration, as Pydantic's config merging only works
with BaseModel subclasses.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from pydantic import Field

from omnimemory.models.contracts.model_handler_routing import (
    ModelHandlerRouting,  # noqa: TC001
)

__all__ = ["MixinHandlerRouting"]


class MixinHandlerRouting:
    """Mixin adding handler_routing field to ONEX contract models.

    This mixin provides only the field definition. Pydantic's model_config
    must be defined in each extended class to ensure extra="ignore" is applied.

    Usage:
        class ModelContractEffectExtended(MixinHandlerRouting, ModelContractEffect):
            model_config = ConfigDict(extra="ignore", ...)
    """

    handler_routing: ModelHandlerRouting | None = Field(
        default=None,
        description="Handler routing configuration for declarative dispatch",
    )
