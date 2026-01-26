# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Extended Orchestrator contract with handler_routing support.

Temporary extension until OMN-1588 adds handler_routing to omnibase_core.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from omnibase_core.models.contracts import ModelContractOrchestrator
from pydantic import ConfigDict, Field

from omnimemory.models.contracts.model_handler_routing import (
    ModelHandlerRouting,  # noqa: TC001
)

__all__ = ["ModelContractOrchestratorExtended"]


class ModelContractOrchestratorExtended(ModelContractOrchestrator):
    """Extended Orchestrator contract with handler_routing support.

    Temporary extension until OMN-1588 adds handler_routing to omnibase_core.

    Note: ModelContractOrchestrator already has consumed_events and
    published_events fields, but this extension allows extra fields that
    may be in different formats from the ONEX infra patterns.

    Adds:
        - handler_routing: Declarative handler dispatch configuration
    """

    handler_routing: ModelHandlerRouting | None = Field(
        default=None,
        description="Handler routing configuration for declarative dispatch",
    )

    model_config = ConfigDict(
        extra="ignore",  # Allow additional ONEX extension fields
        use_enum_values=True,  # Orchestrator uses this
        validate_assignment=True,
    )
