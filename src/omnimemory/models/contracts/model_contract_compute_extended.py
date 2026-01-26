# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Extended Compute contract with handler_routing support.

Temporary extension until OMN-1588 adds handler_routing to omnibase_core.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from omnibase_core.models.contracts import ModelContractCompute
from pydantic import ConfigDict

from omnimemory.models.contracts.mixin_handler_routing import MixinHandlerRouting

__all__ = ["ModelContractComputeExtended"]


class ModelContractComputeExtended(MixinHandlerRouting, ModelContractCompute):
    """Extended Compute contract with handler_routing support.

    Temporary extension until OMN-1588 adds handler_routing to omnibase_core.
    """

    model_config = ConfigDict(
        extra="ignore",  # Allow additional ONEX extension fields
        use_enum_values=False,
        validate_assignment=True,
    )
