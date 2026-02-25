# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Extended contract models with ONEX handler_routing support.

Uses ModelHandlerRoutingSubcontract from omnibase_core for handler routing.
Temporary workaround until OMN-1588 adds handler_routing to base contracts.

OMN-1588 Cleanup: When resolved, delete this entire module and use
omnibase_core contracts directly.
"""

from omnibase_core.models.contracts.subcontracts import (
    ModelHandlerRoutingEntry,
    ModelHandlerRoutingSubcontract,
)

from omnimemory.models.contracts.mixin_handler_routing import MixinHandlerRouting
from omnimemory.models.contracts.model_contract_compute_extended import (
    ModelContractComputeExtended,
)
from omnimemory.models.contracts.model_contract_effect_extended import (
    ModelContractEffectExtended,
)
from omnimemory.models.contracts.model_contract_orchestrator_extended import (
    ModelContractOrchestratorExtended,
)
from omnimemory.models.contracts.model_contract_reducer_extended import (
    ModelContractReducerExtended,
)

__all__ = [
    # Re-exported from omnibase_core for convenience
    "ModelHandlerRoutingEntry",
    "ModelHandlerRoutingSubcontract",
    # Local extended contracts
    "MixinHandlerRouting",
    "ModelContractComputeExtended",
    "ModelContractEffectExtended",
    "ModelContractOrchestratorExtended",
    "ModelContractReducerExtended",
]
