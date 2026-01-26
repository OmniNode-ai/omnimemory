# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Extended contract models with ONEX infra extension support.

This module provides extended contract validation models that add support for
ONEX infrastructure extension fields not yet in omnibase_core. This is a
temporary workaround until OMN-1588 is resolved.

Extended Fields:
    - handler_routing: Declarative handler dispatch configuration

Usage::

    from omnimemory.models.contracts import (
        ModelContractEffectExtended,
        ModelContractComputeExtended,
        ModelContractReducerExtended,
        ModelContractOrchestratorExtended,
        MixinHandlerRouting,
    )

    # Use in test validation
    ModelContractEffectExtended(**contract_data)

    # Create custom extended contracts
    class MyExtendedContract(MixinHandlerRouting, SomeBaseContract):
        pass

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

# TODO(OMN-1588): Remove this entire module when omnibase_core adds native handler_routing support

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
from omnimemory.models.contracts.model_event_model import ModelEventModel
from omnimemory.models.contracts.model_handler_routing import ModelHandlerRouting
from omnimemory.models.contracts.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnimemory.models.contracts.model_handler_spec import ModelHandlerSpec

__all__ = [
    "MixinHandlerRouting",
    "ModelContractComputeExtended",
    "ModelContractEffectExtended",
    "ModelContractOrchestratorExtended",
    "ModelContractReducerExtended",
    "ModelEventModel",
    "ModelHandlerRouting",
    "ModelHandlerRoutingEntry",
    "ModelHandlerSpec",
]
