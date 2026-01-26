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

OMN-1588 Cleanup Checklist
--------------------------

When OMN-1588 is resolved and omnibase_core adds native ``handler_routing``
support, this entire module and related files should be removed. Use this
checklist to ensure complete cleanup:

**Files to DELETE (this module):**

- ``contracts/__init__.py`` - This file (replace with re-exports from omnibase_core)
- ``contracts/mixin_handler_routing.py`` - Mixin class (use omnibase_core version)
- ``contracts/model_contract_compute_extended.py`` - Extended compute contract
- ``contracts/model_contract_effect_extended.py`` - Extended effect contract
- ``contracts/model_contract_orchestrator_extended.py`` - Extended orchestrator contract
- ``contracts/model_contract_reducer_extended.py`` - Extended reducer contract
- ``contracts/model_handler_routing.py`` - Handler routing model
- ``contracts/model_handler_routing_entry.py`` - Routing entry model
- ``contracts/model_handler_spec.py`` - Handler spec model
- ``contracts/model_event_model.py`` - Event model

**Files to UPDATE:**

- ``tests/test_contract_validation.py`` - Update imports to use omnibase_core contracts

**Cleanup Steps:**

1. Verify omnibase_core provides equivalent ``handler_routing`` field support
2. Update all imports in the codebase from ``omnimemory.models.contracts``
   to the appropriate ``omnibase_core`` imports
3. Delete all files listed above
4. Run full test suite to verify no regressions
5. Search codebase for any remaining ``TODO(OMN-1588)`` comments

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
