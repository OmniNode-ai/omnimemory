# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OmniMemory Handlers Package.

This package contains handler implementations and adapter layers for memory operations.
Handlers interface with infrastructure services (databases, graph stores, vector stores)
while adapters translate between memory-domain concepts and handler-level operations.

Available Handlers:
    - HandlerIntent: Direct protocol handler for intent storage and query operations
    - HandlerSubscription: Agent subscription and notification delivery management
    - HandlerSemanticCompute: Semantic analysis with policy hooks (lives in node's handlers/)

Subpackages:
    - adapters: Adapter layers wrapping omnibase_infra handlers

Note:
    HandlerSemanticCompute now lives in its node's handlers directory following ONEX patterns:
    ``omnimemory.nodes.node_semantic_analyzer_compute.handlers.handler_semantic_compute``

    It is re-exported here for import convenience.

Import Direction (CRITICAL):
    This module creates a one-way import dependency:

    ``omnimemory.handlers`` --> imports from --> ``omnimemory.nodes.*/handlers/``

    **Why this exists**: Re-exporting node handlers here provides a convenient single
    import location for consumers (``from omnimemory.handlers import HandlerSemanticCompute``).

    **Circular Import Warning**: Node handler modules (anything under ``nodes/*/handlers/``)
    MUST NOT import from ``omnimemory.handlers``. Doing so would create a circular import:

        - ``omnimemory.handlers.__init__`` imports ``omnimemory.nodes.X.handlers``
        - ``omnimemory.nodes.X.handlers`` imports ``omnimemory.handlers`` (CIRCULAR!)

    If a node handler needs types or utilities from the handlers package, import directly
    from the specific submodule instead:

        # WRONG (causes circular import)
        from omnimemory.handlers import HandlerIntent

        # CORRECT (direct import from specific module)
        from omnimemory.handlers.handler_intent import HandlerIntent

Example::

    from omnimemory.handlers import (
        HandlerIntent,
        ModelHandlerIntentMetadata,
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
        HandlerSemanticCompute,
        ModelHandlerSemanticComputeConfig,
    )
    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        ModelGraphMemoryConfig,
    )

.. versionadded:: 0.1.0
    HandlerSubscription added for OMN-1393.
    HandlerSemanticCompute added for OMN-1390.
    HandlerIntent added for OMN-1536.
"""

from omnimemory.handlers.adapters import (
    AdapterGraphMemory,
    AdapterValkey,
    AdapterValkeyConfig,
    ModelConnectionsResult,
    ModelGraphMemoryConfig,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
    ModelValkeyHealth,
)
from omnimemory.handlers.handler_intent import (
    HandlerIntent,
    ModelHandlerIntentMetadata,
)
from omnimemory.handlers.handler_subscription import (
    HandlerSubscription,
    ModelHandlerSubscriptionConfig,
    ModelSubscriptionHealth,
    ModelSubscriptionMetadata,
    ModelSubscriptionMetrics,
)
from omnimemory.handlers.models import ModelHandlerIntentConfig
from omnimemory.models.config import ModelHandlerSemanticComputeConfig

# Re-export from node's handlers directory for import convenience
from omnimemory.nodes.node_semantic_analyzer_compute.handlers import (
    HandlerSemanticCompute,
    HandlerSemanticComputePolicy,
)

__all__ = [
    # Intent Handler
    "HandlerIntent",
    "ModelHandlerIntentConfig",
    "ModelHandlerIntentMetadata",
    # Semantic Compute Handler
    "HandlerSemanticCompute",
    "HandlerSemanticComputePolicy",
    "ModelHandlerSemanticComputeConfig",
    # Subscription Handler
    "HandlerSubscription",
    "ModelHandlerSubscriptionConfig",
    "ModelSubscriptionHealth",
    "ModelSubscriptionMetadata",
    "ModelSubscriptionMetrics",
    # Graph Memory Adapter
    "AdapterGraphMemory",
    "ModelConnectionsResult",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    # Valkey Adapter
    "AdapterValkey",
    "AdapterValkeyConfig",
    "ModelValkeyHealth",
]
