# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OmniMemory Handlers Package.

This package contains handler implementations and adapter layers for memory operations.
Handlers interface with infrastructure services (databases, graph stores, vector stores)
while adapters translate between memory-domain concepts and handler-level operations.

Available Handlers:
    - HandlerIntent: Direct protocol handler for intent storage and query operations
    - HandlerSubscription: Agent subscription and notification delivery management
    - HandlerSemanticCompute: Semantic analysis with policy hooks

Subpackages:
    - adapters: Adapter layers wrapping omnibase_infra handlers

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
from omnimemory.handlers.handler_semantic_compute import (
    HandlerSemanticCompute,
    HandlerSemanticComputePolicy,
    ModelHandlerSemanticComputeConfig,
)
from omnimemory.handlers.handler_subscription import (
    HandlerSubscription,
    ModelHandlerSubscriptionConfig,
    ModelSubscriptionHealth,
    ModelSubscriptionMetrics,
)
from omnimemory.handlers.models import ModelHandlerIntentConfig

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
