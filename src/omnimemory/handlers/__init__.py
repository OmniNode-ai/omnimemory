# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OmniMemory Handlers Package.

This package contains handler implementations and adapter layers for memory operations.
Handlers interface with infrastructure services (databases, graph stores, vector stores)
while adapters translate between memory-domain concepts and handler-level operations.

Available Handlers:
    - HandlerSubscription: Agent subscription and notification delivery management

Subpackages:
    - adapters: Adapter layers wrapping omnibase_infra handlers

Example::

    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )
    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        AdapterGraphMemoryConfig,
    )

.. versionadded:: 0.1.0
    HandlerSubscription added for OMN-1393.
"""

from omnimemory.handlers.adapters import (
    AdapterGraphMemory,
    AdapterGraphMemoryConfig,
    AdapterValkey,
    AdapterValkeyConfig,
    ModelConnectionsResult,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
    ModelValkeyHealth,
)
from omnimemory.handlers.handler_subscription import (
    HandlerSubscription,
    ModelHandlerSubscriptionConfig,
    ModelSubscriptionHealth,
)

__all__ = [
    # Subscription Handler
    "HandlerSubscription",
    "ModelHandlerSubscriptionConfig",
    "ModelSubscriptionHealth",
    # Graph Memory Adapter
    "AdapterGraphMemory",
    "AdapterGraphMemoryConfig",
    "ModelConnectionsResult",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    # Valkey Adapter
    "AdapterValkey",
    "AdapterValkeyConfig",
    "ModelValkeyHealth",
]
