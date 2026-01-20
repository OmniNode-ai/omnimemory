# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OmniMemory Handlers Package.

This package contains handler implementations and adapter layers for memory operations.
Handlers interface with infrastructure services (databases, graph stores, vector stores)
while adapters translate between memory-domain concepts and handler-level operations.

Subpackages:
    - adapters: Adapter layers wrapping omnibase_infra handlers

Example::

    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        AdapterGraphMemoryConfig,
    )
"""

from omnimemory.handlers.adapters import (
    AdapterGraphMemory,
    AdapterGraphMemoryConfig,
    ModelConnectionsResult,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
)

__all__ = [
    "AdapterGraphMemory",
    "AdapterGraphMemoryConfig",
    "ModelConnectionsResult",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
]
