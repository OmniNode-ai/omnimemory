# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Adapter models for OmniMemory handler adapters.

This module provides Pydantic models used by handler adapters,
following ONEX naming conventions (all models prefixed with "Model").

Graph Memory Models:
    - ModelMemoryConnection: Connection between two memories
    - ModelRelatedMemory: Memory found through relationship traversal
    - ModelRelatedMemoryResult: Result of find_related operation
    - ModelConnectionsResult: Result of get_connections operation
    - ModelGraphMemoryHealth: Health status for graph memory adapter
    - ModelGraphMemoryConfig: Configuration for graph memory adapter

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from omnimemory.models.adapters.model_graph_memory import (
    AdapterGraphMemoryConfig,  # Backward compatibility alias
    ModelConnectionsResult,
    ModelGraphMemoryConfig,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
    PropertyValue,
)

__all__ = [
    # Graph memory models (ONEX naming)
    "ModelConnectionsResult",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    "PropertyValue",
    # Backward compatibility alias
    "AdapterGraphMemoryConfig",
]
