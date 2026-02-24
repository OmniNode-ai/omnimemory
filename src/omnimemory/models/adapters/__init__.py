# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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

FileSystem Models:
    - ModelFileSystemAdapterConfig: Configuration for filesystem storage adapter

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from omnimemory.models.adapters.model_connections_result import ModelConnectionsResult
from omnimemory.models.adapters.model_filesystem_adapter_config import (
    ModelFileSystemAdapterConfig,
)
from omnimemory.models.adapters.model_graph_memory_config import ModelGraphMemoryConfig
from omnimemory.models.adapters.model_graph_memory_health import ModelGraphMemoryHealth
from omnimemory.models.adapters.model_memory_connection import ModelMemoryConnection
from omnimemory.models.adapters.model_related_memory import (
    ModelRelatedMemory,
    PropertyValue,
)
from omnimemory.models.adapters.model_related_memory_result import (
    ModelRelatedMemoryResult,
)

__all__ = [
    "ModelConnectionsResult",
    "ModelFileSystemAdapterConfig",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    "PropertyValue",
]
