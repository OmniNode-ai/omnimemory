# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Effect handlers.

This package contains IHandler implementations for the memory_retrieval_effect node.
Currently provides mock handlers for development and testing. Real handlers
wrapping omnibase_infra can be added when infrastructure is available.

Mock Handlers:
    - HandlerQdrantMock: Simulates semantic similarity search
    - HandlerDbMock: Simulates full-text SQL search
    - HandlerGraphMock: Simulates graph traversal

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from .handler_db_mock import HandlerDbMock, HandlerDbMockConfig
from .handler_graph_mock import (
    GraphRelationship,
    HandlerGraphMock,
    HandlerGraphMockConfig,
)
from .handler_qdrant_mock import HandlerQdrantMock, HandlerQdrantMockConfig

__all__ = [
    # Qdrant (semantic search)
    "HandlerQdrantMock",
    "HandlerQdrantMockConfig",
    # Database (full-text search)
    "HandlerDbMock",
    "HandlerDbMockConfig",
    # Graph (traversal)
    "HandlerGraphMock",
    "HandlerGraphMockConfig",
    "GraphRelationship",
]
