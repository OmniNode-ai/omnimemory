# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Effect handlers.

This package contains IHandler implementations for the memory_retrieval_effect node.
Currently provides mock handlers for development and testing. Real handlers
wrapping omnibase_infra can be added when infrastructure is available.

Handlers:
    - HandlerMemoryRetrieval: Main routing handler that dispatches to backend handlers
    - HandlerQdrantMock: Simulates semantic similarity search
    - HandlerDbMock: Simulates full-text SQL search
    - HandlerGraphMock: Simulates graph traversal

Ranking:
    - HandlerRankFusion: Relevance gating and rank fusion for search_hybrid.
    - HandlerActivationDecay: Activation decay applied after rank fusion.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.

.. versionadded:: 0.18.0
    Rank fusion for hybrid retrieval (OMN-16765).
"""

from ..models import (
    ModelHandlerDbMockConfig,
    ModelHandlerGraphMockConfig,
    ModelHandlerMemoryRetrievalConfig,
    ModelHandlerQdrantMockConfig,
)
from .handler_db_mock import HandlerDbMock
from .handler_decay import HandlerActivationDecay
from .handler_fusion import (
    MIN_LEXICAL_TS_RANK,
    MIN_VECTOR_COSINE,
    RRF_K_DEFAULT,
    HandlerRankFusion,
)
from .handler_graph_mock import (
    HandlerGraphMock,
    HandlerGraphRelationship,
)
from .handler_memory_retrieval import HandlerMemoryRetrieval
from .handler_qdrant_mock import HandlerQdrantMock

__all__ = [
    # Main routing handler
    "HandlerMemoryRetrieval",
    "ModelHandlerMemoryRetrievalConfig",
    # Rank fusion - hybrid search (OMN-16765)
    "MIN_LEXICAL_TS_RANK",
    "MIN_VECTOR_COSINE",
    "RRF_K_DEFAULT",
    "HandlerRankFusion",
    "HandlerActivationDecay",
    # Qdrant - semantic search
    "HandlerQdrantMock",
    "ModelHandlerQdrantMockConfig",
    # Database - full-text search
    "HandlerDbMock",
    "ModelHandlerDbMockConfig",
    # Graph - traversal
    "HandlerGraphMock",
    "ModelHandlerGraphMockConfig",
    "HandlerGraphRelationship",
]
