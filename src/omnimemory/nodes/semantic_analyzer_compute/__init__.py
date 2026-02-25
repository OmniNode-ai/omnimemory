# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Semantic Analyzer Compute - ONEX COMPUTE Node (Core 8 Foundation).

Semantic analysis, embedding generation, and entity extraction.

This node provides compute operations for semantic analysis including:
- Embedding generation via injected ProtocolEmbeddingProvider
- Entity extraction (heuristic or LLM-backed)
- Full semantic analysis combining embeddings, entities, and topics

Components:
    - NodeSemanticAnalyzerCompute: ONEX COMPUTE node wrapping the handler
    - HandlerSemanticCompute: Compute handler for semantic operations
    - ModelSemanticAnalyzerComputeRequest: Request envelope for operations
    - ModelSemanticAnalyzerComputeResponse: Response envelope with results

Example::

    from omnimemory.nodes.semantic_analyzer_compute import (
        NodeSemanticAnalyzerCompute,
        ModelSemanticAnalyzerComputeRequest,
    )
    from omnibase_core.container import ModelONEXContainer

    container = ModelONEXContainer()
    node = NodeSemanticAnalyzerCompute(
        container=container,
        embedding_provider=my_embedding_provider,
    )

    request = ModelSemanticAnalyzerComputeRequest(
        operation="embed",
        content="Hello, world!",
    )
    response = await node.execute(request)
    print(f"Embedding dimension: {response.embedding_dimension}")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1390.
"""

from omnimemory.nodes.semantic_analyzer_compute.handlers import (
    HandlerSemanticCompute,
    HandlerSemanticComputePolicy,
    ModelHandlerSemanticComputeConfig,
)
from omnimemory.nodes.semantic_analyzer_compute.models import (
    ModelSemanticAnalyzerComputeRequest,
    ModelSemanticAnalyzerComputeResponse,
)
from omnimemory.nodes.semantic_analyzer_compute.node_semantic_analyzer_compute import (
    NodeSemanticAnalyzerCompute,
)

__all__ = [
    # Node
    "NodeSemanticAnalyzerCompute",
    # Models
    "ModelSemanticAnalyzerComputeRequest",
    "ModelSemanticAnalyzerComputeResponse",
    # Handler
    "HandlerSemanticCompute",
    "ModelHandlerSemanticComputeConfig",
    "HandlerSemanticComputePolicy",
]
