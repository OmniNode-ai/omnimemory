# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Similarity Compute - ONEX COMPUTE Node (Core 8 Foundation).

Vector similarity calculations and ranking.

This node provides pure compute operations for comparing vectors using
various distance metrics (cosine, euclidean). It performs NO I/O operations.

Components:
    - NodeSimilarityCompute: ONEX COMPUTE node wrapping the handler
    - HandlerSimilarityCompute: Pure compute handler for vector similarity
    - ModelSimilarityComputeRequest: Request envelope for operations
    - ModelSimilarityComputeResponse: Response envelope with results

Example::

    from omnimemory.nodes.similarity_compute import (
        NodeSimilarityCompute,
        ModelSimilarityComputeRequest,
        ModelSimilarityComputeResponse,
    )
    from omnibase_core.container import ModelONEXContainer

    container = ModelONEXContainer()
    node = NodeSimilarityCompute(container)

    request = ModelSimilarityComputeRequest(
        operation="cosine_distance",
        vector_a=[0.1, 0.2, 0.3],
        vector_b=[0.4, 0.5, 0.6],
    )
    response = node.execute(request)
    print(f"Distance: {response.distance}")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1388.
"""

from omnimemory.nodes.similarity_compute.handlers import (
    HandlerSimilarityCompute,
    ModelHandlerSimilarityComputeConfig,
)
from omnimemory.nodes.similarity_compute.models import (
    ModelSimilarityComputeRequest,
    ModelSimilarityComputeResponse,
)
from omnimemory.nodes.similarity_compute.node_similarity_compute import (
    NodeSimilarityCompute,
)

__all__ = [
    # Node
    "NodeSimilarityCompute",
    # Models
    "ModelSimilarityComputeRequest",
    "ModelSimilarityComputeResponse",
    # Handler
    "HandlerSimilarityCompute",
    "ModelHandlerSimilarityComputeConfig",
]
