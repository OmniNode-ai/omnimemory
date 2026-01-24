# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for the similarity_compute node.

This module exports handlers for vector similarity computation operations.

Available Handlers:
    - HandlerSimilarityCompute: Pure compute handler for vector similarity
      and distance calculations (cosine, euclidean).

Example::

    from omnimemory.nodes.similarity_compute.handlers import (
        HandlerSimilarityCompute,
        ModelHandlerSimilarityComputeConfig,
    )

    config = ModelHandlerSimilarityComputeConfig()
    handler = HandlerSimilarityCompute(config)

    vec_a = [0.1, 0.2, 0.3]
    vec_b = [0.2, 0.3, 0.4]
    distance = handler.cosine_distance(vec_a, vec_b)

.. versionadded:: 0.1.0
    Initial handlers for OMN-1388.
"""

from omnimemory.nodes.similarity_compute.handlers.handler_similarity_compute import (
    HandlerSimilarityCompute,
)
from omnimemory.nodes.similarity_compute.models import (
    ModelHandlerSimilarityComputeConfig,
)

__all__ = [
    "HandlerSimilarityCompute",
    "ModelHandlerSimilarityComputeConfig",
]
