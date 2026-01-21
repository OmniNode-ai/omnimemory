# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Similarity Compute Node - ONEX COMPUTE node for vector similarity.

This module provides the ONEX-compliant COMPUTE node for vector similarity
calculations. Following ONEX patterns, the node is a thin wrapper around
the handler - all business logic lives in the handler.

Node Type: COMPUTE
- Pure transformations (no I/O operations)
- Stateless execution
- Deterministic results

Example::

    from omnimemory.nodes.similarity_compute import (
        NodeSimilarityCompute,
        ModelSimilarityComputeRequest,
    )
    from omnimemory.compat import ModelOnexContainer

    container = ModelOnexContainer()
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

from __future__ import annotations

from typing import assert_never

from ..base import BaseComputeNode, ContainerType
from .handlers import HandlerSimilarityCompute, ModelHandlerSimilarityComputeConfig
from .models import ModelSimilarityComputeRequest, ModelSimilarityComputeResponse

__all__ = [
    "NodeSimilarityCompute",
]


class NodeSimilarityCompute(BaseComputeNode):
    """COMPUTE node for vector similarity calculations.

    This node performs pure vector math with no I/O operations. It wraps
    the HandlerSimilarityCompute handler and provides a consistent ONEX
    interface for similarity operations.

    Supported operations:
        - cosine_distance: Calculate cosine distance between two vectors
        - euclidean_distance: Calculate Euclidean (L2) distance
        - compare: Full comparison with optional threshold matching

    Following ONEX patterns:
        - Node is a thin wrapper (minimal logic)
        - All business logic is in the handler
        - Error handling converts exceptions to error responses

    Attributes:
        container: The ONEX container for dependency injection.

    Example::

        container = ModelOnexContainer()
        node = NodeSimilarityCompute(container)

        # Cosine distance
        request = ModelSimilarityComputeRequest(
            operation="cosine_distance",
            vector_a=[1.0, 0.0],
            vector_b=[0.0, 1.0],
        )
        response = node.execute(request)
        assert response.status == "success"
        assert response.distance == 1.0  # Orthogonal vectors

        # Compare with threshold
        request = ModelSimilarityComputeRequest(
            operation="compare",
            vector_a=[0.5, 0.5],
            vector_b=[0.6, 0.4],
            metric="cosine",
            threshold=0.1,
        )
        response = node.execute(request)
        assert response.is_match is not None
    """

    def __init__(self, container: ContainerType) -> None:
        """Initialize the node with container injection.

        Args:
            container: ONEX container for dependency injection.
        """
        super().__init__(container)
        self._handler = HandlerSimilarityCompute(
            config=ModelHandlerSimilarityComputeConfig()
        )

    def execute(
        self,
        request: ModelSimilarityComputeRequest,
    ) -> ModelSimilarityComputeResponse:
        """Execute similarity compute operation.

        Routes the request to the appropriate handler method based on
        the operation type.

        Args:
            request: The compute request with operation and vectors.

        Returns:
            Compute response with results or error information.
        """
        try:
            match request.operation:
                case "cosine_distance":
                    distance = self._handler.cosine_distance(
                        request.vector_a,
                        request.vector_b,
                    )
                    return ModelSimilarityComputeResponse(
                        status="success",
                        distance=distance,
                        dimensions=len(request.vector_a),
                    )

                case "euclidean_distance":
                    distance = self._handler.euclidean_distance(
                        request.vector_a,
                        request.vector_b,
                    )
                    return ModelSimilarityComputeResponse(
                        status="success",
                        distance=distance,
                        dimensions=len(request.vector_a),
                    )

                case "compare":
                    result = self._handler.compare(
                        request.vector_a,
                        request.vector_b,
                        metric=request.metric,
                        threshold=request.threshold,
                    )
                    return ModelSimilarityComputeResponse(
                        status="success",
                        distance=result.distance,
                        similarity=result.similarity,
                        is_match=result.is_match,
                        dimensions=result.dimensions,
                        notes=result.notes,
                    )

                case _:
                    assert_never(request.operation)

        except ValueError as e:
            return ModelSimilarityComputeResponse(
                status="error",
                error_message=str(e),
            )
