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

from __future__ import annotations

from typing import assert_never

from pydantic import BaseModel, ConfigDict, Field

from ..base import BaseComputeNode, ContainerType
from .handlers import (
    HandlerSimilarityCompute,
    ModelHandlerSimilarityComputeConfig,
    ModelSimilarityComputeHealth,
    ModelSimilarityComputeMetadata,
)
from .models import ModelSimilarityComputeRequest, ModelSimilarityComputeResponse

__all__ = [
    "ModelNodeSimilarityComputeHealth",
    "ModelNodeSimilarityComputeMetadata",
    "NodeSimilarityCompute",
]


class ModelNodeSimilarityComputeHealth(  # omnimemory-model-exempt: handler health
    BaseModel
):
    """Health status for the Similarity Compute Node.

    Returned by the node's health_check() method to provide detailed health
    information including the node's status and the underlying handler's health.

    Attributes:
        healthy: Whether the node is healthy.
        node: Node identifier string.
        handler: Detailed health status of the underlying handler.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    healthy: bool = Field(
        ...,
        description="Whether the node is healthy",
    )
    node: str = Field(
        ...,
        description="Node identifier string",
    )
    handler: ModelSimilarityComputeHealth = Field(
        ...,
        description="Detailed health status of the underlying handler",
    )


class ModelNodeSimilarityComputeMetadata(  # omnimemory-model-exempt: handler metadata
    BaseModel
):
    """Metadata describing similarity compute node capabilities and configuration.

    Returned by the node's describe() method to provide introspection information
    about the node's type, capabilities, and underlying handler details.

    Attributes:
        node_type: ONEX node type (COMPUTE for this node).
        node_name: Node identifier string.
        handler: Detailed metadata of the underlying handler.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    node_type: str = Field(
        ...,
        description="ONEX node type (COMPUTE for this node)",
    )
    node_name: str = Field(
        ...,
        description="Node identifier string",
    )
    handler: ModelSimilarityComputeMetadata = Field(
        ...,
        description="Detailed metadata of the underlying handler",
    )


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
        - Handler follows container-driven pattern

    Attributes:
        container: The ONEX container for dependency injection.

    Example::

        container = ModelONEXContainer()
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
        # Handler follows container-driven pattern
        # Config is provided via initialize() or uses defaults
        self._handler = HandlerSimilarityCompute(container)
        self._handler_initialized = False

    async def initialize(
        self,
        config: ModelHandlerSimilarityComputeConfig | None = None,
    ) -> None:
        """Initialize the node and its handler.

        Args:
            config: Optional handler configuration.
        """
        await self._handler.initialize(config)
        self._handler_initialized = True

    async def health_check(self) -> ModelNodeSimilarityComputeHealth:
        """Return health status of the node and handler.

        Returns:
            ModelNodeSimilarityComputeHealth with node and handler status.
        """
        handler_health = await self._handler.health_check()
        return ModelNodeSimilarityComputeHealth(
            healthy=handler_health.healthy,
            node="similarity_compute",
            handler=handler_health,
        )

    async def describe(self) -> ModelNodeSimilarityComputeMetadata:
        """Return node metadata and capabilities.

        Returns:
            ModelNodeSimilarityComputeMetadata describing the node's capabilities.
        """
        handler_desc = await self._handler.describe()
        return ModelNodeSimilarityComputeMetadata(
            node_type="COMPUTE",
            node_name="similarity_compute",
            handler=handler_desc,
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
