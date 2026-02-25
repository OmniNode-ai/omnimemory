# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Semantic Analyzer Compute Node - ONEX COMPUTE node for semantic analysis.

This module provides the ONEX-compliant COMPUTE node for semantic analysis
operations. Following ONEX patterns, the node is a thin wrapper around
the handler - all business logic lives in the handler.

Node Type: COMPUTE
- Orchestrates transformations (delegates I/O to providers)
- Supports deterministic and non-deterministic modes
- Async execution (providers are async)

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

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import BaseComputeNode, ContainerType
from .handlers import HandlerSemanticCompute, ModelHandlerSemanticComputeConfig
from .models import (
    ModelSemanticAnalyzerComputeRequest,
    ModelSemanticAnalyzerComputeResponse,
)

if TYPE_CHECKING:
    from ...protocols import ProtocolEmbeddingProvider, ProtocolLLMProvider

__all__ = [
    "NodeSemanticAnalyzerCompute",
]


class NodeSemanticAnalyzerCompute(BaseComputeNode):
    """COMPUTE node for semantic analysis operations.

    This node provides semantic analysis capabilities including embedding
    generation, entity extraction, and full semantic analysis. It wraps
    the HandlerSemanticCompute handler and provides a consistent ONEX
    interface.

    Unlike pure compute nodes, this node's operations are async because
    they delegate to external providers for embedding and LLM operations.

    Supported operations:
        - embed: Generate embedding vector for content
        - extract_entities: Extract named entities from content
        - analyze: Full semantic analysis

    Following ONEX patterns:
        - Node is a thin wrapper (minimal logic)
        - All business logic is in the handler
        - Error handling converts exceptions to error responses
        - Provider dependencies are injected

    Attributes:
        container: The ONEX container for dependency injection.

    Example::

        container = ModelONEXContainer()
        node = NodeSemanticAnalyzerCompute(
            container=container,
            embedding_provider=my_provider,
        )

        # Embed content
        request = ModelSemanticAnalyzerComputeRequest(
            operation="embed",
            content="Test content",
        )
        response = await node.execute(request)
        assert response.status == "success"
        assert response.embedding is not None
    """

    def __init__(
        self,
        container: ContainerType,
        embedding_provider: ProtocolEmbeddingProvider | None = None,
        llm_provider: ProtocolLLMProvider | None = None,
        config: ModelHandlerSemanticComputeConfig | None = None,
    ) -> None:
        """Initialize the node with container and provider injection.

        The handler is not ready until initialize() is called (either
        explicitly or via execute() which auto-initializes).

        Args:
            container: ONEX container for dependency injection.
            embedding_provider: Optional provider for embedding generation.
                If not provided, resolved from container during initialization.
            llm_provider: Optional provider for LLM-based operations.
                If not provided, resolved from container during initialization.
            config: Optional handler configuration.
        """
        super().__init__(container)
        self._handler = HandlerSemanticCompute(container=container)
        self._pending_config = config
        self._pending_embedding_provider = embedding_provider
        self._pending_llm_provider = llm_provider
        self._handler_initialized = False

    async def _ensure_handler_initialized(self) -> None:
        """Ensure the handler is initialized before operations."""
        if not self._handler_initialized:
            await self._handler.initialize(
                config=self._pending_config,
                embedding_provider=self._pending_embedding_provider,
                llm_provider=self._pending_llm_provider,
            )
            self._handler_initialized = True

    @property
    def handler(self) -> HandlerSemanticCompute:
        """Get the underlying handler."""
        return self._handler

    async def execute(
        self,
        request: ModelSemanticAnalyzerComputeRequest,
    ) -> ModelSemanticAnalyzerComputeResponse:
        """Execute semantic analysis operation.

        Routes the request to the appropriate handler method based on
        the operation type. Auto-initializes the handler if not already done.

        Args:
            request: The compute request with operation and content.

        Returns:
            Compute response with results or error information.
        """
        try:
            # Ensure handler is initialized (lazy initialization)
            await self._ensure_handler_initialized()

            match request.operation:
                case "embed":
                    embedding = await self._handler.embed(
                        content=request.content,
                        model=request.model,
                        correlation_id=request.correlation_id,
                    )
                    return ModelSemanticAnalyzerComputeResponse(
                        status="success",
                        operation="embed",
                        embedding=embedding,
                        embedding_dimension=len(embedding),
                        model_name=self._handler.embedding_provider.model_name,
                    )

                case "extract_entities":
                    entity_list = await self._handler.extract_entities(
                        content=request.content,
                        correlation_id=request.correlation_id,
                    )
                    return ModelSemanticAnalyzerComputeResponse(
                        status="success",
                        operation="extract_entities",
                        entities=entity_list,
                    )

                case "analyze":
                    result = await self._handler.analyze(
                        content=request.content,
                        analysis_type=request.analysis_type,
                        correlation_id=request.correlation_id,
                    )

                    # Use entity_list from the analysis result (already extracted
                    # during analyze()). No duplicate extraction needed.
                    return ModelSemanticAnalyzerComputeResponse(
                        status="success",
                        operation="analyze",
                        embedding=result.semantic_vector
                        if result.semantic_vector
                        else None,
                        embedding_dimension=len(result.semantic_vector)
                        if result.semantic_vector
                        else None,
                        entities=result.entity_list,
                        topics=result.topics,
                        key_concepts=result.key_concepts,
                        confidence_score=result.confidence_score,
                        complexity_score=result.complexity_score,
                        readability_score=result.readability_score,
                        result_id=result.result_id,
                        model_name=result.model_name,
                        processing_time_ms=result.processing_time_ms,
                    )

                case _:
                    # Defensive handling for type safety - Pydantic validates Literal types,
                    # but this provides a clear error if reached (e.g., deserialization bypass)
                    return ModelSemanticAnalyzerComputeResponse(
                        status="error",
                        operation=request.operation,
                        error_message=f"Unknown operation: {request.operation}",
                    )

        except ValueError as e:
            return ModelSemanticAnalyzerComputeResponse(
                status="error",
                operation=request.operation,
                error_message=str(e),
            )
        except Exception as e:
            return ModelSemanticAnalyzerComputeResponse(
                status="error",
                operation=request.operation,
                error_message=f"Unexpected error: {type(e).__name__}: {e}",
            )
