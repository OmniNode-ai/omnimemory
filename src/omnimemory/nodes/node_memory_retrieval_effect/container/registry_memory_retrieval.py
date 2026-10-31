# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the memory retrieval handlers (OMN-21623).

The one place that builds the retrieval handler's collaborators. Callers
(the retrieval handler) take them from these
binders instead of constructing handler classes by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnimemory.nodes.node_memory_retrieval_effect.handlers import (
        HandlerActivationDecay,
        HandlerDbMock,
        HandlerGraphMock,
        HandlerQdrantMock,
        HandlerRankFusion,
    )
    from omnimemory.nodes.node_memory_retrieval_effect.models import (
        ModelHandlerDbMockConfig,
        ModelHandlerGraphMockConfig,
        ModelHandlerQdrantMockConfig,
    )

__all__ = [
    "bind_activation_decay",
    "bind_db_mock",
    "bind_graph_mock",
    "bind_qdrant_mock",
    "bind_rank_fusion",
]


def bind_rank_fusion() -> HandlerRankFusion:
    """Build the stateless rank fusion collaborator."""
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        HandlerRankFusion,
    )

    return HandlerRankFusion()


def bind_activation_decay() -> HandlerActivationDecay:
    """Build the stateless activation decay collaborator."""
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_decay import (
        HandlerActivationDecay,
    )

    return HandlerActivationDecay()


def bind_qdrant_mock(config: ModelHandlerQdrantMockConfig) -> HandlerQdrantMock:
    """Build the mock vector-search handler."""
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant_mock import (
        HandlerQdrantMock,
    )

    return HandlerQdrantMock(config)


def bind_db_mock(config: ModelHandlerDbMockConfig) -> HandlerDbMock:
    """Build the mock full-text-search handler."""
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_db_mock import (
        HandlerDbMock,
    )

    return HandlerDbMock(config)


def bind_graph_mock(config: ModelHandlerGraphMockConfig) -> HandlerGraphMock:
    """Build the mock graph-traversal handler."""
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_graph_mock import (
        HandlerGraphMock,
    )

    return HandlerGraphMock(config)
