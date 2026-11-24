# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the memory retrieval handlers (OMN-21623, OMN-26659).

The one place that builds the retrieval handler and its collaborators. Callers
take them from these binders instead of constructing handler classes by hand.

Stores are resolved from ``ModelRetrievalStoreSettings``. A store the settings do
not describe is left out, and the handler refuses a search that needs it,
naming the store. No binder here builds a stand-in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnimemory.nodes.node_memory_retrieval_effect.handlers import (
        HandlerActivationDecay,
        HandlerMemoryRetrieval,
        HandlerQdrant,
        HandlerRankFusion,
    )
    from omnimemory.nodes.node_memory_retrieval_effect.models.model_retrieval_store_settings import (
        ModelRetrievalStoreSettings,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "bind_activation_decay",
    "bind_memory_retrieval",
    "bind_rank_fusion",
    "bind_vector_store",
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


def bind_vector_store(settings: ModelRetrievalStoreSettings) -> HandlerQdrant | None:
    """Build the Qdrant vector store the settings describe.

    No connection is made here; the retrieval handler connects it on first use,
    so an unreachable Qdrant is a refusal at search time, not a boot failure.

    Returns:
        The store, or None when the host or the embedding server is not set
        (or the values do not form a valid config; the reason is logged).
    """
    from pydantic import ValidationError

    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant import (
        HandlerQdrant,
    )
    from omnimemory.nodes.node_memory_retrieval_effect.models import (
        ModelHandlerQdrantConfig,
    )

    if not settings.qdrant_host or not settings.embedding_server_url:
        return None
    try:
        if settings.qdrant_port is None:
            config = ModelHandlerQdrantConfig(
                qdrant_host=settings.qdrant_host,
                embedding_server_url=settings.embedding_server_url,
            )
        else:
            config = ModelHandlerQdrantConfig(
                qdrant_host=settings.qdrant_host,
                qdrant_port=settings.qdrant_port,
                embedding_server_url=settings.embedding_server_url,
            )
    except ValidationError:
        logger.exception("Vector store settings are not a valid Qdrant config")
        return None
    return HandlerQdrant(config)


def bind_memory_retrieval(
    settings: ModelRetrievalStoreSettings,
) -> HandlerMemoryRetrieval:
    """Build the retrieval handler from the stores the settings describe.

    The database and graph stores are not resolved here: no implementation of
    their protocols exists yet, so a ``search_text`` or ``search_graph`` request
    is refused naming that store.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_memory_retrieval import (
        HandlerMemoryRetrieval,
    )

    return HandlerMemoryRetrieval(vector_store=bind_vector_store(settings))
