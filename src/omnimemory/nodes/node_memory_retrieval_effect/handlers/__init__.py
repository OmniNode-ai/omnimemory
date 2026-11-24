# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Effect handlers.

Handlers:
    - HandlerMemoryRetrieval: Routes a request to the injected vector, database
      or graph store, or refuses by name when one is absent or unreachable
    - HandlerQdrant: The production vector store (Qdrant + embedding server)

Ranking:
    - HandlerRankFusion: Relevance gating and rank fusion for search_hybrid.
    - HandlerActivationDecay: Activation decay applied after rank fusion.

Test doubles for the stores are not part of this package; they live under
``tests/stand_ins`` (OMN-26659).

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.

.. versionadded:: 0.18.0
    Rank fusion for hybrid retrieval (OMN-16765).
"""

from .handler_decay import HandlerActivationDecay
from .handler_fusion import (
    MIN_LEXICAL_TS_RANK,
    MIN_VECTOR_COSINE,
    RRF_K_DEFAULT,
    HandlerRankFusion,
)
from .handler_memory_retrieval import HandlerMemoryRetrieval
from .handler_qdrant import HandlerQdrant

__all__ = [
    # Main routing handler
    "HandlerMemoryRetrieval",
    # Production vector store
    "HandlerQdrant",
    # Rank fusion - hybrid search (OMN-16765)
    "MIN_LEXICAL_TS_RANK",
    "MIN_VECTOR_COSINE",
    "RRF_K_DEFAULT",
    "HandlerRankFusion",
    "HandlerActivationDecay",
]
