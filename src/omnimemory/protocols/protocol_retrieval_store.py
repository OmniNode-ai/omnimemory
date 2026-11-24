# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Protocols for the stores a memory search is answered from (OMN-26659).

``HandlerMemoryRetrieval`` holds no store of its own. The vector, database and
graph stores are handed to it behind these protocols, resolved from settings by
the container binder. A store that is absent is a refusal that names it; the
handler never builds a stand-in in its place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnimemory.nodes.node_memory_retrieval_effect.models import (
        ModelMemoryRetrievalRequest,
        ModelMemoryRetrievalResponse,
    )

__all__ = [
    "ProtocolDatabaseStore",
    "ProtocolGraphStore",
    "ProtocolRetrievalStore",
    "ProtocolVectorStore",
]


@runtime_checkable
class ProtocolRetrievalStore(Protocol):
    """A store that connects on first use and answers retrieval requests."""

    async def initialize(self) -> None:
        """Connect to the backing service. Idempotent; raises when unreachable."""
        ...

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Answer one retrieval request from the backing service."""
        ...

    async def shutdown(self) -> None:
        """Release the connection."""
        ...


@runtime_checkable
class ProtocolVectorStore(ProtocolRetrievalStore, Protocol):
    """Semantic similarity search (``search``)."""


@runtime_checkable
class ProtocolDatabaseStore(ProtocolRetrievalStore, Protocol):
    """Full-text search (``search_text``)."""


@runtime_checkable
class ProtocolGraphStore(ProtocolRetrievalStore, Protocol):
    """Relationship traversal (``search_graph``)."""
