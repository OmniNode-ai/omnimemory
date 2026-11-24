# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""HandlerMemoryRetrieval wired to in-memory stand-in stores, for tests only (OMN-26659).

The runtime builds its retrieval handler from settings and refuses by store name
when a store is absent. Tests that need rows to come back build one here instead,
with the three stand-ins injected through the same constructor the runtime uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnimemory.enums.enum_memory_store import EnumMemoryStore
from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_memory_retrieval import (
    HandlerMemoryRetrieval,
)
from tests.stand_ins.stand_in_database_config import ModelHandlerDbMockConfig
from tests.stand_ins.stand_in_database_store import HandlerDbMock
from tests.stand_ins.stand_in_graph_config import ModelHandlerGraphMockConfig
from tests.stand_ins.stand_in_graph_store import HandlerGraphMock
from tests.stand_ins.stand_in_vector_config import ModelHandlerQdrantMockConfig
from tests.stand_ins.stand_in_vector_store import HandlerQdrantMock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibase_core.models.omnimemory import ModelMemorySnapshot

__all__ = ["StandInHandlerMemoryRetrieval"]


class StandInHandlerMemoryRetrieval(HandlerMemoryRetrieval):
    """The production retrieval handler with stand-in stores injected."""

    def __init__(
        self,
        vector_config: ModelHandlerQdrantMockConfig | None = None,
        database_config: ModelHandlerDbMockConfig | None = None,
        graph_config: ModelHandlerGraphMockConfig | None = None,
    ) -> None:
        self.vector = HandlerQdrantMock(vector_config or ModelHandlerQdrantMockConfig())
        self.database = HandlerDbMock(database_config or ModelHandlerDbMockConfig())
        self.graph = HandlerGraphMock(graph_config or ModelHandlerGraphMockConfig())
        super().__init__(
            vector_store=self.vector,
            database_store=self.database,
            graph_store=self.graph,
        )

    @property
    def is_initialized(self) -> bool:
        """True once any store has been connected."""
        return bool(self._connected)

    async def initialize(self) -> None:
        """Connect all three stand-ins."""
        for store in (
            EnumMemoryStore.VECTOR,
            EnumMemoryStore.DATABASE,
            EnumMemoryStore.GRAPH,
        ):
            refusal = await self._connect(store)
            if refusal is not None:
                raise RuntimeError(refusal.refusal.detail if refusal.refusal else "")

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

    def seed_snapshots(
        self,
        snapshots: Sequence[ModelMemorySnapshot],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Seed every stand-in with the same snapshots."""
        self._require_initialized()
        self.vector.seed_snapshots(snapshots, embeddings)
        self.database.seed_snapshots(snapshots)
        self.graph.seed_snapshots(snapshots)

    def add_graph_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add an edge to the graph stand-in."""
        self._require_initialized()
        self.graph.add_relationship(source_id, target_id, relationship_type, weight)

    def reset(self) -> None:
        """Drop all data from every stand-in."""
        self._require_initialized()
        self.vector.reset()
        self.database.reset()
        self.graph.reset()
