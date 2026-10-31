# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""In-memory store adapter injected into the mock retrieval handlers (OMN-21762).

The mock Qdrant, DB and graph handlers keep no state of their own: seeded
snapshots, embeddings and graph edges live in this collaborator, so a public
handler method never assigns to the handler or mutates one of its attributes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibase_core.models.omnimemory import ModelMemorySnapshot

    from ..models import ModelGraphRelationship

logger = logging.getLogger(__name__)

__all__ = ["MockRetrievalStore"]


class MockRetrievalStore:
    """Snapshots, embeddings and graph edges held for one mock retrieval handler."""

    def __init__(self) -> None:
        self.snapshots: dict[str, ModelMemorySnapshot] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.relationships: list[ModelGraphRelationship] = []
        # id -> [(target, type, weight)]
        self.adjacency: dict[str, list[tuple[str, str, float]]] = {}

    def seed_snapshots(self, snapshots: Sequence[ModelMemorySnapshot]) -> list[str]:
        """Store the snapshots and return the ids stored.

        Snapshots with invalid or empty IDs are skipped with a warning.
        """
        stored: list[str] = []
        for snapshot in snapshots:
            if not snapshot.snapshot_id or not str(snapshot.snapshot_id).strip():
                logger.warning(
                    "Skipping snapshot with invalid/empty ID: %r",
                    snapshot.snapshot_id,
                )
                continue
            snapshot_id = str(snapshot.snapshot_id)
            self.snapshots[snapshot_id] = snapshot
            stored.append(snapshot_id)
        return stored

    def put_embedding(self, snapshot_id: str, embedding: list[float]) -> None:
        """Store one embedding under its snapshot id."""
        self.embeddings[snapshot_id] = embedding

    def ensure_node(self, snapshot_id: str) -> None:
        """Give a snapshot id an adjacency entry when it has none."""
        if snapshot_id not in self.adjacency:
            self.adjacency[snapshot_id] = []

    def put_edge(
        self,
        relationship: ModelGraphRelationship,
        *,
        bidirectional: bool,
    ) -> None:
        """Record a relationship and its adjacency entries."""
        self.relationships.append(relationship)
        self.ensure_node(relationship.source_id)
        self.adjacency[relationship.source_id].append(
            (
                relationship.target_id,
                relationship.relationship_type,
                relationship.weight,
            )
        )
        if bidirectional:
            self.ensure_node(relationship.target_id)
            self.adjacency[relationship.target_id].append(
                (
                    relationship.source_id,
                    relationship.relationship_type,
                    relationship.weight,
                )
            )

    def reset(self) -> None:
        """Drop everything held."""
        self.snapshots.clear()
        self.embeddings.clear()
        self.relationships.clear()
        self.adjacency.clear()
