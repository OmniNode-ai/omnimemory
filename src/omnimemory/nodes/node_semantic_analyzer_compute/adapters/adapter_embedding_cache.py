# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""LRU embedding cache held by the semantic compute handler (OMN-21762).

The handler creates this collaborator when it initializes and only reads and
writes through it, so its public methods never assign to the handler or call a
mutating method on one of its attributes.
"""

from __future__ import annotations

from cachetools import LRUCache

__all__ = ["EmbeddingLruCache"]


class EmbeddingLruCache:
    """Bounded embedding cache keyed by content hash."""

    def __init__(self, max_size: int) -> None:
        self._entries: LRUCache[str, list[float]] = LRUCache(maxsize=max_size)

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return len(self._entries)

    @property
    def max_size(self) -> int:
        """Maximum number of cached embeddings before eviction."""
        return int(self._entries.maxsize)

    def contains(self, key: str) -> bool:
        """Whether an embedding is cached under the key."""
        return key in self._entries

    def get(self, key: str) -> list[float]:
        """Return a copy of the embedding cached under the key."""
        return list(self._entries[key])

    def put(self, key: str, embedding: list[float]) -> None:
        """Cache an embedding, evicting the least recently used when full."""
        self._entries[key] = embedding

    def reset(self) -> None:
        """Drop every cached embedding."""
        self._entries.clear()
