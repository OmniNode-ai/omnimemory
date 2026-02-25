# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Mock Qdrant Handler for semantic search operations.

This module provides a mock handler that simulates `HandlerQdrant` behavior
for semantic similarity search. It allows development and testing of the
memory_retrieval_effect node without requiring a running Qdrant instance.

The mock uses simple text matching to simulate similarity scores, making it
suitable for unit tests and local development. When the real Qdrant service
is available, this can be swapped for the real HandlerQdrant.

Optionally, the handler can use a real MLX embedding server for generating
embeddings instead of mock embeddings. This enables more realistic semantic
search during development while still using the mock storage backend.

Example::

    import asyncio
    import os
    from omnimemory.nodes.memory_retrieval_effect.handlers import (
        HandlerQdrantMock,
        ModelHandlerQdrantMockConfig,
    )

    async def example():
        # Use mock embeddings (default)
        config = ModelHandlerQdrantMockConfig()
        handler = HandlerQdrantMock(config)
        await handler.initialize()

        # Or use real MLX embeddings (URL from environment variable - REQUIRED)
        embedding_url = os.environ["OMNIMEMORY__EMBEDDING__SERVER_URL"]
        config_real = ModelHandlerQdrantMockConfig(
            use_real_embeddings=True,
            embedding_server_url=embedding_url,
        )
        handler_real = HandlerQdrantMock(config_real)
        await handler_real.initialize()

        # Seed with test data
        handler.seed_snapshots([snapshot1, snapshot2])

        # Search
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="authentication decision",
            limit=5,
        )
        response = await handler.execute(request)

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import TYPE_CHECKING

from omnibase_core.models.omnimemory import (
    ModelMemorySnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

from ..clients.embedding_client import (
    EmbeddingClient,
    EmbeddingClientError,
    EmbeddingConnectionError,
)
from ..models import (
    ModelEmbeddingClientConfig,
    ModelHandlerQdrantMockConfig,
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingClientError",
    "EmbeddingConnectionError",
    "HandlerQdrantMock",
    "ModelHandlerQdrantMockConfig",
]


class HandlerQdrantMock:
    """Mock handler that simulates HandlerQdrant for semantic search.

    This handler provides a development-friendly interface for testing
    semantic search functionality without requiring a real Qdrant instance.
    It uses simple text similarity (word overlap) to simulate vector similarity.

    The handler maintains an in-memory store of snapshots and can be seeded
    with test data for reproducible testing.

    Attributes:
        config: The handler configuration.

    Example::

        async def example():
            handler = HandlerQdrantMock(ModelHandlerQdrantMockConfig())
            await handler.initialize()

            # Seed test data
            handler.seed_snapshots([snapshot1, snapshot2])

            # Execute search
            request = ModelMemoryRetrievalRequest(
                operation="search",
                query_text="authentication",
            )
            response = await handler.execute(request)
    """

    def __init__(self, config: ModelHandlerQdrantMockConfig) -> None:
        """Initialize the mock handler with configuration.

        Args:
            config: The handler configuration.
        """
        self._config = config
        self._snapshots: dict[str, ModelMemorySnapshot] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._embedding_client: EmbeddingClient | None = None

    @property
    def config(self) -> ModelHandlerQdrantMockConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    @property
    def snapshot_count(self) -> int:
        """Get the number of stored snapshots."""
        return len(self._snapshots)

    async def initialize(self) -> None:
        """Initialize the mock handler.

        Thread-safe: Uses asyncio.Lock to prevent concurrent initialization.
        If use_real_embeddings is True, connects to the MLX embedding server.

        FAIL-FAST POLICY: If use_real_embeddings is True but configuration
        is invalid or the server is unreachable, this method raises an
        exception immediately. NO silent fallback to mock embeddings.

        Raises:
            ValueError: If use_real_embeddings is True but embedding_server_url
                is empty or invalid.
            EmbeddingConnectionError: If the embedding server is unreachable.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            # Initialize embedding client if real embeddings are requested
            if self._config.use_real_embeddings:
                # Validate configuration - fail fast on invalid config
                if not self._config.embedding_server_url:
                    raise ValueError(
                        "embedding_server_url is required when use_real_embeddings=True"
                    )
                if not self._config.embedding_server_url.startswith(
                    ("http://", "https://")
                ):
                    raise ValueError(
                        f"embedding_server_url must be a valid HTTP(S) URL, "
                        f"got: {self._config.embedding_server_url!r}"
                    )

                embedding_config = ModelEmbeddingClientConfig(
                    base_url=self._config.embedding_server_url,
                    timeout_seconds=self._config.embedding_timeout_seconds,
                    max_retries=self._config.embedding_max_retries,
                    embedding_dimension=self._config.embedding_dimension,
                )
                self._embedding_client = EmbeddingClient(embedding_config)
                # Connect to server - let errors propagate, NO fallback
                await self._embedding_client.connect()
                logger.info(
                    "Mock Qdrant handler initialized with real embeddings from %s",
                    self._config.embedding_server_url,
                )
            else:
                logger.info(
                    "Mock Qdrant handler initialized with mock embeddings, dim=%d",
                    self._config.embedding_dimension,
                )

            self._initialized = True

    def seed_snapshots(
        self,
        snapshots: Sequence[ModelMemorySnapshot],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Seed the mock store with test snapshots.

        Args:
            snapshots: List of snapshots to add to the mock store.
            embeddings: Optional pre-computed embeddings keyed by snapshot_id.
                If not provided, mock embeddings will be generated.

        Note:
            Snapshots with invalid or empty IDs are skipped with a warning.
        """
        valid_count = 0
        for snapshot in snapshots:
            # Validate snapshot ID is non-empty
            if not snapshot.snapshot_id or not str(snapshot.snapshot_id).strip():
                logger.warning(
                    "Skipping snapshot with invalid/empty ID: %r",
                    snapshot.snapshot_id,
                )
                continue

            snapshot_id = str(snapshot.snapshot_id)
            self._snapshots[snapshot_id] = snapshot

            # Use provided embedding or generate mock
            if embeddings and snapshot_id in embeddings:
                self._embeddings[snapshot_id] = embeddings[snapshot_id]
            else:
                self._embeddings[snapshot_id] = self._generate_mock_embedding(snapshot)

            valid_count += 1

        logger.debug("Seeded %d snapshots into mock store", valid_count)

    def clear(self) -> None:
        """Clear all snapshots from the mock store."""
        self._snapshots.clear()
        self._embeddings.clear()

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Execute a semantic search operation.

        Args:
            request: The retrieval request (must have operation="search").

        Returns:
            Response with search results ordered by similarity.

        Raises:
            ValueError: If operation is not "search".
        """
        if not self._initialized:
            await self.initialize()

        if request.operation != "search":
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: Only supports 'search', "
                    f"got '{request.operation}'"
                ),
            )

        # Simulate latency if configured
        if self._config.simulate_latency_ms > 0:
            await asyncio.sleep(self._config.simulate_latency_ms / 1000)

        # Get or generate query embedding
        if request.query_embedding is not None:
            query_embedding = request.query_embedding
        elif request.query_text is not None:
            query_embedding = await self._get_embedding(request.query_text)
        else:
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: Either query_text or query_embedding "
                    f"required for operation '{request.operation}'"
                ),
            )

        # Score all snapshots
        scored_results: list[tuple[str, float]] = []
        for snapshot_id, embedding in self._embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= request.similarity_threshold:
                scored_results.append((snapshot_id, similarity))

        # Sort by similarity (highest first) and limit
        scored_results.sort(key=lambda x: x[1], reverse=True)
        scored_results = scored_results[: request.limit]

        # Build response
        if not scored_results:
            return ModelMemoryRetrievalResponse(
                status="no_results",
                results=[],
                total_count=0,
                query_embedding_used=query_embedding,
            )

        results = [
            ModelSearchResult(
                snapshot=self._snapshots[snapshot_id],
                score=score,
                distance=1.0 - score,  # Convert similarity to distance
            )
            for snapshot_id, score in scored_results
        ]

        return ModelMemoryRetrievalResponse(
            status="success",
            results=results,
            total_count=len(results),
            query_embedding_used=query_embedding,
        )

    def _generate_mock_embedding(self, snapshot: ModelMemorySnapshot) -> list[float]:
        """Generate a mock embedding from snapshot content.

        This creates a deterministic pseudo-embedding based on the snapshot's
        text content, allowing consistent similarity matching in tests.

        Args:
            snapshot: The snapshot to embed.

        Returns:
            A mock embedding vector of the configured dimension.
        """
        # Combine text fields for embedding - extract subject_key for searchable text
        text_parts = []
        if snapshot.subject:
            if snapshot.subject.subject_key:
                text_parts.append(snapshot.subject.subject_key)
        if snapshot.tags:
            text_parts.extend(snapshot.tags)

        text = " ".join(text_parts).lower()
        return self._text_to_mock_embedding(text)

    def _text_to_mock_embedding(self, text: str) -> list[float]:
        """Generate deterministic mock embedding from text using hashlib.

        This creates a fully deterministic embedding based on MD5 hash,
        ensuring reproducible results across Python runs for testing.

        Args:
            text: The text to embed.

        Returns:
            A normalized mock embedding vector.
        """
        # Use deterministic hash (md5 is intentional for reproducible mock data, not security)
        text_hash = hashlib.md5(text.encode()).digest()  # noqa: S324
        dim = self._config.embedding_dimension
        # Convert to reproducible embedding
        embedding = [
            float((text_hash[i % len(text_hash)] % 100) / 100.0) for i in range(dim)
        ]
        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in embedding))
        return [x / norm if norm > 0 else 0.0 for x in embedding]

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, using real embeddings if configured.

        This method provides a unified interface for embedding generation.
        When use_real_embeddings is enabled and the embedding client is
        available, it will use the MLX embedding server. Otherwise, it
        uses deterministic mock embeddings.

        FAIL-FAST POLICY: If real embeddings are requested but the server
        fails, this method raises an exception. NO silent fallback to mock
        embeddings. This ensures production issues are visible immediately.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingClientError: If real embeddings are requested but the
                embedding server fails. Errors propagate without fallback.

        Example::

            async def search_example():
                handler = HandlerQdrantMock(ModelHandlerQdrantMockConfig())
                await handler.initialize()
                embedding = await handler._get_embedding("hello world")
        """
        # Real embeddings requested - no fallback, fail if server fails
        if self._embedding_client is not None:
            embedding = await self._embedding_client.get_embedding(text)
            logger.debug(
                "Generated real embedding for text (len=%d, dim=%d)",
                len(text),
                len(embedding),
            )
            return embedding

        # Mock mode - use mock embeddings
        return self._text_to_mock_embedding(text)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in range [-1, 1], clamped to [0, 1].
        """
        if len(a) != len(b):
            # Truncate or pad to match
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        similarity = dot_product / (magnitude_a * magnitude_b)
        # Clamp to [0, 1] for our use case
        return max(0.0, min(1.0, similarity))

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources.

        Closes the embedding client connection if one was created.
        """
        if self._initialized:
            # Close embedding client if it exists
            if self._embedding_client is not None:
                await self._embedding_client.close()
                self._embedding_client = None
                logger.debug("Embedding client closed")

            self._snapshots.clear()
            self._embeddings.clear()
            self._initialized = False
            logger.debug("Mock Qdrant handler shutdown complete")
