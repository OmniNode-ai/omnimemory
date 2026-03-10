# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Production Qdrant handler for semantic search and document indexing.

Implements semantic similarity search and document ingestion backed by a real
Qdrant vector-store instance. Uses synchronous qdrant-client calls wrapped in
asyncio.to_thread to stay non-blocking.

Operations:
    search  — embed query text (or use pre-computed embedding) → cosine search
              → map scored hits to ModelMemoryRetrievalResponse
    index   — chunk document text → embed each chunk → upsert all points

Chunking strategy:
    1. Split on paragraph boundaries (\\n\\n).
    2. If a paragraph exceeds max_chunk_chars, further split on sentence
       boundaries ('. ', '? ', '! ').

Point IDs:
    str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_index}"))

Double-checked locking:
    initialize() uses asyncio.Lock to prevent concurrent initialisation.

Ticket: OMN-4475
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from omnibase_core.models.omnimemory import ModelMemorySnapshot

from ..clients.embedding_client import EmbeddingClient, ModelEmbeddingClientConfig
from ..models import (
    ModelHandlerQdrantConfig,
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
)

logger = logging.getLogger(__name__)

__all__ = ["HandlerQdrant"]

# Sentinel for "index" operation which is not part of the retrieval literal set
_OP_INDEX = "index"
_OP_SEARCH = "search"


class HandlerQdrant:
    """Production handler backed by a real Qdrant vector-store instance.

    Supports two operations:
    - ``search``: embed query text → cosine search → return ranked results.
    - ``index``: chunk text → embed chunks → upsert points to Qdrant.

    Attributes:
        config: Production Qdrant handler configuration.
    """

    def __init__(self, config: ModelHandlerQdrantConfig) -> None:
        """Initialise the handler with Qdrant configuration.

        Args:
            config: Production Qdrant and embedding server configuration.
        """
        self._config = config
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._embedding_client: EmbeddingClient | None = None
        self._qdrant_client: Any | None = None  # qdrant_client.QdrantClient

    @property
    def config(self) -> ModelHandlerQdrantConfig:
        """Return the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Return True if the handler has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Connect to Qdrant and the embedding server, create collection if absent.

        Double-checked locking prevents concurrent initialisation. Raises on any
        connection or configuration error — no silent fallback.

        Raises:
            RuntimeError: If the Qdrant client cannot be imported.
            qdrant_client.http.exceptions.ResponseHandlingException: On Qdrant errors.
            EmbeddingConnectionError: If the embedding server is unreachable.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                import qdrant_client  # runtime import for optional dep
                from qdrant_client.models import Distance, VectorParams
            except ImportError as exc:
                raise RuntimeError(
                    "qdrant-client is required for HandlerQdrant. "
                    "Install it with: pip install qdrant-client"
                ) from exc

            # Connect to embedding server
            embedding_config = ModelEmbeddingClientConfig(
                base_url=self._config.embedding_server_url,
                timeout_seconds=self._config.embedding_timeout_seconds,
                max_retries=self._config.embedding_max_retries,
            )
            self._embedding_client = EmbeddingClient(embedding_config)
            await self._embedding_client.connect()

            # Connect to Qdrant
            self._qdrant_client = qdrant_client.QdrantClient(
                host=self._config.qdrant_host,
                port=self._config.qdrant_port,
                timeout=int(self._config.qdrant_timeout_seconds),
            )

            # Create collection if it does not exist
            existing = await asyncio.to_thread(
                self._qdrant_client.collection_exists, self._config.collection_name
            )
            if not existing:
                await asyncio.to_thread(
                    self._qdrant_client.create_collection,
                    self._config.collection_name,
                    vectors_config=VectorParams(
                        size=self._config.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (vector_size=%d)",
                    self._config.collection_name,
                    self._config.vector_size,
                )
            else:
                logger.info(
                    "Qdrant collection '%s' already exists — skipping creation",
                    self._config.collection_name,
                )

            self._initialized = True
            logger.info(
                "HandlerQdrant initialized: host=%s, port=%d, collection=%s",
                self._config.qdrant_host,
                self._config.qdrant_port,
                self._config.collection_name,
            )

    async def shutdown(self) -> None:
        """Close connections and release resources."""
        if self._embedding_client is not None:
            await self._embedding_client.close()
            self._embedding_client = None
        if self._qdrant_client is not None:
            self._qdrant_client.close()
            self._qdrant_client = None
        self._initialized = False
        logger.debug("HandlerQdrant shutdown complete")

    async def execute(self, request: object) -> ModelMemoryRetrievalResponse:
        """Execute a search or index operation.

        Dispatches to the appropriate private method based on ``request.operation``.

        Args:
            request: For ``operation="search"``: a ``ModelMemoryRetrievalRequest``.
                     For ``operation="index"``: any object with attributes
                     ``document_id: str`` and ``content: str``.

        Returns:
            ModelMemoryRetrievalResponse (search returns results; index returns
            an empty success response).

        Raises:
            RuntimeError: If the handler has not been initialized.
            ValueError: If the operation is not supported.
        """
        if not self._initialized:
            raise RuntimeError(
                "HandlerQdrant must be initialized before use. "
                "Call await handler.initialize() first."
            )

        operation = getattr(request, "operation", None)
        if operation == _OP_SEARCH:
            search_req = ModelMemoryRetrievalRequest.model_validate(request)
            return await self._search(search_req)
        if operation == _OP_INDEX:
            return await self._index(request)

        return ModelMemoryRetrievalResponse(
            status="error",
            error_message=(
                f"HandlerQdrant: unsupported operation {operation!r}. "
                "Supported: 'search', 'index'."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _search(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Embed query and run cosine search against Qdrant."""
        assert self._embedding_client is not None  # guaranteed by initialize()
        assert self._qdrant_client is not None

        if request.query_embedding is not None:
            query_vector = request.query_embedding
        elif request.query_text is not None:
            query_vector = await self._embedding_client.get_embedding(
                request.query_text
            )
        else:
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message="search requires query_text or query_embedding",
            )

        hits = await asyncio.to_thread(
            self._qdrant_client.search,
            self._config.collection_name,
            query_vector=query_vector,
            limit=request.limit,
            score_threshold=request.similarity_threshold,
        )

        if not hits:
            return ModelMemoryRetrievalResponse(
                status="no_results",
                results=[],
                total_count=0,
                query_embedding_used=query_vector,
            )

        results = [
            ModelSearchResult(
                snapshot=ModelMemorySnapshot.model_validate(hit.payload),
                score=hit.score,
                distance=1.0 - hit.score,
            )
            for hit in hits
        ]
        return ModelMemoryRetrievalResponse(
            status="success",
            results=results,
            total_count=len(results),
            query_embedding_used=query_vector,
        )

    async def _index(self, request: object) -> ModelMemoryRetrievalResponse:
        """Chunk content, embed chunks, upsert to Qdrant."""
        assert self._embedding_client is not None
        assert self._qdrant_client is not None

        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required") from exc

        document_id: str = getattr(request, "document_id", "")
        content: str = getattr(request, "content", "")

        if not document_id or not content:
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message="index requires non-empty document_id and content",
            )

        chunks = self._chunk_text(content)
        points = []
        for chunk_index, chunk in enumerate(chunks):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_index}")
            )
            embedding = await self._embedding_client.get_embedding(chunk)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    },
                )
            )
            await asyncio.to_thread(
                self._qdrant_client.upsert,
                collection_name=self._config.collection_name,
                points=[points[-1]],
            )
            logger.debug(
                "Upserted chunk %d/%d for document_id=%s (point_id=%s)",
                chunk_index + 1,
                len(chunks),
                document_id,
                point_id,
            )

        return ModelMemoryRetrievalResponse(
            status="success",
            results=[],
            total_count=len(points),
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks at paragraph boundaries, then sentence boundaries.

        Args:
            text: The document text to chunk.

        Returns:
            List of non-empty text chunks.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self._config.max_chunk_chars:
                chunks.append(paragraph)
            else:
                # Split on sentence boundaries
                sentences: list[str] = []
                remainder = paragraph
                for separator in (". ", "? ", "! "):
                    parts = remainder.split(separator)
                    if len(parts) > 1:
                        # Rejoin with separator, then split into chunks
                        sentences = [p + separator for p in parts[:-1]] + [parts[-1]]
                        break
                if not sentences:
                    sentences = [paragraph]
                # Accumulate sentences into chunks <= max_chunk_chars
                current = ""
                for sentence in sentences:
                    if (
                        current
                        and len(current) + len(sentence) > self._config.max_chunk_chars
                    ):
                        if current.strip():
                            chunks.append(current.strip())
                        current = sentence
                    else:
                        current += sentence
                if current.strip():
                    chunks.append(current.strip())

        return chunks or [text]
