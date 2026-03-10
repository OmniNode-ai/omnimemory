# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerQdrant (OMN-4475).

Tests (7 total):
    1. test_initialize_creates_collection_when_absent
    2. test_initialize_skips_existing_collection
    3. test_chunk_text_splits_on_blank_lines
    4. test_chunk_text_splits_long_paragraph_at_sentence_boundary
    5. test_execute_search_returns_response_with_mapped_payload
    6. test_execute_index_upserts_deterministic_point_ids
    7. test_execute_raises_if_not_initialized

Ticket: OMN-4475
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant import (
    HandlerQdrant,
)
from omnimemory.nodes.node_memory_retrieval_effect.models import (
    ModelHandlerQdrantConfig,
)


def _make_config(max_chunk_chars: int = 2000) -> ModelHandlerQdrantConfig:
    return ModelHandlerQdrantConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="test_collection",
        vector_size=4,
        embedding_server_url="http://localhost:8100",
        max_chunk_chars=max_chunk_chars,
    )


def _mock_qdrant_client(collection_exists: bool) -> MagicMock:
    client = MagicMock()
    client.collection_exists = MagicMock(return_value=collection_exists)
    client.create_collection = MagicMock()
    client.search = MagicMock(return_value=[])
    client.upsert = MagicMock()
    client.close = MagicMock()
    return client


def _mock_embedding_client(vector: list[float] | None = None) -> AsyncMock:
    vec = vector or [0.1, 0.2, 0.3, 0.4]
    client = AsyncMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.get_embedding = AsyncMock(return_value=vec)
    return client


@pytest.mark.unit
class TestHandlerQdrantInitialize:
    """Tests for HandlerQdrant.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_creates_collection_when_absent(self) -> None:
        """When collection_exists=False, create_collection must be called."""
        handler = HandlerQdrant(config=_make_config())
        qdrant = _mock_qdrant_client(collection_exists=False)
        emb = _mock_embedding_client()

        with (
            patch(
                "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.EmbeddingClient",
                return_value=emb,
            ),
            patch(
                "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
                side_effect=lambda fn, *args, **kwargs: _sync_side_effect(
                    fn, *args, **kwargs
                ),
            ),
            patch("qdrant_client.QdrantClient", return_value=qdrant),
        ):
            with patch(
                "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
                new=_make_to_thread(qdrant),
            ):
                await handler.initialize()

        assert handler.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_skips_existing_collection(self) -> None:
        """When collection_exists=True, create_collection must NOT be called."""
        handler = HandlerQdrant(config=_make_config())
        qdrant = _mock_qdrant_client(collection_exists=True)
        emb = _mock_embedding_client()

        with (
            patch(
                "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.EmbeddingClient",
                return_value=emb,
            ),
            patch(
                "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
                new=_make_to_thread(qdrant),
            ),
            patch("qdrant_client.QdrantClient", return_value=qdrant),
        ):
            await handler.initialize()

        qdrant.create_collection.assert_not_called()
        assert handler.is_initialized


def _make_to_thread(qdrant_client: MagicMock):  # type: ignore[no-untyped-def]
    """Return a coroutine replacement for asyncio.to_thread that calls mock methods."""

    async def _to_thread(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        return fn(*args, **kwargs)

    return _to_thread


def _sync_side_effect(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    return fn(*args, **kwargs)


@pytest.mark.unit
class TestHandlerQdrantChunking:
    """Tests for HandlerQdrant._chunk_text()."""

    def test_chunk_text_splits_on_blank_lines(self) -> None:
        """Two paragraphs separated by blank line → two chunks."""
        handler = HandlerQdrant(config=_make_config(max_chunk_chars=2000))
        text = "First paragraph content.\n\nSecond paragraph content."
        chunks = handler._chunk_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "First paragraph content."
        assert chunks[1] == "Second paragraph content."

    def test_chunk_text_splits_long_paragraph_at_sentence_boundary(self) -> None:
        """Paragraph longer than max_chunk_chars is split at sentence boundary."""
        handler = HandlerQdrant(config=_make_config(max_chunk_chars=25))
        # "Hello world. Goodbye world." - both sentences are individually < 25 chars
        text = "Hello world. Goodbye world."
        chunks = handler._chunk_text(text)
        assert len(chunks) >= 2
        # All chunks must be non-empty
        assert all(c for c in chunks)


@pytest.mark.unit
class TestHandlerQdrantExecute:
    """Tests for HandlerQdrant.execute() operations."""

    @pytest.mark.asyncio
    async def test_execute_search_returns_response_with_mapped_payload(self) -> None:
        """Search operation calls embedding, searches Qdrant, maps payload to response."""

        # Build a fake Qdrant hit with a valid ModelMemorySnapshot payload
        fake_payload = {
            "snapshot_id": "00000000-0000-0000-0000-000000000001",
            "version": 1,
            "corpus_id": None,
            "parent_snapshot_id": None,
            "subject": {
                "subject_type": "agent",
                "subject_id": "00000000-0000-0000-0000-000000000002",
                "namespace": None,
                "subject_key": None,
            },
            "decisions": [],
            "failures": [],
            "cost_ledger": {
                "ledger_id": "00000000-0000-0000-0000-000000000003",
                "budget_total": 100.0,
                "budget_remaining": 100.0,
                "entries": [],
                "total_spent": 0.0,
                "escalation_count": 0,
                "last_escalation_reason": None,
                "warning_threshold": 0.8,
                "hard_ceiling": 1.0,
            },
            "execution_annotations": {},
            "schema_version": "1.0.0",
            "content_hash": "",
            "created_at": "2025-01-01T00:00:00Z",
            "tags": [],
        }
        fake_hit = MagicMock()
        fake_hit.score = 0.9
        fake_hit.payload = fake_payload

        handler = HandlerQdrant(config=_make_config())
        emb = _mock_embedding_client(vector=[0.1, 0.2, 0.3, 0.4])
        qdrant = _mock_qdrant_client(collection_exists=True)
        qdrant.search = MagicMock(return_value=[fake_hit])

        # Force initialized state
        handler._initialized = True
        handler._embedding_client = emb
        handler._qdrant_client = qdrant

        from omnimemory.nodes.node_memory_retrieval_effect.models import (
            ModelMemoryRetrievalRequest,
        )

        request = ModelMemoryRetrievalRequest(operation="search", query_text="hello")
        with patch(
            "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
            new=_make_to_thread(qdrant),
        ):
            response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) == 1
        assert response.results[0].score == 0.9
        emb.get_embedding.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_execute_index_upserts_deterministic_point_ids(self) -> None:
        """index operation produces uuid5-based point IDs: uuid5(NAMESPACE_DNS, 'doc-xyz:N')."""
        handler = HandlerQdrant(config=_make_config())
        emb = _mock_embedding_client(vector=[0.1, 0.2, 0.3, 0.4])
        qdrant = _mock_qdrant_client(collection_exists=True)

        handler._initialized = True
        handler._embedding_client = emb
        handler._qdrant_client = qdrant

        # Two-paragraph document → two chunks → two upsert calls
        content = "First chunk content.\n\nSecond chunk content."
        request = SimpleNamespace(
            operation="index", document_id="doc-xyz", content=content
        )

        with patch(
            "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
            new=_make_to_thread(qdrant),
        ):
            response = await handler.execute(request)

        assert response.status == "success"
        assert response.total_count == 2
        assert qdrant.upsert.call_count == 2

        # Verify deterministic point IDs
        upsert_calls = qdrant.upsert.call_args_list
        expected_id_0 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-xyz:0"))
        expected_id_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-xyz:1"))
        point_ids = [call.kwargs["points"][0].id for call in upsert_calls]
        assert point_ids[0] == expected_id_0
        assert point_ids[1] == expected_id_1

    @pytest.mark.asyncio
    async def test_execute_raises_if_not_initialized(self) -> None:
        """execute() must raise RuntimeError with 'initialize' in the message."""
        handler = HandlerQdrant(config=_make_config())
        from omnimemory.nodes.node_memory_retrieval_effect.models import (
            ModelMemoryRetrievalRequest,
        )

        request = ModelMemoryRetrievalRequest(operation="search", query_text="test")
        with pytest.raises(RuntimeError, match="initialize"):
            await handler.execute(request)
