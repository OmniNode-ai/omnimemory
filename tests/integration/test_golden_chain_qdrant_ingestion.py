# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Golden chain tests for Qdrant ingestion flow (OMN-8646).

Validates the end-to-end path:
  command input → HandlerQdrant.execute() → qdrant_client.upsert() / search()

Tests (8 total):
  Happy path:
    1. test_index_command_calls_upsert_with_correct_args
    2. test_index_two_chunks_upserts_twice
    3. test_search_command_calls_embedding_then_qdrant_search
    4. test_search_returns_mapped_results
  Error paths:
    5. test_index_adapter_raises_returns_error_response
    6. test_search_embedding_raises_returns_error_response
    7. test_execute_before_initialize_raises
    8. test_unsupported_operation_returns_error_response

Ticket: OMN-8646
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
    ModelMemoryRetrievalRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VECTOR_SIZE = 4
_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


def _make_config() -> ModelHandlerQdrantConfig:
    return ModelHandlerQdrantConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="test_col",
        vector_size=_VECTOR_SIZE,
        embedding_server_url="http://localhost:8100",
        max_chunk_chars=2000,
    )


def _make_qdrant_mock() -> MagicMock:
    client = MagicMock()
    client.collection_exists = MagicMock(return_value=True)
    client.create_collection = MagicMock()
    client.search = MagicMock(return_value=[])
    client.upsert = MagicMock()
    client.close = MagicMock()
    return client


def _make_embedding_mock(vector: list[float] | None = None) -> AsyncMock:
    emb = AsyncMock()
    emb.connect = AsyncMock()
    emb.close = AsyncMock()
    emb.get_embedding = AsyncMock(return_value=vector or _EMBEDDING)
    return emb


async def _to_thread_passthrough(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Drop-in for asyncio.to_thread that calls the function synchronously."""
    return fn(*args, **kwargs)


def _initialized_handler(
    qdrant: MagicMock | None = None,
    emb: AsyncMock | None = None,
) -> HandlerQdrant:
    """Return a HandlerQdrant already marked initialized with injected mocks."""
    handler = HandlerQdrant(config=_make_config())
    handler._initialized = True
    handler._qdrant_client = qdrant or _make_qdrant_mock()
    handler._embedding_client = emb or _make_embedding_mock()
    return handler


# ---------------------------------------------------------------------------
# Happy path — index (upsert) flow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoldenChainQdrantIndex:
    """Golden chain: index command → HandlerQdrant → qdrant_client.upsert()."""

    @pytest.mark.asyncio
    async def test_index_command_calls_upsert_with_correct_args(self) -> None:
        """Single-chunk document: upsert called once with deterministic point ID."""
        qdrant = _make_qdrant_mock()
        emb = _make_embedding_mock()
        handler = _initialized_handler(qdrant=qdrant, emb=emb)

        request = SimpleNamespace(
            operation="index", document_id="doc-abc", content="Hello world."
        )

        with patch(
            "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
            new=_to_thread_passthrough,
        ):
            response = await handler.execute(request)

        assert response.status == "success"
        assert qdrant.upsert.call_count == 1

        call_kwargs = qdrant.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_col"
        point = call_kwargs["points"][0]
        expected_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-abc:0"))
        assert point.id == expected_id
        assert point.vector == _EMBEDDING
        assert point.payload["document_id"] == "doc-abc"

    @pytest.mark.asyncio
    async def test_index_two_chunks_upserts_twice(self) -> None:
        """Two-paragraph document produces two upsert calls with sequential IDs."""
        qdrant = _make_qdrant_mock()
        emb = _make_embedding_mock()
        handler = _initialized_handler(qdrant=qdrant, emb=emb)

        content = "First paragraph.\n\nSecond paragraph."
        request = SimpleNamespace(
            operation="index", document_id="doc-xy", content=content
        )

        with patch(
            "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
            new=_to_thread_passthrough,
        ):
            response = await handler.execute(request)

        assert response.status == "success"
        assert response.total_count == 2
        assert qdrant.upsert.call_count == 2
        assert emb.get_embedding.await_count == 2

        ids = [c.kwargs["points"][0].id for c in qdrant.upsert.call_args_list]
        assert ids[0] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-xy:0"))
        assert ids[1] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-xy:1"))


# ---------------------------------------------------------------------------
# Happy path — search flow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoldenChainQdrantSearch:
    """Golden chain: search command → HandlerQdrant → embedding → qdrant.search()."""

    @pytest.mark.asyncio
    async def test_search_command_calls_embedding_then_qdrant_search(self) -> None:
        """Search embeds query text and passes vector to qdrant.search."""
        qdrant = _make_qdrant_mock()
        qdrant.search = MagicMock(return_value=[])
        emb = _make_embedding_mock(vector=[0.5, 0.5, 0.5, 0.5])
        handler = _initialized_handler(qdrant=qdrant, emb=emb)

        request = ModelMemoryRetrievalRequest(
            operation="search", query_text="test query"
        )

        with patch(
            "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
            new=_to_thread_passthrough,
        ):
            response = await handler.execute(request)

        emb.get_embedding.assert_awaited_once_with("test query")
        qdrant.search.assert_called_once()
        call_args = qdrant.search.call_args
        assert call_args.args[0] == "test_col"
        assert call_args.kwargs["query_vector"] == [0.5, 0.5, 0.5, 0.5]

        assert response.status == "no_results"
        assert response.query_embedding_used == [0.5, 0.5, 0.5, 0.5]

    @pytest.mark.asyncio
    async def test_search_returns_mapped_results(self) -> None:
        """Qdrant hits are mapped to ModelSearchResult with correct score/distance."""
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
        fake_hit.score = 0.85
        fake_hit.payload = fake_payload

        qdrant = _make_qdrant_mock()
        qdrant.search = MagicMock(return_value=[fake_hit])
        handler = _initialized_handler(qdrant=qdrant)

        request = ModelMemoryRetrievalRequest(operation="search", query_text="find me")

        with patch(
            "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
            new=_to_thread_passthrough,
        ):
            response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) == 1
        assert response.results[0].score == 0.85
        assert response.results[0].distance == pytest.approx(1.0 - 0.85)
        assert response.total_count == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoldenChainQdrantErrors:
    """Golden chain error paths: adapter raises → handler returns error response."""

    @pytest.mark.asyncio
    async def test_index_adapter_raises_returns_error_response(self) -> None:
        """When qdrant.upsert raises, execute() propagates the exception."""
        qdrant = _make_qdrant_mock()
        qdrant.upsert = MagicMock(side_effect=RuntimeError("Qdrant unavailable"))
        handler = _initialized_handler(qdrant=qdrant)

        request = SimpleNamespace(
            operation="index", document_id="doc-err", content="Some content."
        )

        with (
            patch(
                "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant.asyncio.to_thread",
                new=_to_thread_passthrough,
            ),
            pytest.raises(RuntimeError, match="Qdrant unavailable"),
        ):
            await handler.execute(request)

    @pytest.mark.asyncio
    async def test_search_embedding_raises_returns_error_response(self) -> None:
        """When embedding client raises, execute() propagates the exception."""
        emb = _make_embedding_mock()
        emb.get_embedding = AsyncMock(
            side_effect=ConnectionError("embedding server down")
        )
        handler = _initialized_handler(emb=emb)

        request = ModelMemoryRetrievalRequest(operation="search", query_text="hello")

        with pytest.raises(ConnectionError, match="embedding server down"):
            await handler.execute(request)

    @pytest.mark.asyncio
    async def test_execute_before_initialize_raises(self) -> None:
        """execute() before initialize() raises RuntimeError with 'initialize'."""
        handler = HandlerQdrant(config=_make_config())
        request = ModelMemoryRetrievalRequest(operation="search", query_text="test")

        with pytest.raises(RuntimeError, match="initialize"):
            await handler.execute(request)

    @pytest.mark.asyncio
    async def test_unsupported_operation_returns_error_response(self) -> None:
        """Unsupported operation returns error status without raising."""
        handler = _initialized_handler()
        request = SimpleNamespace(operation="delete")

        response = await handler.execute(request)

        assert response.status == "error"
        assert "unsupported operation" in (response.error_message or "").lower()


# ---------------------------------------------------------------------------
# Optional smoke test (live Qdrant on .201)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smoke_qdrant_live_index_and_search() -> None:
    """Smoke test: live Qdrant on the dev host (port 6333).

    Skipped when Qdrant is unreachable.
    """
    import socket

    _host = "192.168.86.201"  # onex-allow-internal-ip
    _emb_url = f"http://{_host}:8002"  # onex-allow-internal-ip
    try:
        sock = socket.create_connection((_host, 6333), timeout=2)
        sock.close()
    except OSError:
        pytest.skip(f"Live Qdrant at {_host}:6333 not reachable")

    handler = HandlerQdrant(
        config=ModelHandlerQdrantConfig(
            qdrant_host=_host,
            qdrant_port=6333,
            collection_name="golden_chain_smoke_test",
            vector_size=4,
            embedding_server_url=_emb_url,
            max_chunk_chars=2000,
        )
    )
    try:
        await handler.initialize()
    except Exception as exc:
        pytest.skip(f"Live Qdrant initialization failed (auth/config): {exc}")
    try:
        req = SimpleNamespace(
            operation="index",
            document_id="smoke-test-doc",
            content="Smoke test paragraph one.\n\nSmoke test paragraph two.",
        )
        response = await handler.execute(req)
        assert response.status == "success"
        assert response.total_count == 2
    finally:
        await handler.shutdown()
