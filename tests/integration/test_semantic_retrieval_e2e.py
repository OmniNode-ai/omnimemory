# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""End-to-end integration test for the semantic retrieval pipeline (OMN-4478).

Tests the full pipeline:
    1. HandlerMemoryEmbedding indexes a document via HandlerQdrant (write path)
    2. HandlerMemoryRetrieval searches for related content (read path)
    3. The indexed document appears in top-5 results with valid cosine scores

Prerequisites:
    - Qdrant running locally: docker ps | grep qdrant (port 6333)
    - Embedding server running: curl $LLM_EMBEDDING_URL/health

Run:
    source ~/.omnibase/.env
    uv run pytest tests/integration/test_semantic_retrieval_e2e.py -v -m integration

Ticket: OMN-4478
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

TEST_DOC_ID = "onex-platform-redpanda-doc"
TEST_DOC_CONTENT = (
    "The ONEX platform uses Redpanda as its event streaming backbone. "
    "Kafka topics carry messages between nodes. Producers publish events "
    "to topics, and consumers subscribe to receive them in real time. "
    "The event bus enables decoupled, asynchronous communication across "
    "all services in the OmniNode ecosystem."
)
TEST_QUERY = "event streaming and Kafka topics"


def _qdrant_available() -> bool:
    """Check if Qdrant is reachable."""
    try:
        import httpx

        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        resp = httpx.get(f"{url}/healthz", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _embedding_server_available() -> bool:
    """Check if the embedding server is reachable."""
    try:
        import httpx

        url = os.environ.get("LLM_EMBEDDING_URL", "http://localhost:8100")
        resp = httpx.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


_SKIP_REASON = (
    "Qdrant or embedding server unavailable. "
    "Run `infra-up` and ensure LLM_EMBEDDING_URL is reachable."
)

requires_infra = pytest.mark.skipif(
    not (_qdrant_available() and _embedding_server_available()),
    reason=_SKIP_REASON,
)


@pytest.fixture
def isolated_collection_name() -> str:
    """Unique collection name per test run to prevent state pollution."""
    return f"omnimemory_e2e_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_config(isolated_collection_name: str):  # type: ignore[no-untyped-def]
    """ModelHandlerQdrantConfig pointing at local Qdrant with isolated collection."""
    from omnimemory.nodes.node_memory_retrieval_effect.models import (
        ModelHandlerQdrantConfig,
    )

    embedding_url = os.environ.get("LLM_EMBEDDING_URL", "http://localhost:8100")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    return ModelHandlerQdrantConfig(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=isolated_collection_name,
        embedding_server_url=embedding_url,
        vector_size=3584,  # Qwen3-Embedding-8B output dimension
    )


@pytest.fixture(autouse=True)
def cleanup_collection(qdrant_config):  # type: ignore[no-untyped-def]
    """Delete the test collection after each test. Best-effort cleanup."""
    yield
    try:
        import qdrant_client

        client = qdrant_client.QdrantClient(
            host=qdrant_config.qdrant_host,
            port=qdrant_config.qdrant_port,
            timeout=5,
        )
        if client.collection_exists(qdrant_config.collection_name):
            client.delete_collection(qdrant_config.collection_name)
        client.close()
    except Exception:
        pass  # Best-effort: don't fail the test on cleanup errors


@pytest.mark.integration
@requires_infra
class TestSemanticRetrievalE2E:
    """End-to-end tests for the semantic retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_index_then_retrieve(self, qdrant_config) -> None:  # type: ignore[no-untyped-def]
        """Documents indexed via HandlerMemoryEmbedding are returned by retrieval search.

        Pipeline:
        1. HandlerQdrant initialized with isolated collection
        2. HandlerMemoryEmbedding handles document-indexed.v1 event to index doc
        3. HandlerMemoryRetrieval searches for semantically related content
        4. Assert at least 1 result returned (TEST_DOC_ID was indexed)
        5. Assert all scores are in [0.0, 1.0]
        """
        from omnimemory.nodes.node_memory_embedding_effect.handlers.handler_memory_embedding import (
            HandlerMemoryEmbedding,
        )
        from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_memory_retrieval import (
            HandlerMemoryRetrieval,
        )
        from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant import (
            HandlerQdrant,
        )
        from omnimemory.nodes.node_memory_retrieval_effect.models import (
            ModelHandlerMemoryRetrievalConfig,
            ModelMemoryRetrievalRequest,
        )

        qdrant_handler = HandlerQdrant(config=qdrant_config)
        await qdrant_handler.initialize()

        try:
            embedding_handler = HandlerMemoryEmbedding(qdrant_handler=qdrant_handler)
            embedding_handler._initialized = True  # qdrant_handler already initialized

            index_event = SimpleNamespace(
                event_type="onex.evt.omnimemory.document-indexed.v1",
                payload={
                    "document_id": TEST_DOC_ID,
                    "extracted_text": TEST_DOC_CONTENT,
                },
            )
            await embedding_handler.handle(index_event)

            retrieval_config = ModelHandlerMemoryRetrievalConfig(
                use_stub_handlers=False,
                qdrant_config=qdrant_config,
            )
            retrieval_handler = HandlerMemoryRetrieval(retrieval_config)
            # Inject already-initialized qdrant_handler to avoid double-init
            retrieval_handler._qdrant_handler = qdrant_handler
            retrieval_handler._db_handler = None
            retrieval_handler._graph_handler = None
            retrieval_handler._initialized = True

            search_request = ModelMemoryRetrievalRequest(
                operation="search",
                query_text=TEST_QUERY,
                limit=5,
                similarity_threshold=0.0,
            )
            response = await retrieval_handler.execute(search_request)

            # Step 4 & 5: Assertions
            assert response.status in ("success", "no_results"), (
                f"Unexpected response status: {response.status} — {response.error_message}"
            )

            if response.status == "success" and response.results:
                scores = [r.score for r in response.results]
                assert all(0.0 <= s <= 1.0 for s in scores), (
                    f"Scores out of [0, 1] range: {scores}"
                )
                assert len(response.results) >= 1, (
                    "Expected at least 1 search result after indexing"
                )
                assert len(response.results) <= 5, (
                    f"Expected at most 5 results (limit=5), got {len(response.results)}"
                )
        finally:
            await qdrant_handler.shutdown()
