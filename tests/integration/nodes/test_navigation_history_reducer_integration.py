# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for navigation_history_reducer.

These tests require live PostgreSQL and Qdrant connections.
Run with: pytest -m integration tests/integration/nodes/test_navigation_history_reducer_integration.py

Environment:
- PostgreSQL: configured via OMNIMEMORY_PG_DSN env var (default: localhost:5436)
- Qdrant: configured via QDRANT_HOST / QDRANT_PORT env vars (default: localhost:6333)
- Embedding: configured via LLM_EMBEDDING_URL env var (default: localhost:8100)

Markers: integration
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from omnimemory.nodes.node_navigation_history_reducer.handlers import (
    HandlerNavigationHistoryReducer,
)
from omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer import (
    _QDRANT_COLLECTION,
    HandlerNavigationHistoryWriter,
)
from omnimemory.nodes.node_navigation_history_reducer.models import (
    ModelNavigationHistoryRequest,
    ModelNavigationSession,
    ModelPlanStep,
)
from omnimemory.nodes.node_navigation_history_reducer.models.model_navigation_session import (
    ModelNavigationOutcomeFailure,
    ModelNavigationOutcomeSuccess,
)

# ---------------------------------------------------------------------------
# Connection defaults (override via env; no hardcoded internal IPs)
# ---------------------------------------------------------------------------

_PG_DSN = os.environ.get("OMNIMEMORY_PG_DSN", "")
_QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_EMBEDDING_URL = os.environ.get(
    "LLM_EMBEDDING_URL", "http://localhost:8100/v1/embeddings"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OMNIMEMORY_PG_DSN"),
        reason="OMNIMEMORY_PG_DSN not set — requires live PostgreSQL (skipped in CI without infra)",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def writer() -> AsyncGenerator[HandlerNavigationHistoryWriter, None]:
    instance = HandlerNavigationHistoryWriter(
        pg_dsn=_PG_DSN,
        qdrant_host=_QDRANT_HOST,
        qdrant_port=_QDRANT_PORT,
        embedding_url=_EMBEDDING_URL,
    )
    try:
        yield instance
    finally:
        await instance.close()


def _make_success_session() -> ModelNavigationSession:
    return ModelNavigationSession(
        session_id=uuid4(),
        goal_condition="integration_test_goal_reach_state_Z",
        start_state_id="integration_test_state_A",
        end_state_id="integration_test_state_Z",
        executed_steps=[
            ModelPlanStep(
                step_index=0,
                from_state_id="integration_test_state_A",
                to_state_id="integration_test_state_M",
                action="transition_A_to_M",
                executed_at=datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc),
            ),
            ModelPlanStep(
                step_index=1,
                from_state_id="integration_test_state_M",
                to_state_id="integration_test_state_Z",
                action="transition_M_to_Z",
                executed_at=datetime(2026, 2, 24, 12, 0, 5, tzinfo=timezone.utc),
            ),
        ],
        final_outcome=ModelNavigationOutcomeSuccess(
            reached_state_id="integration_test_state_Z"
        ),
        graph_fingerprint="integration_test_fingerprint_abc123",
        created_at=datetime(2026, 2, 24, 12, 0, 10, tzinfo=timezone.utc),
    )


def _make_failure_session() -> ModelNavigationSession:
    return ModelNavigationSession(
        session_id=uuid4(),
        goal_condition="integration_test_goal_reach_state_Z",
        start_state_id="integration_test_state_A",
        end_state_id="integration_test_state_B",
        executed_steps=[
            ModelPlanStep(
                step_index=0,
                from_state_id="integration_test_state_A",
                to_state_id="integration_test_state_B",
                action="transition_A_to_B",
                executed_at=datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        final_outcome=ModelNavigationOutcomeFailure(
            reason="no_path_found",
            details="No route from state_B to state_Z in graph",
        ),
        graph_fingerprint="integration_test_fingerprint_abc123",
        created_at=datetime(2026, 2, 24, 12, 0, 5, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_session_written_to_postgres(
    writer: HandlerNavigationHistoryWriter,
) -> None:
    """Successful session: PostgreSQL row must exist after record()."""
    session = _make_success_session()
    response = await writer.record(session)

    assert response.postgres_written is True, (
        f"Expected postgres_written=True, got: {response}"
    )

    # Verify row exists in PostgreSQL
    pool = await asyncpg.create_pool(dsn=_PG_DSN, min_size=1, max_size=2)
    try:
        row = await pool.fetchrow(
            "SELECT session_id, outcome, step_count FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        assert row is not None, f"Row not found for session_id={session.session_id}"
        assert row["outcome"] == "success"
        assert row["step_count"] == 2
    finally:
        # Cleanup test data
        await pool.execute(
            "DELETE FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        await pool.close()


@pytest.mark.asyncio
async def test_success_session_written_to_qdrant(
    writer: HandlerNavigationHistoryWriter,
) -> None:
    """Successful session: Qdrant point must exist after record()."""
    session = _make_success_session()
    response = await writer.record(session)

    assert response.qdrant_written is True, (
        f"Expected qdrant_written=True, got: {response}"
    )

    qdrant = AsyncQdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=30)
    try:
        results = await qdrant.retrieve(
            collection_name=_QDRANT_COLLECTION,
            ids=[str(session.session_id)],
            with_payload=True,
        )
        assert len(results) == 1, f"Expected 1 Qdrant point, found {len(results)}"
        assert results[0].payload is not None
        assert results[0].payload.get("outcome") == "success"
        assert results[0].payload.get("session_id") == str(session.session_id)
    finally:
        # Cleanup: delete test Qdrant point and PostgreSQL row
        await qdrant.delete(
            collection_name=_QDRANT_COLLECTION,
            points_selector=qdrant_models.PointIdsList(
                points=[str(session.session_id)]
            ),
        )
        await qdrant.close()

        pool = await asyncpg.create_pool(dsn=_PG_DSN, min_size=1, max_size=2)
        await pool.execute(
            "DELETE FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        await pool.close()


@pytest.mark.asyncio
async def test_failure_session_written_to_postgres_only(
    writer: HandlerNavigationHistoryWriter,
) -> None:
    """Failed session: PostgreSQL row must exist; Qdrant must NOT have a point."""
    session = _make_failure_session()
    response = await writer.record(session)

    assert response.postgres_written is True
    assert response.qdrant_written is False

    # Verify PostgreSQL row
    pool = await asyncpg.create_pool(dsn=_PG_DSN, min_size=1, max_size=2)
    try:
        row = await pool.fetchrow(
            "SELECT outcome, failure_reason FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        assert row is not None
        assert row["outcome"] == "failure"
        assert row["failure_reason"] == "no_path_found"
    finally:
        await pool.execute(
            "DELETE FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        await pool.close()

    # Verify Qdrant does NOT have a point for this session
    qdrant = AsyncQdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=30)
    try:
        results = await qdrant.retrieve(
            collection_name=_QDRANT_COLLECTION,
            ids=[str(session.session_id)],
        )
        assert len(results) == 0, (
            f"Failure session must NOT appear in Qdrant, but found {len(results)} points"
        )
    finally:
        await qdrant.close()


@pytest.mark.asyncio
async def test_duplicate_session_id_is_idempotent(
    writer: HandlerNavigationHistoryWriter,
) -> None:
    """Duplicate writes with the same session_id must be no-ops."""
    session = _make_success_session()

    # First write
    response1 = await writer.record(session)
    assert response1.status in ("success", "error")

    # Second write (same session_id)
    response2 = await writer.record(session)
    assert response2.status == "skipped"
    assert response2.idempotent_skip is True

    # Confirm only one row in PostgreSQL
    pool = await asyncpg.create_pool(dsn=_PG_DSN, min_size=1, max_size=2)
    try:
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        assert count == 1, f"Expected exactly 1 row, found {count}"
    finally:
        await pool.execute(
            "DELETE FROM navigation_sessions WHERE session_id = $1",
            str(session.session_id),
        )
        await pool.close()

    # Cleanup Qdrant if write succeeded
    if response1.qdrant_written:
        qdrant = AsyncQdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=30)
        try:
            await qdrant.delete(
                collection_name=_QDRANT_COLLECTION,
                points_selector=qdrant_models.PointIdsList(
                    points=[str(session.session_id)]
                ),
            )
        finally:
            await qdrant.close()


@pytest.mark.asyncio
async def test_handler_fire_and_forget_does_not_propagate_errors() -> None:
    """Fire-and-forget pattern: errors in execute() must not raise."""
    handler = HandlerNavigationHistoryReducer(
        pg_dsn="postgresql://invalid:invalid@127.0.0.1:9999/invalid",
        qdrant_host="127.0.0.1",
        qdrant_port=9998,
    )
    await handler.initialize()

    session = _make_success_session()
    request = ModelNavigationHistoryRequest(session=session)

    # Must not raise — fire-and-forget contract
    response = await handler.execute(request)
    assert response.status == "error"
    assert response.session_id == session.session_id

    await handler.shutdown()
