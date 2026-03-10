# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for the navigation_history_reducer node.

Tests all business logic without requiring live PostgreSQL or Qdrant connections.
All external I/O is mocked.

Markers: unit
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer import (
    HandlerNavigationHistoryReducer,
    HandlerNavigationHistoryWriter,
    _hash_text,
    _uuid_to_qdrant_id,
)
from omnimemory.nodes.node_navigation_history_reducer.models import (
    ModelNavigationHistoryRequest,
    ModelNavigationHistoryResponse,
    ModelNavigationSession,
    ModelPlanStep,
)
from omnimemory.nodes.node_navigation_history_reducer.models.model_navigation_session import (
    ModelNavigationOutcomeFailure,
    ModelNavigationOutcomeSuccess,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_step(index: int = 0) -> ModelPlanStep:
    return ModelPlanStep(
        step_index=index,
        from_state_id=f"state_{index}",
        to_state_id=f"state_{index + 1}",
        action=f"action_{index}",
        executed_at=datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_success_session(session_id: UUID | None = None) -> ModelNavigationSession:
    sid = session_id or uuid4()
    return ModelNavigationSession(
        session_id=sid,
        goal_condition="reach_state_Z",
        start_state_id="state_A",
        end_state_id="state_Z",
        executed_steps=[_make_step(0), _make_step(1)],
        final_outcome=ModelNavigationOutcomeSuccess(reached_state_id="state_Z"),
        graph_fingerprint="abc123fingerprint",
        created_at=datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_failure_session(session_id: UUID | None = None) -> ModelNavigationSession:
    sid = session_id or uuid4()
    return ModelNavigationSession(
        session_id=sid,
        goal_condition="reach_state_Z",
        start_state_id="state_A",
        end_state_id="state_B",
        executed_steps=[_make_step(0)],
        final_outcome=ModelNavigationOutcomeFailure(
            reason="no_path_found", details="No edges from state_B to state_Z"
        ),
        graph_fingerprint="abc123fingerprint",
        created_at=datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNavigationSessionModel:
    """Tests for ModelNavigationSession and related models."""

    def test_success_session_is_successful(self) -> None:
        session = _make_success_session()
        assert session.is_successful is True

    def test_failure_session_is_not_successful(self) -> None:
        session = _make_failure_session()
        assert session.is_successful is False

    def test_step_count(self) -> None:
        session = _make_success_session()
        assert session.step_count == 2

    def test_empty_steps(self) -> None:
        session = ModelNavigationSession(
            session_id=uuid4(),
            goal_condition="reach_Z",
            start_state_id="A",
            end_state_id="A",
            executed_steps=[],
            final_outcome=ModelNavigationOutcomeSuccess(reached_state_id="A"),
            graph_fingerprint="fp",
            created_at=datetime(2026, 2, 24, tzinfo=timezone.utc),
        )
        assert session.step_count == 0
        assert session.is_successful is True

    def test_plan_step_is_frozen(self) -> None:
        step = _make_step()
        with pytest.raises(Exception):
            step.step_index = 99  # type: ignore[misc]

    def test_navigation_session_is_frozen(self) -> None:
        session = _make_success_session()
        with pytest.raises(Exception):
            session.goal_condition = "other"  # type: ignore[misc]

    def test_failure_outcome_has_reason(self) -> None:
        outcome = ModelNavigationOutcomeFailure(reason="no_path_found")
        assert outcome.reason == "no_path_found"
        assert outcome.tag == "failure"

    def test_success_outcome_tag(self) -> None:
        outcome = ModelNavigationOutcomeSuccess(reached_state_id="Z")
        assert outcome.tag == "success"


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUtilityFunctions:
    def test_hash_text_is_stable(self) -> None:
        assert _hash_text("hello") == _hash_text("hello")

    def test_hash_text_is_different_for_different_inputs(self) -> None:
        assert _hash_text("hello") != _hash_text("world")

    def test_hash_text_length(self) -> None:
        # SHA-256 hex is always 64 characters
        assert len(_hash_text("any text")) == 64

    def test_uuid_to_qdrant_id(self) -> None:
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = _uuid_to_qdrant_id(uid)
        assert result == "12345678-1234-5678-1234-567812345678"


# ---------------------------------------------------------------------------
# HandlerNavigationHistoryReducer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerNavigationHistoryReducer:
    """Tests for the ONEX handler wrapper."""

    @pytest.mark.asyncio
    async def test_execute_requires_initialize(self) -> None:
        handler = HandlerNavigationHistoryReducer()
        session = _make_success_session()
        request = ModelNavigationHistoryRequest(session=session)

        response = await handler.execute(request)
        assert response.status == "error"
        assert "not initialized" in (response.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        handler = HandlerNavigationHistoryReducer()
        await handler.initialize()
        await handler.initialize()  # Should not raise
        assert handler.is_initialized is True
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_delegates_to_writer(self) -> None:
        mock_writer = AsyncMock(spec=HandlerNavigationHistoryWriter)
        session = _make_success_session()
        expected_response = ModelNavigationHistoryResponse(
            session_id=session.session_id,
            status="success",
            postgres_written=True,
            qdrant_written=True,
        )
        mock_writer.record.return_value = expected_response

        handler = HandlerNavigationHistoryReducer(writer=mock_writer)
        await handler.initialize()

        request = ModelNavigationHistoryRequest(session=session)
        response = await handler.execute(request)

        mock_writer.record.assert_called_once_with(session)
        assert response.status == "success"
        assert response.postgres_written is True
        assert response.qdrant_written is True

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_handles_writer_exception(self) -> None:
        mock_writer = AsyncMock(spec=HandlerNavigationHistoryWriter)
        mock_writer.record.side_effect = RuntimeError("database exploded")

        handler = HandlerNavigationHistoryReducer(writer=mock_writer)
        await handler.initialize()

        session = _make_success_session()
        request = ModelNavigationHistoryRequest(session=session)
        response = await handler.execute(request)

        assert response.status == "error"
        assert "database exploded" in (response.error_message or "")

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_calls_writer_close(self) -> None:
        mock_writer = AsyncMock(spec=HandlerNavigationHistoryWriter)
        handler = HandlerNavigationHistoryReducer(writer=mock_writer)
        await handler.initialize()
        await handler.shutdown()
        mock_writer.close.assert_called_once()
        assert handler.is_initialized is False


# ---------------------------------------------------------------------------
# HandlerNavigationHistoryWriter tests (mocked external I/O)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerNavigationHistoryWriterUnit:
    """Unit tests for HandlerNavigationHistoryWriter with all I/O mocked."""

    def _make_pg_pool_mock(self, conn: AsyncMock) -> MagicMock:
        """Build a mock asyncpg pool whose acquire() works as async context manager."""
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)

        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)
        return pool

    @pytest.mark.asyncio
    async def test_record_success_writes_postgres_and_qdrant(self) -> None:
        writer = HandlerNavigationHistoryWriter()
        session = _make_success_session()

        mock_conn = AsyncMock()
        # fetchval returns the inserted session_id (non-None = new row inserted)
        mock_conn.fetchval.return_value = str(session.session_id)

        mock_qdrant = AsyncMock()
        mock_qdrant.upsert.return_value = None

        writer._pg_pool = self._make_pg_pool_mock(mock_conn)  # type: ignore[assignment]
        writer._qdrant_client = mock_qdrant

        with patch.object(writer, "_embed_text", return_value=[0.1] * 10):
            response = await writer.record(session)

        assert response.status == "success"
        assert response.postgres_written is True
        assert response.qdrant_written is True
        assert response.idempotent_skip is False
        mock_conn.fetchval.assert_called_once()
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_failure_writes_postgres_only(self) -> None:
        writer = HandlerNavigationHistoryWriter()
        session = _make_failure_session()

        mock_conn = AsyncMock()
        # fetchval returns the inserted session_id (non-None = new row inserted)
        mock_conn.fetchval.return_value = str(session.session_id)

        mock_qdrant = AsyncMock()
        writer._pg_pool = self._make_pg_pool_mock(mock_conn)  # type: ignore[assignment]
        writer._qdrant_client = mock_qdrant

        response = await writer.record(session)

        assert response.status == "success"
        assert response.postgres_written is True
        assert response.qdrant_written is False  # Failure: no Qdrant write
        mock_conn.fetchval.assert_called_once()
        mock_qdrant.upsert.assert_not_called()  # Critical invariant

    @pytest.mark.asyncio
    async def test_record_idempotent_on_duplicate_session_id(self) -> None:
        writer = HandlerNavigationHistoryWriter()
        session = _make_success_session()

        mock_conn = AsyncMock()
        # Simulate duplicate: ON CONFLICT DO NOTHING returns None (no row inserted)
        mock_conn.fetchval.return_value = None

        mock_qdrant = AsyncMock()
        writer._pg_pool = self._make_pg_pool_mock(mock_conn)  # type: ignore[assignment]
        writer._qdrant_client = mock_qdrant

        response = await writer.record(session)

        assert response.status == "skipped"
        assert response.idempotent_skip is True
        assert response.postgres_written is False
        assert response.qdrant_written is False
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_postgres_failure_does_not_write_qdrant(self) -> None:
        writer = HandlerNavigationHistoryWriter()
        session = _make_success_session()

        # Pool.acquire() raises when entering context
        failing_cm = MagicMock()
        failing_cm.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
        failing_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=failing_cm)

        writer._pg_pool = mock_pool  # type: ignore[assignment]
        mock_qdrant = AsyncMock()
        writer._qdrant_client = mock_qdrant

        response = await writer.record(session)

        assert response.status == "error"
        assert response.postgres_written is False
        assert response.qdrant_written is False
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_qdrant_failure_returns_partial_success(self) -> None:
        writer = HandlerNavigationHistoryWriter()
        session = _make_success_session()

        mock_conn = AsyncMock()
        # fetchval returns the inserted session_id (non-None = new row inserted)
        mock_conn.fetchval.return_value = str(session.session_id)

        mock_qdrant = AsyncMock()
        mock_qdrant.upsert.side_effect = Exception("Qdrant unavailable")

        writer._pg_pool = self._make_pg_pool_mock(mock_conn)  # type: ignore[assignment]
        writer._qdrant_client = mock_qdrant

        with patch.object(writer, "_embed_text", return_value=[0.1] * 10):
            response = await writer.record(session)

        assert response.status == "error"
        assert response.postgres_written is True  # PostgreSQL succeeded
        assert response.qdrant_written is False  # Qdrant failed
        assert "Qdrant write failed (PostgreSQL OK)" in (response.error_message or "")

    @pytest.mark.asyncio
    async def test_failure_session_never_writes_to_qdrant_even_if_qdrant_available(
        self,
    ) -> None:
        """Critical invariant: failed paths MUST NOT appear in Qdrant."""
        writer = HandlerNavigationHistoryWriter()
        session = _make_failure_session()

        mock_conn = AsyncMock()
        # fetchval returns the inserted session_id (non-None = new row inserted)
        mock_conn.fetchval.return_value = str(session.session_id)

        mock_qdrant = AsyncMock()
        writer._pg_pool = self._make_pg_pool_mock(mock_conn)  # type: ignore[assignment]
        writer._qdrant_client = mock_qdrant

        response = await writer.record(session)

        # Qdrant must never be called for failure sessions
        mock_qdrant.upsert.assert_not_called()
        assert response.qdrant_written is False

    @pytest.mark.asyncio
    async def test_embed_text_parses_response(self) -> None:
        writer = HandlerNavigationHistoryWriter()
        fake_vector = [0.1, 0.2, 0.3]
        fake_response_json = {"data": [{"embedding": fake_vector}]}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = fake_response_json
            mock_client.post.return_value = mock_response

            result = await writer._embed_text("test input")

        assert result == fake_vector

    @pytest.mark.asyncio
    async def test_embed_text_raises_on_bad_response(self) -> None:
        writer = HandlerNavigationHistoryWriter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"unexpected": "format"}
            mock_client.post.return_value = mock_response

            with pytest.raises(ValueError, match="Unexpected embedding response"):
                await writer._embed_text("test")
