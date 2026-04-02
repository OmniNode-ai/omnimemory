# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for persona storage/retrieval effect nodes and adapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnimemory.enums import EnumPreferredTone, EnumTechnicalLevel
from omnimemory.models.persona import ModelUserPersonaV1
from omnimemory.nodes.node_persona_retrieval_effect import (
    ModelPersonaRetrievalRequest,
    ModelPersonaRetrievalResponse,
)
from omnimemory.nodes.node_persona_storage_effect import (
    AdapterPostgresPersona,
    ModelPersonaStorageRequest,
    ModelPersonaStorageResponse,
)


def _make_persona(
    user_id: str = "test-user",
    version: int = 1,
) -> ModelUserPersonaV1:
    return ModelUserPersonaV1(
        user_id=user_id,
        technical_level=EnumTechnicalLevel.INTERMEDIATE,
        vocabulary_complexity=0.5,
        preferred_tone=EnumPreferredTone.EXPLANATORY,
        domain_familiarity={"omnimemory": 0.3},
        session_count=5,
        persona_version=version,
        created_at=datetime.now(tz=timezone.utc),
        rebuilt_from_signals=10,
    )


@pytest.mark.unit
class TestAdapterPostgresPersonaStore:
    @pytest.mark.asyncio
    async def test_store_returns_true_on_new_insert(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        adapter = AdapterPostgresPersona(conn)
        persona = _make_persona()
        result = await adapter.store(persona, "tick", uuid4())
        assert result is True
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_returns_false_on_duplicate(self) -> None:
        conn = AsyncMock()
        # Simulate asyncpg UniqueViolationError (sqlstate 23505)
        exc = Exception("duplicate key")
        exc.sqlstate = "23505"  # type: ignore[attr-defined]
        conn.execute = AsyncMock(side_effect=exc)
        adapter = AdapterPostgresPersona(conn)
        persona = _make_persona()
        result = await adapter.store(persona, "tick", uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_store_raises_on_other_error(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("connection lost"))
        adapter = AdapterPostgresPersona(conn)
        persona = _make_persona()
        with pytest.raises(RuntimeError, match="connection lost"):
            await adapter.store(persona, "tick", uuid4())


@pytest.mark.unit
class TestAdapterPostgresPersonaRetrieve:
    @pytest.mark.asyncio
    async def test_get_latest_returns_none_when_no_rows(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        adapter = AdapterPostgresPersona(conn)
        result = await adapter.get_latest("nonexistent-user")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_returns_persona(self) -> None:
        now = datetime.now(tz=timezone.utc)
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "user_id": "test-user",
                "agent_id": None,
                "technical_level": "expert",
                "vocabulary_complexity": 0.8,
                "preferred_tone": "concise",
                "domain_familiarity": '{"omnimemory": 0.5}',
                "session_count": 10,
                "persona_version": 3,
                "rebuilt_from_signals": 25,
                "created_at": now,
            }
        )
        adapter = AdapterPostgresPersona(conn)
        result = await adapter.get_latest("test-user")
        assert result is not None
        assert result.technical_level == EnumTechnicalLevel.EXPERT
        assert result.persona_version == 3

    @pytest.mark.asyncio
    async def test_get_users_needing_rebuild_limits_to_100(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[{"user_id": f"user-{i}"} for i in range(100)]
        )
        adapter = AdapterPostgresPersona(conn)
        result = await adapter.get_users_needing_rebuild(since=None)
        assert len(result) == 100


@pytest.mark.unit
class TestPersonaStorageModels:
    def test_storage_request_creation(self) -> None:
        persona = _make_persona()
        request = ModelPersonaStorageRequest(
            persona=persona,
            trigger_reason="tick",
            correlation_id=uuid4(),
        )
        assert request.operation == "store"
        assert request.persona == persona

    def test_storage_response_success(self) -> None:
        response = ModelPersonaStorageResponse(
            status="success",
            is_new_insert=True,
        )
        assert response.is_new_insert is True

    def test_storage_response_duplicate(self) -> None:
        response = ModelPersonaStorageResponse(
            status="duplicate",
            is_new_insert=False,
        )
        assert response.is_new_insert is False


@pytest.mark.unit
class TestPersonaRetrievalModels:
    def test_retrieval_request_creation(self) -> None:
        request = ModelPersonaRetrievalRequest(user_id="test-user")
        assert request.user_id == "test-user"
        assert request.agent_id is None

    def test_retrieval_response_found(self) -> None:
        persona = _make_persona()
        response = ModelPersonaRetrievalResponse(
            status="found",
            persona=persona,
        )
        assert response.persona is not None

    def test_retrieval_response_not_found(self) -> None:
        response = ModelPersonaRetrievalResponse(status="not_found")
        assert response.persona is None
