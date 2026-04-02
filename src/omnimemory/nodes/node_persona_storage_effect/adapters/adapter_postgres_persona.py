# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PostgreSQL adapter for persona snapshot storage.

Shared by both persona_storage_effect and persona_retrieval_effect nodes.
Append-only: no updates or deletes. Duplicate (user_id, persona_version)
inserts are caught and logged as INFO.

Consent enforcement is deferred to Phase B (OMN-3980).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID

from omnimemory.enums import EnumPreferredTone, EnumTechnicalLevel
from omnimemory.models.persona import ModelUserPersonaV1

logger = logging.getLogger(__name__)


class ProtocolAsyncConnection(Protocol):
    """Minimal async DB connection protocol."""

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None: ...
    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]: ...
    async def execute(self, query: str, *args: object) -> str: ...


class AdapterPostgresPersona:
    """PostgreSQL adapter for persona snapshot CRUD.

    Methods:
        store: Insert a new persona snapshot (append-only, idempotent)
        get_latest: Retrieve the latest snapshot for a user
        get_users_needing_rebuild: Find users with stale personas
    """

    def __init__(self, conn: ProtocolAsyncConnection) -> None:
        self._conn = conn

    async def store(
        self,
        persona: ModelUserPersonaV1,
        trigger_reason: str,
        correlation_id: UUID,
    ) -> bool:
        """Store a persona snapshot. Returns True on new insert, False on duplicate.

        Catches UniqueViolationError (23505) from asyncpg for idempotent
        duplicate handling. Event should only be emitted when this returns True.
        """
        try:
            await self._conn.execute(
                """
                INSERT INTO user_persona_snapshots (
                    user_id, agent_id, technical_level, vocabulary_complexity,
                    preferred_tone, domain_familiarity, session_count,
                    persona_version, rebuilt_from_signals, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                persona.user_id,
                persona.agent_id,
                persona.technical_level.value,
                persona.vocabulary_complexity,
                persona.preferred_tone.value,
                json.dumps(persona.domain_familiarity),
                persona.session_count,
                persona.persona_version,
                persona.rebuilt_from_signals,
                persona.created_at,
            )
            return True
        except Exception as e:
            # asyncpg.UniqueViolationError has pgcode "23505"
            if hasattr(e, "sqlstate") and e.sqlstate == "23505":
                logger.info(
                    "Persona snapshot already exists for user=%s version=%d, skipping",
                    persona.user_id,
                    persona.persona_version,
                )
                return False
            raise

    async def get_latest(self, user_id: str) -> ModelUserPersonaV1 | None:
        """Retrieve the latest persona snapshot for a user."""
        row = await self._conn.fetchrow(
            """
            SELECT user_id, agent_id, technical_level, vocabulary_complexity,
                   preferred_tone, domain_familiarity, session_count,
                   persona_version, rebuilt_from_signals, created_at
            FROM user_persona_snapshots
            WHERE user_id = $1
            ORDER BY persona_version DESC
            LIMIT 1
            """,
            user_id,
        )
        if row is None:
            return None

        domain_familiarity = row["domain_familiarity"]
        if isinstance(domain_familiarity, str):
            domain_familiarity = json.loads(domain_familiarity)

        # Row values come as object from the protocol; cast to expected types
        session_count_val: int = int(str(row["session_count"]))
        persona_version_val: int = int(str(row["persona_version"]))
        rebuilt_from_signals_val: int = int(str(row["rebuilt_from_signals"]))
        vocab_val: float = float(str(row["vocabulary_complexity"]))
        created_at_val = cast("datetime", row["created_at"])

        return ModelUserPersonaV1(
            user_id=str(row["user_id"]),
            agent_id=str(row["agent_id"]) if row.get("agent_id") else None,
            technical_level=EnumTechnicalLevel(str(row["technical_level"])),
            vocabulary_complexity=vocab_val,
            preferred_tone=EnumPreferredTone(str(row["preferred_tone"])),
            domain_familiarity=cast("dict[str, float]", domain_familiarity),
            session_count=session_count_val,
            persona_version=persona_version_val,
            rebuilt_from_signals=rebuilt_from_signals_val,
            created_at=created_at_val,
        )

    async def get_users_needing_rebuild(
        self,
        since: datetime | None = None,
    ) -> list[str]:
        """Return up to 100 user_ids with memory items newer than their latest persona.

        If since is None, returns all users with at least one persona snapshot.
        """
        if since is not None:
            rows = await self._conn.fetch(
                """
                SELECT DISTINCT user_id
                FROM user_persona_snapshots
                WHERE created_at < $1
                ORDER BY user_id
                LIMIT 100
                """,
                since,
            )
        else:
            rows = await self._conn.fetch(
                """
                SELECT DISTINCT user_id
                FROM user_persona_snapshots
                ORDER BY user_id
                LIMIT 100
                """,
            )
        return [str(row["user_id"]) for row in rows]
