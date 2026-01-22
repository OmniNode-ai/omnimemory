# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Adapter for tick-based memory lifecycle processing.

This adapter wraps HandlerRuntimeTick semantics but performs direct DB queries
for memory lifecycle state. Uses ADAPTER strategy per handler_reuse_matrix.md.

The adapter identifies memories that need lifecycle transitions without
performing the actual state changes - that responsibility belongs to the
orchestrator which calls the appropriate handlers.

Example::

    from datetime import datetime, timezone
    from uuid import uuid4

    from omnimemory.nodes.memory_lifecycle_orchestrator.adapters import (
        AdapterRuntimeTickMemory,
    )

    async def process_lifecycle_tick():
        adapter = AdapterRuntimeTickMemory()

        # Use authoritative timestamp from envelope
        envelope_timestamp = datetime.now(timezone.utc)

        result = await adapter.process_tick(
            tick_id=uuid4(),
            correlation_id=uuid4(),
            now=envelope_timestamp,
        )

        print(f"Found {result.memories_expired} memories to expire")
        print(f"Found {result.memories_pending_archive} memories to archive")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1392.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelMemoryLifecycleTickResult(BaseModel):
    """Result of processing a lifecycle tick.

    This model captures the outcome of a tick processing operation,
    including counts and IDs of memories identified for state transitions.

    Attributes:
        tick_id: Unique identifier for this tick operation.
        correlation_id: Correlation ID for distributed tracing.
        processed_at: Authoritative timestamp used for processing.
        memories_expired: Count of memories transitioned to EXPIRED.
        expired_memory_ids: List of memory IDs that should be expired.
        memories_pending_archive: Count of EXPIRED memories ready for archive.
        pending_archive_ids: List of memory IDs ready for archival.
        errors: List of error messages encountered during processing.
    """

    model_config = ConfigDict(strict=True)

    tick_id: UUID = Field(description="Unique tick identifier")
    correlation_id: UUID = Field(description="Correlation ID for tracing")
    processed_at: datetime = Field(
        description="Authoritative timestamp used for processing"
    )

    # Expiration results
    memories_expired: int = Field(
        default=0,
        description="Count of memories transitioned to EXPIRED",
    )
    expired_memory_ids: list[UUID] = Field(
        default_factory=list,
        description="Memory IDs that should transition to EXPIRED state",
    )

    # Archive candidates identified
    memories_pending_archive: int = Field(
        default=0,
        description="Count of EXPIRED memories ready for archive",
    )
    pending_archive_ids: list[UUID] = Field(
        default_factory=list,
        description="Memory IDs in EXPIRED state ready for archival",
    )

    # Errors
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages encountered during tick processing",
    )


class AdapterRuntimeTickMemory:
    """
    Adapter for tick-based memory lifecycle event processing.

    Adapts HandlerRuntimeTick patterns for memory domain:
    - Queries ACTIVE memories with expires_at <= now
    - Queries EXPIRED memories pending archive
    - Returns results for orchestrator to act upon

    IMPORTANT: All time comparisons use the passed `now` parameter,
    which must come from envelope.envelope_timestamp. Never use datetime.now().

    This adapter IDENTIFIES candidates for state transitions. The actual
    state changes are performed by the orchestrator calling the appropriate
    handlers (e.g., HandlerMemoryExpire, HandlerMemoryArchive).

    Example::

        adapter = AdapterRuntimeTickMemory()
        result = await adapter.process_tick(
            tick_id=uuid4(),
            correlation_id=uuid4(),
            now=envelope.envelope_timestamp,  # Use envelope timestamp!
        )

        # Orchestrator handles the actual transitions
        for memory_id in result.expired_memory_ids:
            await expire_handler.execute(memory_id)
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._handler_id = "adapter-runtime-tick-memory"

    @property
    def handler_id(self) -> str:
        """Return handler identifier for tracing."""
        return self._handler_id

    async def process_tick(
        self,
        tick_id: UUID,
        correlation_id: UUID,
        now: datetime,
    ) -> ModelMemoryLifecycleTickResult:
        """
        Process a lifecycle tick.

        This method identifies memories that need lifecycle transitions:
        1. ACTIVE memories where expires_at <= now -> should be EXPIRED
        2. EXPIRED memories -> candidates for ARCHIVE

        Args:
            tick_id: Unique identifier for this tick
            correlation_id: Correlation ID for distributed tracing
            now: Authoritative timestamp from envelope (NOT datetime.now())

        Returns:
            ModelMemoryLifecycleTickResult with identified transitions

        Note:
            This adapter IDENTIFIES candidates. The actual state transitions
            are performed by the orchestrator calling the appropriate handlers.
        """
        result = ModelMemoryLifecycleTickResult(
            tick_id=tick_id,
            correlation_id=correlation_id,
            processed_at=now,
        )

        try:
            # Query 1: Find ACTIVE memories past their expiration
            expired_candidates = await self._find_expired_candidates(now)
            result.memories_expired = len(expired_candidates)
            result.expired_memory_ids = expired_candidates

            # Query 2: Find EXPIRED memories ready for archive
            archive_candidates = await self._find_archive_candidates(now)
            result.memories_pending_archive = len(archive_candidates)
            result.pending_archive_ids = archive_candidates

        except Exception as e:
            result.errors.append(f"Tick processing error: {e!s}")

        return result

    async def _find_expired_candidates(self, now: datetime) -> list[UUID]:
        """
        Find ACTIVE memories that should be expired.

        Query: lifecycle_state = 'active' AND expires_at IS NOT NULL AND expires_at <= now

        For P4A: This is a placeholder. Real implementation will query
        the memory store (Postgres/Qdrant) based on configured backend.

        Args:
            now: Authoritative timestamp for comparison (from envelope)

        Returns:
            List of memory UUIDs that should transition to EXPIRED state
        """
        # TODO: Implement actual DB query
        # For now, return empty list - actual query will be:
        # SELECT memory_id FROM memories
        # WHERE lifecycle_state = 'active'
        #   AND expires_at IS NOT NULL
        #   AND expires_at <= :now
        _ = now  # Acknowledge parameter for future use
        return []

    async def _find_archive_candidates(self, now: datetime) -> list[UUID]:
        """
        Find EXPIRED memories ready for archival.

        Query: lifecycle_state = 'expired' AND archived_at IS NULL

        For P4A: This is a placeholder. Real implementation will query
        the memory store based on configured backend.

        Args:
            now: Authoritative timestamp for comparison (from envelope)

        Returns:
            List of memory UUIDs in EXPIRED state ready for archival
        """
        # TODO: Implement actual DB query
        # For now, return empty list - actual query will be:
        # SELECT memory_id FROM memories
        # WHERE lifecycle_state = 'expired'
        #   AND archived_at IS NULL
        _ = now  # Acknowledge parameter for future use
        return []
