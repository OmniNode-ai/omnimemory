# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Adapter for memory deactivation/expiration operations.

This adapter wraps HandlerPostgresDeactivate patterns for the memory domain.
Uses ADAPTER strategy per handler_reuse_matrix.md.

The adapter handles transitioning memories from ACTIVE to EXPIRED state,
including setting the expired_at timestamp and incrementing lifecycle_revision.

Example::

    from datetime import datetime, timezone
    from uuid import uuid4

    from omnimemory.nodes.memory_lifecycle_orchestrator.adapters import (
        AdapterPostgresDeactivateMemory,
        ModelMemoryExpireRequest,
    )

    async def expire_memories():
        adapter = AdapterPostgresDeactivateMemory()

        # Use authoritative timestamp from envelope
        envelope_timestamp = datetime.now(timezone.utc)

        request = ModelMemoryExpireRequest(
            memory_ids=[uuid4(), uuid4()],
            now=envelope_timestamp,
            correlation_id=uuid4(),
            reason="ttl_exceeded",
        )

        result = await adapter.expire_memories(request)
        print(f"Expired {result.expired_count} of {result.requested_count} memories")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1392.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.enum_memory_lifecycle_state import EnumMemoryLifecycleState
from omnimemory.nodes.memory_lifecycle_orchestrator.validators.lifecycle_transition_validator import (
    ModelLifecycleTransitionResult,
    apply_transition,
)


class ModelMemoryExpireRequest(BaseModel):
    """Request to expire one or more memories.

    Attributes:
        memory_ids: List of memory UUIDs to expire.
        now: Authoritative timestamp from envelope (never use datetime.now()).
        correlation_id: Correlation ID for distributed tracing.
        reason: Reason for expiration (default: "ttl_exceeded").
    """

    model_config = ConfigDict(strict=True)

    memory_ids: list[UUID] = Field(description="Memory IDs to expire")
    now: datetime = Field(description="Authoritative timestamp from envelope")
    correlation_id: UUID = Field(description="Correlation ID for tracing")
    reason: str = Field(default="ttl_exceeded", description="Reason for expiration")


class ModelMemoryExpireResult(BaseModel):
    """Result of memory expiration operation.

    Attributes:
        correlation_id: Correlation ID for tracing.
        requested_count: Number of memories requested to expire.
        expired_count: Number of memories successfully expired.
        failed_count: Number of memories that failed to expire.
        expired_ids: List of memory IDs that were successfully expired.
        failed_ids: List of memory IDs that failed to expire.
        transitions: List of transition results for each memory.
        errors: List of error messages encountered during expiration.
    """

    model_config = ConfigDict(strict=True)

    correlation_id: UUID = Field(description="Correlation ID for tracing")
    requested_count: int = Field(description="Number of memories requested to expire")
    expired_count: int = Field(description="Number of memories successfully expired")
    failed_count: int = Field(description="Number of memories that failed to expire")
    expired_ids: list[UUID] = Field(default_factory=list)
    failed_ids: list[UUID] = Field(default_factory=list)
    transitions: list[ModelLifecycleTransitionResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AdapterPostgresDeactivateMemory:
    """
    Adapter for memory deactivation/expiration operations.

    Adapts HandlerPostgresDeactivate patterns for memory domain:
    - Transitions memories from ACTIVE -> EXPIRED
    - Sets expired_at timestamp
    - Increments lifecycle_revision
    - Records transition for audit

    IMPORTANT: All timestamps use the passed `now` parameter from
    envelope.envelope_timestamp. Never use datetime.now().

    The adapter uses optimistic concurrency control via lifecycle_revision.
    If the revision changed between read and write, the operation fails for
    that memory to prevent lost updates.

    Example::

        adapter = AdapterPostgresDeactivateMemory()

        request = ModelMemoryExpireRequest(
            memory_ids=[memory_id],
            now=envelope.envelope_timestamp,  # Use envelope timestamp!
            correlation_id=correlation_id,
            reason="ttl_exceeded",
        )

        result = await adapter.expire_memories(request)

        if result.failed_count > 0:
            # Handle partial failures
            for error in result.errors:
                logger.warning(f"Expiration error: {error}")
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._handler_id = "adapter-postgres-deactivate-memory"

    @property
    def handler_id(self) -> str:
        """Return handler identifier for tracing."""
        return self._handler_id

    async def expire_memories(
        self,
        request: ModelMemoryExpireRequest,
    ) -> ModelMemoryExpireResult:
        """
        Expire one or more memories.

        Transitions memories from ACTIVE -> EXPIRED state:
        - Validates transition is allowed via state machine
        - Sets lifecycle_state = EXPIRED
        - Sets expired_at = now
        - Increments lifecycle_revision

        Args:
            request: Expiration request with memory IDs and timestamp

        Returns:
            ModelMemoryExpireResult with success/failure counts

        Note:
            Uses optimistic concurrency via lifecycle_revision.
            If revision changed between read and write, operation fails for that memory.
        """
        result = ModelMemoryExpireResult(
            correlation_id=request.correlation_id,
            requested_count=len(request.memory_ids),
            expired_count=0,
            failed_count=0,
        )

        for memory_id in request.memory_ids:
            try:
                transition = await self._expire_single(
                    memory_id=memory_id,
                    now=request.now,
                    reason=request.reason,
                )
                result.transitions.append(transition)

                if transition.success:
                    result.expired_count += 1
                    result.expired_ids.append(memory_id)
                else:
                    result.failed_count += 1
                    result.failed_ids.append(memory_id)
                    if transition.error:
                        result.errors.append(f"{memory_id}: {transition.error}")

            except Exception as e:
                result.failed_count += 1
                result.failed_ids.append(memory_id)
                result.errors.append(f"{memory_id}: {e!s}")

        return result

    async def _expire_single(
        self,
        memory_id: UUID,
        now: datetime,
        reason: str,
    ) -> ModelLifecycleTransitionResult:
        """
        Expire a single memory.

        For P4A: This is a placeholder. Real implementation will:
        1. Read current memory state and revision from database
        2. Validate transition using apply_transition
        3. Update with optimistic locking (WHERE revision = expected)

        Args:
            memory_id: UUID of the memory to expire
            now: Authoritative timestamp from envelope
            reason: Reason for expiration (for audit)

        Returns:
            ModelLifecycleTransitionResult with transition outcome
        """
        # Acknowledge parameters for future use
        _ = memory_id
        _ = reason

        # TODO: Implement actual DB operation
        # Step 1: Read current memory state and revision
        # SELECT lifecycle_state, lifecycle_revision
        # FROM memories
        # WHERE memory_id = :memory_id
        # FOR UPDATE SKIP LOCKED  -- Prevent concurrent updates

        # Placeholder: assume memory is ACTIVE with revision 0
        current_state = EnumMemoryLifecycleState.ACTIVE
        current_revision = 0

        # Step 2: Validate and compute transition
        transition = apply_transition(
            current_state=current_state,
            target_state=EnumMemoryLifecycleState.EXPIRED,
            current_revision=current_revision,
            now=now,
            is_promotion=False,
        )

        if transition.success:
            # TODO(OMN-1392): Persist state change with optimistic locking.
            # Update lifecycle_state, expired_at, lifecycle_revision WHERE revision matches.
            # If rowcount is 0, another process updated first -> retry or fail.
            pass

        return transition
