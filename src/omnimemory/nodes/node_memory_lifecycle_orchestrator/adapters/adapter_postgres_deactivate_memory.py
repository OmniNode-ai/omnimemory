# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Adapter for memory deactivation (expiration) operations via PostgreSQL.

This adapter provides a clean abstraction over HandlerMemoryExpire,
encapsulating the ACTIVE -> EXPIRED state transition logic and exposing
a simplified deactivation API for the memory lifecycle orchestrator.

Adapter Pattern:
    AdapterPostgresDeactivateMemory wraps HandlerMemoryExpire and provides:
    - deactivate(): Single expiration with optimistic locking
    - deactivate_with_retry(): Expiration with automatic conflict retry
    - Health reporting from the underlying handler
    - Metadata introspection (describe())
    - Lifecycle management (initialize / shutdown)

Example::

    from omnibase_core.container import ModelONEXContainer
    from omnimemory.nodes.node_memory_lifecycle_orchestrator.adapters import (
        AdapterPostgresDeactivateMemory,
    )
    from uuid import uuid4

    container = ModelONEXContainer()
    adapter = AdapterPostgresDeactivateMemory(container)
    await adapter.initialize(db_pool=pool, max_retries=3)

    # Deactivate a memory (ACTIVE -> EXPIRED)
    result = await adapter.deactivate(
        memory_id=uuid4(),
        expected_revision=5,
        reason="ttl_expired",
    )
    if result.success:
        print(f"Memory deactivated, new revision: {result.new_revision}")
    elif result.conflict:
        print("Concurrent modification, retry needed")

Related Tickets:
    - OMN-1603: Add adapter implementations for memory lifecycle orchestrator
    - OMN-1392: Original lifecycle orchestrator delivery

.. versionadded:: 0.1.0
    Initial implementation for OMN-1603.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
    HandlerMemoryExpire,
    ModelExpireMemoryCommand,
    ModelMemoryExpireHealth,
    ModelMemoryExpireMetadata,
    ModelMemoryExpireResult,
)

if TYPE_CHECKING:
    from asyncpg import Pool
    from omnibase_core.container import ModelONEXContainer

    from omnimemory.utils.concurrency import CircuitBreaker

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterPostgresDeactivateMemory",
    "ModelDeactivateAdapterHealth",
    "ModelDeactivateAdapterMetadata",
]


class ModelDeactivateAdapterHealth(
    BaseModel
):  # omnimemory-model-exempt: adapter health
    """Health status for AdapterPostgresDeactivateMemory.

    Aggregates health from the underlying HandlerMemoryExpire.

    Attributes:
        initialized: Whether the adapter has been initialized.
        handler_health: Health status from the underlying expire handler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )
    handler_health: ModelMemoryExpireHealth = Field(
        ...,
        description="Health status from the underlying HandlerMemoryExpire",
    )


class ModelDeactivateAdapterMetadata(
    BaseModel
):  # omnimemory-model-exempt: adapter config
    """Metadata for AdapterPostgresDeactivateMemory.

    Attributes:
        name: Adapter class name.
        description: Brief description of adapter purpose.
        handler_metadata: Metadata from the underlying expire handler.
        initialized: Whether the adapter has been initialized.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ...,
        description="Adapter class name",
    )
    description: str = Field(
        ...,
        description="Brief description of adapter purpose",
    )
    handler_metadata: ModelMemoryExpireMetadata = Field(
        ...,
        description="Metadata from the underlying HandlerMemoryExpire",
    )
    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )


class AdapterPostgresDeactivateMemory:
    """Adapter for memory deactivation (expiration) via PostgreSQL.

    Wraps HandlerMemoryExpire to provide a clean interface for the lifecycle
    orchestrator. This adapter is responsible for:

    1. Initializing the underlying expire handler with a database pool.
    2. Delegating ACTIVE -> EXPIRED state transitions to the handler.
    3. Exposing health and metadata for observability.

    The adapter uses the term "deactivate" to emphasize the semantic of
    removing a memory from the active set, which aligns with the lifecycle
    orchestrator's responsibility.

    Container-Driven Pattern:
        Follows the ONEX container-driven initialization pattern:
        - Constructor takes only ModelONEXContainer
        - Dependencies injected via async initialize()
        - health_check() and describe() for introspection

    Attributes:
        _handler: Underlying HandlerMemoryExpire instance.
        _initialized: Whether initialize() has been called.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize AdapterPostgresDeactivateMemory with ONEX container.

        Args:
            container: ONEX container for dependency injection.

        Note:
            Call initialize() with a db_pool before using deactivate().
        """
        self._handler = HandlerMemoryExpire(container)
        self._initialized: bool = False

    async def initialize(
        self,
        db_pool: Pool,
        max_retries: int = 3,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the adapter and the underlying expire handler.

        Args:
            db_pool: PostgreSQL connection pool for database operations.
            max_retries: Maximum retry attempts for deactivate_with_retry().
                Defaults to 3 attempts.
            circuit_breaker: Optional circuit breaker for database operation
                resilience. A default is created if not provided.

        Raises:
            ValueError: If max_retries is less than 1. Validation is delegated
                to HandlerMemoryExpire.initialize(), which enforces this constraint.
        """
        await self._handler.initialize(
            db_pool=db_pool,
            max_retries=max_retries,
            circuit_breaker=circuit_breaker,
        )
        self._initialized = True
        logger.info(
            "AdapterPostgresDeactivateMemory initialized",
            extra={"max_retries": max_retries},
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter and the underlying handler.

        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            await self._handler.shutdown()
            self._initialized = False
            logger.info("AdapterPostgresDeactivateMemory shutdown complete")

    @property
    def initialized(self) -> bool:
        """Return whether the adapter has been initialized."""
        return self._initialized

    async def deactivate(
        self,
        memory_id: UUID,
        expected_revision: int,
        reason: str = "ttl_expired",
        expired_at: datetime | None = None,
    ) -> ModelMemoryExpireResult:
        """Deactivate a memory by transitioning it from ACTIVE to EXPIRED.

        Performs a single attempt with optimistic locking. On revision conflict
        (concurrent modification), the caller is responsible for re-reading
        and retrying with the updated revision. For automatic retry behavior,
        use deactivate_with_retry().

        Args:
            memory_id: UUID of the memory to deactivate.
            expected_revision: Expected lifecycle revision for optimistic lock.
                Must match the current revision in the database.
            reason: Reason for deactivation (for audit trail).
                Defaults to "ttl_expired".
            expired_at: Optional explicit expiration timestamp.
                Defaults to the current UTC time if not provided.

        Returns:
            ModelMemoryExpireResult indicating:
            - success=True: Transition completed, new_revision populated.
            - success=False, conflict=True: Revision mismatch, retry eligible.
            - success=False, conflict=False: Hard failure (invalid state, not found).

        Raises:
            RuntimeError: If the adapter has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "AdapterPostgresDeactivateMemory not initialized. "
                "Call initialize() before deactivate()."
            )
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=expected_revision,
            reason=reason,
            expired_at=expired_at,
        )
        return await self._handler.handle(command)

    async def deactivate_with_retry(
        self,
        memory_id: UUID,
        initial_revision: int,
        reason: str = "ttl_expired",
        expired_at: datetime | None = None,
    ) -> ModelMemoryExpireResult:
        """Deactivate a memory with automatic retry on conflict.

        Delegates retry logic to the handler's handle_with_retry() method,
        which re-reads the current revision and retries up to max_retries times.

        Args:
            memory_id: UUID of the memory to deactivate.
            initial_revision: Starting revision for the first attempt.
            reason: Reason for deactivation (for audit trail).
            expired_at: Optional explicit expiration timestamp.

        Returns:
            ModelMemoryExpireResult with final outcome:
            - success=True: Deactivation eventually succeeded.
            - success=False, conflict=True: Max retries exceeded.
            - success=False, conflict=False: Hard failure.

        Raises:
            RuntimeError: If the adapter has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "AdapterPostgresDeactivateMemory not initialized. "
                "Call initialize() before deactivate_with_retry()."
            )
        return await self._handler.handle_with_retry(
            memory_id=memory_id,
            initial_revision=initial_revision,
            reason=reason,
            expired_at=expired_at,
        )

    async def health_check(self) -> ModelDeactivateAdapterHealth:
        """Return health status of the adapter and its handler.

        Returns:
            ModelDeactivateAdapterHealth with initialization status
            and underlying handler health.
        """
        handler_health = await self._handler.health_check()
        return ModelDeactivateAdapterHealth(
            initialized=self._initialized,
            handler_health=handler_health,
        )

    async def describe(self) -> ModelDeactivateAdapterMetadata:
        """Return metadata and capabilities of this adapter.

        Returns:
            ModelDeactivateAdapterMetadata with adapter information
            and underlying handler metadata.
        """
        handler_metadata = await self._handler.describe()
        return ModelDeactivateAdapterMetadata(
            name="AdapterPostgresDeactivateMemory",
            description=(
                "Adapter for memory deactivation (expiration) via PostgreSQL. "
                "Wraps HandlerMemoryExpire to provide ACTIVE -> EXPIRED "
                "state transitions with optimistic locking."
            ),
            handler_metadata=handler_metadata,
            initialized=self._initialized,
        )
