# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Adapter for tick-based memory lifecycle detection.

This adapter provides a clean abstraction over HandlerMemoryTick,
encapsulating the tick-based TTL evaluation and archive detection logic.
It exposes high-level operations for the memory lifecycle orchestrator
without requiring callers to manage handler internals.

Adapter Pattern:
    AdapterRuntimeTickMemory wraps HandlerMemoryTick and provides:
    - A simplified API for processing runtime ticks
    - Health reporting from the underlying handler
    - Metadata introspection (describe())
    - Lifecycle management (initialize / shutdown)

Example::

    from omnibase_core.container import ModelONEXContainer
    from omnimemory.nodes.node_memory_lifecycle_orchestrator.adapters import (
        AdapterRuntimeTickMemory,
    )
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

    container = ModelONEXContainer()
    adapter = AdapterRuntimeTickMemory(container)
    await adapter.initialize(projection_reader=reader, batch_size=50)

    # Process a runtime tick envelope
    output = await adapter.process_tick(envelope)
    print(f"Expired: {output.metrics['expired_count']}")

Related Tickets:
    - OMN-1603: Add adapter implementations for memory lifecycle orchestrator
    - OMN-1392: Original lifecycle orchestrator delivery

.. versionadded:: 0.1.0
    Initial implementation for OMN-1603.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_tick import (
    HandlerMemoryTick,
    ModelMemoryTickHealth,
    ModelMemoryTickMetadata,
    ProtocolMemoryLifecycleProjectionReader,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

    from omnimemory.utils.concurrency import CircuitBreaker

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterRuntimeTickMemory",
    "ModelRuntimeTickAdapterHealth",
    "ModelRuntimeTickAdapterMetadata",
]


class ModelRuntimeTickAdapterHealth(
    BaseModel
):  # omnimemory-model-exempt: adapter health
    """Health status for AdapterRuntimeTickMemory.

    Aggregates health from the underlying HandlerMemoryTick.

    Attributes:
        initialized: Whether the adapter has been initialized.
        handler_health: Health status from the underlying tick handler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )
    handler_health: ModelMemoryTickHealth = Field(
        ...,
        description="Health status from the underlying HandlerMemoryTick",
    )


class ModelRuntimeTickAdapterMetadata(
    BaseModel
):  # omnimemory-model-exempt: adapter config
    """Metadata for AdapterRuntimeTickMemory.

    Attributes:
        name: Adapter class name.
        description: Brief description of adapter purpose.
        handler_metadata: Metadata from the underlying tick handler.
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
    handler_metadata: ModelMemoryTickMetadata = Field(
        ...,
        description="Metadata from the underlying HandlerMemoryTick",
    )
    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )


class AdapterRuntimeTickMemory:
    """Adapter for tick-based memory lifecycle detection.

    Wraps HandlerMemoryTick to provide a clean interface for the lifecycle
    orchestrator. This adapter is responsible for:

    1. Initializing the underlying tick handler with configured dependencies.
    2. Delegating tick processing to the handler.
    3. Exposing health and metadata for observability.

    Container-Driven Pattern:
        Follows the ONEX container-driven initialization pattern:
        - Constructor takes only ModelONEXContainer
        - Dependencies injected via async initialize()
        - health_check() and describe() for introspection

    Attributes:
        _handler: Underlying HandlerMemoryTick instance.
        _initialized: Whether initialize() has been called.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize AdapterRuntimeTickMemory with ONEX container.

        Args:
            container: ONEX container for dependency injection.

        Note:
            Call initialize() before using process_tick().
        """
        self._handler = HandlerMemoryTick(container)
        self._initialized: bool = False

    async def initialize(
        self,
        projection_reader: ProtocolMemoryLifecycleProjectionReader | None = None,
        batch_size: int = 100,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the adapter and the underlying tick handler.

        Args:
            projection_reader: Reader for memory lifecycle projection state.
                If None, handler will return empty results (no-op mode).
            batch_size: Maximum memories to process per tick.
            circuit_breaker: Optional circuit breaker for projection reader
                resilience. A default is created if not provided.
        """
        await self._handler.initialize(
            projection_reader=projection_reader,
            batch_size=batch_size,
            circuit_breaker=circuit_breaker,
        )
        self._initialized = True
        logger.info(
            "AdapterRuntimeTickMemory initialized",
            extra={
                "batch_size": batch_size,
                "has_projection_reader": projection_reader is not None,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter and the underlying handler.

        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            await self._handler.shutdown()
            self._initialized = False
            logger.info("AdapterRuntimeTickMemory shutdown complete")

    @property
    def initialized(self) -> bool:
        """Return whether the adapter has been initialized."""
        return self._initialized

    async def process_tick(
        self,
        envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> ModelHandlerOutput[None]:
        """Process a runtime tick and emit lifecycle transition events.

        Delegates to HandlerMemoryTick.handle() to detect memory expirations
        and archive candidates within the current tick window.

        Args:
            envelope: Event envelope containing the runtime tick payload.

        Returns:
            ModelHandlerOutput containing lifecycle transition events
            (ModelMemoryExpiredEvent, ModelMemoryArchiveInitiated).

        Raises:
            RuntimeError: If the adapter has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "AdapterRuntimeTickMemory not initialized. "
                "Call initialize() before process_tick()."
            )
        return await self._handler.handle(envelope)

    async def health_check(self) -> ModelRuntimeTickAdapterHealth:
        """Return health status of the adapter and its handler.

        Returns:
            ModelRuntimeTickAdapterHealth with initialization status
            and underlying handler health.
        """
        handler_health = await self._handler.health_check()
        return ModelRuntimeTickAdapterHealth(
            initialized=self._initialized,
            handler_health=handler_health,
        )

    async def describe(self) -> ModelRuntimeTickAdapterMetadata:
        """Return metadata and capabilities of this adapter.

        Returns:
            ModelRuntimeTickAdapterMetadata with adapter information
            and underlying handler metadata.
        """
        handler_metadata = await self._handler.describe()
        return ModelRuntimeTickAdapterMetadata(
            name="AdapterRuntimeTickMemory",
            description=(
                "Adapter for tick-based memory lifecycle detection. "
                "Wraps HandlerMemoryTick to provide TTL expiration "
                "and archive candidate detection via runtime tick events."
            ),
            handler_metadata=handler_metadata,
            initialized=self._initialized,
        )
