# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for intent handler operations.

This module defines the ProtocolHandlerIntent interface that establishes the
contract for intent storage and query operations. Following the omnibase_infra
container-driven pattern, this protocol enables contract-driven development
and allows for multiple implementations.

The protocol defines:
    - Lifecycle management (initialize, shutdown)
    - Intent storage operations (store_intent)
    - Intent query operations (query_session, query_distribution)
    - Introspection capabilities (health_check, describe)

Example::

    from omnimemory.protocols import ProtocolHandlerIntent

    async def process_intents(handler: ProtocolHandlerIntent) -> None:
        '''Process intents using any conforming handler implementation.'''
        # Verify handler is ready
        if not handler.is_initialized:
            await handler.initialize(
                connection_uri="bolt://localhost:7687",
            )

        # Store an intent
        from uuid import uuid4
        from omnimemory.handlers.adapters.models import ModelIntentClassificationOutput

        result = await handler.store_intent(
            session_id="session_123",
            intent_data=ModelIntentClassificationOutput(
                intent_category="debugging",
                confidence=0.92,
                keywords=["error", "traceback"],
            ),
            correlation_id=uuid4(),
        )

        # Query session intents
        query_result = await handler.query_session(
            session_id="session_123",
            min_confidence=0.5,
        )

.. versionadded:: 0.1.0
    Initial implementation for OMN-1536.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from omnimemory.handlers.adapters.models import (
        ModelIntentClassificationOutput,
        ModelIntentDistributionResult,
        ModelIntentGraphHealth,
        ModelIntentQueryResult,
        ModelIntentStorageResult,
    )
    from omnimemory.handlers.handler_intent import ModelHandlerIntentMetadata

__all__ = [
    "ProtocolHandlerIntent",
]


@runtime_checkable
class ProtocolHandlerIntent(Protocol):
    """Protocol defining the contract for intent handler operations.

    This protocol establishes the interface for handlers that manage intent
    classification storage and retrieval in a graph database. It follows
    the omnibase_infra container-driven pattern for ONEX compliance.

    Implementations must provide:
        - handler_type: Property returning a string identifier for the handler type.
        - is_initialized: Property indicating whether the handler is ready for operations.
        - initialize(): Async method to establish connections and prepare the handler.
        - shutdown(): Async method to gracefully close connections and release resources.
        - store_intent(): Async method to store an intent classification for a session.
        - query_session(): Async method to retrieve intents for a specific session.
        - query_distribution(): Async method to get intent category statistics.
        - health_check(): Async method to verify handler and adapter health.
        - describe(): Async method to return handler metadata and capabilities.

    Thread Safety:
        Implementations should use appropriate synchronization (e.g., asyncio.Lock)
        for initialization to prevent race conditions when multiple coroutines
        call initialize() concurrently.

    Error Handling:
        Business operation methods (store_intent, query_session, query_distribution)
        should return error status in response models rather than raising exceptions.
        This allows API layers to handle errors gracefully without try/except blocks.

    Example::

        class CustomIntentHandler:
            '''Custom implementation of ProtocolHandlerIntent.'''

            @property
            def handler_type(self) -> str:
                return "custom-intent"

            @property
            def is_initialized(self) -> bool:
                return self._initialized

            async def initialize(
                self,
                connection_uri: str,
                auth: tuple[str, str] | None = None,
                *,
                options: Mapping[str, object] | None = None,
            ) -> None:
                # Implementation details...
                pass

            # ... other required methods

        # Verify conformance at runtime
        handler = CustomIntentHandler()
        assert isinstance(handler, ProtocolHandlerIntent)
    """

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            String identifying this handler type (e.g., "intent").
            Used for logging, registration, and handler discovery.
        """
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized.

        Returns:
            True if handler is initialized and ready for operations,
            False otherwise.
        """
        ...

    async def initialize(
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize handler and establish connections.

        Establishes connection to the graph database by creating and
        initializing the underlying adapter. This method should be
        idempotent - calling it multiple times after successful
        initialization should be a no-op.

        Args:
            connection_uri: Graph database URI (e.g., "bolt://localhost:7687").
            auth: Optional (username, password) tuple for authentication.
            options: Additional configuration options. Common options include:
                - timeout_seconds: Operation timeout (default: 30.0)
                - max_intents_per_session: Max intents per query (default: 100)
                - default_confidence_threshold: Min confidence filter (default: 0.0)
                - auto_create_indexes: Create indexes on init (default: True)

        Raises:
            RuntimeError: If initialization fails or times out.
            ValueError: If connection_uri is malformed.
        """
        ...

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Shutdown handler and close connections.

        Gracefully shuts down the handler by closing the underlying
        adapter and releasing resources. Safe to call multiple times.

        Args:
            timeout_seconds: Maximum time to wait for shutdown. Defaults to 30.0.
        """
        ...

    async def store_intent(
        self,
        session_id: str,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: UUID,
        user_context: str = "",
    ) -> ModelIntentStorageResult:
        """Store an intent classification linked to a session.

        Creates or updates an intent node in the graph database and links
        it to the specified session using MERGE semantics.

        Args:
            session_id: Unique identifier for the session.
            intent_data: The intent classification output to store.
            correlation_id: Correlation ID for request tracing.
            user_context: Optional user context string for the session.

        Returns:
            ModelIntentStorageResult indicating success or failure.
            On success, includes the intent_id and whether a new
            intent was created vs merged.

        Note:
            This method should not raise on business errors - it should
            return an error status in the result model instead.
        """
        ...

    async def query_session(
        self,
        session_id: str,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> ModelIntentQueryResult:
        """Retrieve intents for a specific session.

        Queries the graph database for intents associated with the
        specified session, with optional filtering by confidence
        threshold and result limit.

        Args:
            session_id: The session identifier to query.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                If None, uses the handler's default threshold.
            limit: Maximum number of results to return.
                If None, uses the handler's default limit.

        Returns:
            ModelIntentQueryResult with the list of intents or error status.

        Note:
            This method should not raise on business errors - it should
            return an error status in the result model instead.
        """
        ...

    async def query_distribution(
        self,
        time_range_hours: int = 24,
    ) -> ModelIntentDistributionResult:
        """Get intent category distribution for analytics.

        Returns the count of intents per category within the specified
        time range, useful for analytics and reporting.

        Args:
            time_range_hours: Number of hours to look back from now.
                Defaults to 24 hours.

        Returns:
            ModelIntentDistributionResult with distribution data or error status.

        Note:
            This method should not raise on business errors - it should
            return an error status in the result model instead.
        """
        ...

    async def health_check(self) -> ModelIntentGraphHealth:
        """Check if the handler and adapter are healthy.

        Verifies connectivity to the graph database and gathers
        statistics about stored data.

        Returns:
            ModelIntentGraphHealth with detailed health status.
            This method should not raise - errors should be captured
            in the result model.
        """
        ...

    async def describe(self) -> ModelHandlerIntentMetadata:
        """Return handler metadata and capabilities.

        Provides information about the handler type, supported operations,
        and current initialization state.

        Returns:
            ModelHandlerIntentMetadata with handler information including:
            - handler_type: The handler type identifier
            - capabilities: List of supported operations
            - adapter_type: Type of adapter being wrapped
            - initialized: Whether the handler is currently initialized
        """
        ...
