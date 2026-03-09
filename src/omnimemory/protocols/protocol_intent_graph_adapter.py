# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Protocol definition for intent graph adapter implementations.

This module defines the contract boundary protocol for intent graph storage
operations. All intent graph adapter implementations must conform to this
protocol for proper contract-driven design and testability.

IMPORTANT: This protocol enables HandlerIntent to depend on an abstraction
rather than a concrete AdapterIntentGraph implementation. This supports:
- Mock implementations for unit testing
- Alternative graph backends (Neo4j, Memgraph, etc.)
- Dependency injection via ONEX container

Example::

    from omnimemory.protocols import ProtocolIntentGraphAdapter

    class MockIntentGraphAdapter:
        '''Test double conforming to ProtocolIntentGraphAdapter.'''

        async def initialize(
            self,
            connection_uri: str,
            auth: tuple[str, str] | None = None,
            *,
            options: Mapping[str, object] | None = None,
        ) -> None:
            pass  # No-op for tests

        async def shutdown(self) -> None:
            pass

        async def store_intent(
            self,
            session_id: str,
            intent_data: ModelIntentClassificationOutput,
            correlation_id: str,
        ) -> ModelIntentStorageResult:
            return ModelIntentStorageResult(
                status="success",
                session_id=session_id,
                intent_id=uuid4(),
                created=True,
            )

        # ... other method implementations

.. versionadded:: 0.2.0
    Initial implementation for OMN-1536.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_core.models.intelligence import (
        ModelIntentClassificationOutput,
        ModelIntentQueryResult,
        ModelIntentStorageResult,
    )

    from omnimemory.handlers.adapters.models import (
        ModelIntentDistributionResult,
    )
    from omnimemory.handlers.adapters.models import (
        ModelIntentQueryResult as LocalModelIntentQueryResult,
    )

__all__ = [
    "ProtocolIntentGraphAdapter",
]


@runtime_checkable
class ProtocolIntentGraphAdapter(Protocol):
    """Protocol for intent graph adapter implementations.

    Defines the contract for all intent graph storage adapters in OmniMemory.
    Implementations wrap a graph database (Memgraph, Neo4j, etc.) and provide
    intent-specific operations:

    - store_intent(): Store an intent classification linked to a session
    - get_session_intents(): Retrieve intents for a given session
    - get_recent_intents(): Query recent intents across all sessions
    - get_intent_distribution(): Get aggregate intent statistics

    Implementations are expected to:
    - Handle session and intent node creation/merging
    - Track relationships between sessions and intents
    - Support confidence-based filtering
    - Provide temporal queries for analytics

    Lifecycle:
        1. Create instance with configuration
        2. Call initialize() with connection parameters
        3. Use store/query methods
        4. Call shutdown() to release resources

    The adapter supports async context manager protocol for automatic
    cleanup via ``async with``.
    """

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter has been initialized.

        Returns:
            True if initialize() has been called successfully and
            shutdown() has not been called, False otherwise.
        """
        ...

    async def initialize(
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the adapter and establish graph database connection.

        Establishes connection to the graph database and prepares
        the adapter for intent storage operations. Creates indexes
        for optimal query performance.

        This method is idempotent - calling it multiple times after
        successful initialization is a no-op.

        Args:
            connection_uri: Graph database URI (e.g., "bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}").
            auth: Optional (username, password) tuple for authentication.
            options: Additional connection options passed to the underlying
                graph handler.

        Raises:
            RuntimeError: If initialization fails or times out.
            ValueError: If connection_uri is malformed.
        """
        ...

    async def shutdown(self) -> None:
        """Shutdown the adapter and release resources.

        Closes the connection to the graph database and cleans up
        internal state. Safe to call multiple times - subsequent
        calls are no-ops.
        """
        ...

    async def store_intent(
        self,
        session_id: str,
        intent_data: ModelIntentClassificationOutput,
        correlation_id: str,
    ) -> ModelIntentStorageResult:
        """Store an intent classification linked to a session.

        Uses MERGE semantics to create or update the session and intent
        nodes. If an intent with the same category already exists for
        the session, its confidence and keywords are updated.

        Args:
            session_id: Unique identifier for the session.
            intent_data: The intent classification output to store.
            correlation_id: Correlation ID for request tracing.

        Returns:
            ModelIntentStorageResult indicating success or failure.
            On success, includes the intent_id and whether a new
            intent was created vs merged.

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        ...

    async def get_session_intents(
        self,
        session_id: str,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> ModelIntentQueryResult:
        """Get intents for a session with optional filtering.

        Retrieves intent classifications associated with the specified
        session, ordered by creation time (most recent first).

        Args:
            session_id: The session identifier to query.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                Defaults to implementation-specific threshold.
            limit: Maximum number of results to return.
                Defaults to implementation-specific maximum.

        Returns:
            Core ``ModelIntentQueryResult`` with bool success field.

        Note:
            This method never raises on business errors - it returns
            success=False with error_message in the result model instead.
        """
        ...

    async def get_recent_intents(
        self,
        time_range_hours: int = 24,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> LocalModelIntentQueryResult:
        """Get recent intents across all sessions within a time range.

        Retrieves intent classifications created within the specified time
        range, optionally filtered by confidence threshold. Results are
        ordered by creation time (most recent first) and include the
        associated session reference.

        Args:
            time_range_hours: Number of hours to look back from now.
                Defaults to 24 hours. Values less than 1 may be clamped
                to a minimum by implementations.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                If None, no confidence filtering is applied.
            limit: Maximum number of results to return.
                Defaults to implementation-specific maximum.

        Returns:
            ModelIntentQueryResult with the list of intents or error status.
            Each intent record includes session_ref populated with the
            session_id it belongs to.

            Possible status values:
            - "success": Query completed with results
            - "no_results": No intents found matching criteria
            - "error": Query failed

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        ...

    async def get_intent_distribution(
        self,
        time_range_hours: int = 24,
    ) -> ModelIntentDistributionResult:
        """Get intent category distribution for analytics.

        Returns the count of intents per category within the specified
        time range. Useful for dashboards and understanding user intent
        patterns.

        Args:
            time_range_hours: Number of hours to look back from now.
                Defaults to 24 hours. Values less than 1 may be clamped
                to a minimum by implementations.

        Returns:
            ModelIntentDistributionResult with distribution data or error status.
            On success, includes the distribution dictionary and total count.

        Example::

            result = await adapter.get_intent_distribution(time_range_hours=48)
            if result.status == "success":
                print(result.distribution)
                # {"debugging": 150, "code_generation": 89, "explanation": 45}

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the graph connection is healthy.

        Returns:
            True if the graph is healthy and accessible, False otherwise.
        """
        ...
