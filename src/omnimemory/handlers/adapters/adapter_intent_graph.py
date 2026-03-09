# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Adapter for storing intent classifications in Memgraph.

This adapter implements ProtocolIntentGraphAdapter from
omnimemory.protocols.protocol_intent_graph_adapter,
providing a Memgraph-backed implementation for storing and retrieving intent
classifications.

The storage boundary accepts **classification output** (category, confidence,
keywords), not raw input. Classification happens upstream; this adapter
persists the results.

Operations:
- store_intent(): Store an intent classification linked to a session
- get_session_intents(): Retrieve intents for a given session
- health_check(): Check if the graph connection is healthy
- get_recent_intents(): Query recent intents across all sessions (extension)
- get_intent_distribution(): Get aggregate intent statistics (extension)
- get_health_details(): Get detailed health status (extension)

The adapter handles:
- Session and Intent node creation/merging
- Relationship tracking between sessions and intents
- Confidence-based filtering
- Temporal queries for analytics

Example::

    async def example():
        config = ModelAdapterIntentGraphConfig(timeout_seconds=30.0)
        adapter = AdapterIntentGraph(config)
        await adapter.initialize(
            connection_uri="bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}",
        )

        # Store an intent (classification already happened upstream)
        classification = ModelIntentClassificationOutput(
            intent_category="debugging",
            confidence=0.92,
            keywords=["error", "traceback"],
        )
        result = await adapter.store_intent(
            session_id="session_123",
            intent_data=classification,
            correlation_id="corr-456",
        )
        if result.status == "success":
            print(f"Stored intent: {result.intent_id}")

        await adapter.shutdown()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1457.

.. versionchanged:: 0.2.0
    Migrated to ProtocolIntentGraphAdapter from omnimemory.protocols.protocol_intent_graph_adapter (OMN-1476).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import cast
from urllib.parse import urlparse
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.intelligence.enum_intent_category import EnumIntentCategory
from omnibase_core.models.intelligence import (
    ModelIntentClassificationOutput as CoreModelIntentClassificationOutput,
)
from omnibase_core.models.intelligence import (
    ModelIntentQueryResult as CoreModelIntentQueryResult,
)
from omnibase_core.models.intelligence import (
    ModelIntentRecord as CoreModelIntentRecord,
)
from omnibase_core.models.intelligence import (
    ModelIntentStorageResult as CoreModelIntentStorageResult,
)
from omnibase_core.types.type_json import JsonType
from omnibase_spi.protocols.storage.protocol_graph_database_handler import (
    ProtocolGraphDatabaseHandler,
)

from omnimemory.handlers.adapters.models import (
    ModelAdapterIntentGraphConfig,
    ModelIntentDistributionResult,
    ModelIntentGraphHealth,
    ModelIntentQueryResult,
    ModelIntentRecord,
)
from omnimemory.protocols.protocol_intent_graph_adapter import (
    ProtocolIntentGraphAdapter,
)

__all__ = ["AdapterIntentGraph", "IntentCypherTemplates"]

logger = logging.getLogger(__name__)


def _create_graph_handler(
    container: ModelONEXContainer,
) -> ProtocolGraphDatabaseHandler:
    """Lazily resolve and instantiate the concrete HandlerGraph at runtime.

    Uses ``importlib`` to import the concrete ``HandlerGraph`` class from
    ``omnibase_infra`` so that this module depends only on the SPI protocol
    at import time, not on the concrete infrastructure implementation.

    Args:
        container: The ONEX container for dependency injection.

    Returns:
        A concrete handler instance satisfying ProtocolGraphDatabaseHandler.

    Raises:
        RuntimeError: If the concrete class cannot be imported or instantiated.
    """
    try:
        module = importlib.import_module("omnibase_infra.handlers.handler_graph")
        handler_cls = module.HandlerGraph
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "Failed to resolve concrete HandlerGraph from omnibase_infra. "
            "Ensure omnibase_infra is installed."
        ) from e
    handler: ProtocolGraphDatabaseHandler = handler_cls(container)
    return handler


class IntentCypherTemplates:
    """Parameterized Cypher query templates for intent graph operations.

    All template methods accept label/type parameters to allow configurable
    graph schema (e.g., "Session", "Intent", "HAD_INTENT").

    Design Rationale:
        - Session nodes are MERGE'd by session_id to avoid duplicates
        - Intent nodes are MERGE'd by (session)->(intent_category) to allow
          updating confidence/keywords when same intent category is detected
          again within the same session
        - Relationship properties track when each intent was detected

    Security:
        All dynamic data values use Cypher parameters ($param syntax) to prevent
        injection attacks - these are handled safely by the database driver.

        Label/type parameters (session_label, intent_label, rel_type) use f-string
        interpolation, which is SAFE because of the validation chain:

        1. Labels/types originate from ModelAdapterIntentGraphConfig fields
        2. Config validates via _CYPHER_IDENTIFIER_PATTERN regex: ^[A-Za-z_][A-Za-z0-9_]*$
        3. This pattern ONLY allows: letters, numbers, underscores (starting with letter/_)
        4. Injection payloads require special chars (quotes, braces, semicolons, etc.)
        5. Since those chars are rejected by validation, injection is impossible

        Example: "Session" passes validation, "Session'; DROP" fails validation.

        NEVER bypass config validation or accept unvalidated strings for labels.
    """

    @staticmethod
    def store_intent_query(session_label: str, intent_label: str, rel_type: str) -> str:
        """Generate query to store an intent classification for a session."""
        return f"""
        MERGE (s:{session_label} {{session_id: $session_id}})
        ON CREATE SET s.started_at_utc = $started_at_utc, s.user_context = $user_context
        MERGE (s)-[r:{rel_type}]->(i:{intent_label} {{intent_category: $intent_category}})
        ON CREATE SET i.intent_id = $intent_id, i.created_at_utc = $created_at_utc, i.confidence = $confidence, i.keywords = $keywords
        ON MATCH SET i.confidence = $confidence, i.keywords = $keywords
        SET r.timestamp_utc = $timestamp_utc, r.confidence = $confidence, r.correlation_id = $correlation_id
        RETURN i.intent_id AS intent_id, i.created_at_utc = $created_at_utc AS was_created
        """

    @staticmethod
    def get_session_intents_query(
        session_label: str, intent_label: str, rel_type: str
    ) -> str:
        """Generate query to retrieve intents for a session."""
        return f"""
        MATCH (s:{session_label} {{session_id: $session_id}})-[r:{rel_type}]->(i:{intent_label})
        WHERE i.confidence >= $min_confidence
        RETURN i.intent_id AS intent_id, i.intent_category AS intent_category, i.confidence AS confidence,
               i.keywords AS keywords, i.created_at_utc AS created_at_utc, r.correlation_id AS correlation_id
        ORDER BY i.created_at_utc DESC
        LIMIT $limit
        """

    @staticmethod
    def get_intent_distribution_query(intent_label: str) -> str:
        """Generate query to get intent distribution by category."""
        return f"""
        MATCH (i:{intent_label})
        WHERE i.created_at_utc >= $since_utc
        RETURN i.intent_category AS category, count(i) AS count
        ORDER BY count DESC
        """

    @staticmethod
    def get_recent_intents_query(
        session_label: str, intent_label: str, rel_type: str
    ) -> str:
        """Generate query to retrieve recent intents across all sessions.

        Returns intents created within a time range, optionally filtered by
        confidence threshold, ordered by creation time (most recent first).

        Args:
            session_label: Label for session nodes (e.g., "Session").
            intent_label: Label for intent nodes (e.g., "Intent").
            rel_type: Relationship type connecting sessions to intents
                (e.g., "HAD_INTENT").

        Returns:
            Parameterized Cypher query string expecting:
            - $cutoff_time: ISO datetime string for time boundary
            - $min_confidence: Minimum confidence threshold (or null to skip)
            - $limit: Maximum number of results to return
        """
        return f"""
        MATCH (s:{session_label})-[r:{rel_type}]->(i:{intent_label})
        WHERE i.created_at_utc >= $cutoff_time
          AND ($min_confidence IS NULL OR i.confidence >= $min_confidence)
        RETURN s.session_id AS session_id, i.intent_id AS intent_id,
               i.intent_category AS intent_category, i.confidence AS confidence,
               i.keywords AS keywords, i.created_at_utc AS created_at_utc,
               r.correlation_id AS correlation_id
        ORDER BY i.created_at_utc DESC
        LIMIT $limit
        """

    @staticmethod
    def create_indexes_queries(
        session_label: str, intent_label: str, rel_type: str
    ) -> list[str]:
        """Generate index creation queries for intent graph schema.

        Uses ``CREATE INDEX IF NOT EXISTS`` syntax (Memgraph 2.0+) to ensure
        idempotent index creation without relying on error handling for
        duplicate index detection.

        Args:
            session_label: Label for session nodes (e.g., "Session").
            intent_label: Label for intent nodes (e.g., "Intent").
            rel_type: Relationship type connecting sessions to intents
                (e.g., "HAD_INTENT").

        Returns:
            List of index creation queries for both node properties and
            relationship properties.

        Note:
            Memgraph 2.0+ supports edge property indexes via the
            ``CREATE EDGE INDEX`` syntax. The ``timestamp_utc`` property
            on relationships is indexed to optimize temporal queries that
            filter or order by relationship timestamp.
        """
        return [
            # Node property indexes
            f"CREATE INDEX IF NOT EXISTS ON :{session_label}(session_id);",
            f"CREATE INDEX IF NOT EXISTS ON :{intent_label}(intent_id);",
            f"CREATE INDEX IF NOT EXISTS ON :{intent_label}(intent_category);",
            f"CREATE INDEX IF NOT EXISTS ON :{intent_label}(created_at_utc);",
            # Edge property index for temporal queries on relationship timestamp
            f"CREATE EDGE INDEX IF NOT EXISTS ON :{rel_type}(timestamp_utc);",
        ]

    @staticmethod
    def count_all_query(session_label: str, intent_label: str) -> str:
        """Generate query to count both session and intent nodes in one call."""
        return f"""
        OPTIONAL MATCH (s:{session_label})
        WITH count(s) AS session_count
        OPTIONAL MATCH (i:{intent_label})
        RETURN session_count, count(i) AS intent_count
        """


class AdapterIntentGraph(ProtocolIntentGraphAdapter):
    """Adapter that wraps a ProtocolGraphDatabaseHandler for intent classification storage.

    Implements the ProtocolIntentGraphAdapter protocol from
    omnimemory.protocols.protocol_intent_graph_adapter, providing a
    Memgraph-backed implementation for storing and retrieving intent
    classifications.

    This adapter provides an intent-domain interface on top of the generic
    graph handler, translating intent storage and retrieval operations
    into graph queries:

    - store_intent(session_id, intent_data): Store intent linked to session
    - get_session_intents(session_id): Retrieve intents for a session
    - get_recent_intents(time_range, min_confidence): Query recent intents across sessions
    - get_intent_distribution(time_range): Get intent category statistics
    - health_check(): Check if the graph connection is healthy

    The adapter handles:
    - Session node creation with MERGE semantics
    - Intent node creation/update with MERGE semantics
    - Relationship properties for correlation tracking
    - Confidence-based filtering and time-range queries

    Note:
        The storage boundary accepts **classification output** (category,
        confidence, keywords), not raw input. Classification happens upstream;
        this adapter persists the results.

    Attributes:
        config: The adapter configuration.
        handler: The underlying ProtocolGraphDatabaseHandler instance.

    Example::

        async def example():
            config = ModelAdapterIntentGraphConfig(
                timeout_seconds=30.0,
                max_intents_per_session=100,
            )
            adapter = AdapterIntentGraph(config)
            await adapter.initialize(
                connection_uri="bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}",
            )

            # Store intent classification (classification happened upstream)
            result = await adapter.store_intent(
                session_id="sess_123",
                intent_data=ModelIntentClassificationOutput(
                    intent_category="code_generation",
                    confidence=0.95,
                    keywords=["python", "function"],
                ),
                correlation_id="corr-456",
            )

            # Query intents for session
            query_result = await adapter.get_session_intents(
                session_id="sess_123",
                min_confidence=0.8,
            )
            for intent in query_result.intents:
                print(f"{intent.intent_category}: {intent.confidence}")

            await adapter.shutdown()
    """

    def __init__(
        self,
        config: ModelAdapterIntentGraphConfig,
        container: ModelONEXContainer | None = None,
    ) -> None:
        """Initialize the adapter with configuration.

        Args:
            config: The adapter configuration controlling timeouts, labels,
                and query limits.
            container: Optional ONEX container for dependency injection.
                If not provided, a minimal container will be created during
                initialization.
        """
        self._config = config
        self._container = container
        self._handler: ProtocolGraphDatabaseHandler | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def __aenter__(self) -> AdapterIntentGraph:
        """Enter async context manager.

        Note: initialize() must still be called separately as it requires
        connection parameters.

        Returns:
            Self for use in async with statement.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, ensuring shutdown is called.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        await self.shutdown()

    @property
    def config(self) -> ModelAdapterIntentGraphConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def handler(self) -> ProtocolGraphDatabaseHandler | None:
        """Get the underlying graph handler (None if not initialized)."""
        return self._handler

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter has been initialized."""
        return self._initialized

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"AdapterIntentGraph(initialized={self._initialized})"

    async def initialize(
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the adapter and underlying graph handler.

        Establishes connection to the graph database and prepares
        the handler for intent storage operations. Creates indexes
        for optimal query performance.

        This method is idempotent - calling it multiple times after
        successful initialization is a no-op.

        Args:
            connection_uri: Graph database URI (e.g., "bolt://{OMNIMEMORY_MEMGRAPH_HOST}:{OMNIMEMORY_MEMGRAPH_PORT}").
            auth: Optional (username, password) tuple for authentication.
            options: Additional connection options passed to the graph handler.

        Raises:
            RuntimeError: If initialization fails or times out.
            ValueError: If connection_uri is malformed.
        """
        # Validate URI format before attempting connection
        parsed_uri = urlparse(connection_uri)
        if not parsed_uri.scheme or not parsed_uri.hostname:
            raise ValueError(
                f"Invalid connection_uri: '{connection_uri}'. "
                "Expected format: 'bolt://hostname:port' or 'bolt+s://hostname:port'"
            )
        if parsed_uri.scheme not in ("bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s"):
            logger.warning(
                "Unexpected URI scheme '%s' in connection_uri. "
                "Expected bolt, bolt+s, bolt+ssc, neo4j, or neo4j+s.",
                parsed_uri.scheme,
            )

        try:
            # Timeout covers both lock acquisition and initialization work
            async with asyncio.timeout(self._config.timeout_seconds):
                async with self._init_lock:
                    # Early return for already-initialized (idempotent)
                    if self._initialized:
                        return

                    try:
                        # Create container if not provided
                        if self._container is None:
                            self._container = ModelONEXContainer()

                        # Create and initialize handler via lazy resolution
                        self._handler = _create_graph_handler(self._container)
                        # Assert for type narrowing: pyright doesn't narrow instance
                        # attributes after assignment due to potential concurrent modification
                        assert self._handler is not None

                        init_options: dict[str, JsonType] = {
                            "timeout_seconds": self._config.timeout_seconds,
                        }
                        if options:
                            init_options.update(cast("Mapping[str, JsonType]", options))

                        await self._handler.initialize(
                            connection_uri=connection_uri,
                            auth=auth,
                            options=init_options,
                        )

                        # Log safe URI (without credentials)
                        safe_uri = f"{parsed_uri.scheme}://{parsed_uri.hostname}"
                        if parsed_uri.port:
                            safe_uri += f":{parsed_uri.port}"
                        logger.info(
                            "AdapterIntentGraph initialized with connection to %s",
                            safe_uri,
                        )

                        # Ensure indexes exist for optimal query performance
                        # self._initialized is set to True only after this completes
                        # to prevent other coroutines from seeing a partially-initialized
                        # adapter (race condition guard).
                        await self._ensure_indexes()

                        self._initialized = True

                    except Exception as e:
                        logger.error(
                            "Failed to initialize AdapterIntentGraph: %s",
                            e,
                        )
                        raise RuntimeError(f"Initialization failed: {e}") from e

        except TimeoutError as e:
            raise RuntimeError(
                f"Initialization timed out after {self._config.timeout_seconds}s. "
                "Possible causes: (1) Lock contention - another coroutine may be "
                "holding the initialization lock; (2) Database connection issue - "
                "the graph database may be slow or unresponsive. Suggestions: "
                "Check if another initialization is in progress, verify the "
                "database is reachable, or increase timeout_seconds in config."
            ) from e

    async def _ensure_indexes(self) -> None:
        """Create indexes for optimal query performance.

        Index creation is idempotent via ``CREATE INDEX IF NOT EXISTS`` syntax
        (Memgraph 2.0+). This method is safe to call multiple times.

        The method respects the ``auto_create_indexes`` config option - if set
        to False, index creation is skipped entirely. This is useful for:
        - Testing environments where indexes are not needed
        - Deployments where indexes are managed externally (e.g., migrations)
        - Databases that don't support the IF NOT EXISTS syntax
        """
        if self._handler is None:
            return

        if not self._config.auto_create_indexes:
            logger.debug(
                "Skipping automatic index creation (auto_create_indexes=False)"
            )
            return

        index_queries = IntentCypherTemplates.create_indexes_queries(
            session_label=self._config.session_node_label,
            intent_label=self._config.intent_node_label,
            rel_type=self._config.relationship_type,
        )

        successful = 0
        failed = 0

        for query in index_queries:
            try:
                await self._handler.execute_query(query=query, parameters={})
                successful += 1
                logger.debug("Index ensured: %s", query.strip()[:60])
            except Exception as e:
                failed += 1
                # Log warning but don't fail initialization - indexes improve
                # performance but are not required for correctness
                logger.warning(
                    "Index creation failed (non-fatal): query=%s error=%s",
                    query.strip()[:60],
                    e,
                )

        if failed > 0:
            logger.warning(
                "Index creation completed with errors: %d successful, %d failed",
                successful,
                failed,
            )
        else:
            logger.info(
                "All %d indexes created or verified successfully",
                successful,
            )

    async def shutdown(self) -> None:
        """Shutdown the adapter and release resources.

        Closes the connection to the graph database and cleans up
        internal state. Safe to call multiple times.
        """
        if self._initialized and self._handler is not None:
            await self._handler.shutdown()
            self._handler = None
            self._initialized = False
            logger.info("AdapterIntentGraph shutdown complete")

    def _ensure_initialized(self) -> ProtocolGraphDatabaseHandler:
        """Ensure adapter is initialized and return handler.

        Returns:
            The initialized ProtocolGraphDatabaseHandler.

        Raises:
            RuntimeError: If adapter is not initialized.
        """
        if not self._initialized or self._handler is None:
            raise RuntimeError(
                "AdapterIntentGraph not initialized. Call initialize() first."
            )
        return self._handler

    async def store_intent(
        self,
        session_id: str,
        intent_data: CoreModelIntentClassificationOutput,
        correlation_id: str,
    ) -> CoreModelIntentStorageResult:
        """Store an intent classification linked to a session.

        Implements ProtocolIntentGraphAdapter.store_intent.

        Uses MERGE semantics to create or update the session and intent
        nodes. If an intent with the same category already exists for
        the session, its confidence and keywords are updated.

        Args:
            session_id: Unique identifier for the session.
            intent_data: The classification output containing category,
                confidence, and keywords.
            correlation_id: Correlation ID for request tracing.

        Returns:
            ModelIntentStorageResult indicating success or failure with
            metadata about the stored intent.

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        # intent_data.intent_category is EnumIntentCategory in CoreModelIntentClassificationOutput.
        intent_category_str = intent_data.intent_category.value

        # Validate session_id is non-empty
        if not session_id or not session_id.strip():
            return CoreModelIntentStorageResult(
                success=False,
                error_message="session_id cannot be empty",
            )

        try:
            handler = self._ensure_initialized()
        except RuntimeError as e:
            return CoreModelIntentStorageResult(
                success=False,
                error_message=str(e),
            )

        start_time = time.perf_counter()
        intent_id = uuid4()
        timestamp_utc = datetime.now(UTC)

        # Validate correlation_id as UUID
        parsed_correlation_id: UUID | None = None
        if correlation_id:
            try:
                parsed_correlation_id = UUID(correlation_id)
            except ValueError:
                logger.warning("Invalid correlation_id format: %s", correlation_id)
                parsed_correlation_id = None

        # Extract confidence and keywords defensively
        confidence_raw = intent_data.confidence
        confidence_val = (
            float(confidence_raw) if isinstance(confidence_raw, int | float) else 0.0
        )
        keywords_raw = intent_data.keywords
        keywords_val = list(keywords_raw) if keywords_raw else []

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                query = IntentCypherTemplates.store_intent_query(
                    session_label=self._config.session_node_label,
                    intent_label=self._config.intent_node_label,
                    rel_type=self._config.relationship_type,
                )

                timestamp_utc_str = timestamp_utc.isoformat()
                parameters: dict[str, JsonType] = {
                    "session_id": session_id,
                    "started_at_utc": timestamp_utc_str,
                    "user_context": "",
                    "intent_id": str(intent_id),
                    "intent_category": intent_category_str,
                    "confidence": confidence_val,
                    "keywords": cast("list[JsonType]", keywords_val),
                    "created_at_utc": timestamp_utc_str,
                    "timestamp_utc": timestamp_utc_str,
                    "correlation_id": str(parsed_correlation_id)
                    if parsed_correlation_id
                    else None,
                }

                result = await handler.execute_query(
                    query=query,
                    parameters=parameters,
                )

                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000

                # Determine if this was a create or merge operation
                was_created = False
                returned_intent_id: UUID = intent_id
                if result.records:
                    record = result.records[0]
                    was_created = bool(record.get("was_created", False))
                    # Parse UUID from database string.
                    # We generated intent_id above and passed it to the query, so the
                    # database should return it unchanged. If it returns an invalid UUID,
                    # that indicates a database issue (corruption, schema mismatch, etc.)
                    # We use our generated UUID since we know it's valid.
                    db_intent_id = record.get("intent_id")
                    if isinstance(db_intent_id, str):
                        try:
                            returned_intent_id = UUID(db_intent_id)
                        except ValueError:
                            # Database returned invalid UUID - this is unexpected since we
                            # passed a valid UUID in the query. Log as warning for
                            # investigation but continue with our generated UUID.
                            logger.warning(
                                "Database returned invalid intent_id UUID: %s. "
                                "This may indicate database corruption or schema issues. "
                                "Using generated UUID %s instead.",
                                db_intent_id,
                                intent_id,
                            )

                logger.info(
                    "Stored intent for session %s: category=%s, created=%s (%.2fms)",
                    session_id,
                    intent_category_str,
                    was_created,
                    execution_time_ms,
                )

                return CoreModelIntentStorageResult(
                    success=True,
                    intent_id=returned_intent_id,
                    created=was_created,
                )

        except TimeoutError:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.warning(
                "Timeout storing intent for session %s after %.2fms",
                session_id,
                execution_time_ms,
            )
            return CoreModelIntentStorageResult(
                success=False,
                error_message=f"Operation timed out after {self._config.timeout_seconds}s",
            )

        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.error(
                "Error storing intent for session %s: %s",
                session_id,
                e,
            )
            return CoreModelIntentStorageResult(
                success=False,
                error_message=f"Storage failed: {e}",
            )

    async def get_session_intents(
        self,
        session_id: str,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> CoreModelIntentQueryResult:
        """Get intents for a session with optional filtering.

        Implements ProtocolIntentGraphAdapter.get_session_intents.

        Retrieves intent classifications associated with the specified
        session, ordered by creation time (most recent first).

        Args:
            session_id: The session identifier to query.
            min_confidence: Minimum confidence threshold (0.0-1.0).
            limit: Maximum number of results to return.
                Defaults to config.max_intents_per_session.

        Returns:
            ModelIntentQueryResult with the list of intents or error status.

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        try:
            handler = self._ensure_initialized()
        except RuntimeError as e:
            return CoreModelIntentQueryResult(
                success=False,
                error_message=str(e),
            )

        start_time = time.perf_counter()

        # Apply defaults from config for min_confidence
        effective_min_confidence = (
            min_confidence
            if min_confidence > 0.0
            else self._config.default_confidence_threshold
        )
        effective_min_confidence = max(0.0, min(effective_min_confidence, 1.0))
        effective_limit = (
            limit if limit is not None else self._config.max_intents_per_session
        )
        effective_limit = max(
            1, min(effective_limit, self._config.max_intents_per_session)
        )

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                query = IntentCypherTemplates.get_session_intents_query(
                    session_label=self._config.session_node_label,
                    intent_label=self._config.intent_node_label,
                    rel_type=self._config.relationship_type,
                )

                parameters: dict[str, JsonType] = {
                    "session_id": session_id,
                    "min_confidence": effective_min_confidence,
                    "limit": effective_limit,
                }

                result = await handler.execute_query(
                    query=query,
                    parameters=parameters,
                )

                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000

                if not result.records:
                    logger.debug(
                        "No intents found for session %s (%.2fms)",
                        session_id,
                        execution_time_ms,
                    )
                    return CoreModelIntentQueryResult(
                        success=True,
                    )

                # Convert records to core intent models
                core_intents: list[CoreModelIntentRecord] = []
                for record in result.records:
                    intent_id_raw = record.get("intent_id")
                    if not isinstance(intent_id_raw, str):
                        logger.warning(
                            "Skipping intent record with missing or non-string intent_id: %s",
                            intent_id_raw,
                        )
                        continue

                    # Parse UUID from database string
                    try:
                        intent_id = UUID(intent_id_raw)
                    except ValueError:
                        logger.warning(
                            "Skipping intent record with invalid intent_id UUID: %s",
                            intent_id_raw,
                        )
                        continue

                    keywords_raw = record.get("keywords", [])
                    keywords: list[str] = (
                        [str(k) for k in keywords_raw]
                        if isinstance(keywords_raw, list)
                        else []
                    )

                    # Extract and validate confidence (defaults to 0.0 if not a number)
                    confidence_raw = record.get("confidence", 0.0)
                    confidence_val = (
                        float(confidence_raw)
                        if isinstance(confidence_raw, int | float)
                        else 0.0
                    )

                    # Parse correlation_id UUID from database string
                    correlation_id_raw = record.get("correlation_id")
                    correlation_id: UUID | None = None
                    if correlation_id_raw is not None:
                        try:
                            correlation_id = UUID(str(correlation_id_raw))
                        except ValueError:
                            logger.warning(
                                "Invalid correlation_id UUID: %s", correlation_id_raw
                            )

                    # Parse datetime from ISO string
                    created_at_raw = record.get("created_at_utc", "")
                    try:
                        created_at = datetime.fromisoformat(str(created_at_raw))
                    except ValueError:
                        logger.warning(
                            "Skipping intent record with invalid created_at_utc: %s",
                            created_at_raw,
                        )
                        continue

                    # Map category string to EnumIntentCategory; fall back to unknown
                    intent_category_str = str(record.get("intent_category", "unknown"))
                    try:
                        intent_category = EnumIntentCategory(intent_category_str)
                    except ValueError:
                        intent_category = EnumIntentCategory.UNKNOWN

                    core_intents.append(
                        CoreModelIntentRecord(
                            intent_id=intent_id,
                            session_id=session_id,
                            intent_category=intent_category,
                            confidence=confidence_val,
                            keywords=keywords,
                            created_at=created_at,
                            correlation_id=correlation_id,
                        )
                    )

                logger.debug(
                    "Found %d intents for session %s (%.2fms)",
                    len(core_intents),
                    session_id,
                    execution_time_ms,
                )
                return CoreModelIntentQueryResult(
                    success=True,
                    intents=core_intents,
                    total_count=len(core_intents),
                )

        except TimeoutError:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.warning(
                "Timeout querying intents for session %s after %.2fms",
                session_id,
                execution_time_ms,
            )
            return CoreModelIntentQueryResult(
                success=False,
                error_message=f"Query timed out after {self._config.timeout_seconds}s",
            )

        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.error(
                "Error querying intents for session %s: %s",
                session_id,
                e,
            )
            return CoreModelIntentQueryResult(
                success=False,
                error_message=f"Query failed: {e}",
            )

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
                Defaults to 24 hours. Values less than 1 are clamped to
                a minimum of 1 hour.

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
        start_time = time.perf_counter()

        try:
            handler = self._ensure_initialized()
        except RuntimeError as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            return ModelIntentDistributionResult(
                status="error",
                time_range_hours=time_range_hours,
                execution_time_ms=execution_time_ms,
                error_message=str(e),
            )

        # Clamp time_range_hours to valid range (minimum 1 hour)
        time_range_hours = max(1, time_range_hours)

        # Calculate time boundary
        since_utc = (datetime.now(UTC) - timedelta(hours=time_range_hours)).isoformat()

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                query = IntentCypherTemplates.get_intent_distribution_query(
                    intent_label=self._config.intent_node_label,
                )

                parameters: dict[str, JsonType] = {
                    "since_utc": since_utc,
                }

                result = await handler.execute_query(
                    query=query,
                    parameters=parameters,
                )

                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000

                distribution: dict[str, int] = {}
                for record in result.records:
                    category = record.get("category")
                    count = record.get("count")
                    if isinstance(category, str) and isinstance(count, int):
                        distribution[category] = count

                total_intents = sum(distribution.values())

                logger.debug(
                    "Retrieved intent distribution: %d categories, %d total intents",
                    len(distribution),
                    total_intents,
                )

                return ModelIntentDistributionResult(
                    status="success",
                    distribution=distribution,
                    total_intents=total_intents,
                    time_range_hours=time_range_hours,
                    execution_time_ms=execution_time_ms,
                )

        except TimeoutError:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.warning(
                "Timeout getting intent distribution after %ss",
                self._config.timeout_seconds,
            )
            return ModelIntentDistributionResult(
                status="error",
                time_range_hours=time_range_hours,
                execution_time_ms=execution_time_ms,
                error_message=f"Query timed out after {self._config.timeout_seconds}s",
            )

        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.error("Error getting intent distribution: %s", e)
            return ModelIntentDistributionResult(
                status="error",
                time_range_hours=time_range_hours,
                execution_time_ms=execution_time_ms,
                error_message=f"Query failed: {e}",
            )

    async def get_recent_intents(
        self,
        time_range_hours: int = 24,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> ModelIntentQueryResult:
        """Get recent intents across all sessions within a time range.

        Retrieves intent classifications created within the specified time
        range, optionally filtered by confidence threshold. Results are
        ordered by creation time (most recent first) and include the
        associated session ID.

        Args:
            time_range_hours: Number of hours to look back from now.
                Defaults to 24 hours. Values less than 1 are clamped to
                a minimum of 1 hour.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                If None, no confidence filtering is applied.
            limit: Maximum number of results to return.
                Defaults to config.max_intents_per_session.
                Clamped to config.max_intents_per_session maximum.

        Returns:
            ModelIntentQueryResult with the list of intents or error status.
            Each intent record includes session_id populated with the
            session it belongs to.

        Example::

            # Get all intents from the last 48 hours with high confidence
            result = await adapter.get_recent_intents(
                time_range_hours=48,
                min_confidence=0.8,
                limit=100,
            )
            if result.status == "success":
                for intent in result.intents:
                    print(f"Session {intent.session_ref}: {intent.intent_category}")

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.

        .. versionadded:: 0.1.0
            Added for OMN-1504 to support querying recent intents across sessions.
        """
        start_time = time.perf_counter()

        try:
            handler = self._ensure_initialized()
        except RuntimeError as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            return ModelIntentQueryResult(
                status="error",
                execution_time_ms=execution_time_ms,
                error_message=str(e),
            )

        # Clamp time_range_hours to valid range (minimum 1 hour)
        effective_time_range = max(1, time_range_hours)

        # Apply defaults from config for limit
        effective_limit = (
            limit if limit is not None else self._config.max_intents_per_session
        )
        # Clamp limit to valid range
        effective_limit = max(
            1, min(effective_limit, self._config.max_intents_per_session)
        )

        # Clamp min_confidence if provided
        effective_min_confidence: float | None = None
        if min_confidence is not None:
            effective_min_confidence = max(0.0, min(min_confidence, 1.0))

        # Calculate time boundary
        cutoff_time = (
            datetime.now(UTC) - timedelta(hours=effective_time_range)
        ).isoformat()

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                query = IntentCypherTemplates.get_recent_intents_query(
                    session_label=self._config.session_node_label,
                    intent_label=self._config.intent_node_label,
                    rel_type=self._config.relationship_type,
                )

                parameters: dict[str, JsonType] = {
                    "cutoff_time": cutoff_time,
                    "min_confidence": effective_min_confidence,
                    "limit": effective_limit,
                }

                result = await handler.execute_query(
                    query=query,
                    parameters=parameters,
                )

                if not result.records:
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000
                    logger.debug(
                        "No recent intents found in last %d hours",
                        effective_time_range,
                    )
                    return ModelIntentQueryResult(
                        status="no_results",
                        execution_time_ms=execution_time_ms,
                    )

                # Convert records to intent models
                intents: list[ModelIntentRecord] = []
                for record in result.records:
                    intent_id_raw = record.get("intent_id")
                    if not isinstance(intent_id_raw, str):
                        logger.warning(
                            "Skipping intent record with missing or non-string intent_id: %s",
                            intent_id_raw,
                        )
                        continue

                    # Parse UUID from database string
                    try:
                        intent_id = UUID(intent_id_raw)
                    except ValueError:
                        logger.warning(
                            "Skipping intent record with invalid intent_id UUID: %s",
                            intent_id_raw,
                        )
                        continue

                    # Extract session_id
                    session_id_raw = record.get("session_id")
                    if session_id_raw is None:
                        logger.warning("Skipping intent record with missing session_id")
                        continue
                    record_session_id = str(session_id_raw)

                    keywords_raw = record.get("keywords", [])
                    keywords: list[str] = (
                        [str(k) for k in keywords_raw]
                        if isinstance(keywords_raw, list)
                        else []
                    )

                    # Extract and validate confidence (defaults to 0.0 if not a number)
                    confidence_raw = record.get("confidence", 0.0)
                    confidence_val = (
                        float(confidence_raw)
                        if isinstance(confidence_raw, int | float)
                        else 0.0
                    )

                    # Parse correlation_id UUID from database string
                    correlation_id_raw = record.get("correlation_id")
                    correlation_id: UUID | None = None
                    if correlation_id_raw is not None:
                        try:
                            correlation_id = UUID(str(correlation_id_raw))
                        except ValueError:
                            logger.warning(
                                "Invalid correlation_id UUID: %s", correlation_id_raw
                            )

                    # Parse datetime from ISO string
                    created_at_raw = record.get("created_at_utc", "")
                    try:
                        created_at = datetime.fromisoformat(str(created_at_raw))
                    except ValueError:
                        logger.warning(
                            "Skipping intent record with invalid created_at_utc: %s",
                            created_at_raw,
                        )
                        continue

                    # Use category string directly for local model
                    intent_category_str = str(record.get("intent_category", "unknown"))

                    intents.append(
                        ModelIntentRecord(
                            intent_id=intent_id,
                            session_ref=record_session_id,
                            intent_category=intent_category_str,
                            confidence=confidence_val,
                            keywords=keywords,
                            created_at_utc=created_at,
                            correlation_id=correlation_id,
                        )
                    )

                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000

                logger.debug(
                    "Retrieved %d recent intents from last %d hours (%.2fms)",
                    len(intents),
                    effective_time_range,
                    execution_time_ms,
                )

                return ModelIntentQueryResult(
                    status="success",
                    intents=intents,
                    total_count=len(intents),
                    execution_time_ms=execution_time_ms,
                )

        except TimeoutError:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.warning(
                "Timeout querying recent intents after %ss",
                self._config.timeout_seconds,
            )
            return ModelIntentQueryResult(
                status="error",
                execution_time_ms=execution_time_ms,
                error_message=f"Query timed out after {self._config.timeout_seconds}s",
            )

        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.error(
                "Error querying recent intents: %s",
                e,
            )
            return ModelIntentQueryResult(
                status="error",
                execution_time_ms=execution_time_ms,
                error_message=f"Query failed: {e}",
            )

    async def health_check(self) -> bool:
        """Check if the intent graph storage is healthy and accessible.

        Implements ProtocolIntentGraphAdapter.health_check.

        Returns:
            True if the storage is healthy, False otherwise.

        Note:
            For detailed health information (counts, timestamps, error messages),
            use get_health_details() instead.
        """
        if not self._initialized or self._handler is None:
            return False

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                health = await self._handler.health_check()
                return bool(health.healthy)
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    async def get_health_details(self) -> ModelIntentGraphHealth:
        """Get detailed health status of the graph connection.

        including session/intent counts and timestamps. Use this when you
        need more than a simple healthy/unhealthy indicator.

        Returns:
            ModelIntentGraphHealth with detailed health status.
            This method never raises - errors are captured in the
            result model.
        """
        timestamp = datetime.now(UTC)

        if not self._initialized or self._handler is None:
            return ModelIntentGraphHealth(
                is_healthy=False,
                initialized=False,
                handler_healthy=None,
                error_message="Adapter not initialized",
                last_check_timestamp=timestamp,
            )

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                # Check handler health
                health = await self._handler.health_check()
                handler_healthy = bool(health.healthy)

                if not handler_healthy:
                    return ModelIntentGraphHealth(
                        is_healthy=False,
                        initialized=True,
                        handler_healthy=False,
                        error_message="Handler reports unhealthy",
                        last_check_timestamp=timestamp,
                    )

                # Get counts for detailed health info using single combined query
                session_count: int | None = None
                intent_count: int | None = None

                try:
                    count_query = IntentCypherTemplates.count_all_query(
                        session_label=self._config.session_node_label,
                        intent_label=self._config.intent_node_label,
                    )
                    count_result = await self._handler.execute_query(
                        query=count_query,
                        parameters={},
                    )
                    if count_result.records:
                        record = count_result.records[0]
                        session_val = record.get("session_count")
                        intent_val = record.get("intent_count")
                        if isinstance(session_val, int):
                            session_count = session_val
                        if isinstance(intent_val, int):
                            intent_count = intent_val

                except Exception as e:
                    # Log but don't fail health check for count errors
                    logger.debug("Failed to get counts during health check: %s", e)

                return ModelIntentGraphHealth(
                    is_healthy=True,
                    initialized=True,
                    handler_healthy=True,
                    session_count=session_count,
                    intent_count=intent_count,
                    last_check_timestamp=timestamp,
                )

        except TimeoutError:
            logger.warning(
                "Health check timed out after %ss",
                self._config.timeout_seconds,
            )
            return ModelIntentGraphHealth(
                is_healthy=False,
                initialized=True,
                handler_healthy=None,
                error_message=f"Health check timed out after {self._config.timeout_seconds}s",
                last_check_timestamp=timestamp,
            )

        except Exception as e:
            logger.warning(
                "Health check failed with %s: %s",
                type(e).__name__,
                e,
            )
            logger.debug("Health check exception traceback", exc_info=True)
            return ModelIntentGraphHealth(
                is_healthy=False,
                initialized=True,
                handler_healthy=None,
                error_message=f"Health check failed: {e}",
                last_check_timestamp=timestamp,
            )
