# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Adapter for storing intent classifications in Memgraph.

This adapter provides a domain-specific interface for storing and retrieving
intent classification results in a graph database. It wraps the generic
HandlerGraph to provide intent-specific operations:

- store_intent(): Store an intent classification linked to a session
- get_session_intents(): Retrieve intents for a given session
- get_intent_distribution(): Get aggregate intent statistics

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
            connection_uri="bolt://localhost:7687",
        )

        # Store an intent
        classification = ModelIntentClassificationOutput(
            intent_category="debugging",
            confidence=0.92,
            keywords=["error", "traceback"],
        )
        result = await adapter.store_intent(
            session_id="session_123",
            intent_data=classification,
            correlation_id="corr_abc",
        )
        if result.status == "success":
            print(f"Stored intent: {result.intent_id}")

        await adapter.shutdown()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1457.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse
from uuid import uuid4

from omnibase_core.types.type_json import JsonType

from omnimemory.handlers.adapters.models import (
    ModelAdapterIntentGraphConfig,
    ModelIntentClassificationOutput,
    ModelIntentDistributionResult,
    ModelIntentGraphHealth,
    ModelIntentQueryResult,
    ModelIntentRecord,
    ModelIntentStorageResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.handlers.handler_graph import HandlerGraph

# Runtime conditional import - omnibase_infra is a dev dependency
_OMNIBASE_INFRA_AVAILABLE: bool = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

if not TYPE_CHECKING:
    try:
        from omnibase_infra.handlers.handler_graph import HandlerGraph

        _OMNIBASE_INFRA_AVAILABLE = True
    except ImportError as e:
        _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

        class HandlerGraph:  # type: ignore[no-redef]
            """Stub for HandlerGraph when omnibase_infra is not installed."""

            def __init__(self, container: object) -> None:
                raise ImportError(
                    f"omnibase_infra is required for AdapterIntentGraph. "
                    f"Install it with: poetry install --with dev. "
                    f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
                )


__all__ = ["AdapterIntentGraph", "IntentCypherTemplates"]

logger = logging.getLogger(__name__)


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
        All queries use parameters ($param) instead of string interpolation.
        The label/type parameters are safe from injection as they come from
        config validation (string type with pydantic validation).
        NEVER construct queries by concatenating user input.
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
    def create_indexes_queries(session_label: str, intent_label: str) -> list[str]:
        """Generate index creation queries for intent graph schema.

        Uses ``CREATE INDEX IF NOT EXISTS`` syntax (Memgraph 2.0+) to ensure
        idempotent index creation without relying on error handling for
        duplicate index detection.
        """
        return [
            f"CREATE INDEX IF NOT EXISTS ON :{session_label}(session_id);",
            f"CREATE INDEX IF NOT EXISTS ON :{intent_label}(intent_id);",
            f"CREATE INDEX IF NOT EXISTS ON :{intent_label}(intent_category);",
            f"CREATE INDEX IF NOT EXISTS ON :{intent_label}(created_at_utc);",
        ]

    @staticmethod
    def count_sessions_query(session_label: str) -> str:
        """Generate query to count session nodes."""
        return f"""
        MATCH (s:{session_label}) RETURN count(s) AS count
        """

    @staticmethod
    def count_intents_query(intent_label: str) -> str:
        """Generate query to count intent nodes."""
        return f"""
        MATCH (i:{intent_label}) RETURN count(i) AS count
        """


class AdapterIntentGraph:
    """Adapter that wraps HandlerGraph for intent classification storage.

    This adapter provides an intent-domain interface on top of the generic
    graph handler, translating intent storage and retrieval operations
    into graph queries:

    - store_intent(session_id, intent_data): Store intent linked to session
    - get_session_intents(session_id): Retrieve intents for a session
    - get_intent_distribution(time_range): Get intent category statistics

    The adapter handles:
    - Session node creation with MERGE semantics
    - Intent node creation/update with MERGE semantics
    - Relationship properties for correlation tracking
    - Confidence-based filtering and time-range queries

    Attributes:
        config: The adapter configuration.
        handler: The underlying HandlerGraph instance.

    Example::

        async def example():
            config = ModelAdapterIntentGraphConfig(
                timeout_seconds=30.0,
                max_intents_per_session=100,
            )
            adapter = AdapterIntentGraph(config)
            await adapter.initialize(
                connection_uri="bolt://localhost:7687",
            )

            # Store intent classification
            result = await adapter.store_intent(
                session_id="sess_123",
                intent_data=ModelIntentClassificationOutput(
                    intent_category="code_generation",
                    confidence=0.95,
                    keywords=["python", "function"],
                ),
                correlation_id="corr_abc",
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
        self._handler: HandlerGraph | None = None
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
    def handler(self) -> HandlerGraph | None:
        """Get the underlying graph handler (None if not initialized)."""
        return self._handler

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter has been initialized."""
        return self._initialized

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
            connection_uri: Graph database URI (e.g., "bolt://localhost:7687").
            auth: Optional (username, password) tuple for authentication.
            options: Additional connection options passed to HandlerGraph.

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
                            from omnibase_core.container import ModelONEXContainer

                            self._container = ModelONEXContainer()

                        # Create and initialize handler
                        if not _OMNIBASE_INFRA_AVAILABLE:
                            raise RuntimeError(
                                "HandlerGraph not available. "
                                "Install omnibase_infra to use graph features."
                            )

                        self._handler = HandlerGraph(self._container)

                        init_options: dict[str, JsonType] = {
                            "timeout_seconds": self._config.timeout_seconds,
                        }
                        if options:
                            init_options.update(cast(Mapping[str, JsonType], options))

                        await self._handler.initialize(
                            connection_uri=connection_uri,
                            auth=auth,
                            options=init_options,
                        )

                        self._initialized = True

                        # Log safe URI (without credentials)
                        safe_uri = f"{parsed_uri.scheme}://{parsed_uri.hostname}"
                        if parsed_uri.port:
                            safe_uri += f":{parsed_uri.port}"
                        logger.info(
                            "AdapterIntentGraph initialized with connection to %s",
                            safe_uri,
                        )

                        # Ensure indexes exist for optimal query performance
                        await self._ensure_indexes()

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

    def _ensure_initialized(self) -> HandlerGraph:
        """Ensure adapter is initialized and return handler.

        Returns:
            The initialized HandlerGraph.

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
        intent_data: ModelIntentClassificationOutput,
        correlation_id: str,
        *,
        user_context: str = "",
    ) -> ModelIntentStorageResult:
        """Store an intent classification linked to a session.

        Uses MERGE semantics to create or update the session and intent
        nodes. If an intent with the same category already exists for
        the session, its confidence and keywords are updated.

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
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        # Validate session_id is non-empty
        if not session_id or not session_id.strip():
            return ModelIntentStorageResult(
                status="error",
                session_id=session_id,
                error_message="session_id cannot be empty",
            )

        try:
            handler = self._ensure_initialized()
        except RuntimeError as e:
            return ModelIntentStorageResult(
                status="error",
                session_id=session_id,
                error_message=str(e),
            )

        start_time = time.perf_counter()
        intent_id = str(uuid4())
        timestamp_utc = datetime.now(UTC).isoformat()

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                query = IntentCypherTemplates.store_intent_query(
                    session_label=self._config.session_node_label,
                    intent_label=self._config.intent_node_label,
                    rel_type=self._config.relationship_type,
                )

                parameters: dict[str, JsonType] = {
                    "session_id": session_id,
                    "started_at_utc": timestamp_utc,
                    "user_context": user_context,
                    "intent_id": intent_id,
                    "intent_category": intent_data.intent_category,
                    "confidence": intent_data.confidence,
                    "keywords": cast(list[JsonType], intent_data.keywords),
                    "created_at_utc": timestamp_utc,
                    "timestamp_utc": timestamp_utc,
                    "correlation_id": correlation_id,
                }

                result = await handler.execute_query(
                    query=query,
                    parameters=parameters,
                )

                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000

                # Determine if this was a create or merge operation
                was_created = False
                returned_intent_id = intent_id
                if result.records:
                    record = result.records[0]
                    was_created = bool(record.get("was_created", False))
                    returned_intent_id = str(record.get("intent_id", intent_id))

                logger.info(
                    "Stored intent for session %s: category=%s, confidence=%.2f, created=%s",
                    session_id,
                    intent_data.intent_category,
                    intent_data.confidence,
                    was_created,
                )

                return ModelIntentStorageResult(
                    status="success",
                    intent_id=returned_intent_id,
                    session_id=session_id,
                    created=was_created,
                    execution_time_ms=execution_time_ms,
                )

        except TimeoutError:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.warning(
                "Timeout storing intent for session %s after %.2fms",
                session_id,
                execution_time_ms,
            )
            return ModelIntentStorageResult(
                status="error",
                session_id=session_id,
                execution_time_ms=execution_time_ms,
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
            return ModelIntentStorageResult(
                status="error",
                session_id=session_id,
                execution_time_ms=execution_time_ms,
                error_message=f"Storage failed: {e}",
            )

    async def get_session_intents(
        self,
        session_id: str,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> ModelIntentQueryResult:
        """Get intents for a session with optional filtering.

        Retrieves intent classifications associated with the specified
        session, ordered by creation time (most recent first).

        Args:
            session_id: The session identifier to query.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                Defaults to config.default_confidence_threshold.
            limit: Maximum number of results to return.
                Defaults to config.max_intents_per_session.

        Returns:
            ModelIntentQueryResult with the list of intents or error status.
            Possible status values:
            - "success": Query completed with results
            - "no_results": Session exists but has no intents matching criteria
            - "not_found": Session not found (reserved for future use)
            - "error": Query failed

        Note:
            This method never raises on business errors - it returns
            an error status in the result model instead.
        """
        try:
            handler = self._ensure_initialized()
        except RuntimeError as e:
            return ModelIntentQueryResult(
                status="error",
                error_message=str(e),
            )

        start_time = time.perf_counter()

        # Apply defaults from config
        effective_min_confidence = (
            min_confidence
            if min_confidence is not None
            else self._config.default_confidence_threshold
        )
        effective_limit = (
            limit if limit is not None else self._config.max_intents_per_session
        )

        # Clamp values to valid ranges
        effective_min_confidence = max(0.0, min(effective_min_confidence, 1.0))
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
                    return ModelIntentQueryResult(
                        status="no_results",
                        intents=[],
                        total_count=0,
                        execution_time_ms=execution_time_ms,
                    )

                # Convert records to intent models
                intents: list[ModelIntentRecord] = []
                for record in result.records:
                    intent_id = record.get("intent_id")
                    if not isinstance(intent_id, str):
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

                    # Extract correlation_id with proper type narrowing
                    correlation_id_raw = record.get("correlation_id")
                    correlation_id = (
                        str(correlation_id_raw)
                        if correlation_id_raw is not None
                        else None
                    )

                    intents.append(
                        ModelIntentRecord(
                            intent_id=intent_id,
                            intent_category=str(record.get("intent_category", "")),
                            confidence=confidence_val,
                            keywords=keywords,
                            created_at_utc=str(record.get("created_at_utc", "")),
                            correlation_id=correlation_id,
                        )
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
                "Timeout querying intents for session %s after %.2fms",
                session_id,
                execution_time_ms,
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
                "Error querying intents for session %s: %s",
                session_id,
                e,
            )
            return ModelIntentQueryResult(
                status="error",
                execution_time_ms=execution_time_ms,
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
                Defaults to 24 hours.

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

    async def health_check(self) -> ModelIntentGraphHealth:
        """Check if the graph connection is healthy.

        Performs connectivity check and optionally gathers graph
        statistics (session and intent counts).

        Returns:
            ModelIntentGraphHealth with detailed health status.
            This method never raises - errors are captured in the
            result model.
        """
        timestamp = datetime.now(UTC).isoformat()

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

                # Get counts for detailed health info
                session_count: int | None = None
                intent_count: int | None = None

                try:
                    # Count sessions
                    session_query = IntentCypherTemplates.count_sessions_query(
                        session_label=self._config.session_node_label,
                    )
                    session_result = await self._handler.execute_query(
                        query=session_query,
                        parameters={},
                    )
                    if session_result.records:
                        count_val = session_result.records[0].get("count")
                        if isinstance(count_val, int):
                            session_count = count_val

                    # Count intents
                    intent_query = IntentCypherTemplates.count_intents_query(
                        intent_label=self._config.intent_node_label,
                    )
                    intent_result = await self._handler.execute_query(
                        query=intent_query,
                        parameters={},
                    )
                    if intent_result.records:
                        count_val = intent_result.records[0].get("count")
                        if isinstance(count_val, int):
                            intent_count = count_val

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
