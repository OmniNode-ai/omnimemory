"""PostgreSQL Adapter Integration for OmniMemory event-driven operations."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# Import ONEX core components
try:
    from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
    from omnibase_core.core.protocol_event_bus import ProtocolEventBus
    from omnibase_core.model.core.model_event_envelope import ModelEventEnvelope
    from omnibase_core.model.core.model_onex_event import ModelOnexEvent
    from omnibase_core.model.core.model_route_spec import ModelRouteSpec
except ImportError:
    # Fallback for development
    class CoreErrorCode:
        DEPENDENCY_RESOLUTION_ERROR = "DEPENDENCY_RESOLUTION_ERROR"
        SERVICE_UNAVAILABLE_ERROR = "SERVICE_UNAVAILABLE_ERROR"
        VALIDATION_ERROR = "VALIDATION_ERROR"

    class OnexError(Exception):
        def __init__(self, code: str, message: str, details: Optional[Dict] = None):
            self.code = code
            self.message = message
            self.details = details
            super().__init__(message)

    class ProtocolEventBus:
        async def publish_async(self, event):
            pass


# Import omnibase_infra models (with fallbacks)
try:
    from omnibase_infra.enums.enum_omninode_topic_class import EnumOmniNodeTopicClass
    from omnibase_infra.models.event_publishing.model_omninode_topic_spec import (
        ModelOmniNodeTopicSpec,
    )
    from omnibase_infra.models.postgres.model_postgres_query_request import (
        ModelPostgresQueryRequest,
    )
except ImportError:
    # Fallback models for development
    class ModelPostgresQueryRequest(BaseModel):
        query: str
        parameters: List[Any] = Field(default_factory=list)
        correlation_id: Optional[UUID] = None
        timeout: Optional[int] = None
        context: Optional[Dict[str, Any]] = None

    class ModelOmniNodeTopicSpec(BaseModel):
        def to_topic_string(self) -> str:
            return "dev.omnibase.onex.cmd.postgres-query.v1"

        @classmethod
        def for_postgres_query_command(cls, correlation_id: Optional[str] = None):
            return cls()


from ..models.events.model_omnimemory_event_data import (
    ModelOmniMemoryRetrieveData,
    ModelOmniMemoryStoreData,
    ModelOmniMemoryVectorSearchData,
)


class PostgresMemorySchema(BaseModel):
    """Schema definition for OmniMemory PostgreSQL tables."""

    @staticmethod
    def get_create_tables_sql() -> List[str]:
        """Get SQL statements to create OmniMemory tables."""
        return [
            """
            CREATE TABLE IF NOT EXISTS omnimemory_persistent_storage (
                memory_key VARCHAR(255) PRIMARY KEY,
                content JSONB NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                memory_type VARCHAR(50) NOT NULL DEFAULT 'persistent',
                metadata JSONB DEFAULT '{}',
                storage_size INTEGER NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP WITH TIME ZONE NULL,
                vector_dimensions INTEGER NULL
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_omnimemory_memory_type
            ON omnimemory_persistent_storage(memory_type);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_omnimemory_created_at
            ON omnimemory_persistent_storage(created_at);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_omnimemory_expires_at
            ON omnimemory_persistent_storage(expires_at)
            WHERE expires_at IS NOT NULL;
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_omnimemory_content_gin
            ON omnimemory_persistent_storage USING GIN(content);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_omnimemory_metadata_gin
            ON omnimemory_persistent_storage USING GIN(metadata);
            """,
        ]

    @staticmethod
    def get_cleanup_expired_sql() -> str:
        """Get SQL to clean up expired memories."""
        return """
        DELETE FROM omnimemory_persistent_storage
        WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
        """


class PostgresAdapterIntegration(BaseModel):
    """
    Integration layer for PostgreSQL adapter operations via event bus.

    Maps OmniMemory operations to PostgreSQL queries and publishes them
    to the infrastructure PostgreSQL adapter via RedPanda event bus.
    """

    node_id: str = Field(
        default="omnimemory_postgres_integration", description="Integration node ID"
    )
    event_bus: Optional[ProtocolEventBus] = Field(
        default=None, description="Protocol event bus for PostgreSQL operations"
    )
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("omnimemory.postgres_integration"),
        description="Logger instance",
    )
    initialized: bool = Field(default=False, description="Initialization status")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def initialize(self, event_bus: ProtocolEventBus) -> None:
        """
        Initialize PostgreSQL adapter integration.

        Args:
            event_bus: ProtocolEventBus for infrastructure communication

        Raises:
            OnexError: If initialization fails
        """
        if not event_bus:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus not available for PostgreSQL integration",
            )

        self.event_bus = event_bus
        self.initialized = True

        self.logger.info(
            "PostgreSQL adapter integration initialized",
            extra={
                "node_id": self.node_id,
                "component": "postgres_integration",
                "operation": "initialization",
            },
        )

    async def execute_store_memory_query(
        self, correlation_id: UUID, store_data: ModelOmniMemoryStoreData, content: Any
    ) -> None:
        """
        Execute PostgreSQL store query via infrastructure adapter.

        Args:
            correlation_id: Operation correlation ID
            store_data: Memory store data
            content: Content to store

        Raises:
            OnexError: If query execution fails
        """
        self._validate_initialization()

        # Build INSERT/UPSERT query
        query = """
        INSERT INTO omnimemory_persistent_storage
        (memory_key, content, content_hash, memory_type, metadata, storage_size,
         created_at, updated_at, expires_at, vector_dimensions)
        VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, $7, $8)
        ON CONFLICT (memory_key)
        DO UPDATE SET
            content = EXCLUDED.content,
            content_hash = EXCLUDED.content_hash,
            memory_type = EXCLUDED.memory_type,
            metadata = EXCLUDED.metadata,
            storage_size = EXCLUDED.storage_size,
            updated_at = CURRENT_TIMESTAMP,
            expires_at = EXCLUDED.expires_at,
            vector_dimensions = EXCLUDED.vector_dimensions
        """

        # Calculate expires_at if TTL is specified
        expires_at = None
        if store_data.ttl_seconds:
            from datetime import timedelta

            expires_at = (
                datetime.utcnow() + timedelta(seconds=store_data.ttl_seconds)
            ).isoformat()

        # Prepare parameters
        parameters = [
            store_data.memory_key,
            json.dumps(content),
            store_data.content_hash,
            store_data.memory_type,
            json.dumps(store_data.metadata),
            store_data.storage_size,
            expires_at,
            store_data.vector_dimensions,
        ]

        # Create PostgreSQL query request
        query_request = ModelPostgresQueryRequest(
            query=query,
            parameters=parameters,
            correlation_id=correlation_id,
            timeout=30,  # 30 second timeout
            context={
                "operation": "store_memory",
                "memory_key": store_data.memory_key,
                "memory_type": store_data.memory_type,
            },
        )

        await self._publish_postgres_query(correlation_id, query_request)

        self.logger.info(
            f"Store memory query published to PostgreSQL adapter",
            extra={
                "correlation_id": str(correlation_id),
                "memory_key": store_data.memory_key,
                "memory_type": store_data.memory_type,
                "storage_size": store_data.storage_size,
                "component": "postgres_integration",
                "operation": "execute_store_memory_query",
            },
        )

    async def execute_retrieve_memory_query(
        self, correlation_id: UUID, retrieve_data: ModelOmniMemoryRetrieveData
    ) -> None:
        """
        Execute PostgreSQL retrieve query via infrastructure adapter.

        Args:
            correlation_id: Operation correlation ID
            retrieve_data: Memory retrieve data

        Raises:
            OnexError: If query execution fails
        """
        self._validate_initialization()

        query_type = retrieve_data.query_type
        parameters = []

        if query_type == "key":
            # Simple key-based retrieval
            query = """
            SELECT memory_key, content, content_hash, memory_type, metadata,
                   storage_size, created_at, updated_at, expires_at, vector_dimensions
            FROM omnimemory_persistent_storage
            WHERE memory_key = $1
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
            parameters = [retrieve_data.memory_key]

        elif query_type == "temporal":
            # Time-based query
            query = """
            SELECT memory_key, content, content_hash, memory_type, metadata,
                   storage_size, created_at, updated_at, expires_at, vector_dimensions
            FROM omnimemory_persistent_storage
            WHERE created_at >= $1
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
            if retrieve_data.time_range and "start_time" in retrieve_data.time_range:
                parameters = [retrieve_data.time_range["start_time"]]
            else:
                # Default to last 24 hours
                from datetime import timedelta

                start_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
                parameters = [start_time]

            # Add end time if specified
            if retrieve_data.time_range and "end_time" in retrieve_data.time_range:
                query += " AND created_at <= $2"
                parameters.append(retrieve_data.time_range["end_time"])

            query += " ORDER BY created_at DESC LIMIT 100"

        elif query_type == "metadata":
            # Metadata-based query
            query_params = retrieve_data.query_parameters or {}
            if "metadata_key" in query_params and "metadata_value" in query_params:
                query = """
                SELECT memory_key, content, content_hash, memory_type, metadata,
                       storage_size, created_at, updated_at, expires_at, vector_dimensions
                FROM omnimemory_persistent_storage
                WHERE metadata ->> $1 = $2
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY created_at DESC LIMIT 100
                """
                parameters = [
                    query_params["metadata_key"],
                    query_params["metadata_value"],
                ]
            else:
                # Generic metadata query
                query = """
                SELECT memory_key, content, content_hash, memory_type, metadata,
                       storage_size, created_at, updated_at, expires_at, vector_dimensions
                FROM omnimemory_persistent_storage
                WHERE metadata @> $1::jsonb
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY created_at DESC LIMIT 100
                """
                parameters = [json.dumps(query_params)]

        else:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Unsupported query type: {query_type}",
            )

        # Create PostgreSQL query request
        query_request = ModelPostgresQueryRequest(
            query=query,
            parameters=parameters,
            correlation_id=correlation_id,
            timeout=30,
            context={
                "operation": "retrieve_memory",
                "query_type": query_type,
                "memory_key": retrieve_data.memory_key,
            },
        )

        await self._publish_postgres_query(correlation_id, query_request)

        self.logger.info(
            f"Retrieve memory query published to PostgreSQL adapter",
            extra={
                "correlation_id": str(correlation_id),
                "query_type": query_type,
                "memory_key": retrieve_data.memory_key,
                "component": "postgres_integration",
                "operation": "execute_retrieve_memory_query",
            },
        )

    async def execute_health_check_query(self, correlation_id: UUID) -> None:
        """
        Execute PostgreSQL health check query.

        Args:
            correlation_id: Operation correlation ID
        """
        self._validate_initialization()

        # Simple health check query
        query = """
        SELECT
            COUNT(*) as total_memories,
            COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) as active_memories,
            pg_database_size(current_database()) as database_size,
            CURRENT_TIMESTAMP as check_time
        FROM omnimemory_persistent_storage
        """

        query_request = ModelPostgresQueryRequest(
            query=query,
            parameters=[],
            correlation_id=correlation_id,
            timeout=10,
            context={"operation": "health_check"},
        )

        await self._publish_postgres_query(correlation_id, query_request)

        self.logger.info(
            f"Health check query published to PostgreSQL adapter",
            extra={
                "correlation_id": str(correlation_id),
                "component": "postgres_integration",
                "operation": "execute_health_check_query",
            },
        )

    async def execute_cleanup_expired_query(self, correlation_id: UUID) -> None:
        """
        Execute cleanup query for expired memories.

        Args:
            correlation_id: Operation correlation ID
        """
        self._validate_initialization()

        query = PostgresMemorySchema.get_cleanup_expired_sql()

        query_request = ModelPostgresQueryRequest(
            query=query,
            parameters=[],
            correlation_id=correlation_id,
            timeout=60,  # Longer timeout for cleanup
            context={"operation": "cleanup_expired"},
        )

        await self._publish_postgres_query(correlation_id, query_request)

        self.logger.info(
            f"Cleanup expired query published to PostgreSQL adapter",
            extra={
                "correlation_id": str(correlation_id),
                "component": "postgres_integration",
                "operation": "execute_cleanup_expired_query",
            },
        )

    async def initialize_database_schema(self, correlation_id: UUID) -> None:
        """
        Initialize database schema for OmniMemory.

        Args:
            correlation_id: Operation correlation ID
        """
        self._validate_initialization()

        # Execute each schema creation statement
        schema_queries = PostgresMemorySchema.get_create_tables_sql()

        for i, query in enumerate(schema_queries):
            schema_correlation_id = uuid4()

            query_request = ModelPostgresQueryRequest(
                query=query,
                parameters=[],
                correlation_id=schema_correlation_id,
                timeout=30,
                context={
                    "operation": "initialize_schema",
                    "parent_correlation_id": str(correlation_id),
                    "schema_step": i + 1,
                    "total_steps": len(schema_queries),
                },
            )

            await self._publish_postgres_query(schema_correlation_id, query_request)

            # Small delay between schema operations to prevent conflicts
            await asyncio.sleep(0.1)

        self.logger.info(
            f"Database schema initialization queries published",
            extra={
                "correlation_id": str(correlation_id),
                "schema_queries": len(schema_queries),
                "component": "postgres_integration",
                "operation": "initialize_database_schema",
            },
        )

    async def _publish_postgres_query(
        self, correlation_id: UUID, query_request: ModelPostgresQueryRequest
    ) -> None:
        """
        Publish PostgreSQL query to infrastructure adapter via event bus.

        Args:
            correlation_id: Operation correlation ID
            query_request: PostgreSQL query request

        Raises:
            OnexError: If publishing fails
        """
        try:
            # Create ONEX event for PostgreSQL adapter
            event_payload = ModelOnexEvent.create_core_event(
                event_type="database.postgres.query_request",
                node_id=self.node_id,
                correlation_id=correlation_id,
                data={
                    "query": query_request.query,
                    "parameters": query_request.parameters,
                    "timeout": query_request.timeout,
                    "context": query_request.context or {},
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Create topic spec for PostgreSQL adapter
            topic_spec = ModelOmniNodeTopicSpec.for_postgres_query_command(
                str(correlation_id)
            )

            # Create route spec
            route_spec = ModelRouteSpec.create_direct_route(
                topic_spec.to_topic_string()
            )

            # Create event envelope
            envelope = ModelEventEnvelope(
                payload=event_payload,
                route_spec=route_spec,
                source_node_id=self.node_id,
                correlation_id=correlation_id,
                metadata={
                    "topic_spec": topic_spec.to_topic_string(),
                    "database_operation": "postgres_query",
                    "omninode_namespace": "dev.omnibase.onex",
                },
            )

            # Add source hop
            envelope.add_source_hop(self.node_id, "OmniMemory PostgreSQL Integration")

            # Publish to event bus
            await self.event_bus.publish_async(envelope.payload)

            self.logger.debug(
                f"PostgreSQL query published via event bus",
                extra={
                    "correlation_id": str(correlation_id),
                    "topic": topic_spec.to_topic_string(),
                    "query_length": len(query_request.query),
                    "parameter_count": len(query_request.parameters),
                    "component": "postgres_integration",
                    "operation": "publish_postgres_query",
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to publish PostgreSQL query",
                extra={
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "component": "postgres_integration",
                    "operation": "publish_postgres_query_error",
                },
                exc_info=True,
            )
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to publish PostgreSQL query: {str(e)}",
                details={"correlation_id": str(correlation_id)},
            ) from e

    def _validate_initialization(self) -> None:
        """Validate that integration is properly initialized."""
        if not self.initialized or not self.event_bus:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="PostgreSQL adapter integration not initialized",
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of PostgreSQL integration.

        Returns:
            Health check results
        """
        try:
            return {
                "status": "healthy" if self.initialized else "unhealthy",
                "component": "postgres_adapter_integration",
                "initialized": self.initialized,
                "event_bus_available": self.event_bus is not None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "postgres_adapter_integration",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
