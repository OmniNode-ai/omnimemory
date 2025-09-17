"""Redis Adapter Integration for OmniMemory temporal and cache operations."""

import json
import logging
from datetime import datetime, timedelta
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

    class ModelOnexEvent(BaseModel):
        @classmethod
        def create_core_event(
            cls, event_type: str, node_id: str, correlation_id: UUID, data: dict
        ):
            return cls(
                event_type=event_type,
                node_id=node_id,
                correlation_id=correlation_id,
                data=data,
            )

    class ModelRouteSpec(BaseModel):
        @classmethod
        def create_direct_route(cls, topic: str):
            return cls(topic=topic)

    class ModelEventEnvelope(BaseModel):
        def __init__(self, **data):
            super().__init__(**data)

        def add_source_hop(self, node_id: str, description: str):
            pass


from ..models.events.model_omnimemory_event_data import (
    ModelOmniMemoryRetrieveData,
    ModelOmniMemoryStoreData,
)


class RedisAdapterIntegration(BaseModel):
    """
    Integration layer for Redis adapter operations via event bus.

    Maps OmniMemory temporal operations to Redis commands and publishes them
    to the infrastructure Redis adapter via RedPanda event bus.
    """

    node_id: str = Field(
        default="omnimemory_redis_integration", description="Integration node ID"
    )
    event_bus: Optional[ProtocolEventBus] = Field(
        default=None, description="Protocol event bus for Redis operations"
    )
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("omnimemory.redis_integration"),
        description="Logger instance",
    )
    initialized: bool = Field(default=False, description="Initialization status")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def initialize(self, event_bus: ProtocolEventBus) -> None:
        """
        Initialize Redis adapter integration.

        Args:
            event_bus: ProtocolEventBus for infrastructure communication

        Raises:
            OnexError: If initialization fails
        """
        if not event_bus:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus not available for Redis integration",
            )

        self.event_bus = event_bus
        self.initialized = True

        self.logger.info(
            "Redis adapter integration initialized",
            extra={
                "node_id": self.node_id,
                "component": "redis_integration",
                "operation": "initialization",
            },
        )

    async def execute_store_temporal_memory(
        self, correlation_id: UUID, store_data: ModelOmniMemoryStoreData, content: Any
    ) -> None:
        """
        Execute Redis SET operation for temporal memory storage.

        Args:
            correlation_id: Operation correlation ID
            store_data: Memory store data
            content: Content to store

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        # Create Redis SET command with TTL
        redis_key = f"omnimemory:temporal:{store_data.memory_key}"
        redis_value = {
            "content": content,
            "metadata": store_data.metadata,
            "content_hash": store_data.content_hash,
            "storage_size": store_data.storage_size,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Redis command
        if store_data.ttl_seconds:
            command = "SETEX"
            parameters = [redis_key, store_data.ttl_seconds, json.dumps(redis_value)]
        else:
            command = "SET"
            parameters = [redis_key, json.dumps(redis_value)]

        await self._publish_redis_command(
            correlation_id=correlation_id,
            command=command,
            parameters=parameters,
            context={
                "operation": "store_temporal_memory",
                "memory_key": store_data.memory_key,
                "ttl_seconds": store_data.ttl_seconds,
            },
        )

        self.logger.info(
            f"Store temporal memory command published to Redis adapter",
            extra={
                "correlation_id": str(correlation_id),
                "memory_key": store_data.memory_key,
                "ttl_seconds": store_data.ttl_seconds,
                "command": command,
                "component": "redis_integration",
                "operation": "execute_store_temporal_memory",
            },
        )

    async def execute_retrieve_temporal_memory(
        self, correlation_id: UUID, retrieve_data: ModelOmniMemoryRetrieveData
    ) -> None:
        """
        Execute Redis GET operation for temporal memory retrieval.

        Args:
            correlation_id: Operation correlation ID
            retrieve_data: Memory retrieve data

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        query_type = retrieve_data.query_type

        if query_type == "key":
            # Simple key retrieval
            redis_key = f"omnimemory:temporal:{retrieve_data.memory_key}"
            command = "GET"
            parameters = [redis_key]

        elif query_type == "pattern":
            # Pattern-based retrieval
            query_params = retrieve_data.query_parameters or {}
            pattern = query_params.get("pattern", "omnimemory:temporal:*")
            command = "KEYS"
            parameters = [pattern]

        elif query_type == "scan":
            # Efficient scanning for large datasets
            query_params = retrieve_data.query_parameters or {}
            cursor = query_params.get("cursor", 0)
            match_pattern = query_params.get("pattern", "omnimemory:temporal:*")
            count = query_params.get("count", 100)
            command = "SCAN"
            parameters = [cursor, "MATCH", match_pattern, "COUNT", count]

        else:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Unsupported Redis query type: {query_type}",
            )

        await self._publish_redis_command(
            correlation_id=correlation_id,
            command=command,
            parameters=parameters,
            context={
                "operation": "retrieve_temporal_memory",
                "query_type": query_type,
                "memory_key": retrieve_data.memory_key,
            },
        )

        self.logger.info(
            f"Retrieve temporal memory command published to Redis adapter",
            extra={
                "correlation_id": str(correlation_id),
                "query_type": query_type,
                "memory_key": retrieve_data.memory_key,
                "command": command,
                "component": "redis_integration",
                "operation": "execute_retrieve_temporal_memory",
            },
        )

    async def execute_cache_operation(
        self,
        correlation_id: UUID,
        operation: str,
        cache_key: str,
        cache_data: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Execute Redis cache operation.

        Args:
            correlation_id: Operation correlation ID
            operation: Cache operation (set, get, delete, exists)
            cache_key: Cache key
            cache_data: Data to cache (for set operations)
            ttl_seconds: TTL for cached data

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        redis_key = f"omnimemory:cache:{cache_key}"

        if operation == "set":
            if cache_data is None:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message="Cache data required for SET operation",
                )

            if ttl_seconds:
                command = "SETEX"
                parameters = [redis_key, ttl_seconds, json.dumps(cache_data)]
            else:
                command = "SET"
                parameters = [redis_key, json.dumps(cache_data)]

        elif operation == "get":
            command = "GET"
            parameters = [redis_key]

        elif operation == "delete":
            command = "DEL"
            parameters = [redis_key]

        elif operation == "exists":
            command = "EXISTS"
            parameters = [redis_key]

        elif operation == "expire":
            if ttl_seconds is None:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message="TTL required for EXPIRE operation",
                )
            command = "EXPIRE"
            parameters = [redis_key, ttl_seconds]

        else:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Unsupported cache operation: {operation}",
            )

        await self._publish_redis_command(
            correlation_id=correlation_id,
            command=command,
            parameters=parameters,
            context={
                "operation": f"cache_{operation}",
                "cache_key": cache_key,
                "ttl_seconds": ttl_seconds,
            },
        )

        self.logger.info(
            f"Cache {operation} command published to Redis adapter",
            extra={
                "correlation_id": str(correlation_id),
                "operation": operation,
                "cache_key": cache_key,
                "command": command,
                "component": "redis_integration",
                "operation_method": "execute_cache_operation",
            },
        )

    async def execute_health_check(self, correlation_id: UUID) -> None:
        """
        Execute Redis health check.

        Args:
            correlation_id: Operation correlation ID
        """
        self._validate_initialization()

        # Redis PING command for health check
        command = "PING"
        parameters = []

        await self._publish_redis_command(
            correlation_id=correlation_id,
            command=command,
            parameters=parameters,
            context={"operation": "health_check"},
        )

        self.logger.info(
            f"Health check command published to Redis adapter",
            extra={
                "correlation_id": str(correlation_id),
                "command": command,
                "component": "redis_integration",
                "operation": "execute_health_check",
            },
        )

    async def execute_cleanup_expired(self, correlation_id: UUID) -> None:
        """
        Execute cleanup of expired temporal memories.

        Args:
            correlation_id: Operation correlation ID
        """
        self._validate_initialization()

        # Use SCAN to find keys and check TTL
        command = "SCAN"
        parameters = [0, "MATCH", "omnimemory:temporal:*", "COUNT", 1000]

        await self._publish_redis_command(
            correlation_id=correlation_id,
            command=command,
            parameters=parameters,
            context={"operation": "cleanup_expired"},
        )

        self.logger.info(
            f"Cleanup expired command published to Redis adapter",
            extra={
                "correlation_id": str(correlation_id),
                "command": command,
                "component": "redis_integration",
                "operation": "execute_cleanup_expired",
            },
        )

    async def _publish_redis_command(
        self,
        correlation_id: UUID,
        command: str,
        parameters: List[Any],
        context: Dict[str, Any],
    ) -> None:
        """
        Publish Redis command to infrastructure adapter via event bus.

        Args:
            correlation_id: Operation correlation ID
            command: Redis command
            parameters: Command parameters
            context: Additional context

        Raises:
            OnexError: If publishing fails
        """
        try:
            # Create ONEX event for Redis adapter
            event_payload = ModelOnexEvent.create_core_event(
                event_type="cache.redis.command_request",
                node_id=self.node_id,
                correlation_id=correlation_id,
                data={
                    "command": command,
                    "parameters": parameters,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Create topic for Redis adapter
            # Note: This would need to be implemented in omnibase_infra
            topic_string = "dev.omnibase.onex.cmd.redis-command.v1"

            # Create route spec
            route_spec = ModelRouteSpec.create_direct_route(topic_string)

            # Create event envelope
            envelope = ModelEventEnvelope(
                payload=event_payload,
                route_spec=route_spec,
                source_node_id=self.node_id,
                correlation_id=correlation_id,
                metadata={
                    "topic_spec": topic_string,
                    "cache_operation": command.lower(),
                    "omninode_namespace": "dev.omnibase.onex",
                },
            )

            # Add source hop
            envelope.add_source_hop(self.node_id, "OmniMemory Redis Integration")

            # Publish to event bus
            await self.event_bus.publish_async(envelope.payload)

            self.logger.debug(
                f"Redis command published via event bus",
                extra={
                    "correlation_id": str(correlation_id),
                    "topic": topic_string,
                    "command": command,
                    "parameter_count": len(parameters),
                    "component": "redis_integration",
                    "operation": "publish_redis_command",
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to publish Redis command",
                extra={
                    "correlation_id": str(correlation_id),
                    "command": command,
                    "error": str(e),
                    "component": "redis_integration",
                    "operation": "publish_redis_command_error",
                },
                exc_info=True,
            )
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to publish Redis command: {str(e)}",
                details={"correlation_id": str(correlation_id), "command": command},
            ) from e

    def _validate_initialization(self) -> None:
        """Validate that integration is properly initialized."""
        if not self.initialized or not self.event_bus:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Redis adapter integration not initialized",
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of Redis integration.

        Returns:
            Health check results
        """
        try:
            return {
                "status": "healthy" if self.initialized else "unhealthy",
                "component": "redis_adapter_integration",
                "initialized": self.initialized,
                "event_bus_available": self.event_bus is not None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "redis_adapter_integration",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
