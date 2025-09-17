"""Qdrant Adapter Integration for OmniMemory vector search operations."""

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
    ModelOmniMemoryStoreData,
    ModelOmniMemoryVectorSearchData,
)


class QdrantAdapterIntegration(BaseModel):
    """
    Integration layer for Qdrant vector database operations via event bus.

    Maps OmniMemory vector operations to Qdrant API calls and publishes them
    to the infrastructure Qdrant adapter via RedPanda event bus.
    """

    node_id: str = Field(
        default="omnimemory_qdrant_integration", description="Integration node ID"
    )
    default_collection: str = Field(
        default="omnimemory_vectors", description="Default vector collection"
    )
    event_bus: Optional[ProtocolEventBus] = Field(
        default=None, description="Protocol event bus for Qdrant operations"
    )
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger("omnimemory.qdrant_integration"),
        description="Logger instance",
    )
    initialized: bool = Field(default=False, description="Initialization status")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def initialize(self, event_bus: ProtocolEventBus) -> None:
        """
        Initialize Qdrant adapter integration.

        Args:
            event_bus: ProtocolEventBus for infrastructure communication

        Raises:
            OnexError: If initialization fails
        """
        if not event_bus:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus not available for Qdrant integration",
            )

        self.event_bus = event_bus
        self.initialized = True

        self.logger.info(
            "Qdrant adapter integration initialized",
            extra={
                "node_id": self.node_id,
                "default_collection": self.default_collection,
                "component": "qdrant_integration",
                "operation": "initialization",
            },
        )

    async def execute_store_vector(
        self,
        correlation_id: UUID,
        store_data: ModelOmniMemoryStoreData,
        vector: List[float],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Execute Qdrant upsert operation for vector storage.

        Args:
            correlation_id: Operation correlation ID
            store_data: Memory store data
            vector: Vector to store
            collection_name: Qdrant collection name

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        collection = collection_name or self.default_collection

        # Create Qdrant point
        point = {
            "id": store_data.memory_key,
            "vector": vector,
            "payload": {
                "content_hash": store_data.content_hash,
                "memory_type": store_data.memory_type,
                "metadata": store_data.metadata,
                "storage_size": store_data.storage_size,
                "created_at": datetime.utcnow().isoformat(),
                "vector_dimensions": store_data.vector_dimensions,
            },
        }

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="upsert",
            collection_name=collection,
            data={"points": [point]},
            context={
                "operation": "store_vector",
                "memory_key": store_data.memory_key,
                "vector_dimensions": len(vector),
            },
        )

        self.logger.info(
            f"Store vector operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "memory_key": store_data.memory_key,
                "collection": collection,
                "vector_dimensions": len(vector),
                "component": "qdrant_integration",
                "operation": "execute_store_vector",
            },
        )

    async def execute_vector_search(
        self,
        correlation_id: UUID,
        search_data: ModelOmniMemoryVectorSearchData,
        query_vector: List[float],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Execute Qdrant vector search operation.

        Args:
            correlation_id: Operation correlation ID
            search_data: Vector search data
            query_vector: Query vector
            collection_name: Qdrant collection name

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        collection = (
            collection_name or search_data.index_name or self.default_collection
        )

        # Create Qdrant search request
        search_request = {
            "vector": query_vector,
            "limit": search_data.max_results,
            "score_threshold": search_data.similarity_threshold,
            "with_payload": True,
            "with_vector": False,  # Don't return vectors by default to save bandwidth
        }

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="search",
            collection_name=collection,
            data=search_request,
            context={
                "operation": "vector_search",
                "search_type": search_data.search_type,
                "similarity_threshold": search_data.similarity_threshold,
                "max_results": search_data.max_results,
            },
        )

        self.logger.info(
            f"Vector search operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "collection": collection,
                "similarity_threshold": search_data.similarity_threshold,
                "max_results": search_data.max_results,
                "vector_dimensions": search_data.vector_dimensions,
                "component": "qdrant_integration",
                "operation": "execute_vector_search",
            },
        )

    async def execute_similarity_search_with_filter(
        self,
        correlation_id: UUID,
        query_vector: List[float],
        filter_conditions: Dict[str, Any],
        similarity_threshold: float = 0.8,
        max_results: int = 10,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Execute Qdrant vector search with metadata filtering.

        Args:
            correlation_id: Operation correlation ID
            query_vector: Query vector
            filter_conditions: Metadata filter conditions
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum results to return
            collection_name: Qdrant collection name

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        collection = collection_name or self.default_collection

        # Create Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filter_conditions)

        # Create search request with filter
        search_request = {
            "vector": query_vector,
            "filter": qdrant_filter,
            "limit": max_results,
            "score_threshold": similarity_threshold,
            "with_payload": True,
            "with_vector": False,
        }

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="search",
            collection_name=collection,
            data=search_request,
            context={
                "operation": "similarity_search_with_filter",
                "filter_conditions": filter_conditions,
                "similarity_threshold": similarity_threshold,
                "max_results": max_results,
            },
        )

        self.logger.info(
            f"Filtered vector search operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "collection": collection,
                "similarity_threshold": similarity_threshold,
                "max_results": max_results,
                "filter_conditions": filter_conditions,
                "component": "qdrant_integration",
                "operation": "execute_similarity_search_with_filter",
            },
        )

    async def execute_get_vector(
        self,
        correlation_id: UUID,
        point_id: str,
        collection_name: Optional[str] = None,
        with_vector: bool = True,
    ) -> None:
        """
        Execute Qdrant get point operation.

        Args:
            correlation_id: Operation correlation ID
            point_id: Point ID to retrieve
            collection_name: Qdrant collection name
            with_vector: Whether to include vector in response

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        collection = collection_name or self.default_collection

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="get",
            collection_name=collection,
            data={"ids": [point_id], "with_payload": True, "with_vector": with_vector},
            context={
                "operation": "get_vector",
                "point_id": point_id,
                "with_vector": with_vector,
            },
        )

        self.logger.info(
            f"Get vector operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "collection": collection,
                "point_id": point_id,
                "with_vector": with_vector,
                "component": "qdrant_integration",
                "operation": "execute_get_vector",
            },
        )

    async def execute_delete_vector(
        self, correlation_id: UUID, point_id: str, collection_name: Optional[str] = None
    ) -> None:
        """
        Execute Qdrant delete point operation.

        Args:
            correlation_id: Operation correlation ID
            point_id: Point ID to delete
            collection_name: Qdrant collection name

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        collection = collection_name or self.default_collection

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="delete",
            collection_name=collection,
            data={"points": [point_id]},
            context={"operation": "delete_vector", "point_id": point_id},
        )

        self.logger.info(
            f"Delete vector operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "collection": collection,
                "point_id": point_id,
                "component": "qdrant_integration",
                "operation": "execute_delete_vector",
            },
        )

    async def execute_create_collection(
        self, correlation_id: UUID, collection_name: str, vector_config: Dict[str, Any]
    ) -> None:
        """
        Execute Qdrant create collection operation.

        Args:
            correlation_id: Operation correlation ID
            collection_name: Name of collection to create
            vector_config: Vector configuration (size, distance, etc.)

        Raises:
            OnexError: If operation fails
        """
        self._validate_initialization()

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="create_collection",
            collection_name=collection_name,
            data={"vectors": vector_config},
            context={"operation": "create_collection", "vector_config": vector_config},
        )

        self.logger.info(
            f"Create collection operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "collection_name": collection_name,
                "vector_config": vector_config,
                "component": "qdrant_integration",
                "operation": "execute_create_collection",
            },
        )

    async def execute_health_check(self, correlation_id: UUID) -> None:
        """
        Execute Qdrant health check.

        Args:
            correlation_id: Operation correlation ID
        """
        self._validate_initialization()

        await self._publish_qdrant_operation(
            correlation_id=correlation_id,
            operation="health_check",
            collection_name="",  # Not needed for health check
            data={},
            context={"operation": "health_check"},
        )

        self.logger.info(
            f"Health check operation published to Qdrant adapter",
            extra={
                "correlation_id": str(correlation_id),
                "component": "qdrant_integration",
                "operation": "execute_health_check",
            },
        )

    def _build_qdrant_filter(self, filter_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Qdrant filter from filter conditions.

        Args:
            filter_conditions: Filter conditions

        Returns:
            Qdrant filter structure
        """
        must_conditions = []

        for key, value in filter_conditions.items():
            if isinstance(value, dict):
                # Handle range conditions
                if "gte" in value or "gt" in value or "lte" in value or "lt" in value:
                    range_condition = {"key": key, "range": value}
                    must_conditions.append(range_condition)
                elif "in" in value:
                    # Handle "in" conditions
                    match_condition = {"key": key, "match": {"any": value["in"]}}
                    must_conditions.append(match_condition)
            else:
                # Simple match condition
                match_condition = {"key": key, "match": {"value": value}}
                must_conditions.append(match_condition)

        return {"must": must_conditions} if must_conditions else {}

    async def _publish_qdrant_operation(
        self,
        correlation_id: UUID,
        operation: str,
        collection_name: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """
        Publish Qdrant operation to infrastructure adapter via event bus.

        Args:
            correlation_id: Operation correlation ID
            operation: Qdrant operation type
            collection_name: Target collection
            data: Operation data
            context: Additional context

        Raises:
            OnexError: If publishing fails
        """
        try:
            # Create ONEX event for Qdrant adapter
            event_payload = ModelOnexEvent.create_core_event(
                event_type="vector.qdrant.operation_request",
                node_id=self.node_id,
                correlation_id=correlation_id,
                data={
                    "operation": operation,
                    "collection_name": collection_name,
                    "data": data,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Create topic for Qdrant adapter
            # Note: This would need to be implemented in omnibase_infra
            topic_string = "dev.omnibase.onex.cmd.qdrant-operation.v1"

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
                    "vector_operation": operation,
                    "collection_name": collection_name,
                    "omninode_namespace": "dev.omnibase.onex",
                },
            )

            # Add source hop
            envelope.add_source_hop(self.node_id, "OmniMemory Qdrant Integration")

            # Publish to event bus
            await self.event_bus.publish_async(envelope.payload)

            self.logger.debug(
                f"Qdrant operation published via event bus",
                extra={
                    "correlation_id": str(correlation_id),
                    "topic": topic_string,
                    "operation": operation,
                    "collection": collection_name,
                    "component": "qdrant_integration",
                    "operation_method": "publish_qdrant_operation",
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to publish Qdrant operation",
                extra={
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "collection": collection_name,
                    "error": str(e),
                    "component": "qdrant_integration",
                    "operation_method": "publish_qdrant_operation_error",
                },
                exc_info=True,
            )
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to publish Qdrant operation: {str(e)}",
                details={
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "collection": collection_name,
                },
            ) from e

    def _validate_initialization(self) -> None:
        """Validate that integration is properly initialized."""
        if not self.initialized or not self.event_bus:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Qdrant adapter integration not initialized",
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of Qdrant integration.

        Returns:
            Health check results
        """
        try:
            return {
                "status": "healthy" if self.initialized else "unhealthy",
                "component": "qdrant_adapter_integration",
                "initialized": self.initialized,
                "event_bus_available": self.event_bus is not None,
                "default_collection": self.default_collection,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "qdrant_adapter_integration",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
