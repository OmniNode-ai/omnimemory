"""Qdrant Adapter Input Model for operation requests."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelQdrantVectorOperationRequest(BaseModel):
    """Model for vector operation requests to Qdrant."""

    operation_type: str = Field(description="Type of vector operation")
    collection_name: str = Field(description="Target Qdrant collection")
    correlation_id: Optional[UUID] = Field(
        default=None, description="Request correlation ID"
    )
    timeout: Optional[int] = Field(
        default=None, description="Operation timeout in seconds"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context"
    )

    # Vector data for operations
    vector_id: Optional[str] = Field(default=None, description="Vector point ID")
    vector_data: Optional[List[float]] = Field(default=None, description="Vector data")
    payload: Optional[Dict[str, Any]] = Field(
        default=None, description="Vector payload/metadata"
    )

    # Search parameters
    query_vector: Optional[List[float]] = Field(
        default=None, description="Query vector for searches"
    )
    search_limit: Optional[int] = Field(
        default=10, description="Maximum search results"
    )
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum similarity score"
    )
    search_filter: Optional[Dict[str, Any]] = Field(
        default=None, description="Search filter conditions"
    )
    with_payload: bool = Field(
        default=True, description="Include payload in search results"
    )
    with_vector: bool = Field(
        default=False, description="Include vector in search results"
    )

    # Batch operations
    batch_points: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Batch of points for operations"
    )
    batch_ids: Optional[List[str]] = Field(
        default=None, description="Batch of IDs for operations"
    )

    # Collection management
    vector_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Vector configuration for collection creation"
    )
    collection_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Collection configuration parameters"
    )


class ModelQdrantAdapterInput(BaseModel):
    """
    Input model for Qdrant adapter operations.

    Encapsulates all necessary data for processing Qdrant vector operations
    through the NodeEffectService pattern.
    """

    operation_type: str = Field(
        description="Type of operation (vector_search, store_vector, health_check, etc.)"
    )
    correlation_id: Optional[UUID] = Field(
        default=None, description="Request correlation ID for tracing"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context data"
    )

    # Vector operation request
    vector_request: Optional[ModelQdrantVectorOperationRequest] = Field(
        default=None, description="Vector operation request data"
    )

    # Health check parameters
    health_check_type: Optional[str] = Field(
        default="basic", description="Type of health check to perform"
    )
    include_collections_info: bool = Field(
        default=False, description="Include collection information in health check"
    )

    def get_operation_timeout(self) -> Optional[int]:
        """Get operation timeout from request or default."""
        if self.vector_request and self.vector_request.timeout:
            return self.vector_request.timeout
        return None

    def get_collection_name(self) -> Optional[str]:
        """Get collection name from request."""
        if self.vector_request:
            return self.vector_request.collection_name
        return None

    def validate_for_operation(self) -> bool:
        """
        Validate input data for the specified operation type.

        Returns:
            True if valid for the operation, False otherwise
        """
        if self.operation_type == "vector_search":
            return (
                self.vector_request is not None
                and self.vector_request.query_vector is not None
                and self.vector_request.collection_name is not None
            )
        elif self.operation_type == "store_vector":
            return (
                self.vector_request is not None
                and self.vector_request.vector_id is not None
                and self.vector_request.vector_data is not None
                and self.vector_request.collection_name is not None
            )
        elif self.operation_type == "get_vector":
            return (
                self.vector_request is not None
                and self.vector_request.vector_id is not None
                and self.vector_request.collection_name is not None
            )
        elif self.operation_type == "delete_vector":
            return (
                self.vector_request is not None
                and (
                    self.vector_request.vector_id is not None
                    or self.vector_request.batch_ids is not None
                )
                and self.vector_request.collection_name is not None
            )
        elif self.operation_type == "batch_upsert":
            return (
                self.vector_request is not None
                and self.vector_request.batch_points is not None
                and self.vector_request.collection_name is not None
            )
        elif self.operation_type == "create_collection":
            return (
                self.vector_request is not None
                and self.vector_request.collection_name is not None
                and self.vector_request.vector_config is not None
            )
        elif self.operation_type == "health_check":
            return True
        else:
            return False

    def get_request_summary(self) -> Dict[str, Any]:
        """
        Get summary of request for logging.

        Returns:
            Dictionary with request summary
        """
        summary = {
            "operation_type": self.operation_type,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "has_context": self.context is not None,
        }

        if self.vector_request:
            summary.update(
                {
                    "collection_name": self.vector_request.collection_name,
                    "has_vector_data": self.vector_request.vector_data is not None,
                    "has_query_vector": self.vector_request.query_vector is not None,
                    "vector_id": self.vector_request.vector_id,
                    "search_limit": self.vector_request.search_limit,
                    "has_filter": self.vector_request.search_filter is not None,
                }
            )

        if self.operation_type == "health_check":
            summary.update(
                {
                    "health_check_type": self.health_check_type,
                    "include_collections_info": self.include_collections_info,
                }
            )

        return summary
