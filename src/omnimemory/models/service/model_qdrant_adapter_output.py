"""Qdrant Adapter Output Model for operation responses."""

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelQdrantPoint(BaseModel):
    """Model for Qdrant point data."""

    id: str = Field(description="Point ID")
    vector: Optional[List[float]] = Field(default=None, description="Vector data")
    payload: Optional[Dict[str, Any]] = Field(
        default=None, description="Point payload/metadata"
    )
    score: Optional[float] = Field(
        default=None, description="Similarity score (for search results)"
    )


class ModelQdrantSearchResult(BaseModel):
    """Model for Qdrant search operation results."""

    points: List[ModelQdrantPoint] = Field(
        default_factory=list, description="Search result points"
    )
    total_count: int = Field(description="Total number of results")
    search_time_ms: float = Field(description="Search execution time in milliseconds")
    collection_name: str = Field(description="Collection that was searched")
    query_vector_hash: Optional[str] = Field(
        default=None, description="Hash of query vector"
    )


class ModelQdrantOperationResult(BaseModel):
    """Model for general Qdrant operation results."""

    operation_type: str = Field(description="Type of operation performed")
    success: bool = Field(description="Operation success status")
    affected_points: int = Field(default=0, description="Number of points affected")
    collection_name: Optional[str] = Field(
        default=None, description="Target collection name"
    )
    operation_time_ms: float = Field(description="Operation execution time")
    result_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Operation-specific result data"
    )


class ModelQdrantCollectionInfo(BaseModel):
    """Model for Qdrant collection information."""

    name: str = Field(description="Collection name")
    status: str = Field(description="Collection status")
    vectors_count: int = Field(description="Number of vectors in collection")
    indexed_vectors_count: int = Field(description="Number of indexed vectors")
    points_count: int = Field(description="Number of points in collection")
    segments_count: int = Field(description="Number of segments")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Collection configuration"
    )


class ModelQdrantHealthStatus(BaseModel):
    """Model for Qdrant health check status."""

    status: str = Field(description="Overall health status")
    version: Optional[str] = Field(default=None, description="Qdrant server version")
    collections: Optional[List[ModelQdrantCollectionInfo]] = Field(
        default=None, description="Collection information if requested"
    )
    response_time_ms: float = Field(description="Health check response time")
    connection_status: str = Field(description="Connection status")
    error_details: Optional[str] = Field(
        default=None, description="Error details if unhealthy"
    )


class ModelQdrantAdapterOutput(BaseModel):
    """
    Output model for Qdrant adapter operations.

    Contains operation results, timing information, and metadata
    following ONEX patterns for infrastructure tools.
    """

    operation_type: str = Field(description="Type of operation performed")
    success: bool = Field(description="Operation success status")
    correlation_id: Optional[UUID] = Field(
        default=None, description="Request correlation ID"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Operation completion timestamp"
    )
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context data"
    )

    # Operation results
    search_result: Optional[ModelQdrantSearchResult] = Field(
        default=None, description="Search operation results"
    )
    operation_result: Optional[ModelQdrantOperationResult] = Field(
        default=None, description="General operation results"
    )
    health_status: Optional[ModelQdrantHealthStatus] = Field(
        default=None, description="Health check results"
    )
    collection_info: Optional[ModelQdrantCollectionInfo] = Field(
        default=None, description="Collection information"
    )

    # Error information
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    error_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error context"
    )

    def add_search_result(
        self,
        points: List[Dict[str, Any]],
        search_time_ms: float,
        collection_name: str,
        query_vector_hash: Optional[str] = None,
    ) -> None:
        """
        Add search results to output.

        Args:
            points: Search result points
            search_time_ms: Search execution time
            collection_name: Collection that was searched
            query_vector_hash: Hash of query vector
        """
        qdrant_points = []
        for point_data in points:
            point = ModelQdrantPoint(
                id=str(point_data.get("id", "")),
                vector=point_data.get("vector"),
                payload=point_data.get("payload", {}),
                score=point_data.get("score"),
            )
            qdrant_points.append(point)

        self.search_result = ModelQdrantSearchResult(
            points=qdrant_points,
            total_count=len(qdrant_points),
            search_time_ms=search_time_ms,
            collection_name=collection_name,
            query_vector_hash=query_vector_hash,
        )

    def add_operation_result(
        self,
        operation_type: str,
        affected_points: int,
        operation_time_ms: float,
        collection_name: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add operation results to output.

        Args:
            operation_type: Type of operation
            affected_points: Number of points affected
            operation_time_ms: Operation execution time
            collection_name: Target collection name
            result_data: Operation-specific result data
        """
        self.operation_result = ModelQdrantOperationResult(
            operation_type=operation_type,
            success=self.success,
            affected_points=affected_points,
            collection_name=collection_name,
            operation_time_ms=operation_time_ms,
            result_data=result_data,
        )

    def add_health_status(
        self,
        status: str,
        response_time_ms: float,
        connection_status: str,
        version: Optional[str] = None,
        collections: Optional[List[Dict[str, Any]]] = None,
        error_details: Optional[str] = None,
    ) -> None:
        """
        Add health check results to output.

        Args:
            status: Overall health status
            response_time_ms: Health check response time
            connection_status: Connection status
            version: Qdrant server version
            collections: Collection information
            error_details: Error details if unhealthy
        """
        collection_info = None
        if collections:
            collection_info = []
            for coll_data in collections:
                coll_info = ModelQdrantCollectionInfo(
                    name=coll_data.get("name", ""),
                    status=coll_data.get("status", "unknown"),
                    vectors_count=coll_data.get("vectors_count", 0),
                    indexed_vectors_count=coll_data.get("indexed_vectors_count", 0),
                    points_count=coll_data.get("points_count", 0),
                    segments_count=coll_data.get("segments_count", 0),
                    config=coll_data.get("config", {}),
                )
                collection_info.append(coll_info)

        self.health_status = ModelQdrantHealthStatus(
            status=status,
            version=version,
            collections=collection_info,
            response_time_ms=response_time_ms,
            connection_status=connection_status,
            error_details=error_details,
        )

    def add_error(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add error information to output.

        Args:
            error_message: Error message
            error_code: Error code
            error_context: Additional error context
        """
        self.success = False
        self.error_message = error_message
        self.error_code = error_code
        self.error_context = error_context

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for logging.

        Returns:
            Summary dictionary
        """
        summary = {
            "operation_type": self.operation_type,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
        }

        if self.search_result:
            summary.update(
                {
                    "result_type": "search",
                    "results_count": self.search_result.total_count,
                    "collection": self.search_result.collection_name,
                    "search_time_ms": self.search_result.search_time_ms,
                }
            )

        if self.operation_result:
            summary.update(
                {
                    "result_type": "operation",
                    "affected_points": self.operation_result.affected_points,
                    "collection": self.operation_result.collection_name,
                    "operation_time_ms": self.operation_result.operation_time_ms,
                }
            )

        if self.health_status:
            summary.update(
                {
                    "result_type": "health_check",
                    "health_status": self.health_status.status,
                    "connection_status": self.health_status.connection_status,
                    "response_time_ms": self.health_status.response_time_ms,
                }
            )

        if not self.success:
            summary.update(
                {
                    "error_message": self.error_message,
                    "error_code": self.error_code,
                }
            )

        return summary
