# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event models for intent query operations via Kafka.

This module defines Pydantic models for intent query request and response
events used in the event-driven architecture. These models are used for
Kafka event transmission between OmniDash and OmniMemory.

Models:
    ModelIntentRecordPayload: Payload model for intent records in events.
    ModelIntentQueryRequestedEvent: Request event for intent queries.
    ModelIntentQueryResponseEvent: Response event with query results.

Example::

    from omnimemory.nodes.intent_query_effect.models import (
        ModelIntentRecordPayload,
        ModelIntentQueryRequestedEvent,
        ModelIntentQueryResponseEvent,
    )

    # Create a query request
    request = ModelIntentQueryRequestedEvent(
        query_type="distribution",
        time_range_hours=24,
    )

    # Process and create response
    response = ModelIntentQueryResponseEvent.create_distribution_response(
        query_id=request.query_id,
        distribution={"debugging": 10, "code_generation": 5},
        time_range_hours=24,
        execution_time_ms=45.2,
    )

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ModelIntentRecordPayload",
    "ModelIntentQueryRequestedEvent",
    "ModelIntentQueryResponseEvent",
]


class ModelIntentRecordPayload(BaseModel):
    """Payload model for intent records in Kafka events.

    This model represents an intent record formatted for event transmission.
    Unlike ModelIntentRecord (internal), session_ref is required here since
    events must have complete context.

    Attributes:
        intent_id: Unique identifier for the intent.
        session_ref: Session reference (required for event context).
        intent_category: The classified intent category.
        confidence: Confidence score from 0.0 to 1.0.
        keywords: Keywords associated with this intent.
        created_at: UTC timestamp when the intent was created.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    intent_id: UUID = Field(
        ...,
        description="Unique identifier for the intent",
    )
    session_ref: str = Field(
        ...,
        min_length=1,
        description="Session reference (required for event context)",
    )
    intent_category: str = Field(
        ...,
        min_length=1,
        description="The classified intent category",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords associated with this intent",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the intent was created",
    )


class ModelIntentQueryRequestedEvent(BaseModel):
    """Request event for intent queries via Kafka.

    Sent by clients (e.g., OmniDash) to request intent data from OmniMemory.
    Supports three query types: distribution, session, and recent.

    Attributes:
        query_id: Unique identifier for this query request.
        query_type: Type of query to perform.
        session_ref: Session reference for session queries (required when
            query_type is "session").
        time_range_hours: Time window for distribution and recent queries.
        min_confidence: Minimum confidence threshold for filtering.
        limit: Maximum number of results to return.
        correlation_id: Optional correlation ID for request tracing.
        requester_name: Name of the requesting service/component.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this query request",
    )
    query_type: Literal["distribution", "session", "recent"] = Field(
        ...,
        description="Type of intent query to perform",
    )
    session_ref: str | None = Field(
        default=None,
        description="Session reference for session queries",
    )
    time_range_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Time window in hours for queries",
    )
    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for filtering",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results to return",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for request tracing",
    )
    requester_name: str | None = Field(
        default=None,
        description="Name of the requesting service/component",
    )

    @model_validator(mode="after")
    def validate_session_ref_required(self) -> ModelIntentQueryRequestedEvent:
        """Validate session_ref is provided for session queries.

        Raises:
            ValueError: If query_type is "session" but session_ref is not provided.

        Returns:
            The validated model instance.
        """
        if self.query_type == "session" and not self.session_ref:
            raise ValueError("session_ref is required when query_type is 'session'")
        return self

    @classmethod
    def create_distribution_query(
        cls,
        *,
        time_range_hours: int = 24,
        min_confidence: float | None = None,
        correlation_id: UUID | None = None,
        requester_name: str | None = None,
    ) -> ModelIntentQueryRequestedEvent:
        """Create a distribution query request.

        Args:
            time_range_hours: Time window in hours.
            min_confidence: Optional minimum confidence threshold.
            correlation_id: Optional correlation ID.
            requester_name: Name of requester.

        Returns:
            Request event for distribution query.
        """
        return cls(
            query_type="distribution",
            time_range_hours=time_range_hours,
            min_confidence=min_confidence,
            correlation_id=correlation_id or uuid4(),
            requester_name=requester_name,
        )

    @classmethod
    def create_session_query(
        cls,
        *,
        session_ref: str,
        min_confidence: float | None = None,
        limit: int = 100,
        correlation_id: UUID | None = None,
        requester_name: str | None = None,
    ) -> ModelIntentQueryRequestedEvent:
        """Create a session query request.

        Args:
            session_ref: Session reference to query.
            min_confidence: Optional minimum confidence threshold.
            limit: Maximum results to return.
            correlation_id: Optional correlation ID.
            requester_name: Name of requester.

        Returns:
            Request event for session query.
        """
        return cls(
            query_type="session",
            session_ref=session_ref,
            min_confidence=min_confidence,
            limit=limit,
            correlation_id=correlation_id or uuid4(),
            requester_name=requester_name,
        )

    @classmethod
    def create_recent_query(
        cls,
        *,
        time_range_hours: int = 24,
        min_confidence: float | None = None,
        limit: int = 100,
        correlation_id: UUID | None = None,
        requester_name: str | None = None,
    ) -> ModelIntentQueryRequestedEvent:
        """Create a recent intents query request.

        Args:
            time_range_hours: Time window in hours.
            min_confidence: Optional minimum confidence threshold.
            limit: Maximum results to return.
            correlation_id: Optional correlation ID.
            requester_name: Name of requester.

        Returns:
            Request event for recent query.
        """
        return cls(
            query_type="recent",
            time_range_hours=time_range_hours,
            min_confidence=min_confidence,
            limit=limit,
            correlation_id=correlation_id or uuid4(),
            requester_name=requester_name,
        )


class ModelIntentQueryResponseEvent(BaseModel):
    """Response event for intent queries via Kafka.

    Sent by OmniMemory in response to query requests. Contains query results
    or error information.

    Attributes:
        query_id: The query ID from the original request.
        query_type: The type of query that was executed.
        status: Result status of the query.
        distribution: Category counts for distribution queries.
        intents: Intent records for session and recent queries.
        total_count: Total number of results.
        time_range_hours: Time range that was queried.
        execution_time_ms: Query execution time in milliseconds.
        error_message: Error details if status is "error".
        correlation_id: Correlation ID from the request.
        timestamp: When the response was created.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query_id: UUID = Field(
        ...,
        description="Query ID from the original request",
    )
    query_type: Literal["distribution", "session", "recent"] = Field(
        ...,
        description="Type of query that was executed",
    )
    status: Literal["success", "error", "no_results"] = Field(
        ...,
        description="Result status of the query",
    )
    distribution: dict[str, int] | None = Field(
        default=None,
        description="Category counts for distribution queries",
    )
    intents: list[ModelIntentRecordPayload] | None = Field(
        default=None,
        description="Intent records for session and recent queries",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of results",
    )
    time_range_hours: int | None = Field(
        default=None,
        ge=1,
        le=720,
        description="Time range that was queried",
    )
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if status is 'error'",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID from the request",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the response was created",
    )

    @classmethod
    def from_error(
        cls,
        *,
        query_id: UUID,
        query_type: Literal["distribution", "session", "recent"],
        error_message: str,
        correlation_id: UUID | None = None,
    ) -> ModelIntentQueryResponseEvent:
        """Create an error response.

        Args:
            query_id: The query ID from the request.
            query_type: The type of query that failed.
            error_message: Description of the error.
            correlation_id: Optional correlation ID.

        Returns:
            Response event with error status.
        """
        return cls(
            query_id=query_id,
            query_type=query_type,
            status="error",
            error_message=error_message,
            correlation_id=correlation_id,
        )

    @classmethod
    def create_distribution_response(
        cls,
        *,
        query_id: UUID,
        distribution: dict[str, int],
        time_range_hours: int,
        execution_time_ms: float,
        correlation_id: UUID | None = None,
    ) -> ModelIntentQueryResponseEvent:
        """Create a distribution query response.

        Args:
            query_id: The query ID from the request.
            distribution: Category counts.
            time_range_hours: Time range that was queried.
            execution_time_ms: Query execution time.
            correlation_id: Optional correlation ID.

        Returns:
            Response event with distribution data.
        """
        total = sum(distribution.values())
        return cls(
            query_id=query_id,
            query_type="distribution",
            status="success" if total > 0 else "no_results",
            distribution=distribution,
            total_count=total,
            time_range_hours=time_range_hours,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id,
        )

    @classmethod
    def create_session_response(
        cls,
        *,
        query_id: UUID,
        intents: list[ModelIntentRecordPayload],
        execution_time_ms: float,
        correlation_id: UUID | None = None,
    ) -> ModelIntentQueryResponseEvent:
        """Create a session query response.

        Args:
            query_id: The query ID from the request.
            intents: Intent records for the session.
            execution_time_ms: Query execution time.
            correlation_id: Optional correlation ID.

        Returns:
            Response event with session intents.
        """
        return cls(
            query_id=query_id,
            query_type="session",
            status="success" if intents else "no_results",
            intents=intents,
            total_count=len(intents),
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id,
        )

    @classmethod
    def create_recent_response(
        cls,
        *,
        query_id: UUID,
        intents: list[ModelIntentRecordPayload],
        time_range_hours: int,
        execution_time_ms: float,
        correlation_id: UUID | None = None,
    ) -> ModelIntentQueryResponseEvent:
        """Create a recent query response.

        Args:
            query_id: The query ID from the request.
            intents: Recent intent records.
            time_range_hours: Time range that was queried.
            execution_time_ms: Query execution time.
            correlation_id: Optional correlation ID.

        Returns:
            Response event with recent intents.
        """
        return cls(
            query_id=query_id,
            query_type="recent",
            status="success" if intents else "no_results",
            intents=intents,
            total_count=len(intents),
            time_range_hours=time_range_hours,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id,
        )
