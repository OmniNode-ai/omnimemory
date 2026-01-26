# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent domain models for AdapterIntentGraph operations.

This module defines Pydantic models for intent classification storage and retrieval
operations in the graph database. These models provide structured input/output
contracts for the AdapterIntentGraph adapter.

Models:
    ModelIntentClassificationOutput: Input model for storing classified intents.
    ModelIntentStorageResult: Result of an intent storage operation.
    ModelIntentRecord: A single intent record returned from queries.
    ModelIntentQueryResult: Result of an intent query operation.

Example::

    from omnimemory.handlers.adapters.models import (
        ModelIntentClassificationOutput,
        ModelIntentStorageResult,
    )

    # Create classification output to store
    classification = ModelIntentClassificationOutput(
        intent_category="debugging",
        confidence=0.92,
        keywords=["error", "traceback", "fix"],
        raw_text="Help me debug this error",
    )

    # Receive storage result
    from uuid import UUID
    result = ModelIntentStorageResult(
        status="success",
        intent_id=UUID("12345678-1234-5678-1234-567812345678"),
        session_id="session_xyz789",
        created=True,
        execution_time_ms=12.5,
    )

.. versionadded:: 0.1.0
    Initial implementation for OMN-1457.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from omnibase_core.types.type_json import JsonType  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelIntentClassificationOutput",
    "ModelIntentDistributionResult",
    "ModelIntentQueryResult",
    "ModelIntentRecord",
    "ModelIntentStorageResult",
]


class ModelIntentClassificationOutput(  # omnimemory-model-exempt: adapter internal
    BaseModel
):
    """Input model for intent classification data to be stored.

    This model represents the output from an intent classifier that will be
    stored in the graph database and linked to a session.

    Attributes:
        intent_category: The classified intent category (e.g., "debugging",
            "code_generation", "explanation", "refactoring").
        confidence: Confidence score from the classifier, ranging from 0.0
            (no confidence) to 1.0 (full confidence).
        keywords: List of keywords extracted from the classified text that
            contributed to the classification decision.
        raw_text: The original text that was classified. Pass-through field
            for client use only; not stored in the graph database.
        metadata: Additional key-value metadata associated with the
            classification. Pass-through field for client use only; not
            stored in the graph database.

    Note:
        The ``raw_text`` and ``metadata`` fields are pass-through fields that
        are not stored in the graph database. They can be used by clients for
        local processing or logging but will not be persisted.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    intent_category: str = Field(
        ...,
        min_length=1,
        description=(
            "The classified intent category (e.g., 'debugging', "
            "'code_generation', 'explanation', 'refactoring')"
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (no confidence) to 1.0 (full confidence)",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords extracted from the text that contributed to classification",
    )
    raw_text: str | None = Field(
        default=None,
        description=(
            "Original text that was classified. Pass-through field for client "
            "use only; not stored in the graph database."
        ),
    )
    metadata: dict[str, JsonType] = Field(
        default_factory=dict,
        description=(
            "Additional key-value metadata (e.g., model version, timestamp). "
            "Pass-through field for client use only; not stored in the graph database."
        ),
    )


class ModelIntentStorageResult(BaseModel):  # omnimemory-model-exempt: adapter internal
    """Result of an intent storage operation.

    Returned by AdapterIntentGraph.store_intent() to indicate the outcome
    of storing an intent classification in the graph database.

    Attributes:
        status: Operation status - "success" if stored successfully,
            "error" if the operation failed.
        intent_id: Unique identifier of the created or updated intent node.
            None if the operation failed.
        session_id: The session identifier the intent was linked to.
        created: True if a new intent node was created, False if an
            existing intent was merged/updated.
        execution_time_ms: Time taken to execute the storage operation
            in milliseconds.
        error_message: Detailed error message if status is "error".
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: Literal["success", "error"] = Field(
        ...,
        description="Operation status - 'success' or 'error'",
    )
    intent_id: UUID | None = Field(
        default=None,
        description="Unique identifier of the created/updated intent node",
    )
    session_id: str = Field(
        ...,
        description="Session identifier the intent was linked to",
    )
    created: bool = Field(
        default=False,
        description="True if new intent created, False if merged with existing",
    )
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Storage operation execution time in milliseconds",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if status is 'error'",
    )


class ModelIntentRecord(BaseModel):  # omnimemory-model-exempt: adapter internal
    """A single intent record returned from query operations.

    Represents an intent classification that was previously stored in the
    graph database, including its metadata and timestamps.

    Attributes:
        intent_id: Unique identifier (UUID) for the intent node in the graph.
        session_ref: Optional session reference this intent belongs to, used
            for mapping to IntentRecordPayload.
        intent_category: The classified intent category.
        confidence: Confidence score from the original classification.
        keywords: Keywords associated with this intent.
        created_at_utc: UTC datetime when the intent was created.
        correlation_id: Optional correlation ID (UUID) linking this intent to
            a specific request or conversation turn.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    intent_id: UUID = Field(
        ...,
        description="Unique identifier for the intent node",
    )
    session_ref: str | None = Field(
        default=None,
        description="Session reference this intent belongs to",
    )
    intent_category: str = Field(
        ...,
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
    created_at_utc: datetime = Field(
        ...,
        description="UTC timestamp when the intent was created",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID linking to a specific request",
    )


class ModelIntentQueryResult(BaseModel):  # omnimemory-model-exempt: adapter internal
    """Result of an intent query operation.

    Returned by AdapterIntentGraph.get_session_intents() to provide the
    list of intents associated with a session.

    Attributes:
        status: Query status indicating the outcome:
            - "success": Query completed and found results.
            - "error": Query failed due to an error.
            - "not_found": The specified session was not found.
            - "no_results": Session exists but has no associated intents.
        intents: List of intent records found, ordered by creation time
            (most recent first by default).
        total_count: Total number of intent records returned.
        execution_time_ms: Time taken to execute the query in milliseconds.
        error_message: Detailed error message if status is "error".
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: Literal["success", "error", "not_found", "no_results"] = Field(
        ...,
        description="Query status: 'success', 'error', 'not_found', or 'no_results'",
    )
    intents: list[ModelIntentRecord] = Field(
        default_factory=list,
        description="Intent records found, ordered by creation time",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of intent records returned",
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


class ModelIntentDistributionResult(  # omnimemory-model-exempt: adapter internal
    BaseModel
):
    """Result of an intent distribution query.

    Returned by AdapterIntentGraph.get_intent_distribution() to provide
    intent category statistics for analytics.

    Attributes:
        status: Query status - "success" or "error".
        distribution: Dictionary mapping intent categories to counts.
        total_intents: Total number of intents across all categories.
        time_range_hours: The time range that was queried.
        execution_time_ms: Time taken to execute the query in milliseconds.
        error_message: Detailed error message if status is "error".
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: Literal["success", "error"] = Field(
        ...,
        description="Query status - 'success' or 'error'",
    )
    distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Intent category counts",
    )
    total_intents: int = Field(
        default=0,
        ge=0,
        description="Total intents across all categories",
    )
    time_range_hours: int = Field(
        default=24,
        ge=1,
        description="Time range in hours that was queried",
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
