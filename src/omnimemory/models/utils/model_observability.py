"""
Observability-related Pydantic models for OmniMemory ONEX architecture.

This module contains models for structured logging and correlation tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from ..foundation.model_typed_collections import ModelMetadata

__all__ = [
    "ModelCorrelationContext",
    "ModelStructuredLogEntry",
    "TraceLevel",
    # Backward compatibility aliases
    "CorrelationContext",
    "StructuredLogEntry",
]


class TraceLevel(Enum):
    """Trace level enumeration for different types of operations."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelStructuredLogEntry(BaseModel):
    """Pydantic model for validating structured log entries.

    This model enforces the log schema documented at the module level.
    All handler operation log events MUST conform to this schema for
    downstream ingestion by ELK, Datadog, or other log aggregators.

    Required fields (ALWAYS present):
        correlation_id: Unique request correlation identifier
        operation: Operation name (e.g., "store", "retrieve", "delete")
        handler: Handler name (e.g., "filesystem", "postgresql")
        status: Operation status, one of "success" or "failure"
        latency_ms: Operation latency in milliseconds (rounded to 2 decimal places)
        timestamp: ISO8601 timestamp in UTC (format: YYYY-MM-DDTHH:MM:SS.sssZ)

    Optional fields (only present on failure):
        error_type: Exception class name (e.g., "ValueError", "IOError")
        error_message: Sanitized error message (PII-safe)

    Example (success):
        entry = ModelStructuredLogEntry(
            correlation_id="abc123-def456",
            operation="store",
            handler="filesystem",
            status="success",
            latency_ms=45.23,
            timestamp="2025-01-19T12:34:56.789Z"
        )

    Example (failure):
        entry = ModelStructuredLogEntry(
            correlation_id="abc123-def456",
            operation="store",
            handler="filesystem",
            status="failure",
            latency_ms=102.5,
            timestamp="2025-01-19T12:34:56.789Z",
            error_type="IOError",
            error_message="Permission denied"
        )
    """

    correlation_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9\-_]+$",
        description="Unique request correlation identifier",
    )
    operation: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Operation name (e.g., store, retrieve, delete)",
    )
    handler: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Handler name (e.g., filesystem, postgresql)",
    )
    status: Literal["success", "failure"] = Field(
        ...,
        description="Operation status (success or failure)",
    )
    latency_ms: float = Field(
        ...,
        ge=0,
        description="Operation latency in milliseconds",
    )
    timestamp: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
        description="ISO8601 timestamp in UTC (YYYY-MM-DDTHH:MM:SS.sssZ)",
    )

    # Optional fields (only on failure)
    error_type: str | None = Field(
        default=None,
        max_length=256,
        description="Exception class name (only on failure)",
    )
    error_message: str | None = Field(
        default=None,
        max_length=1000,
        description="Sanitized error message (only on failure, PII-safe)",
    )


class ModelCorrelationContext(BaseModel):
    """Context information for correlation tracking."""

    correlation_id: str = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    request_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    operation: str | None = Field(default=None)
    parent_correlation_id: str | None = Field(default=None)
    trace_level: TraceLevel = Field(default=TraceLevel.INFO)
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Backward compatibility aliases
StructuredLogEntry = ModelStructuredLogEntry
CorrelationContext = ModelCorrelationContext
