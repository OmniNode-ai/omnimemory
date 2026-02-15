"""
Audit-related Pydantic models for OmniMemory ONEX architecture.

This module contains models for audit event tracking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.model_audit_metadata import (
    AuditEventDetails,
    ResourceUsageMetadata,
)

__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "ModelAuditEvent",
]


class AuditEventType(str, Enum):
    """Types of auditable events."""

    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_DELETE = "memory_delete"
    CONFIG_CHANGE = "config_change"
    PII_DETECTED = "pii_detected"
    PII_SANITIZED = "pii_sanitized"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    ACCESS_DENIED = "access_denied"
    SYSTEM_ERROR = "system_error"
    SECURITY_VIOLATION = "security_violation"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelAuditEvent(BaseModel):
    """Structured audit event model."""

    # Event identification
    event_id: str = Field(description="Unique event identifier")
    timestamp: datetime = Field(description="Event timestamp in UTC")
    event_type: AuditEventType = Field(description="Type of event")
    severity: AuditSeverity = Field(description="Event severity level")

    # Context information
    operation: str = Field(description="Operation being performed")
    component: str = Field(description="Component generating the event")
    user_context: str | None = Field(
        default=None, description="User context if available"
    )
    session_id: str | None = Field(default=None, description="Session identifier")

    # Event details
    message: str = Field(description="Human-readable event description")
    details: AuditEventDetails = Field(
        default_factory=AuditEventDetails,
        description="Additional event details",
    )

    # Security context
    source_ip: str | None = Field(default=None, description="Source IP address")
    user_agent: str | None = Field(default=None, description="User agent string")

    # Performance data
    duration_ms: float | None = Field(
        default=None, ge=0.0, description="Operation duration in milliseconds"
    )
    resource_usage: ResourceUsageMetadata | None = Field(
        default=None, description="Resource usage metrics"
    )

    # Compliance tracking
    data_classification: str | None = Field(
        default=None, description="Data classification level"
    )
    pii_detected: bool = Field(default=False, description="Whether PII was detected")
    sanitized: bool = Field(default=False, description="Whether data was sanitized")

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )
