"""
Error details model following ONEX standards.

Uses the standard ONEX error patterns from omnibase_core when available.
"""

from datetime import datetime
from typing import Union
from uuid import UUID

# Import standard ONEX error types from omnibase_core
try:
    from omnibase_core.core.errors.core_errors import (
        OnexErrorCode as CoreErrorCode,  # type: ignore[import-untyped]
    )
    from omnibase_core.enums.enum_log_level import (
        EnumLogLevel as CoreSeverity,  # type: ignore[import-untyped]
    )
except ImportError:
    # Fallback for development environments without omnibase_core
    from enum import Enum

    class CoreErrorCode(str, Enum):
        """Base class for ONEX error codes (fallback implementation)."""

        pass

    class CoreSeverity(str, Enum):
        """Base class for severity levels (fallback implementation)."""

        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"


from pydantic import BaseModel, Field, field_validator

# Local omnimemory-specific error codes
from ...enums import EnumErrorCode

# Type aliases for error codes and severity
ErrorCodeType = Union[CoreErrorCode, EnumErrorCode, str]
SeverityType = Union[CoreSeverity, str]  # Type alias instead of variable


class ModelErrorDetails(BaseModel):
    """
    Error details model following ONEX standards with omnibase_core integration.
    """

    # Error identification
    error_id: UUID = Field(
        description="Unique identifier for this error instance",
    )
    error_code: ErrorCodeType = Field(
        description="Standardized error code (core or omnimemory-specific)",
    )
    error_type: str = Field(
        max_length=100,
        description="Type or category of the error (VALIDATION, AUTHENTICATION, etc.)",
    )

    # Error information
    message: str = Field(
        max_length=500,
        description="Human-readable error message (sanitized for security)",
    )
    detailed_message: str | None = Field(
        default=None,
        description="Detailed technical error message",
    )
    severity: SeverityType = Field(
        description="Severity level of the error (using core severity)",
    )

    # Context information
    component: str = Field(
        max_length=100,
        description="System component where the error occurred (cache, database, etc.)",
    )
    operation: str = Field(
        description="Operation that was being performed",
    )
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Additional context for the error",
    )

    # Correlation and tracing
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing related operations",
    )
    parent_error_id: UUID | None = Field(
        default=None,
        description="ID of parent error if this is a cascading error",
    )
    trace_id: str | None = Field(
        default=None,
        description="Distributed tracing identifier",
    )

    # Stack trace and debugging
    stack_trace: list[str] = Field(
        default_factory=list,
        description="Stack trace lines",
    )
    inner_error: str | None = Field(
        default=None,
        description="Inner exception details",
    )

    # Resolution information
    is_retryable: bool = Field(
        default=False,
        description="Whether this error can be retried",
    )
    retry_after_seconds: int | None = Field(
        default=None,
        ge=0,
        le=3600,
        description="Suggested retry delay in seconds (0-3600 max)",
    )
    resolution_hint: str | None = Field(
        default=None,
        description="Hint on how to resolve this error",
    )

    # User information
    user_message: str | None = Field(
        default=None,
        description="User-friendly error message",
    )
    user_action_required: bool = Field(
        default=False,
        description="Whether user action is required to resolve",
    )

    # Temporal information
    occurred_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred",
    )
    resolved_at: datetime | None = Field(
        default=None,
        description="When the error was resolved (if applicable)",
    )

    # Recovery information
    recovery_attempted: bool = Field(
        default=False,
        description="Whether automatic recovery was attempted",
    )
    recovery_successful: bool = Field(
        default=False,
        description="Whether recovery was successful",
    )
    recovery_details: str | None = Field(
        default=None,
        description="Details about recovery attempts",
    )

    # Metrics and monitoring
    occurrence_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this error has occurred (for monitoring)",
    )
    first_occurrence: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this error first occurred",
    )
    last_occurrence: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this error last occurred",
    )

    # Additional metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the error",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional error metadata (sanitized for security)",
    )

    @field_validator("detailed_message", "inner_error", "resolution_hint")
    @classmethod
    def validate_sensitive_content(cls, v: str | None) -> str | None:
        """Sanitize potentially sensitive content in error details."""
        if v is None:
            return v
        # Check for sensitive patterns and sanitize
        v_lower = v.lower()
        sensitive_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "api_key",
            "auth",
            "private_key",
            "certificate",
        ]
        if any(pattern in v_lower for pattern in sensitive_patterns):
            return "[REDACTED_SENSITIVE_CONTENT]"
        # Limit length to prevent excessive logging
        if len(v) > 2000:
            return v[:2000] + "[TRUNCATED]"
        return v

    @field_validator("stack_trace")
    @classmethod
    def validate_stack_trace(cls, v: list[str]) -> list[str]:
        """Validate and sanitize stack trace for security."""
        if not v:
            return v
        # Limit stack trace depth and line length
        sanitized = []
        for i, line in enumerate(v[:50]):  # Max 50 stack frames
            if len(line) > 500:
                line = line[:500] + "[TRUNCATED]"
            sanitized.append(line)
        return sanitized

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate error tags for security and performance."""
        if not v:
            return v
        # Limit number and length of tags
        return [tag[:50] for tag in v[:20] if tag.strip()]
