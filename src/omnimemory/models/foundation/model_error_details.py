"""
Error details model following ONEX standards.

Uses the standard ONEX error patterns from omnibase_core when available.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

# Import standard ONEX error types from omnibase_core
try:
    from omnibase_core.core.errors.core_errors import OnexErrorCode as CoreErrorCode
    from omnibase_core.enums.enum_log_level import EnumLogLevel as CoreSeverity

    # Local omnimemory-specific error codes
    from ...enums.enum_error_code import OmniMemoryErrorCode

    # Union type for error codes
    ErrorCodeType = CoreErrorCode | OmniMemoryErrorCode | str
    SeverityType = CoreSeverity
except ImportError:
    # Fallback for development environments
    from ...enums.enum_error_code import OmniMemoryErrorCode as ErrorCodeType
    from ...enums.enum_severity import EnumSeverity as SeverityType


class ModelErrorDetails(BaseModel):
    """Error details model following ONEX standards with omnibase_core integration."""

    # Error identification
    error_id: UUID = Field(
        description="Unique identifier for this error instance",
    )
    error_code: ErrorCodeType = Field(
        description="Standardized error code (core or omnimemory-specific)",
    )
    error_type: str = Field(
        description="Type or category of the error",
    )

    # Error information
    message: str = Field(
        description="Human-readable error message",
    )
    detailed_message: str | None = Field(
        default=None,
        description="Detailed technical error message",
    )
    severity: SeverityType = Field(
        description="Severity level of the error (using core severity levels)",
    )

    # Context information
    component: str = Field(
        description="System component where the error occurred",
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
        description="Suggested retry delay in seconds",
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
        description="Number of times this error has occurred",
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
        description="Additional error metadata",
    )
