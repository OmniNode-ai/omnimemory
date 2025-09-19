"""
Event data model for ONEX Foundation Architecture.

Provides strongly typed event data for system events.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelEventData(BaseModel):
    """Strongly typed event data for system events."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    event_type: str = Field(
        description="Type of event (creation, update, deletion, etc.)"
    )
    timestamp: str = Field(description="ISO 8601 timestamp of the event")
    source: str = Field(description="Source system or component generating the event")
    severity: str = Field(
        default="info",
        description="Event severity level (debug, info, warning, error, critical)",
    )
    message: str = Field(description="Human-readable event message")
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for tracking related events"
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type format."""
        if not v or not v.strip():
            raise ValueError("event_type cannot be empty")
        return v.strip().lower()

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity level."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"severity must be one of: {valid_levels}")
        return v.lower()
