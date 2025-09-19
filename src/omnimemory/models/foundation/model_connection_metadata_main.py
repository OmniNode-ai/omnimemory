"""
Connection metadata model following ONEX standards.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelConnectionMetadata(BaseModel):
    """Strongly typed metadata for connection objects."""

    connection_id: UUID = Field(description="Unique identifier for this connection")

    created_at: datetime = Field(
        default_factory=datetime.now, description="When this connection was created"
    )

    last_used_at: Optional[datetime] = Field(
        default=None, description="When this connection was last used"
    )

    usage_count: int = Field(
        default=0, description="Number of times this connection has been used"
    )

    connection_string: Optional[str] = Field(
        default=None, description="Connection string (sanitized)"
    )

    database_name: Optional[str] = Field(
        default=None, description="Name of the database"
    )

    server_version: Optional[str] = Field(
        default=None, description="Server version information"
    )

    is_healthy: bool = Field(
        default=True, description="Whether the connection is healthy"
    )

    last_health_check: Optional[datetime] = Field(
        default=None, description="When the connection was last health checked"
    )

    error_count: int = Field(
        default=0, description="Number of errors encountered with this connection"
    )

    last_error: Optional[str] = Field(
        default=None, description="Last error message (sanitized)"
    )
