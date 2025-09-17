"""
Individual tag model following ONEX standards.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelTag(BaseModel):
    """Individual tag model with metadata."""

    name: str = Field(
        description="Tag name",
        min_length=1,
        max_length=100,
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional tag category for organization",
        max_length=50,
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the tag was created",
    )
    created_by: Optional[UUID] = Field(
        default=None,
        description="User who created the tag",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Tag importance weight",
    )

    @field_validator("name")
    @classmethod
    def validate_tag_name(cls, v):
        """Validate tag name format."""
        # Remove whitespace and convert to lowercase
        v = v.strip().lower()

        # Check for invalid characters
        invalid_chars = set("!@#$%^&*()+={}[]|\\:\";'<>?,/`~")
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Tag name contains invalid characters: {v}")

        # Replace spaces with underscores
        v = v.replace(" ", "_").replace("-", "_")

        return v
