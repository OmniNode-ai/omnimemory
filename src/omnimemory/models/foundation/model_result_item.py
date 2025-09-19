"""
Result item model for ONEX Foundation Architecture.

Provides strongly typed result item for operation results.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .model_structured_data import ModelStructuredData


class ModelResultItem(BaseModel):
    """Strongly typed result item for operation results."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    id: str = Field(description="Unique identifier for this result item")
    status: str = Field(
        description="Status of this specific item (success, failure, pending)"
    )
    message: str = Field(description="Human-readable message about this item")
    data: Optional[ModelStructuredData] = Field(
        default=None, description="Structured data associated with this item"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status values."""
        valid_statuses = {"success", "failure", "pending", "partial", "cancelled"}
        if v.lower() not in valid_statuses:
            raise ValueError(f"status must be one of: {valid_statuses}")
        return v.lower()
