"""
Structured field model for ONEX Foundation Architecture.

Provides strongly typed field for structured data replacing generic dict fields.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelStructuredField(BaseModel):
    """Strongly typed field for structured data."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    name: str = Field(description="Field name identifier")
    value: str = Field(description="Field value content")
    field_type: str = Field(
        default="string",
        description="Field type indicator (string, number, boolean, etc.)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate field name format."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()
