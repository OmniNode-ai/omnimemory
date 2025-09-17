"""
Key-value pair model following ONEX standards.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelKeyValuePair(BaseModel):
    """Strongly typed key-value pair for metadata."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    key: str = Field(description="Metadata key identifier")
    value: str = Field(description="Metadata value content")

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        """Validate metadata key format."""
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return v.strip()
