"""
Configuration option model for ONEX Foundation Architecture.

Provides strongly typed configuration option for system configuration.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfigurationOption(BaseModel):
    """Strongly typed configuration option."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    key: str = Field(description="Configuration option key")
    value: str = Field(description="Configuration option value")
    description: Optional[str] = Field(
        default=None, description="Option description for documentation"
    )
    is_sensitive: bool = Field(
        default=False, description="Whether this option contains sensitive data"
    )

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate configuration key format."""
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return v.strip()
