"""
String list collection model following ONEX standards.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelStringList(BaseModel):
    """Strongly typed list of strings following ONEX standards."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    values: List[str] = Field(
        default_factory=list,
        description="List of string values with validation and deduplication",
    )

    @field_validator("values")
    @classmethod
    def validate_strings(cls, v):
        """Validate and deduplicate string values."""
        if not isinstance(v, list):
            raise ValueError("values must be a list")

        # Remove empty strings and duplicates while preserving order
        # Use O(1) set operations for efficient deduplication
        seen = set()
        result = []
        for item in v:
            if item:
                stripped_item = item.strip()
                if stripped_item and stripped_item not in seen:
                    seen.add(stripped_item)
                    result.append(stripped_item)

        return result
