"""
Optional strongly typed string list model for ONEX Foundation Architecture.

Provides optional strongly typed list of strings with validation and deduplication.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelOptionalStringList(BaseModel):
    """Optional strongly typed list of strings."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    values: Optional[List[str]] = Field(
        default=None, description="Optional list of string values, None if not set"
    )

    @field_validator("values")
    @classmethod
    def validate_optional_strings(cls, v: list[str] | None) -> list[str] | None:
        """Validate optional string values."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("values must be a list or None")

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

        return result if result else None
