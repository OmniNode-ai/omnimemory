"""
Cache value model for OmniMemory following ONEX standards.
"""

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, field_validator


class ModelCacheValue(BaseModel):
    """Strongly typed cache value with validation and serialization support."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    # Value type and content
    value_type: Literal[
        "string", "integer", "float", "boolean", "dict", "list"
    ] = Field(description="Type of the cached value for runtime validation")
    string_value: str | None = Field(default=None, description="String value")
    integer_value: int | None = Field(default=None, description="Integer value")
    float_value: float | None = Field(default=None, description="Float value")
    boolean_value: bool | None = Field(default=None, description="Boolean value")
    dict_value: Dict[str, Any] | None = Field(
        default=None, description="Dictionary value"
    )
    list_value: List[Any] | None = Field(default=None, description="List value")

    # Metadata
    is_sanitized: bool = Field(default=False, description="Whether value was sanitized")
    original_type: str = Field(description="Original Python type name")

    @field_validator("value_type")
    @classmethod
    def validate_value_type(cls, v: str) -> str:
        """Validate value type is supported."""
        valid_types = {"string", "integer", "float", "boolean", "dict", "list"}
        if v not in valid_types:
            raise ValueError(f"value_type must be one of: {valid_types}")
        return v

    @classmethod
    def from_raw_value(
        cls, value: Union[str, int, float, bool, Dict, List]
    ) -> "ModelCacheValue":
        """Create ModelCacheValue from raw Python value."""
        if isinstance(value, str):
            return cls(
                value_type="string",
                string_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, int):
            return cls(
                value_type="integer",
                integer_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, float):
            return cls(
                value_type="float",
                float_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, bool):
            return cls(
                value_type="boolean",
                boolean_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, dict):
            return cls(
                value_type="dict", dict_value=value, original_type=type(value).__name__
            )
        elif isinstance(value, list):
            return cls(
                value_type="list", list_value=value, original_type=type(value).__name__
            )
        else:
            raise ValueError(f"Unsupported cache value type: {type(value)}")

    def to_raw_value(self) -> Union[str, int, float, bool, Dict, List]:
        """Convert back to raw Python value."""
        if self.value_type == "string":
            if self.string_value is None:
                raise ValueError("string_value is None but value_type is 'string'")
            return self.string_value
        elif self.value_type == "integer":
            if self.integer_value is None:
                raise ValueError("integer_value is None but value_type is 'integer'")
            return self.integer_value
        elif self.value_type == "float":
            if self.float_value is None:
                raise ValueError("float_value is None but value_type is 'float'")
            return self.float_value
        elif self.value_type == "boolean":
            if self.boolean_value is None:
                raise ValueError("boolean_value is None but value_type is 'boolean'")
            return self.boolean_value
        elif self.value_type == "dict":
            if self.dict_value is None:
                raise ValueError("dict_value is None but value_type is 'dict'")
            return self.dict_value
        elif self.value_type == "list":
            if self.list_value is None:
                raise ValueError("list_value is None but value_type is 'list'")
            return self.list_value
        else:
            raise ValueError(f"Invalid value_type: {self.value_type}")

    def validate_value_consistency(self) -> None:
        """Validate that exactly one value field is set and matches value_type."""
        value_fields = [
            self.string_value,
            self.integer_value,
            self.float_value,
            self.boolean_value,
            self.dict_value,
            self.list_value,
        ]
        non_none_count = sum(1 for v in value_fields if v is not None)

        if non_none_count != 1:
            raise ValueError(
                f"Exactly one value field must be set, found {non_none_count}"
            )
