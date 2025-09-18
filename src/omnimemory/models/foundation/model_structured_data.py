"""
Structured data model for ONEX Foundation Architecture.

Provides strongly typed structured data replacing List[Dict[str, Any]].
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .model_structured_field import ModelStructuredField


class ModelStructuredData(BaseModel):
    """Strongly typed structured data replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    fields: List[ModelStructuredField] = Field(
        default_factory=list,
        description="List of structured fields with type information",
    )
    schema_version: str = Field(
        default="1.0", description="Schema version for compatibility tracking"
    )

    def get_field_value(self, name: str) -> Optional[str]:
        """Get field value by name."""
        for field in self.fields:
            if field.name == name:
                return field.value
        return None

    def set_field_value(
        self, name: str, value: str, field_type: str = "string"
    ) -> None:
        """Set field value by name."""
        # Update existing or add new
        for field in self.fields:
            if field.name == name:
                field.value = value
                field.field_type = field_type
                return

        self.fields.append(
            ModelStructuredField(name=name, value=value, field_type=field_type)
        )
