"""
Configuration model for ONEX Foundation Architecture.

Provides strongly typed configuration replacing Dict[str, Any].
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .model_configuration_option import ModelConfigurationOption


class ModelConfiguration(BaseModel):
    """Strongly typed configuration replacing Dict[str, Any]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    options: List[ModelConfigurationOption] = Field(
        default_factory=list, description="List of configuration options with metadata"
    )

    def get_option(self, key: str) -> Optional[str]:
        """Get configuration option value by key."""
        for option in self.options:
            if option.key == key:
                return option.value
        return None

    def set_option(
        self,
        key: str,
        value: str,
        description: Optional[str] = None,
        is_sensitive: bool = False,
    ) -> None:
        """Set configuration option with metadata."""
        # Update existing or add new
        for option in self.options:
            if option.key == key:
                option.value = value
                if description is not None:
                    option.description = description
                option.is_sensitive = is_sensitive
                return

        self.options.append(
            ModelConfigurationOption(
                key=key, value=value, description=description, is_sensitive=is_sensitive
            )
        )
