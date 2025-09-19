"""
ONEX-compliant typed model for configuration change metadata.

This module provides strongly typed replacement for Dict[str, Any] patterns
in health management, ensuring type safety and validation.
"""

from typing import Dict, List

from pydantic import BaseModel, Field


class ModelConfigurationChangeMetadata(BaseModel):
    """Strongly typed metadata for configuration changes."""

    changed_keys: List[str] = Field(
        description="List of configuration keys that were modified"
    )

    change_source: str = Field(description="Source of the configuration change")

    validation_results: Dict[str, bool] = Field(
        description="Validation results for each changed configuration"
    )

    requires_restart: bool = Field(
        default=False, description="Whether changes require service restart"
    )

    backup_created: bool = Field(
        default=False, description="Whether configuration backup was created"
    )

    rollback_available: bool = Field(
        default=False, description="Whether rollback is available for this change"
    )
