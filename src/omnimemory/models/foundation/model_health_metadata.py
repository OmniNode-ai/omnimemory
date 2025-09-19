"""
ONEX-compliant typed models for health check metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in health management, ensuring type safety and validation.

NOTICE: This module has been refactored to follow one-model-per-file standards.
Individual models are now in separate files but re-exported here for backwards compatibility.
"""

from .model_aggregate_health_metadata import ModelAggregateHealthMetadata
from .model_configuration_change_metadata import ModelConfigurationChangeMetadata

# Re-export individual models for backwards compatibility
from .model_health_check_metadata import ModelHealthCheckMetadata

__all__ = [
    "ModelHealthCheckMetadata",
    "ModelAggregateHealthMetadata",
    "ModelConfigurationChangeMetadata",
]
