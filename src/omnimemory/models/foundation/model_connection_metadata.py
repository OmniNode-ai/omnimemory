"""
ONEX-compliant typed models for connection pool metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in connection pooling, ensuring type safety and validation.
"""

# Import and re-export models for backwards compatibility
from .model_connection_metadata_main import ModelConnectionMetadata
from .model_connection_pool_stats import ModelConnectionPoolStats
from .model_semaphore_metrics import ModelSemaphoreMetrics

__all__ = [
    "ModelConnectionMetadata",
    "ModelConnectionPoolStats",
    "ModelSemaphoreMetrics",
]
