"""
Service domain models for OmniMemory following ONEX standards.

This module provides models for service configurations, orchestration,
and coordination in the ONEX 4-node architecture.
"""

from __future__ import annotations

try:
    from omnibase_core.enums import EnumHealthStatus
except ImportError:
    # Fallback for development environments without omnibase_core
    from enum import Enum

    class EnumHealthStatus(str, Enum):  # type: ignore[no-redef]
        """Fallback health status levels (use omnibase_core.enums.EnumHealthStatus in production)."""

        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        UNKNOWN = "unknown"


from .model_service_config import ModelServiceConfig
from .model_service_health import ModelServiceHealth
from .model_service_registry import ModelServiceRegistry

__all__ = [
    "EnumHealthStatus",
    "ModelServiceConfig",
    "ModelServiceHealth",
    "ModelServiceRegistry",
]
