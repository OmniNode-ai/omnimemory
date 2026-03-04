# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Service domain models for OmniMemory following ONEX standards.

and coordination in the ONEX 4-node architecture.
"""

from __future__ import annotations

# NOTE: omnibase_core >= 0.1.0 required for EnumHealthStatus
# The fallback below mirrors the production enum values for development environments
try:
    from omnibase_core.enums import EnumHealthStatus
except ImportError:
    # Fallback for development environments without omnibase_core installed
    from enum import Enum

    class EnumHealthStatus(str, Enum):  # type: ignore[no-redef]
        """Fallback health status levels for development.

        Production code MUST use omnibase_core.enums.EnumHealthStatus.
        This fallback is only for isolated testing/development.
        """

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
