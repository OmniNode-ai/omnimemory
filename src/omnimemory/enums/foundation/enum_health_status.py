"""
Health status enumeration.
"""

from enum import Enum


class EnumHealthStatus(Enum):
    """Health status enumeration."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    OFFLINE = "offline"
