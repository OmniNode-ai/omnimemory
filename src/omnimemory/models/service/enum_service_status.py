"""
Enum for service status following ONEX standards.
"""

from enum import Enum


class EnumServiceStatus(str, Enum):
    """Status of services in the ONEX memory system."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"