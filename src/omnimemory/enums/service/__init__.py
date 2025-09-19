"""
Service domain enums for OmniMemory following ONEX standards.

This module provides service-specific enums for configurations,
orchestration, and coordination operations.
"""

from ..core.enum_node_type import EnumNodeType

# Import from foundation for service-related enums
from ..foundation.enum_health_status import EnumHealthStatus
from .enum_cache_eviction_policy import EnumCacheEvictionPolicy
from .enum_circuit_breaker_state import EnumCircuitBreakerState
from .enum_discovery_method import EnumDiscoveryMethod
from .enum_environment import EnumEnvironment

# Import service-specific enums
from .enum_protocol import EnumProtocol
from .enum_region import EnumRegion
from .enum_service_type import EnumServiceType
from .enum_trend_direction import EnumTrendDirection

__all__ = [
    # Re-exported from other domains
    "EnumHealthStatus",
    "EnumNodeType",
    # Service-specific enums
    "EnumProtocol",
    "EnumCircuitBreakerState",
    "EnumDiscoveryMethod",
    "EnumEnvironment",
    "EnumRegion",
    "EnumServiceType",
    "EnumTrendDirection",
    "EnumCacheEvictionPolicy",
]
