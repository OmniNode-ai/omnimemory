"""
Service type enum for OmniMemory following ONEX standards.

Defines service types within the ONEX 4-node architecture.
"""

from enum import Enum


class EnumServiceType(str, Enum):
    """Service types for ONEX 4-node architecture."""

    # EFFECT node services
    STORAGE = "storage"
    PERSISTENCE = "persistence"
    EXTERNAL_API = "external_api"

    # COMPUTE node services
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    COMPUTATION = "computation"
    INTELLIGENCE = "intelligence"

    # REDUCER node services
    AGGREGATION = "aggregation"
    CONSOLIDATION = "consolidation"
    STREAM_PROCESSING = "stream_processing"

    # ORCHESTRATOR node services
    ORCHESTRATOR = "orchestrator"
    COORDINATOR = "coordinator"
    WORKFLOW = "workflow"

    # Cross-cutting services
    MONITORING = "monitoring"
    LOGGING = "logging"
    METRICS = "metrics"
    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"
    DISCOVERY = "discovery"
    LOAD_BALANCER = "load_balancer"
    GATEWAY = "gateway"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def default(cls) -> "EnumServiceType":
        """Return default service type."""
        return cls.PROCESSING

    @property
    def onex_node_type(self) -> str:
        """Get the ONEX node type this service belongs to."""
        node_mapping = {
            # EFFECT services
            self.STORAGE: "EFFECT",
            self.PERSISTENCE: "EFFECT",
            self.EXTERNAL_API: "EFFECT",
            # COMPUTE services
            self.PROCESSING: "COMPUTE",
            self.ANALYSIS: "COMPUTE",
            self.COMPUTATION: "COMPUTE",
            self.INTELLIGENCE: "COMPUTE",
            # REDUCER services
            self.AGGREGATION: "REDUCER",
            self.CONSOLIDATION: "REDUCER",
            self.STREAM_PROCESSING: "REDUCER",
            # ORCHESTRATOR services
            self.ORCHESTRATOR: "ORCHESTRATOR",
            self.COORDINATOR: "ORCHESTRATOR",
            self.WORKFLOW: "ORCHESTRATOR",
            # Cross-cutting services (can be in any node)
            self.MONITORING: "CROSS_CUTTING",
            self.LOGGING: "CROSS_CUTTING",
            self.METRICS: "CROSS_CUTTING",
            self.HEALTH_CHECK: "CROSS_CUTTING",
            self.CONFIGURATION: "CROSS_CUTTING",
            self.DISCOVERY: "CROSS_CUTTING",
            self.LOAD_BALANCER: "CROSS_CUTTING",
            self.GATEWAY: "CROSS_CUTTING",
        }
        return node_mapping[self]

    @property
    def is_stateful(self) -> bool:
        """Check if service type is typically stateful."""
        stateful_services = {
            self.STORAGE,
            self.PERSISTENCE,
            self.AGGREGATION,
            self.CONSOLIDATION,
            self.CONFIGURATION,
        }
        return self in stateful_services

    @property
    def requires_high_availability(self) -> bool:
        """Check if service type requires high availability."""
        ha_services = {
            self.ORCHESTRATOR,
            self.COORDINATOR,
            self.DISCOVERY,
            self.LOAD_BALANCER,
            self.GATEWAY,
            self.HEALTH_CHECK,
        }
        return self in ha_services
