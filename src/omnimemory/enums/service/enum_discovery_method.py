"""
Discovery method enum for OmniMemory following ONEX standards.

Defines service discovery methods supported by ONEX services.
"""

from enum import Enum


class EnumDiscoveryMethod(str, Enum):
    """Service discovery methods for ONEX service registry."""

    CONSUL = "consul"
    DNS = "dns"
    STATIC = "static"
    KUBERNETES = "kubernetes"
    EUREKA = "eureka"
    ETCD = "etcd"
    ZOOKEEPER = "zookeeper"
    MANUAL = "manual"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def default(cls) -> "EnumDiscoveryMethod":
        """Return default discovery method."""
        return cls.CONSUL

    @property
    def is_dynamic(self) -> bool:
        """Check if discovery method supports dynamic registration."""
        return self in {
            self.CONSUL,
            self.KUBERNETES,
            self.EUREKA,
            self.ETCD,
            self.ZOOKEEPER,
        }

    @property
    def supports_health_checks(self) -> bool:
        """Check if discovery method supports health checks."""
        return self in {
            self.CONSUL,
            self.KUBERNETES,
            self.EUREKA,
        }

    @property
    def requires_agent(self) -> bool:
        """Check if discovery method requires agent installation."""
        return self in {
            self.CONSUL,
            self.EUREKA,
        }
