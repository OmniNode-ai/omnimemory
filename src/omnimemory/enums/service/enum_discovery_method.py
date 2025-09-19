"""
Discovery method enum for ONEX standards.
"""

from enum import Enum


class EnumDiscoveryMethod(str, Enum):
    """Service discovery methods for ONEX architecture."""

    MANUAL = "manual"
    DNS = "dns"
    CONSUL = "consul"
    KUBERNETES = "kubernetes"
    EUREKA = "eureka"
    ETCD = "etcd"
    ZOOKEEPER = "zookeeper"
    SERVICE_MESH = "service_mesh"
    MULTICAST = "multicast"
