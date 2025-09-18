"""
Protocol enum for OmniMemory following ONEX standards.

Defines communication protocols supported by ONEX services.
"""

from enum import Enum


class EnumProtocol(str, Enum):
    """Protocol types for service communication."""

    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    WEBSOCKET_SECURE = "wss"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def default(cls) -> "EnumProtocol":
        """Return default protocol."""
        return cls.HTTPS

    @property
    def is_secure(self) -> bool:
        """Check if protocol is secure."""
        return self in {self.HTTPS, self.WEBSOCKET_SECURE}

    @property
    def default_port(self) -> int:
        """Get default port for protocol."""
        port_mapping = {
            self.HTTP: 80,
            self.HTTPS: 443,
            self.GRPC: 443,
            self.TCP: 8080,
            self.UDP: 8080,
            self.WEBSOCKET: 80,
            self.WEBSOCKET_SECURE: 443,
        }
        return port_mapping[self]
