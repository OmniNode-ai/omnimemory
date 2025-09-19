"""
Protocol enum for ONEX standards.
"""

from enum import Enum


class EnumProtocol(str, Enum):
    """Network protocols for ONEX services."""

    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    WEBSOCKET_SECURE = "wss"
    GRPC = "grpc"
    GRPC_SECURE = "grpcs"
