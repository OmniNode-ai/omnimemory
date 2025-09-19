"""
Encoding format enum for OmniMemory following ONEX standards.
"""

from enum import Enum


class EnumEncodingFormat(str, Enum):
    """Data encoding formats for memory operations."""

    JSON = "json"
    BINARY = "binary"
    COMPRESSED = "compressed"
    PICKLE = "pickle"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    MSGPACK = "msgpack"
    XML = "xml"
