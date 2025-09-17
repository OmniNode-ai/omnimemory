"""
Data type enumeration following ONEX standards.
"""

from enum import Enum


class EnumDataType(str, Enum):
    """Data types for memory data values."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    BYTES = "bytes"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    OTHER = "other"
