"""
Data type classification enumeration.
"""

from enum import Enum


class EnumDataType(Enum):
    """Data type classifications."""

    TEXT = "text"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATE = "date"
    JSON = "json"
    BINARY = "binary"
    VECTOR = "vector"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"
