"""
Trust level and decay function enumerations.
"""

from enum import Enum


class EnumTrustLevel(Enum):
    """Trust level classifications."""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"
    SYSTEM = "system"


class EnumDecayFunction(Enum):
    """Time-based decay function types."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    STEP = "step"
    NONE = "none"
