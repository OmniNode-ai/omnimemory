"""
Trust and decay function enumerations for ONEX compliance.

This module contains trust scoring and time decay enum types following ONEX standards.
"""

from enum import Enum


class EnumTrustLevel(str, Enum):
    """Trust level categories."""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRUSTED = "trusted"
    VERIFIED = "verified"


class EnumDecayFunction(str, Enum):
    """Time decay function types."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    NONE = "none"