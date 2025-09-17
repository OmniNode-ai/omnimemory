"""
Trust and decay function enumerations for ONEX compliance.

This module contains trust scoring and time decay enum types following ONEX standards.
"""

from enum import Enum


class EnumTrustLevel(str, Enum):
    """
    Trust level categories for memory and intelligence scoring.

    These levels represent the confidence and reliability of data or operations:
    - UNTRUSTED: Score below 0.2, data should not be used
    - LOW: Score 0.2-0.5, data may be unreliable
    - MEDIUM: Score 0.5-0.7, moderate confidence level
    - HIGH: Score 0.7-0.9, high confidence, good for most uses
    - TRUSTED: Score 0.9-0.95, very reliable data
    - VERIFIED: Score 0.95+, externally validated, highest confidence
    """

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRUSTED = "trusted"
    VERIFIED = "verified"


class EnumDecayFunction(str, Enum):
    """
    Time decay function types for trust score deterioration.

    Different mathematical functions for modeling how trust decays over time:
    - LINEAR: Constant rate of decay (score -= rate * time)
    - EXPONENTIAL: Exponential decay using half-life (most realistic)
    - LOGARITHMIC: Logarithmic decay (slower initial decay, then faster)
    - NONE: No time-based decay applied
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    NONE = "none"
