"""
Severity level enumeration following ONEX standards.

This module contains severity level enum types for logging and error classification.
"""

from enum import Enum

__all__ = ["EnumSeverity"]


class EnumSeverity(str, Enum):
    """
    Severity levels for ONEX operations and logging.

    Standard severity levels used throughout OmniMemory for:
    - Logging classification
    - Error severity reporting
    - Alert prioritization
    - Audit event classification
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"
