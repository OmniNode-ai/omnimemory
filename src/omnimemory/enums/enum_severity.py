"""
Severity level enumeration following ONEX standards.
"""

from enum import Enum


class EnumSeverity(str, Enum):
    """Severity levels for errors, alerts, and notifications."""
    
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"