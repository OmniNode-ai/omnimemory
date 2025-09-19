"""
Environment enum for ONEX standards.
"""

from enum import Enum


class EnumEnvironment(str, Enum):
    """Deployment environments for ONEX services."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
    SANDBOX = "sandbox"
