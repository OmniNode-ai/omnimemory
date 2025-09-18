"""
Environment enumeration for ONEX Foundation Architecture.

Defines standardized environment types for deployment and operation contexts.
"""

from enum import Enum


class EnumEnvironment(str, Enum):
    """Standardized environment types for ONEX operations."""

    # Development environments
    DEVELOPMENT = "development"
    DEV = "dev"
    LOCAL = "local"

    # Testing environments
    TESTING = "testing"
    TEST = "test"
    INTEGRATION = "integration"
    STAGING = "staging"

    # Production environments
    PRODUCTION = "production"
    PROD = "prod"

    # Specialized environments
    SANDBOX = "sandbox"
    DEMO = "demo"
    PREVIEW = "preview"
    CANARY = "canary"

    # CI/CD environments
    CI = "ci"
    CD = "cd"
    BUILD = "build"
