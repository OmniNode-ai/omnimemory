"""
Environment enum for OmniMemory following ONEX standards.

Defines deployment environments for ONEX services.
"""

from enum import Enum


class EnumEnvironment(str, Enum):
    """Deployment environments for ONEX services."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def default(cls) -> "EnumEnvironment":
        """Return default environment."""
        return cls.PRODUCTION

    @property
    def is_production_like(self) -> bool:
        """Check if environment is production-like."""
        return self in {self.STAGING, self.PRODUCTION}

    @property
    def allows_debug(self) -> bool:
        """Check if environment allows debug features."""
        return self in {self.DEVELOPMENT, self.TESTING, self.LOCAL}

    @property
    def requires_security(self) -> bool:
        """Check if environment requires full security measures."""
        return self in {self.STAGING, self.PRODUCTION}

    @property
    def log_level(self) -> str:
        """Get recommended log level for environment."""
        level_mapping = {
            self.DEVELOPMENT: "DEBUG",
            self.TESTING: "DEBUG",
            self.LOCAL: "DEBUG",
            self.STAGING: "INFO",
            self.PRODUCTION: "WARNING",
        }
        return level_mapping[self]
