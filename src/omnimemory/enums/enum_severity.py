"""
Severity level enumeration following ONEX standards.

Re-exports EnumSeverity from omnibase_core for convenient access.
"""

from __future__ import annotations

__all__ = ["EnumSeverity"]

# Import EnumSeverity directly from omnibase_core
try:
    from omnibase_core.enums import EnumSeverity
except ImportError:
    # Fallback for development environments without omnibase_core
    from enum import Enum

    class EnumSeverity(str, Enum):  # type: ignore[no-redef]
        """Fallback severity levels (use omnibase_core.enums.EnumSeverity in production)."""

        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
        FATAL = "fatal"
