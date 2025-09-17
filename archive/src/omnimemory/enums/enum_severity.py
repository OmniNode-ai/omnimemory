"""
Severity level enumeration following ONEX standards.

Uses the standard severity levels from omnibase_core.
This file maintained for backward compatibility during migration.
"""

# Import standard ONEX severity levels from omnibase_core
try:
    from omnibase_core.enums.enum_log_level import EnumLogLevel as EnumSeverity
except ImportError:
    # Fallback for development environments without omnibase_core
    from enum import Enum

    class EnumSeverity(str, Enum):
        """Fallback severity levels (use omnibase_core.enums.EnumLogLevel in production)."""

        CRITICAL = "critical"
        ERROR = "error"
        WARNING = "warning"
        INFO = "info"
        DEBUG = "debug"
