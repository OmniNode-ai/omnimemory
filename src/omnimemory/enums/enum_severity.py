"""
Severity level enumeration following ONEX standards.

Standard severity levels imported from omnibase_core for unified logging.
"""

# Import standard ONEX severity levels from omnibase_core
from omnibase_core.enums.enum_log_level import EnumLogLevel as EnumSeverity

# Make available for import
__all__ = ["EnumSeverity"]
