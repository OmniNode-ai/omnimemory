"""
Resource type enumeration for ONEX Foundation Architecture.

Defines standardized resource types for audit logging and system operations.
"""

from enum import Enum


class EnumResourceType(str, Enum):
    """Standardized resource types for system operations and audit logging."""

    # Core memory resources
    MEMORY = "memory"
    CACHE = "cache"
    VECTOR = "vector"
    INTELLIGENCE = "intelligence"

    # Data storage resources
    DATABASE = "database"
    FILE = "file"
    CONTENT = "content"

    # System resources
    CONFIGURATION = "configuration"
    SERVICE = "service"
    API = "api"
    HEALTH = "health"
    AUDIT = "audit"

    # User-related resources
    USER = "user"
    SESSION = "session"
    AUTHENTICATION = "authentication"

    # Processing resources
    MIGRATION = "migration"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    SECURITY = "security"
