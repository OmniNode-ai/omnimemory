"""
Operation context enumeration for ONEX Foundation Architecture.

Defines standardized operation contexts for error tracking and debugging.
"""

from enum import Enum


class EnumOperationContext(str, Enum):
    """Standardized operation contexts for error tracking and debugging."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"

    # Memory operations
    STORE = "store"
    RETRIEVE = "retrieve"
    SEARCH = "search"
    INDEX = "index"
    CACHE = "cache"

    # Processing operations
    PROCESS = "process"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    SERIALIZE = "serialize"
    DESERIALIZE = "deserialize"

    # Network operations
    REQUEST = "request"
    RESPONSE = "response"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    SEND = "send"
    RECEIVE = "receive"

    # Authentication operations
    LOGIN = "login"
    LOGOUT = "logout"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    REFRESH_TOKEN = "refresh_token"

    # Configuration operations
    LOAD_CONFIG = "load_config"
    UPDATE_CONFIG = "update_config"
    VALIDATE_CONFIG = "validate_config"

    # Health and monitoring operations
    HEALTH_CHECK = "health_check"
    METRICS_COLLECTION = "metrics_collection"
    PERFORMANCE_MONITORING = "performance_monitoring"

    # Migration operations
    MIGRATE = "migrate"
    ROLLBACK = "rollback"
    BACKUP = "backup"
    RESTORE = "restore"

    # Cleanup operations
    CLEANUP = "cleanup"
    GARBAGE_COLLECT = "garbage_collect"
    EVICT = "evict"
    PURGE = "purge"

    # Initialization operations
    INITIALIZE = "initialize"
    SETUP = "setup"
    CONFIGURE = "configure"
    BOOTSTRAP = "bootstrap"

    # Shutdown operations
    SHUTDOWN = "shutdown"
    CLEANUP_RESOURCES = "cleanup_resources"
    FINALIZE = "finalize"
