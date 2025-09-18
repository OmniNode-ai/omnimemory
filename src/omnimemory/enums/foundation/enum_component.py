"""
Component enumeration for ONEX Foundation Architecture.

Defines standardized system components for error tracking and operation context.
"""

from enum import Enum


class EnumComponent(str, Enum):
    """Standardized system components for error tracking and operations."""

    # Core memory components
    MEMORY_MANAGER = "memory_manager"
    MEMORY_STORE = "memory_store"
    MEMORY_CACHE = "memory_cache"
    MEMORY_INDEX = "memory_index"

    # Database components
    DATABASE = "database"
    DATABASE_CONNECTION = "database_connection"
    DATABASE_MIGRATION = "database_migration"
    DATABASE_POOL = "database_pool"

    # Cache components
    CACHE = "cache"
    CACHE_SUBCONTRACT = "cache_subcontract"
    CACHE_MANAGER = "cache_manager"
    CACHE_EVICTION = "cache_eviction"

    # Intelligence components
    INTELLIGENCE_ANALYZER = "intelligence_analyzer"
    PATTERN_RECOGNIZER = "pattern_recognizer"
    SEMANTIC_ANALYZER = "semantic_analyzer"
    KNOWLEDGE_EXTRACTOR = "knowledge_extractor"

    # API components
    API_GATEWAY = "api_gateway"
    API_ROUTER = "api_router"
    API_MIDDLEWARE = "api_middleware"
    API_VALIDATOR = "api_validator"

    # Service components
    SERVICE_REGISTRY = "service_registry"
    SERVICE_DISCOVERY = "service_discovery"
    SERVICE_HEALTH = "service_health"
    SERVICE_CONFIG = "service_config"

    # Security components
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    AUDIT_LOGGER = "audit_logger"

    # Monitoring components
    METRICS_COLLECTOR = "metrics_collector"
    HEALTH_MONITOR = "health_monitor"
    PERFORMANCE_TRACKER = "performance_tracker"
    ERROR_TRACKER = "error_tracker"

    # Infrastructure components
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_HANDLER = "retry_handler"
    RATE_LIMITER = "rate_limiter"
    LOAD_BALANCER = "load_balancer"

    # External integrations
    EXTERNAL_API = "external_api"
    THIRD_PARTY_SERVICE = "third_party_service"
    WEBHOOK_HANDLER = "webhook_handler"

    # Utility components
    CONFIG_MANAGER = "config_manager"
    ERROR_SANITIZER = "error_sanitizer"
    VALIDATOR = "validator"
    SERIALIZER = "serializer"
