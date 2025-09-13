"""
Container configuration model following ONEX standards.
"""

from pydantic import BaseModel, Field, SecretStr


class ModelContainerConfig(BaseModel):
    """Configuration for ONEX dependency injection container."""

    # Container identification
    container_id: str = Field(
        description="Unique identifier for the container",
    )
    container_name: str = Field(
        description="Human-readable name for the container",
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level for the container",
    )
    enable_performance_logging: bool = Field(
        default=True,
        description="Whether to enable performance logging",
    )

    # Database configuration
    database_url: str | None = Field(
        default=None,
        description="Database connection URL",
    )
    supabase_url: str | None = Field(
        default=None,
        description="Supabase URL for cloud database",
    )
    supabase_anon_key: SecretStr | None = Field(
        default=None,
        description="Supabase anonymous key - protected with SecretStr",
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )

    # Vector database configuration
    pinecone_api_key: SecretStr | None = Field(
        default=None,
        description="Pinecone API key for vector database - protected with SecretStr",
    )
    pinecone_environment: str | None = Field(
        default=None,
        description="Pinecone environment",
    )
    pinecone_index_name: str = Field(
        default="omnimemory-vectors",
        description="Pinecone index name",
    )

    # Service discovery configuration
    consul_host: str = Field(
        default="localhost",
        description="Consul host for service discovery",
    )
    consul_port: int = Field(
        default=8500,
        description="Consul port",
    )
    consul_datacenter: str = Field(
        default="dc1",
        description="Consul datacenter",
    )
    consul_timeout: int = Field(
        default=10,
        description="Consul timeout in seconds",
    )

    # Performance configuration
    max_concurrent_operations: int = Field(
        default=100,
        description="Maximum concurrent operations",
    )
    operation_timeout_ms: int = Field(
        default=30000,
        description="Operation timeout in milliseconds",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live in seconds",
    )

    # Memory configuration
    max_memory_size_mb: int = Field(
        default=1024,
        description="Maximum memory size in megabytes",
    )
    enable_compression: bool = Field(
        default=True,
        description="Whether to enable data compression",
    )
    enable_encryption: bool = Field(
        default=True,
        description="Whether to enable data encryption",
    )

    # Development settings
    development_mode: bool = Field(
        default=False,
        description="Whether running in development mode",
    )
    debug_enabled: bool = Field(
        default=False,
        description="Whether debug logging is enabled",
    )

    # Container behavior
    auto_wire_services: bool = Field(
        default=True,
        description="Whether to automatically wire service dependencies",
    )
    lazy_initialization: bool = Field(
        default=True,
        description="Whether to use lazy initialization for services",
    )
    enable_circular_dependency_detection: bool = Field(
        default=True,
        description="Whether to detect circular dependencies",
    )

    # Health monitoring
    enable_health_checks: bool = Field(
        default=True,
        description="Whether to enable health checking",
    )
    health_check_interval_ms: int = Field(
        default=30000,
        description="Health check interval in milliseconds",
    )

    # Custom configuration
    custom_settings: dict[str, str] = Field(
        default_factory=dict,
        description="Custom configuration settings",
    )