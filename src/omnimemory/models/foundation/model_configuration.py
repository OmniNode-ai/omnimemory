"""
Configuration model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelDatabaseConfig(BaseModel):
    """Database configuration settings."""

    host: str = Field(description="Database host address")
    port: int = Field(description="Database port number")
    database_name: str = Field(description="Database name")
    username: str = Field(description="Database username")
    max_connections: int = Field(
        default=10, description="Maximum number of connections"
    )
    connection_timeout_seconds: int = Field(
        default=30, description="Connection timeout in seconds"
    )
    enable_ssl: bool = Field(
        default=True, description="Whether to enable SSL connections"
    )


class ModelCacheConfig(BaseModel):
    """Cache configuration settings."""

    enabled: bool = Field(default=True, description="Whether caching is enabled")
    max_size_mb: int = Field(default=100, description="Maximum cache size in megabytes")
    ttl_seconds: int = Field(
        default=3600, description="Time to live for cached items in seconds"
    )
    eviction_policy: str = Field(
        default="LRU", description="Cache eviction policy (LRU, FIFO, etc.)"
    )


class ModelPerformanceConfig(BaseModel):
    """Performance configuration settings."""

    max_concurrent_operations: int = Field(
        default=100, description="Maximum concurrent operations"
    )
    operation_timeout_seconds: int = Field(
        default=30, description="Operation timeout in seconds"
    )
    rate_limit_per_minute: int = Field(
        default=1000, description="Rate limit per minute"
    )
    batch_size: int = Field(
        default=50, description="Default batch size for bulk operations"
    )


class ModelObservabilityConfig(BaseModel):
    """Observability configuration settings."""

    metrics_enabled: bool = Field(
        default=True, description="Whether metrics collection is enabled"
    )
    tracing_enabled: bool = Field(
        default=True, description="Whether distributed tracing is enabled"
    )
    logging_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARN, ERROR)"
    )
    metrics_export_interval_seconds: int = Field(
        default=60, description="Metrics export interval in seconds"
    )


class ModelSystemConfiguration(BaseModel):
    """Complete system configuration following ONEX standards."""

    database: ModelDatabaseConfig = Field(description="Database configuration")
    cache: ModelCacheConfig = Field(description="Cache configuration")
    performance: ModelPerformanceConfig = Field(description="Performance configuration")
    observability: ModelObservabilityConfig = Field(
        description="Observability configuration"
    )
    environment: str = Field(
        description="Deployment environment (development, staging, production)"
    )
    debug_mode: bool = Field(default=False, description="Whether debug mode is enabled")
