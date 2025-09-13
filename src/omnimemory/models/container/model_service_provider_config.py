"""
Service provider configuration model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelServiceProviderConfig(BaseModel):
    """Configuration for service provider following ONEX standards."""

    # Provider identification
    provider_id: str = Field(
        description="Unique identifier for the service provider",
    )
    provider_name: str = Field(
        description="Human-readable name for the service provider",
    )

    # Provider behavior
    enable_caching: bool = Field(
        default=True,
        description="Whether to cache service instances",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live in seconds",
    )
    max_cached_instances: int = Field(
        default=100,
        description="Maximum number of cached instances",
    )

    # Performance settings
    max_concurrent_resolutions: int = Field(
        default=50,
        description="Maximum concurrent service resolutions",
    )
    resolution_timeout_ms: int = Field(
        default=10000,
        description="Service resolution timeout in milliseconds",
    )
    enable_performance_tracking: bool = Field(
        default=True,
        description="Whether to track performance metrics",
    )

    # Error handling
    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for service resolution",
    )
    retry_delay_ms: int = Field(
        default=1000,
        description="Delay between retry attempts in milliseconds",
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Whether to enable circuit breaker pattern",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        description="Circuit breaker failure threshold",
    )

    # Logging and monitoring
    enable_detailed_logging: bool = Field(
        default=False,
        description="Whether to enable detailed operation logging",
    )
    log_successful_resolutions: bool = Field(
        default=False,
        description="Whether to log successful resolutions",
    )
    log_failed_resolutions: bool = Field(
        default=True,
        description="Whether to log failed resolutions",
    )

    # Health monitoring
    enable_health_monitoring: bool = Field(
        default=True,
        description="Whether to monitor service health",
    )
    health_check_interval_ms: int = Field(
        default=30000,
        description="Health check interval in milliseconds",
    )
    auto_remove_unhealthy_services: bool = Field(
        default=False,
        description="Whether to auto-remove unhealthy services",
    )

    # Service lifecycle
    auto_dispose_services: bool = Field(
        default=True,
        description="Whether to auto-dispose unused services",
    )
    disposal_delay_ms: int = Field(
        default=60000,
        description="Delay before disposing unused services",
    )
    track_service_usage: bool = Field(
        default=True,
        description="Whether to track service usage statistics",
    )

    # Dependency management
    resolve_dependencies_async: bool = Field(
        default=True,
        description="Whether to resolve dependencies asynchronously",
    )
    validate_dependencies: bool = Field(
        default=True,
        description="Whether to validate service dependencies",
    )
    enable_circular_dependency_detection: bool = Field(
        default=True,
        description="Whether to detect circular dependencies",
    )

    # Provider metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the provider configuration was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When the provider configuration was last updated",
    )

    # Statistics collection
    collect_resolution_statistics: bool = Field(
        default=True,
        description="Whether to collect resolution statistics",
    )
    statistics_retention_hours: int = Field(
        default=24,
        description="How long to retain statistics in hours",
    )

    # Custom settings
    custom_options: dict[str, str] = Field(
        default_factory=dict,
        description="Custom provider options",
    )