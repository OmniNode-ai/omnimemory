"""
Service configuration model following ONEX standards.
"""

from pydantic import BaseModel, Field

from ...enums.enum_health_status import EnumHealthStatus, EnumNodeType


class ModelServiceConfig(BaseModel):
    """Configuration for ONEX memory services following standards."""

    # Service identification
    service_id: str = Field(
        description="Unique identifier for the service",
    )
    service_name: str = Field(
        description="Human-readable name for the service",
    )
    service_type: str = Field(
        description="Type of service (storage, retrieval, processing, etc.)",
    )

    # ONEX architecture information
    node_type: EnumNodeType = Field(
        description="ONEX node type for this service",
    )
    node_priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority of this service within its node type",
    )

    # Service configuration
    host: str = Field(
        description="Host address for the service",
    )
    port: int = Field(
        description="Port number for the service",
    )
    endpoint: str = Field(
        description="Service endpoint path",
    )

    # Resource configuration
    max_memory_mb: int = Field(
        default=1024,
        description="Maximum memory allocation in megabytes",
    )
    max_cpu_percent: int = Field(
        default=80,
        description="Maximum CPU usage percentage",
    )
    max_connections: int = Field(
        default=100,
        description="Maximum number of concurrent connections",
    )

    # Timeout configuration
    request_timeout_ms: int = Field(
        default=30000,
        description="Request timeout in milliseconds",
    )
    health_check_timeout_ms: int = Field(
        default=5000,
        description="Health check timeout in milliseconds",
    )
    shutdown_timeout_ms: int = Field(
        default=10000,
        description="Graceful shutdown timeout in milliseconds",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
    )
    retry_delay_ms: int = Field(
        default=1000,
        description="Delay between retry attempts in milliseconds",
    )
    exponential_backoff: bool = Field(
        default=True,
        description="Whether to use exponential backoff for retries",
    )

    # Monitoring configuration
    enable_metrics: bool = Field(
        default=True,
        description="Whether to enable metrics collection",
    )
    enable_logging: bool = Field(
        default=True,
        description="Whether to enable detailed logging",
    )
    enable_tracing: bool = Field(
        default=False,
        description="Whether to enable distributed tracing",
    )

    # Security configuration
    require_authentication: bool = Field(
        default=True,
        description="Whether authentication is required",
    )
    require_authorization: bool = Field(
        default=True,
        description="Whether authorization is required",
    )
    enable_tls: bool = Field(
        default=True,
        description="Whether to enable TLS encryption",
    )

    # Service dependencies
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of service dependencies",
    )
    optional_dependencies: list[str] = Field(
        default_factory=list,
        description="List of optional service dependencies",
    )

    # Environment configuration
    environment: str = Field(
        default="production",
        description="Environment (development, staging, production)",
    )
    region: str = Field(
        default="us-west-2",
        description="Deployment region",
    )

    # Feature flags
    feature_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags for the service",
    )

    # Additional configuration
    custom_config: dict[str, str] = Field(
        default_factory=dict,
        description="Custom configuration parameters",
    )
