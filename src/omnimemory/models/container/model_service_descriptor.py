"""
Service descriptor model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelServiceDescriptor(BaseModel):
    """Descriptor for a service registration following ONEX standards."""

    # Service identification
    service_id: str = Field(
        description="Unique identifier for the service",
    )
    service_name: str = Field(
        description="Human-readable name for the service",
    )
    protocol_type: str = Field(
        description="Protocol interface implemented by the service",
    )
    service_class: str = Field(
        description="Implementation class name",
    )

    # Service lifecycle
    singleton: bool = Field(
        default=True,
        description="Whether to create a singleton instance",
    )
    lazy_initialization: bool = Field(
        default=True,
        description="Whether to use lazy initialization",
    )
    auto_dispose: bool = Field(
        default=True,
        description="Whether to automatically dispose resources",
    )

    # Dependencies
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of service dependencies",
    )
    optional_dependencies: list[str] = Field(
        default_factory=list,
        description="List of optional dependencies",
    )

    # Configuration
    configuration: dict[str, str] = Field(
        default_factory=dict,
        description="Service-specific configuration",
    )
    initialization_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Parameters for service initialization",
    )

    # Registration metadata
    registered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the service was registered",
    )
    registered_by: str = Field(
        description="Who or what registered the service",
    )

    # Usage tracking
    initialization_count: int = Field(
        default=0,
        description="Number of times service has been initialized",
    )
    last_access_time: datetime | None = Field(
        default=None,
        description="When the service was last accessed",
    )
    access_count: int = Field(
        default=0,
        description="Number of times service has been accessed",
    )

    # Quality and validation
    validated: bool = Field(
        default=False,
        description="Whether the service descriptor has been validated",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="List of validation errors if any",
    )

    # Performance tracking
    average_initialization_time_ms: float = Field(
        default=0.0,
        description="Average initialization time in milliseconds",
    )
    last_initialization_time_ms: float = Field(
        default=0.0,
        description="Last initialization time in milliseconds",
    )

    # Health and status
    is_healthy: bool = Field(
        default=True,
        description="Whether the service is healthy",
    )
    last_health_check: datetime | None = Field(
        default=None,
        description="When the service was last health checked",
    )
    health_status: str = Field(
        default="unknown",
        description="Current health status",
    )

    # Tagging and categorization
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the service",
    )
    category: str | None = Field(
        default=None,
        description="Service category",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Service priority for initialization order",
    )