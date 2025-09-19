"""
Memory service subcontract model following ONEX standards.
"""

from datetime import datetime
from typing import Dict, List, Union
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums import (
    EnumDiscoveryMethod,
    EnumEnvironment,
    EnumHealthStatus,
    EnumNodeType,
    EnumProtocol,
)
from ...enums.memory import EnumMemoryStorageType, EnumStorageBackend
from ...enums.service import EnumRegion, EnumServiceType
from ...types import ConfigDict


class ModelServiceSubcontract(BaseModel):
    """Service subcontract configuration for ONEX memory services."""

    # Service identification
    service_id: UUID = Field(
        description="Unique identifier for the memory service",
    )
    service_name: str = Field(
        description="Human-readable name for the memory service",
    )
    service_type: EnumServiceType = Field(
        description="Type of memory service using ONEX 4-node architecture",
    )

    # ONEX architecture information
    node_type: EnumNodeType = Field(
        description="ONEX node type for this memory service",
    )
    node_priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority of this memory service within its node type",
    )

    # Service configuration
    host: str = Field(
        description="Host address for the memory service",
    )
    port: int = Field(
        description="Port number for the memory service",
    )
    endpoint: str = Field(
        description="Memory service endpoint path",
    )
    protocol: EnumProtocol = Field(
        default=EnumProtocol.HTTPS,
        description="Protocol used by the memory service",
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

    # Memory service dependencies
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of memory service dependencies",
    )
    optional_dependencies: List[str] = Field(
        default_factory=list,
        description="List of optional memory service dependencies",
    )

    # Environment configuration
    environment: EnumEnvironment = Field(
        default=EnumEnvironment.PRODUCTION,
        description="Environment (development, staging, production)",
    )
    region: EnumRegion = Field(
        default=EnumRegion.US_WEST_2,
        description="Deployment region using ONEX enum types",
    )

    # Feature flags
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags for the memory service",
    )

    # Service status and health
    status: EnumHealthStatus = Field(
        default=EnumHealthStatus.UNKNOWN,
        description="Current status of the memory service",
    )
    is_available: bool = Field(
        default=False,
        description="Whether the memory service is available for requests",
    )

    # Registration and heartbeat
    registered_at: datetime | None = Field(
        default=None,
        description="When the memory service was registered",
    )
    last_heartbeat: datetime | None = Field(
        default=None,
        description="Last heartbeat from the memory service",
    )
    heartbeat_interval_ms: int = Field(
        default=30000,
        description="Expected heartbeat interval in milliseconds",
    )

    # Load balancing
    weight: int = Field(
        default=1,
        description="Weight for load balancing (higher = more traffic)",
    )
    max_load: int = Field(
        default=100,
        description="Maximum load this memory service can handle",
    )

    # Memory service-specific configuration
    memory_type_supported: List[EnumMemoryStorageType] = Field(
        default_factory=list,
        description="Types of memory operations supported using ONEX enum types",
    )
    storage_backends: List[EnumStorageBackend] = Field(
        default_factory=list,
        description="Supported storage backends using ONEX enum types",
    )
    max_memory_capacity_mb: int | None = Field(
        default=None,
        description="Maximum memory capacity in megabytes",
    )

    # Discovery and lifecycle
    discovery_method: EnumDiscoveryMethod = Field(
        default=EnumDiscoveryMethod.MANUAL,
        description="How the memory service was discovered",
    )
    auto_deregister: bool = Field(
        default=True,
        description="Whether to auto-deregister if health checks fail",
    )
    ttl_seconds: int = Field(
        default=300,
        description="Time to live for the registration",
    )

    # Additional configuration
    custom_config: ConfigDict = Field(
        default_factory=dict,
        description="Custom memory service configuration parameters",
    )
