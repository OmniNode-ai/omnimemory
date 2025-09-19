"""
Memory service registry model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from ...enums import EnumHealthStatus, EnumNodeType


class ModelMemoryServiceRegistry(BaseModel):
    """Memory service registry entry following ONEX standards."""

    # Service identification
    service_id: str = Field(
        description="Unique identifier for the memory service",
    )
    service_name: str = Field(
        description="Human-readable name for the memory service",
    )
    service_version: str = Field(
        description="Version of the memory service",
    )

    # Service location
    host: str = Field(
        description="Host address for the memory service",
    )
    port: int = Field(
        description="Port number for the memory service",
    )
    endpoint: str = Field(
        description="Memory service endpoint path",
    )
    protocol: str = Field(
        default="https",
        description="Protocol used by the memory service",
    )

    # ONEX architecture information
    node_type: EnumNodeType = Field(
        description="ONEX node type for this memory service",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Memory capabilities provided by this service",
    )

    # Service status
    status: EnumHealthStatus = Field(
        description="Current status of the memory service",
    )
    is_available: bool = Field(
        description="Whether the memory service is available for requests",
    )

    # Registration information
    registered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory service was registered",
    )
    last_heartbeat: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last heartbeat from the memory service",
    )
    heartbeat_interval_ms: int = Field(
        default=30000,
        description="Expected heartbeat interval in milliseconds",
    )

    # Service metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the memory service",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the memory service",
    )

    # Load balancing information
    weight: int = Field(
        default=1,
        description="Weight for load balancing (higher = more traffic)",
    )
    max_load: int = Field(
        default=100,
        description="Maximum load this memory service can handle",
    )
    current_load: int = Field(
        default=0,
        description="Current load on the memory service",
    )

    # Health information
    health_check_url: str = Field(
        description="URL for memory service health checks",
    )
    last_health_check: datetime | None = Field(
        default=None,
        description="When the last health check was performed",
    )
    health_status: str = Field(
        default="unknown",
        description="Result of the last health check",
    )

    # Circuit breaker information
    circuit_breaker_state: str = Field(
        default="closed",
        description="State of the circuit breaker (closed, open, half-open)",
    )
    failure_count: int = Field(
        default=0,
        description="Number of consecutive failures",
    )
    failure_threshold: int = Field(
        default=5,
        description="Failure threshold for circuit breaker",
    )

    # Memory service-specific information
    memory_type_supported: list[str] = Field(
        default_factory=list,
        description="Types of memory operations supported (temporal, persistent, vector)",
    )
    storage_backends: list[str] = Field(
        default_factory=list,
        description="Supported storage backends (memory_cache, postgresql, pinecone)",
    )
    max_memory_capacity_mb: int = Field(
        description="Maximum memory capacity in megabytes",
    )

    # Discovery information
    discovery_method: str = Field(
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
