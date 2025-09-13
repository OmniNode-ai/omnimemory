"""
Memory storage configuration model following ONEX standards.
"""

from pydantic import BaseModel, Field

from .enum_memory_storage_type import EnumMemoryStorageType


class ModelMemoryStorageConfig(BaseModel):
    """Configuration for memory storage systems following ONEX standards."""

    # Storage identification
    storage_id: str = Field(
        description="Unique identifier for the storage system",
    )
    storage_name: str = Field(
        description="Human-readable name for the storage system",
    )
    storage_type: EnumMemoryStorageType = Field(
        description="Type of storage system",
    )

    # Connection configuration
    connection_string: str = Field(
        description="Connection string for the storage system",
    )
    host: str = Field(
        description="Host address for the storage system",
    )
    port: int = Field(
        description="Port number for the storage system",
    )
    database_name: str = Field(
        description="Name of the database or collection",
    )

    # Authentication
    username: str | None = Field(
        default=None,
        description="Username for authentication",
    )
    password_hash: str | None = Field(
        default=None,
        description="Hashed password for authentication",
        exclude=True  # Prevent serialization for security
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
        exclude=True  # Prevent serialization for security
    )

    # Connection pool settings
    max_connections: int = Field(
        default=10,
        description="Maximum number of concurrent connections",
    )
    connection_timeout_ms: int = Field(
        default=5000,
        description="Connection timeout in milliseconds",
    )
    idle_timeout_ms: int = Field(
        default=30000,
        description="Idle connection timeout in milliseconds",
    )

    # Performance settings
    batch_size: int = Field(
        default=100,
        description="Default batch size for operations",
    )
    enable_compression: bool = Field(
        default=True,
        description="Whether to enable data compression",
    )
    enable_encryption: bool = Field(
        default=True,
        description="Whether to enable data encryption",
    )

    # Operational settings
    enable_metrics: bool = Field(
        default=True,
        description="Whether to collect performance metrics",
    )
    enable_logging: bool = Field(
        default=True,
        description="Whether to enable operation logging",
    )
    backup_enabled: bool = Field(
        default=False,
        description="Whether automatic backups are enabled",
    )