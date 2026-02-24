# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Memory storage configuration model following ONEX standards.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ...enums.enum_memory_storage_type import EnumMemoryStorageType  # noqa: TC001

# Validation limits for storage configuration
MAX_CONNECTIONS_LIMIT = 1000
MIN_TIMEOUT_MS = 100
MAX_TIMEOUT_MS = 300000  # 5 minutes
MAX_BATCH_SIZE = 10000


class ModelMemoryStorageConfig(BaseModel):
    """Configuration for memory storage systems following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid")

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
    password_hash: SecretStr | None = Field(
        default=None,
        description="Hashed password for authentication - protected with SecretStr",
        exclude=True,  # Never serialize sensitive data
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for authentication - protected with SecretStr",
        exclude=True,  # Never serialize sensitive data
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

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int) -> int:
        """Validate max_connections is within reasonable bounds."""
        if v < 1:
            raise ValueError("max_connections must be at least 1")
        if v > MAX_CONNECTIONS_LIMIT:
            raise ValueError(f"max_connections cannot exceed {MAX_CONNECTIONS_LIMIT}")
        return v

    @field_validator("connection_timeout_ms", "idle_timeout_ms")
    @classmethod
    def validate_timeout_values(cls, v: int) -> int:
        """Validate timeout values are positive and reasonable."""
        if v < MIN_TIMEOUT_MS:
            raise ValueError(f"Timeout values must be at least {MIN_TIMEOUT_MS}ms")
        if v > MAX_TIMEOUT_MS:
            raise ValueError(
                f"Timeout values cannot exceed {MAX_TIMEOUT_MS:,}ms (5 minutes)"
            )
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is within reasonable bounds."""
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        if v > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size cannot exceed {MAX_BATCH_SIZE:,}")
        return v
