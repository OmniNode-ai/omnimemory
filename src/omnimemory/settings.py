"""Environment-based settings loading for OmniMemory.

Uses pydantic-settings to automatically load configuration from environment
variables with the OMNIMEMORY__ prefix and __ nested delimiter.

Example environment variables:
    OMNIMEMORY__FILESYSTEM__BASE_PATH=/data/omnimemory
    OMNIMEMORY__FILESYSTEM__MAX_FILE_SIZE_BYTES=20971520
    OMNIMEMORY__POSTGRES__DSN=postgresql://user@localhost/db
    OMNIMEMORY__POSTGRES__PASSWORD=secret
    OMNIMEMORY__QDRANT__URL=http://localhost:6333
    OMNIMEMORY__POSTGRES_ENABLED=true
    OMNIMEMORY__QDRANT_ENABLED=true
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, HttpUrl, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnimemory.models.config import (
    ModelFilesystemConfig,
    ModelMemoryServiceConfig,
    ModelPostgresConfig,
    ModelQdrantConfig,
)


class FilesystemSettings(BaseSettings):
    """Filesystem config loaded from environment.

    Required for Phase 1. All memory operations use filesystem storage.

    Environment variables (prefix: OMNIMEMORY__FILESYSTEM__):
        BASE_PATH: Base directory for memory storage (required, must be absolute)
        MAX_FILE_SIZE_BYTES: Maximum file size (default: 10MB)
        ALLOWED_EXTENSIONS: JSON array of extensions (default: [".json",".txt",".md"])
        CREATE_IF_MISSING: Create directory if missing (default: true)
        ENABLE_COMPRESSION: Enable gzip compression (default: false)
        BUFFER_SIZE_BYTES: I/O buffer size (default: 64KB)
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIMEMORY__FILESYSTEM__",
        extra="forbid",
    )

    base_path: Path = Field(
        ...,
        description="Base directory for memory storage (must be absolute)",
    )
    max_file_size_bytes: int = Field(
        default=10_485_760,
        ge=1,
        le=1_073_741_824,
        description="Maximum file size in bytes (default 10MB, max 1GB)",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".json", ".txt", ".md"],
        description="Allowed file extensions for memory storage",
    )
    create_if_missing: bool = Field(
        default=True,
        description="Create base_path directory if it does not exist",
    )
    enable_compression: bool = Field(
        default=False,
        description="Enable gzip compression for stored files",
    )
    buffer_size_bytes: int = Field(
        default=65536,
        ge=4096,
        le=1_048_576,
        description="I/O buffer size in bytes (default 64KB)",
    )

    def to_config(self) -> ModelFilesystemConfig:
        """Convert settings to config model.

        Returns:
            ModelFilesystemConfig with validated configuration
        """
        return ModelFilesystemConfig(
            base_path=self.base_path,
            max_file_size_bytes=self.max_file_size_bytes,
            allowed_extensions=self.allowed_extensions,
            create_if_missing=self.create_if_missing,
            enable_compression=self.enable_compression,
            buffer_size_bytes=self.buffer_size_bytes,
        )


class PostgresSettings(BaseSettings):
    """Postgres config loaded from environment.

    Optional backend for persistent memory storage.

    Environment variables (prefix: OMNIMEMORY__POSTGRES__):
        DSN: PostgreSQL connection DSN (required when enabled)
        PASSWORD: Database password (required when enabled)
        POOL_SIZE: Connection pool size (default: 5)
        POOL_TIMEOUT_SECONDS: Pool acquisition timeout (default: 30)
        POOL_RECYCLE_SECONDS: Connection recycle time (default: 3600)
        STATEMENT_TIMEOUT_SECONDS: Max query execution time (default: 30)
        LOCK_TIMEOUT_SECONDS: Max time to wait for locks (default: 10)
        SSL_MODE: SSL mode (default: prefer)
        SCHEMA_NAME: PostgreSQL schema name (default: omnimemory)
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIMEMORY__POSTGRES__",
        extra="forbid",
    )

    dsn: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection DSN",
    )
    password: SecretStr = Field(
        ...,
        description="Database password (stored securely)",
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Connection pool size",
    )
    pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Pool connection acquisition timeout in seconds",
    )
    pool_recycle_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Connection recycle time in seconds",
    )
    statement_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum query execution time in seconds",
    )
    lock_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Maximum time to wait for locks in seconds",
    )
    ssl_mode: str = Field(
        default="prefer",
        description="SSL mode: disable, allow, prefer, require, "
        "verify-ca, verify-full",
    )
    schema_name: str = Field(
        default="omnimemory",
        description="PostgreSQL schema name for memory tables",
    )

    def to_config(self) -> ModelPostgresConfig:
        """Convert settings to config model.

        Returns:
            ModelPostgresConfig with validated configuration
        """
        return ModelPostgresConfig(
            dsn=self.dsn,
            password=self.password,
            pool_size=self.pool_size,
            pool_timeout_seconds=self.pool_timeout_seconds,
            pool_recycle_seconds=self.pool_recycle_seconds,
            statement_timeout_seconds=self.statement_timeout_seconds,
            lock_timeout_seconds=self.lock_timeout_seconds,
            ssl_mode=self.ssl_mode,
            schema_name=self.schema_name,
        )


class QdrantSettings(BaseSettings):
    """Qdrant config loaded from environment.

    Optional backend for vector memory storage.

    Environment variables (prefix: OMNIMEMORY__QDRANT__):
        URL: Qdrant server URL (default: http://localhost:6333)
        API_KEY: Qdrant API key (optional)
        COLLECTION_NAME: Default collection name (default: omnimemory)
        VECTOR_SIZE: Vector embedding dimensions (default: 1024)
        TIMEOUT_SECONDS: Request timeout (default: 30)
        GRPC_PORT: gRPC port for high-performance ops (optional)
        PREFER_GRPC: Prefer gRPC over HTTP (default: false)
        DEFAULT_LIMIT: Default number of results (default: 10)
        SCORE_THRESHOLD: Minimum similarity score (default: 0.7)
        DISTANCE_METRIC: Distance metric (default: Cosine)
        ON_DISK: Store vectors on disk (default: false)
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIMEMORY__QDRANT__",
        extra="forbid",
    )

    url: HttpUrl = Field(
        default="http://localhost:6333",  # type: ignore[assignment]
        description="Qdrant server URL",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key for authentication",
    )
    collection_name: str = Field(
        default="omnimemory",
        description="Default collection name for memory vectors",
    )
    vector_size: int = Field(
        default=1024,
        ge=1,
        le=65536,
        description="Vector embedding dimensions",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    grpc_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="gRPC port for high-performance operations",
    )
    prefer_grpc: bool = Field(
        default=False,
        description="Prefer gRPC over HTTP for operations",
    )
    default_limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Default number of results to return",
    )
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    distance_metric: str = Field(
        default="Cosine",
        description="Distance metric (Cosine, Euclid, Dot)",
    )
    on_disk: bool = Field(
        default=False,
        description="Store vectors on disk instead of RAM",
    )

    def to_config(self) -> ModelQdrantConfig:
        """Convert settings to config model.

        Returns:
            ModelQdrantConfig with validated configuration
        """
        return ModelQdrantConfig(
            url=self.url,
            api_key=self.api_key,
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            timeout_seconds=self.timeout_seconds,
            grpc_port=self.grpc_port,
            prefer_grpc=self.prefer_grpc,
            default_limit=self.default_limit,
            score_threshold=self.score_threshold,
            distance_metric=self.distance_metric,
            on_disk=self.on_disk,
        )


class EmbeddingSettings(BaseSettings):
    """Embedding server configuration loaded from environment.

    Required when using real embeddings (use_real_embeddings=True in handlers).

    Environment variables (prefix: OMNIMEMORY__EMBEDDING__):
        SERVER_URL: URL of the embedding server (REQUIRED - no default)
        TIMEOUT_SECONDS: Request timeout (default: 5.0)
        MAX_RETRIES: Maximum retry attempts (default: 3)
        DIMENSION: Expected embedding vector dimension (default: 1024)

    Example:
        export OMNIMEMORY__EMBEDDING__SERVER_URL=http://192.168.86.200:8102
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIMEMORY__EMBEDDING__",
        extra="forbid",
    )

    server_url: str = Field(
        ...,
        description="URL of the embedding server (REQUIRED - no default)",
    )
    timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient failures",
    )
    dimension: int = Field(
        default=1024,
        gt=0,
        description="Expected embedding vector dimension",
    )


class MemoryServiceSettings(BaseSettings):
    """Top-level settings for OmniMemory service.

    Loads configuration from environment variables with OMNIMEMORY__ prefix.

    Required for Phase 1:
        - OMNIMEMORY__FILESYSTEM__BASE_PATH (absolute path)

    Optional backend enablement:
        - OMNIMEMORY__POSTGRES_ENABLED=true (then set OMNIMEMORY__POSTGRES__* vars)
        - OMNIMEMORY__QDRANT_ENABLED=true (then set OMNIMEMORY__QDRANT__* vars)

    Embedding server (required when use_real_embeddings=True):
        - OMNIMEMORY__EMBEDDING__SERVER_URL (REQUIRED - no default)

    Service-level settings:
        - OMNIMEMORY__SERVICE_NAME (default: omnimemory)
        - OMNIMEMORY__ENABLE_METRICS (default: true)
        - OMNIMEMORY__ENABLE_LOGGING (default: true)
        - OMNIMEMORY__DEBUG_MODE (default: false)

    Example:
        # Set required env var
        export OMNIMEMORY__FILESYSTEM__BASE_PATH=/data/omnimemory

        # Load settings
        settings = MemoryServiceSettings()
        config = settings.to_config()
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIMEMORY__",
        extra="forbid",
    )

    # Backend enablement flags
    postgres_enabled: bool = Field(
        default=False,
        description="Enable PostgreSQL backend",
    )
    qdrant_enabled: bool = Field(
        default=False,
        description="Enable Qdrant backend",
    )
    embedding_enabled: bool = Field(
        default=False,
        description="Enable real embedding server (requires EMBEDDING__SERVER_URL)",
    )

    # Service-level settings
    service_name: str = Field(
        default="omnimemory",
        description="Name of the memory service instance",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics collection",
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable operation logging",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for verbose output",
    )

    def to_config(self) -> ModelMemoryServiceConfig:
        """Convert settings to config model.

        Loads filesystem settings (required), and optionally postgres
        and qdrant settings based on enablement flags.

        Returns:
            ModelMemoryServiceConfig ready for bootstrap

        Raises:
            pydantic.ValidationError: If required settings are missing
        """
        # Filesystem is always required
        # BaseSettings loads required fields from environment variables
        # pyright doesn't understand pydantic-settings env var loading
        filesystem_settings = FilesystemSettings()  # pyright: ignore[reportCallIssue]
        filesystem_config = filesystem_settings.to_config()

        # Load optional backends based on enablement flags
        postgres_config: ModelPostgresConfig | None = None
        qdrant_config: ModelQdrantConfig | None = None

        if self.postgres_enabled:
            # BaseSettings loads required fields from environment variables
            # pyright doesn't understand pydantic-settings env var loading
            postgres_settings = PostgresSettings()  # pyright: ignore[reportCallIssue]
            postgres_config = postgres_settings.to_config()

        if self.qdrant_enabled:
            qdrant_settings = QdrantSettings()
            qdrant_config = qdrant_settings.to_config()

        return ModelMemoryServiceConfig(
            filesystem=filesystem_config,
            postgres=postgres_config,
            qdrant=qdrant_config,
            service_name=self.service_name,
            enable_metrics=self.enable_metrics,
            enable_logging=self.enable_logging,
            debug_mode=self.debug_mode,
        )


def load_settings() -> MemoryServiceSettings:
    """Load settings from environment variables.

    This is the primary entry point for loading configuration.
    Fails fast with clear error messages if required settings are missing.

    Returns:
        MemoryServiceSettings with validated configuration

    Raises:
        pydantic.ValidationError: If required settings missing or invalid

    Example:
        >>> import os
        >>> os.environ["OMNIMEMORY__FILESYSTEM__BASE_PATH"] = "/tmp/omnimemory"
        >>> settings = load_settings()
        >>> config = settings.to_config()
        >>> config.filesystem.base_path
        PosixPath('/tmp/omnimemory')
    """
    return MemoryServiceSettings()
