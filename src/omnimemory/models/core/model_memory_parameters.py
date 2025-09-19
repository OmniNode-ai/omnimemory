"""
Memory operation parameters model following ONEX standards.
"""

from pydantic import BaseModel, Field

from ...enums import (
    EnumCompressionLevel,
    EnumEncodingFormat,
    EnumMemoryStorageType,
    EnumMigrationStrategy,
    EnumRetentionPolicy,
    EnumStorageBackend,
)


class ModelMemoryParameters(BaseModel):
    """Structured parameters for memory operations following ONEX standards."""

    # Memory operation parameters (using proper enums and types)
    memory_type: EnumMemoryStorageType | None = Field(
        default=None,
        description="Type of memory storage with strongly typed enum values",
    )
    storage_backend: EnumStorageBackend | None = Field(
        default=None,
        description="Storage backend to use with strongly typed enum values",
    )
    encoding_format: EnumEncodingFormat | None = Field(
        default=None,
        description="Data encoding format with strongly typed enum values",
    )
    retention_policy: EnumRetentionPolicy | None = Field(
        default=None,
        description="Memory retention policy with strongly typed enum values",
    )
    compression_level: EnumCompressionLevel | None = Field(
        default=None,
        description="Compression level for storage optimization with enum values",
    )
    encryption_key: str | None = Field(
        default=None,
        description="Encryption key identifier for secure storage",
    )

    # Intelligence-specific parameters (proper numeric types)
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model to use for semantic processing",
    )
    similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic matching (0.0-1.0)",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Maximum number of results to return",
    )

    # Migration-specific parameters (proper types)
    batch_size: int | None = Field(
        default=None,
        ge=1,
        le=100000,
        description="Batch size for migration operations",
    )
    migration_strategy: EnumMigrationStrategy | None = Field(
        default=None,
        description="Migration strategy with strongly typed enum values",
    )
