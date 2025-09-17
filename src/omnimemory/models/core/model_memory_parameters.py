"""
Memory operation parameters model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelMemoryParameters(BaseModel):
    """Structured parameters for memory operations following ONEX standards."""

    # Memory operation parameters (string values for type safety)
    memory_type: str | None = Field(
        default=None,
        description="Type of memory operation (temporal, persistent, vector, etc.)",
    )
    storage_backend: str | None = Field(
        default=None,
        description="Storage backend to use (redis, postgresql, pinecone)",
    )
    encoding_format: str | None = Field(
        default=None,
        description="Data encoding format (json, binary, compressed)",
    )
    retention_policy: str | None = Field(
        default=None,
        description="Memory retention policy (permanent, ttl, lru)",
    )
    compression_level: str | None = Field(
        default=None,
        description="Compression level for storage (none, low, medium, high)",
    )
    encryption_key: str | None = Field(
        default=None,
        description="Encryption key identifier for secure storage",
    )

    # Intelligence-specific parameters
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model to use for semantic processing",
    )
    similarity_threshold: str | None = Field(
        default=None,
        description="Similarity threshold for semantic matching (0.0-1.0 as string)",
    )
    max_results: str | None = Field(
        default=None,
        description="Maximum number of results to return (as string for consistency)",
    )

    # Migration-specific parameters
    batch_size: str | None = Field(
        default=None,
        description="Batch size for migration operations (as string)",
    )
    migration_strategy: str | None = Field(
        default=None,
        description="Migration strategy (incremental, bulk, intelligent)",
    )
