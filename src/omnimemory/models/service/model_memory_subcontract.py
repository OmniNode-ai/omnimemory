"""
Memory operations subcontract model following ONEX standards.
"""

from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums.memory import (
    EnumCompressionLevel,
    EnumEncodingFormat,
    EnumMemoryItemType,
    EnumMigrationStrategy,
    EnumRetentionPolicy,
    EnumStorageBackend,
)


class ModelMemorySubcontract(BaseModel):
    """Memory operations subcontract configuration for ONEX architecture."""

    # Memory operation identification
    operation_id: UUID = Field(
        description="Unique identifier for this memory operation",
    )
    operation_type: str = Field(
        description="Type of memory operation (store, retrieve, search, migrate)",
    )
    memory_type: EnumMemoryItemType = Field(
        description="Type of memory item being processed",
    )

    # Storage configuration
    storage_backend: EnumStorageBackend = Field(
        description="Primary storage backend for this operation",
    )
    fallback_backends: List[EnumStorageBackend] = Field(
        default_factory=list,
        description="Fallback storage backends if primary fails",
    )

    # Data processing configuration
    encoding_format: EnumEncodingFormat = Field(
        default=EnumEncodingFormat.JSON,
        description="Data encoding format for storage",
    )
    compression_level: EnumCompressionLevel = Field(
        default=EnumCompressionLevel.BALANCED,
        description="Compression level for storage optimization",
    )

    # Retention and lifecycle
    retention_policy: EnumRetentionPolicy = Field(
        default=EnumRetentionPolicy.TIME_BASED,
        description="Data retention policy to apply",
    )
    ttl_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Time to live for stored data in seconds",
    )
    max_versions: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Maximum number of versions to retain",
    )

    # Performance configuration
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Batch size for bulk operations",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=100,
        le=300000,
        description="Operation timeout in milliseconds",
    )
    max_parallel_operations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of parallel operations",
    )

    # Search and retrieval configuration
    similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for search operations",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Maximum number of results to return",
    )
    enable_fuzzy_search: bool = Field(
        default=False,
        description="Whether to enable fuzzy search capabilities",
    )

    # Migration configuration
    migration_strategy: EnumMigrationStrategy | None = Field(
        default=None,
        description="Strategy for data migration operations",
    )
    source_backend: EnumStorageBackend | None = Field(
        default=None,
        description="Source backend for migration operations",
    )
    target_backend: EnumStorageBackend | None = Field(
        default=None,
        description="Target backend for migration operations",
    )

    # Memory hierarchy configuration
    enable_caching: bool = Field(
        default=True,
        description="Whether to enable memory caching",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time to live in seconds",
    )
    cache_size_mb: int = Field(
        default=256,
        ge=16,
        le=8192,
        description="Maximum cache size in megabytes",
    )

    # Vector memory configuration
    vector_dimensions: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Vector dimensions for semantic operations",
    )
    vector_index_type: str | None = Field(
        default=None,
        description="Vector index type (cosine, euclidean, dot_product)",
    )
    enable_vector_quantization: bool = Field(
        default=False,
        description="Whether to enable vector quantization",
    )

    # Cross-modal memory configuration
    enable_cross_modal: bool = Field(
        default=False,
        description="Whether to enable cross-modal memory bridging",
    )
    modal_types: List[str] = Field(
        default_factory=list,
        description="Supported modal types (text, image, audio, video)",
    )
    modal_alignment_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for cross-modal alignment",
    )

    # Quality and validation
    enable_validation: bool = Field(
        default=True,
        description="Whether to enable data validation",
    )
    enable_sanitization: bool = Field(
        default=True,
        description="Whether to enable data sanitization",
    )
    enable_encryption: bool = Field(
        default=False,
        description="Whether to enable data encryption at rest",
    )

    # Monitoring and observability
    enable_metrics: bool = Field(
        default=True,
        description="Whether to enable operation metrics",
    )
    enable_tracing: bool = Field(
        default=False,
        description="Whether to enable distributed tracing",
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Whether to enable audit logging",
    )

    # Context and metadata
    operation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the memory operation",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the memory operation",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Operation priority (1=lowest, 10=highest)",
    )

    # Temporal configuration
    created_at: datetime | None = Field(
        default=None,
        description="When the memory operation was created",
    )
    scheduled_at: datetime | None = Field(
        default=None,
        description="When the operation is scheduled to run",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the operation expires",
    )

    # Error handling and recovery
    max_retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts on failure",
    )
    retry_backoff_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Backoff time between retries in milliseconds",
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Whether to enable circuit breaker pattern",
    )
