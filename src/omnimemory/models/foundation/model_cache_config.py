"""
Cache configuration model for ONEX Foundation Architecture.

Provides strongly typed cache configuration for ModelCachingSubcontract.
"""

from pydantic import BaseModel, Field

from ...enums import EnumCacheEvictionPolicy


class ModelCacheConfig(BaseModel):
    """Configuration for the ModelCachingSubcontract caching system."""

    enabled: bool = Field(default=True, description="Whether caching is enabled")
    max_size_mb: float = Field(
        default=100.0, ge=1.0, le=1000.0, description="Maximum cache size in megabytes"
    )
    max_entries: int = Field(
        default=1000, ge=10, le=100000, description="Maximum number of cache entries"
    )
    max_entry_size_mb: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="Maximum size per entry in megabytes",
    )
    ttl_seconds: int = Field(
        default=3600,
        ge=0,
        le=86400,
        description="Default TTL for cache entries in seconds",
    )
    eviction_policy: EnumCacheEvictionPolicy = Field(
        default=EnumCacheEvictionPolicy.LRU,
        description="Cache eviction policy (currently only 'lru' supported)",
    )
    sanitize_values: bool = Field(
        default=True, description="Whether to sanitize cached values for security"
    )
    circuit_breaker_enabled: bool = Field(
        default=True, description="Whether to enable circuit breaker pattern"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, ge=1, le=20, description="Failures before opening circuit breaker"
    )
