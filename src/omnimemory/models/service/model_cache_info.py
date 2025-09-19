"""
Cache information model for OmniMemory following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelCacheInfo(BaseModel):
    """Strongly typed cache information for monitoring."""

    enabled: bool = Field(description="Whether caching is enabled")
    max_size_mb: float = Field(description="Maximum cache size in megabytes")
    current_size_mb: float = Field(description="Current cache size in megabytes")
    entry_count: int = Field(description="Number of entries in cache")
    hit_rate_percent: float = Field(description="Cache hit rate percentage")
    total_hits: int = Field(description="Total cache hits")
    total_misses: int = Field(description="Total cache misses")
    total_evictions: int = Field(description="Total cache evictions")
    eviction_policy: str = Field(description="Eviction policy in use")
    default_ttl_seconds: int = Field(description="Default TTL in seconds")
