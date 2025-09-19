"""
Cache entry model for OmniMemory following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from .model_cache_value import ModelCacheValue


class ModelCacheEntry(BaseModel):
    """Individual cache entry with metadata."""

    value: ModelCacheValue = Field(description="Strongly typed cached value")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(default=None)
    access_count: int = Field(default=0)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: int = Field(default=0, description="Approximate size in bytes")
