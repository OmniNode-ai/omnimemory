"""
Cache statistics model for OmniMemory following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelCacheStats(BaseModel):
    """Cache performance statistics."""

    hits: int = Field(default=0)
    misses: int = Field(default=0)
    evictions: int = Field(default=0)
    current_size_mb: float = Field(default=0.0)
    max_size_mb: float = Field(default=100.0)
    entry_count: int = Field(default=0)
    circuit_breaker_trips: int = Field(
        default=0, description="Number of circuit breaker trips"
    )

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0
