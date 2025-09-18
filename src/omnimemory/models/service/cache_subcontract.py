"""
ModelCachingSubcontract implementation for OmniMemory ONEX architecture.

This module provides standardized caching infrastructure that replaces external
Redis dependencies with centralized omnibase_core patterns.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, Field

from ..models.foundation.model_configuration import ModelCacheConfig
from ..utils.error_sanitizer import SanitizationLevel, sanitize_error

logger = structlog.get_logger(__name__)


class ModelCacheEntry(BaseModel):
    """Individual cache entry with metadata."""

    value: Any = Field(description="Cached value")
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(default=None)
    access_count: int = Field(default=0)
    last_accessed: datetime = Field(default_factory=datetime.now)
    size_bytes: int = Field(default=0, description="Approximate size in bytes")


class ModelCacheStats(BaseModel):
    """Cache performance statistics."""

    hits: int = Field(default=0)
    misses: int = Field(default=0)
    evictions: int = Field(default=0)
    current_size_mb: float = Field(default=0.0)
    max_size_mb: float = Field(default=100.0)
    entry_count: int = Field(default=0)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0


class ModelCachingSubcontract:
    """
    Centralized caching infrastructure replacing Redis dependencies.

    Features:
    - In-memory caching with configurable TTL
    - LRU eviction policy with size limits
    - Thread-safe operations with asyncio support
    - Health monitoring and performance metrics
    - ONEX standards compliance
    """

    def __init__(self, config: ModelCacheConfig):
        """Initialize caching subcontract with configuration."""
        self.config = config
        self._cache: Dict[str, ModelCacheEntry] = {}
        self._access_order: Dict[str, float] = {}  # key -> timestamp for LRU
        self._stats = ModelCacheStats(max_size_mb=config.max_size_mb)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                logger.debug("cache_miss", key=key)
                return None

            # Check expiration
            if entry.expires_at and datetime.now() > entry.expires_at:
                await self._remove_entry(key)
                self._stats.misses += 1
                logger.debug("cache_expired", key=key, expired_at=entry.expires_at)
                return None

            # Update access tracking
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._access_order[key] = time.time()
            self._stats.hits += 1

            logger.debug("cache_hit", key=key, access_count=entry.access_count)
            return entry.value

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (uses config default if None)

        Returns:
            True if successfully stored
        """
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl_seconds is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            elif self.config.ttl_seconds > 0:
                expires_at = datetime.now() + timedelta(seconds=self.config.ttl_seconds)

            # Estimate size (rough approximation)
            size_bytes = len(str(value).encode("utf-8"))

            # Check size limits before adding
            projected_size = self._stats.current_size_mb + (size_bytes / 1024 / 1024)
            if projected_size > self.config.max_size_mb:
                await self._evict_lru_entries()

            # Create entry
            entry = ModelCacheEntry(
                value=value, expires_at=expires_at, size_bytes=size_bytes
            )

            # Update existing or add new
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.current_size_mb -= old_entry.size_bytes / 1024 / 1024
            else:
                self._stats.entry_count += 1

            self._cache[key] = entry
            self._access_order[key] = time.time()
            self._stats.current_size_mb += size_bytes / 1024 / 1024

            logger.debug(
                "cache_set",
                key=key,
                size_bytes=size_bytes,
                expires_at=expires_at,
                total_entries=self._stats.entry_count,
            )
            return True

    async def delete(self, key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key existed and was removed
        """
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                logger.debug("cache_delete", key=key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = ModelCacheStats(max_size_mb=self.config.max_size_mb)
            logger.info("cache_cleared")

    async def ping(self) -> bool:
        """Health check for cache availability."""
        try:
            # Simple test operation
            test_key = "__health_check__"
            await self.set(test_key, "ok", ttl_seconds=1)
            result = await self.get(test_key)
            await self.delete(test_key)
            return result == "ok"
        except Exception as e:
            logger.error(
                "cache_health_check_failed",
                error=sanitize_error(
                    e, context="health_check", level=SanitizationLevel.STANDARD
                ),
            )
            return False

    def get_stats(self) -> ModelCacheStats:
        """Get current cache statistics."""
        return self._stats.model_copy()

    async def get_info(self) -> Dict[str, Any]:
        """Get cache information for monitoring."""
        stats = self.get_stats()
        return {
            "enabled": self.config.enabled,
            "max_size_mb": stats.max_size_mb,
            "current_size_mb": round(stats.current_size_mb, 2),
            "entry_count": stats.entry_count,
            "hit_rate_percent": round(stats.hit_rate, 2),
            "total_hits": stats.hits,
            "total_misses": stats.misses,
            "total_evictions": stats.evictions,
            "eviction_policy": self.config.eviction_policy,
            "default_ttl_seconds": self.config.ttl_seconds,
        }

    async def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats (must be called with lock)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._access_order.pop(key, None)
            self._stats.current_size_mb -= entry.size_bytes / 1024 / 1024
            self._stats.entry_count -= 1

    async def _evict_lru_entries(self) -> None:
        """Evict least recently used entries to free space."""
        if not self._access_order:
            return

        # Sort by access time (oldest first)
        sorted_keys = sorted(self._access_order.items(), key=lambda x: x[1])

        # Evict oldest entries until we're under the size limit
        target_size = self.config.max_size_mb * 0.8  # Leave 20% buffer

        for key, _ in sorted_keys:
            if self._stats.current_size_mb <= target_size:
                break

            await self._remove_entry(key)
            self._stats.evictions += 1
            logger.debug("cache_evicted_lru", key=key)

    async def _cleanup_expired(self) -> None:
        """Remove expired entries periodically."""
        if not self._cache:
            return

        now = datetime.now()
        expired_keys = []

        async with self._lock:
            for key, entry in self._cache.items():
                if entry.expires_at and now > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                await self._remove_entry(key)
                logger.debug("cache_expired_cleanup", key=key)

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""

        async def cleanup_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(60)  # Run cleanup every minute
                    await self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "cache_cleanup_error",
                        error=sanitize_error(
                            e, context="cleanup", level=SanitizationLevel.STANDARD
                        ),
                    )

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def close(self) -> None:
        """Cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.clear()


# Global cache instance (lazy initialization)
_global_cache: Optional[ModelCachingSubcontract] = None
_cache_lock = asyncio.Lock()


async def get_memory_cache(
    config: Optional[ModelCacheConfig] = None,
) -> ModelCachingSubcontract:
    """
    Get or create global memory cache instance.

    Args:
        config: Cache configuration (uses defaults if None)

    Returns:
        ModelCachingSubcontract instance
    """
    global _global_cache

    async with _cache_lock:
        if _global_cache is None:
            if config is None:
                config = ModelCacheConfig()
            _global_cache = ModelCachingSubcontract(config)

        return _global_cache


async def close_memory_cache() -> None:
    """Close global memory cache instance."""
    global _global_cache

    async with _cache_lock:
        if _global_cache is not None:
            await _global_cache.close()
            _global_cache = None
