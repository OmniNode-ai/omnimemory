"""
ModelCachingSubcontract implementation for OmniMemory ONEX architecture.

This module provides standardized in-memory caching infrastructure following
centralized omnibase_core patterns for high-performance memory management.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Union

import structlog
from pydantic import BaseModel, Field, field_validator

from ...utils.error_sanitizer import SanitizationLevel, sanitize_error
from ..foundation.model_configuration import ModelCacheConfig

logger = structlog.get_logger(__name__)


class ModelCacheValue(BaseModel):
    """Strongly typed cache value with validation and serialization support."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    # Value type and content
    value_type: Literal[
        "string", "integer", "float", "boolean", "dict", "list"
    ] = Field(description="Type of the cached value for runtime validation")
    string_value: Optional[str] = Field(default=None, description="String value")
    integer_value: Optional[int] = Field(default=None, description="Integer value")
    float_value: Optional[float] = Field(default=None, description="Float value")
    boolean_value: Optional[bool] = Field(default=None, description="Boolean value")
    dict_value: Optional[Dict[str, Any]] = Field(
        default=None, description="Dictionary value"
    )
    list_value: Optional[List[Any]] = Field(default=None, description="List value")

    # Metadata
    is_sanitized: bool = Field(default=False, description="Whether value was sanitized")
    original_type: str = Field(description="Original Python type name")

    @field_validator("value_type")
    @classmethod
    def validate_value_type(cls, v: str) -> str:
        """Validate value type is supported."""
        valid_types = {"string", "integer", "float", "boolean", "dict", "list"}
        if v not in valid_types:
            raise ValueError(f"value_type must be one of: {valid_types}")
        return v

    @classmethod
    def from_raw_value(
        cls, value: Union[str, int, float, bool, Dict, List]
    ) -> "ModelCacheValue":
        """Create ModelCacheValue from raw Python value."""
        if isinstance(value, str):
            return cls(
                value_type="string",
                string_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, int):
            return cls(
                value_type="integer",
                integer_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, float):
            return cls(
                value_type="float",
                float_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, bool):
            return cls(
                value_type="boolean",
                boolean_value=value,
                original_type=type(value).__name__,
            )
        elif isinstance(value, dict):
            return cls(
                value_type="dict", dict_value=value, original_type=type(value).__name__
            )
        elif isinstance(value, list):
            return cls(
                value_type="list", list_value=value, original_type=type(value).__name__
            )
        else:
            raise ValueError(f"Unsupported cache value type: {type(value)}")

    def to_raw_value(self) -> Union[str, int, float, bool, Dict, List]:
        """Convert back to raw Python value."""
        if self.value_type == "string":
            return self.string_value
        elif self.value_type == "integer":
            return self.integer_value
        elif self.value_type == "float":
            return self.float_value
        elif self.value_type == "boolean":
            return self.boolean_value
        elif self.value_type == "dict":
            return self.dict_value
        elif self.value_type == "list":
            return self.list_value
        else:
            raise ValueError(f"Invalid value_type: {self.value_type}")

    def validate_value_consistency(self) -> None:
        """Validate that exactly one value field is set and matches value_type."""
        value_fields = [
            self.string_value,
            self.integer_value,
            self.float_value,
            self.boolean_value,
            self.dict_value,
            self.list_value,
        ]
        non_none_count = sum(1 for v in value_fields if v is not None)

        if non_none_count != 1:
            raise ValueError(
                f"Exactly one value field must be set, found {non_none_count}"
            )


class ModelCircuitBreakerState(BaseModel):
    """Circuit breaker state tracking."""

    is_open: bool = Field(default=False, description="Whether circuit is open")
    failure_count: int = Field(default=0, description="Current failure count")
    last_failure_time: Optional[datetime] = Field(
        default=None, description="Time of last failure"
    )
    next_retry_time: Optional[datetime] = Field(
        default=None, description="When to retry next"
    )


class ModelCacheEntry(BaseModel):
    """Individual cache entry with metadata."""

    value: ModelCacheValue = Field(description="Strongly typed cached value")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)
    access_count: int = Field(default=0)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: int = Field(default=0, description="Approximate size in bytes")


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


class ModelCachingSubcontract:
    """
    Centralized in-memory caching infrastructure for ONEX architecture.

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
        self._access_order: OrderedDict[
            str, float
        ] = OrderedDict()  # key -> timestamp for LRU
        self._stats = ModelCacheStats(max_size_mb=config.max_size_mb)
        self._circuit_breaker = ModelCircuitBreakerState()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_task_started = False

    async def _ensure_cleanup_task_started(self) -> None:
        """Safely start cleanup task if not already started."""
        # Use lock to prevent race condition in task initialization
        async with self._lock:
            if not self._cleanup_task_started:
                self._start_cleanup_task()
                self._cleanup_task_started = True

    async def get(self, key: str) -> Optional[Union[str, int, float, bool, Dict, List]]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        await self._ensure_cleanup_task_started()

        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                logger.debug("cache_miss", key=key)
                return None

            # Check expiration
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                await self._remove_entry(key)
                self._stats.misses += 1
                logger.debug("cache_expired", key=key, expired_at=entry.expires_at)
                return None

            # Update access tracking with bounds checking
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()

            # Move to end of OrderedDict for O(1) LRU tracking
            self._access_order.move_to_end(key)

            # Bounds checking: prevent _access_order from growing too large
            if len(self._access_order) > self.config.max_entries * 1.2:
                # Remove 20% oldest access entries to prevent memory bloat
                entries_to_remove = int(self.config.max_entries * 0.2)
                for _ in range(entries_to_remove):
                    if self._access_order:
                        # Remove oldest access record that's not in current cache
                        oldest_key = next(iter(self._access_order))
                        if oldest_key not in self._cache:
                            self._access_order.pop(oldest_key, None)
                        else:
                            break

            self._stats.hits += 1

            logger.debug("cache_hit", key=key, access_count=entry.access_count)
            return entry.value.to_raw_value()

    async def set(
        self,
        key: str,
        value: Union[str, int, float, bool, Dict, List],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Store value in cache with security and performance validation.

        Args:
            key: Cache key
            value: Value to cache (will be validated and converted to ModelCacheValue)
            ttl_seconds: Time to live in seconds (uses config default if None)

        Returns:
            True if successfully stored
        """
        await self._ensure_cleanup_task_started()

        try:
            # Type validation and conversion to strongly typed model
            try:
                typed_value = ModelCacheValue.from_raw_value(value)
                typed_value.validate_value_consistency()
            except (ValueError, TypeError) as e:
                logger.warning(
                    "cache_set_rejected_invalid_type",
                    key=key,
                    value_type=type(value).__name__,
                    error=str(e),
                )
                return False

            # Check circuit breaker
            if self._is_circuit_breaker_open():
                logger.warning("cache_set_rejected_circuit_breaker_open", key=key)
                return False

            async with self._lock:
                # Security: Check entry count limit to prevent key flooding
                if len(self._cache) >= self.config.max_entries:
                    logger.warning(
                        "cache_set_rejected_max_entries",
                        key=key,
                        current_entries=len(self._cache),
                        max_entries=self.config.max_entries,
                    )
                    await self._evict_lru_entries()
                    if len(self._cache) >= self.config.max_entries:
                        return False

                # Security: Sanitize value if enabled
                sanitized_raw_value = (
                    self._sanitize_value(value)
                    if self.config.sanitize_values
                    else value
                )

                # Create final typed value (re-validate after sanitization)
                try:
                    final_typed_value = ModelCacheValue.from_raw_value(
                        sanitized_raw_value
                    )
                    final_typed_value.is_sanitized = self.config.sanitize_values
                    final_typed_value.validate_value_consistency()
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "cache_set_rejected_sanitization_failed", key=key, error=str(e)
                    )
                    return False

                # Performance: Better size calculation using sys.getsizeof
                try:
                    size_bytes = sys.getsizeof(sanitized_raw_value)
                except (TypeError, OSError):
                    # Fallback to string encoding for complex objects
                    size_bytes = len(str(sanitized_raw_value).encode("utf-8"))

                # Security: Check individual entry size limit
                size_mb = size_bytes / 1024 / 1024
                if size_mb > self.config.max_entry_size_mb:
                    logger.warning(
                        "cache_set_rejected_entry_too_large",
                        key=key,
                        size_mb=round(size_mb, 2),
                        max_size_mb=self.config.max_entry_size_mb,
                    )
                    return False

                # Calculate expiration
                expires_at = None
                if ttl_seconds is not None:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
                elif self.config.ttl_seconds > 0:
                    expires_at = datetime.utcnow() + timedelta(
                        seconds=self.config.ttl_seconds
                    )

                # Check total size limits before adding
                projected_size = self._stats.current_size_mb + size_mb
                if projected_size > self.config.max_size_mb:
                    await self._evict_lru_entries()

                # Create entry with strongly typed value
                entry = ModelCacheEntry(
                    value=final_typed_value,
                    expires_at=expires_at,
                    size_bytes=size_bytes,
                )

                # Update existing or add new
                if key in self._cache:
                    old_entry = self._cache[key]
                    self._stats.current_size_mb -= old_entry.size_bytes / 1024 / 1024
                else:
                    self._stats.entry_count += 1

                self._cache[key] = entry
                # Add to end of OrderedDict for O(1) LRU tracking
                self._access_order[key] = time.time()

                # Bounds checking: prevent _access_order from growing too large
                if len(self._access_order) > self.config.max_entries * 1.2:
                    # Remove 20% oldest access entries to prevent memory bloat
                    entries_to_remove = int(self.config.max_entries * 0.2)
                    for _ in range(entries_to_remove):
                        if self._access_order:
                            # Remove oldest access record that's not in current cache
                            oldest_key = next(iter(self._access_order))
                            if oldest_key not in self._cache:
                                self._access_order.pop(oldest_key, None)
                            else:
                                break

                self._stats.current_size_mb += size_mb

                # Reset circuit breaker on success
                await self._record_success()

                logger.debug(
                    "cache_set_success",
                    key=key,
                    size_bytes=size_bytes,
                    expires_at=expires_at,
                    total_entries=self._stats.entry_count,
                    sanitized=self.config.sanitize_values,
                )
                return True

        except Exception as e:
            await self._record_failure()
            logger.error(
                "cache_set_error",
                key=key,
                error=sanitize_error(
                    e, context="cache_set", level=SanitizationLevel.STANDARD
                ),
            )
            return False

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

    async def get_info(self) -> ModelCacheInfo:
        """Get cache information for monitoring."""
        stats = self.get_stats()
        return ModelCacheInfo(
            enabled=self.config.enabled,
            max_size_mb=stats.max_size_mb,
            current_size_mb=round(stats.current_size_mb, 2),
            entry_count=stats.entry_count,
            hit_rate_percent=round(stats.hit_rate, 2),
            total_hits=stats.hits,
            total_misses=stats.misses,
            total_evictions=stats.evictions,
            eviction_policy=self.config.eviction_policy,
            default_ttl_seconds=self.config.ttl_seconds,
        )

    async def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats (must be called with lock)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._access_order.pop(key, None)
            self._stats.current_size_mb -= entry.size_bytes / 1024 / 1024
            self._stats.entry_count -= 1

    async def _evict_lru_entries(self) -> None:
        """Evict least recently used entries to free space - O(1) performance."""
        if not self._access_order:
            return

        # Evict oldest entries until we're under the size limit
        target_size = self.config.max_size_mb * 0.8  # Leave 20% buffer

        # Pop from beginning of OrderedDict (oldest entries) for O(1) eviction
        while self._stats.current_size_mb > target_size and self._access_order:
            # Get the first (oldest) key from OrderedDict
            key = next(iter(self._access_order))
            await self._remove_entry(key)
            self._stats.evictions += 1
            logger.debug("cache_evicted_lru", key=key)

    async def _cleanup_expired(self) -> None:
        """Remove expired entries periodically."""
        if not self._cache:
            return

        now = datetime.utcnow()
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

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.config.circuit_breaker_enabled:
            return False

        if not self._circuit_breaker.is_open:
            return False

        # Check if we should retry
        if (
            self._circuit_breaker.next_retry_time
            and datetime.utcnow() > self._circuit_breaker.next_retry_time
        ):
            self._circuit_breaker.is_open = False
            logger.info("circuit_breaker_half_open")
            return False

        return True

    async def _record_success(self) -> None:
        """Record successful operation for circuit breaker."""
        if self.config.circuit_breaker_enabled:
            self._circuit_breaker.failure_count = 0
            if self._circuit_breaker.is_open:
                self._circuit_breaker.is_open = False
                logger.info("circuit_breaker_closed_after_success")

    async def _record_failure(self) -> None:
        """Record failed operation for circuit breaker."""
        if not self.config.circuit_breaker_enabled:
            return

        self._circuit_breaker.failure_count += 1
        self._circuit_breaker.last_failure_time = datetime.utcnow()

        if (
            self._circuit_breaker.failure_count
            >= self.config.circuit_breaker_failure_threshold
        ):
            self._circuit_breaker.is_open = True
            # Retry after exponential backoff (max 5 minutes)
            backoff_seconds = min(300, 2**self._circuit_breaker.failure_count)
            self._circuit_breaker.next_retry_time = datetime.utcnow() + timedelta(
                seconds=backoff_seconds
            )
            self._stats.circuit_breaker_trips += 1
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self._circuit_breaker.failure_count,
                retry_in_seconds=backoff_seconds,
            )

    def _sanitize_value(
        self, value: Union[str, int, float, bool, Dict, List]
    ) -> Union[str, int, float, bool, Dict, List]:
        """Sanitize cache value for security."""
        if value is None:
            return value

        # Convert sensitive types to safe representations
        if isinstance(value, dict):
            sanitized = {}
            for k, v in value.items():
                # Skip potentially sensitive keys
                key_lower = str(k).lower()
                if any(
                    sensitive in key_lower
                    for sensitive in ["password", "secret", "key", "token", "auth"]
                ):
                    sanitized[k] = "[REDACTED]"
                else:
                    sanitized[k] = self._sanitize_value(v)
            return sanitized
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(item) for item in value]
        elif isinstance(value, str) and len(value) > 1000:
            # Truncate very long strings to prevent memory issues
            return value[:1000] + "[TRUNCATED]"
        else:
            return value

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
