"""
Unit tests for ModelCachingSubcontract security and performance features.

Tests critical functionality identified in PR review:
- Import path fixes
- Security validation for cache values
- Max entry count protection
- Circuit breaker patterns
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from omnimemory.models.foundation.model_configuration import ModelCacheConfig
from omnimemory.models.service.cache_subcontract import (
    ModelCacheEntry,
    ModelCacheStats,
    ModelCachingSubcontract,
    close_memory_cache,
    get_memory_cache,
)


class TestModelCacheConfig:
    """Test ModelCacheConfig with security enhancements."""

    def test_default_config_security_settings(self):
        """Test that default config includes security settings."""
        config = ModelCacheConfig()

        assert config.enabled is True
        assert config.max_entries == 10000  # Key flooding protection
        assert config.max_entry_size_mb == 10.0  # Memory exhaustion protection
        assert config.sanitize_values is True  # Security sanitization
        assert config.circuit_breaker_enabled is True  # Circuit breaker pattern


class TestModelCachingSubcontract:
    """Test ModelCachingSubcontract security and performance features."""

    @pytest.fixture
    async def cache_config(self):
        """Create test cache configuration."""
        return ModelCacheConfig(
            max_size_mb=10,
            max_entries=5,  # Small for testing
            max_entry_size_mb=1.0,
            ttl_seconds=60,
            sanitize_values=True,
            circuit_breaker_enabled=True,
        )

    @pytest.fixture
    async def cache(self, cache_config):
        """Create test cache instance."""
        cache_instance = ModelCachingSubcontract(cache_config)
        yield cache_instance
        await cache_instance.close()

    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_config):
        """Test cache initializes with security features."""
        cache = ModelCachingSubcontract(cache_config)

        assert cache.config.sanitize_values is True
        assert cache.config.circuit_breaker_enabled is True
        assert cache._circuit_breaker is not None

        await cache.close()

    @pytest.mark.asyncio
    async def test_max_entries_protection(self, cache):
        """Test max entries limit prevents key flooding."""
        # Fill cache to max entries
        for i in range(5):
            result = await cache.set(f"key_{i}", f"value_{i}")
            assert result is True

        # Next entry should trigger eviction, not rejection
        result = await cache.set("overflow_key", "overflow_value")
        assert result is True  # Should succeed after eviction

        # Verify total entries doesn't exceed limit
        stats = cache.get_stats()
        assert stats.entry_count <= cache.config.max_entries

    @pytest.mark.asyncio
    async def test_max_entry_size_protection(self, cache):
        """Test max entry size prevents memory exhaustion."""
        # Create a large value that exceeds max_entry_size_mb (1.0 MB)
        large_value = "x" * (1024 * 1024 + 1)  # 1MB + 1 byte

        result = await cache.set("large_key", large_value)
        assert result is False  # Should be rejected

        # Verify entry wasn't stored
        retrieved = await cache.get("large_key")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_value_sanitization(self, cache):
        """Test sensitive values are sanitized."""
        sensitive_data = {
            "username": "admin",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "normal_field": "normal_value",
        }

        result = await cache.set("sensitive_key", sensitive_data)
        assert result is True

        retrieved = await cache.get("sensitive_key")
        assert retrieved is not None
        assert retrieved["normal_field"] == "normal_value"
        assert retrieved["password"] == "[REDACTED]"
        assert retrieved["api_key"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, cache):
        """Test circuit breaker opens after failures."""
        # Force some failures by using invalid operations
        # This is a basic test - in real scenarios, failures would come from storage issues
        original_set = cache.set

        async def failing_set(*args, **kwargs):
            raise Exception("Simulated failure")

        # Simulate failures
        cache.set = failing_set
        for _ in range(cache.config.circuit_breaker_failure_threshold):
            try:
                await cache.set("test_key", "test_value")
            except:
                await cache._record_failure()

        # Verify circuit breaker is open
        assert cache._is_circuit_breaker_open() is True

        # Restore original method for cleanup
        cache.set = original_set

    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache):
        """Test basic cache operations work correctly."""
        # Test set and get
        result = await cache.set("test_key", "test_value")
        assert result is True

        retrieved = await cache.get("test_key")
        assert retrieved == "test_value"

        # Test delete
        result = await cache.delete("test_key")
        assert result is True

        retrieved = await cache.get("test_key")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_health_check(self, cache):
        """Test cache health check functionality."""
        result = await cache.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Test cache statistics tracking."""
        # Initial stats
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0

        # Add some entries and access them
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Access existing key (hit)
        await cache.get("key1")
        # Access non-existing key (miss)
        await cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entry_count == 2


class TestCacheGlobalInstance:
    """Test global cache instance management."""

    @pytest.mark.asyncio
    async def test_get_memory_cache(self):
        """Test global cache instance creation."""
        cache1 = await get_memory_cache()
        cache2 = await get_memory_cache()

        # Should return same instance
        assert cache1 is cache2

        # Cleanup
        await close_memory_cache()

    @pytest.mark.asyncio
    async def test_close_memory_cache(self):
        """Test global cache cleanup."""
        cache = await get_memory_cache()
        await cache.set("test", "value")

        await close_memory_cache()

        # New instance should be clean
        new_cache = await get_memory_cache()
        result = await new_cache.get("test")
        assert result is None  # Should be clean instance

        await close_memory_cache()


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/models/service/test_cache_subcontract.py -v
    pytest.main([__file__, "-v"])
