# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ProviderRateLimiter.

This module tests the rate limiter adapter that enforces
per-provider rate limits for external API calls.

Test Categories:
    - Configuration: Config validation and defaults
    - Acquire: Rate limit acquisition with blocking
    - Try Acquire: Non-blocking rate limit checks
    - Sliding Window: Window cleanup and reset
    - Registry: Rate limiter registry operations

Usage:
    pytest tests/handlers/adapters/test_adapter_rate_limiter.py -v
    pytest tests/handlers/adapters/ -v -k "rate_limiter"

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from omnimemory.handlers.adapters.adapter_rate_limiter import (
    ModelRateLimiterConfig,
    ProviderRateLimiter,
    RateLimiterRegistry,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> ModelRateLimiterConfig:
    """Create a default rate limiter configuration."""
    return ModelRateLimiterConfig(
        provider="openai",
        model="text-embedding-3-small",
        requests_per_minute=60,
        tokens_per_minute=0,
    )


@pytest.fixture
def config_with_tokens() -> ModelRateLimiterConfig:
    """Create a config with token limiting enabled."""
    return ModelRateLimiterConfig(
        provider="openai",
        model="text-embedding-3-small",
        requests_per_minute=60,
        tokens_per_minute=1000,
    )


@pytest.fixture
def limiter(config: ModelRateLimiterConfig) -> ProviderRateLimiter:
    """Create a rate limiter with default config."""
    return ProviderRateLimiter(config)


@pytest.fixture
def registry() -> RateLimiterRegistry:
    """Create a rate limiter registry."""
    return RateLimiterRegistry()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestModelRateLimiterConfig:
    """Tests for ModelRateLimiterConfig validation."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelRateLimiterConfig(
            provider="test",
            model="test-model",
        )
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 0
        assert config.burst_multiplier == 1.0

    def test_provider_normalization(self) -> None:
        """Test provider identifier is normalized to lowercase."""
        config = ModelRateLimiterConfig(
            provider="  OpenAI  ",
            model="Test-Model",
        )
        assert config.provider == "openai"
        assert config.model == "test-model"

    def test_key_property(self) -> None:
        """Test the (provider, model) key property."""
        config = ModelRateLimiterConfig(
            provider="openai",
            model="text-embedding-3-small",
        )
        assert config.key == ("openai", "text-embedding-3-small")

    def test_validation_min_requests(self) -> None:
        """Test validation rejects zero requests per minute."""
        with pytest.raises(ValueError):
            ModelRateLimiterConfig(
                provider="test",
                model="test",
                requests_per_minute=0,
            )

    def test_validation_burst_multiplier_bounds(self) -> None:
        """Test burst multiplier must be >= 1.0."""
        with pytest.raises(ValueError):
            ModelRateLimiterConfig(
                provider="test",
                model="test",
                burst_multiplier=0.5,
            )


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestProviderRateLimiter:
    """Tests for ProviderRateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_try_acquire_success(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test successful acquisition."""
        result = await limiter.try_acquire()
        assert result is True
        assert limiter.get_remaining() == 59

    @pytest.mark.asyncio
    async def test_try_acquire_at_limit(
        self,
        config: ModelRateLimiterConfig,
    ) -> None:
        """Test acquisition fails when at limit."""
        # Create limiter with very low limit
        low_config = ModelRateLimiterConfig(
            provider="test",
            model="test",
            requests_per_minute=2,
        )
        limiter = ProviderRateLimiter(low_config)

        # Exhaust the limit
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is False

    @pytest.mark.asyncio
    async def test_try_acquire_with_correlation_id(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test acquisition with correlation ID for logging."""
        cid = uuid4()
        result = await limiter.try_acquire(correlation_id=cid)
        assert result is True

    @pytest.mark.asyncio
    async def test_try_acquire_token_limiting(
        self,
        config_with_tokens: ModelRateLimiterConfig,
    ) -> None:
        """Test token-based rate limiting."""
        limiter = ProviderRateLimiter(config_with_tokens)

        # Request 500 tokens - should succeed
        assert await limiter.try_acquire(tokens=500) is True

        # Request 600 more tokens - should fail (500 + 600 > 1000)
        assert await limiter.try_acquire(tokens=600) is False

        # Request 400 tokens - should succeed (500 + 400 < 1000)
        assert await limiter.try_acquire(tokens=400) is True

    def test_get_remaining(self, limiter: ProviderRateLimiter) -> None:
        """Test remaining requests count."""
        assert limiter.get_remaining() == 60

    def test_get_reset_time_empty_window(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test reset time with empty window."""
        assert limiter.get_reset_time() == 0.0

    @pytest.mark.asyncio
    async def test_get_reset_time_with_requests(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test reset time after requests.

        Note:
            Uses a wide tolerance (55-60 seconds) to handle CI environments
            where scheduling delays may occur between try_acquire() and
            get_reset_time() calls. The key invariant is that reset time
            should be close to 60 seconds, not exactly 60.
        """
        await limiter.try_acquire()
        reset_time = limiter.get_reset_time()
        # Reset time should be close to 60 seconds.
        # Use wide tolerance (55s) for CI environments with potential delays.
        # Upper bound of 60.1s accounts for floating-point precision.
        assert (
            55.0 <= reset_time <= 60.1
        ), f"Reset time {reset_time:.2f}s outside expected range [55.0, 60.1]"

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_limited(self) -> None:
        """Test that acquire blocks when rate limited."""
        # Create limiter with very low limit
        config = ModelRateLimiterConfig(
            provider="test",
            model="test",
            requests_per_minute=1,
        )
        limiter = ProviderRateLimiter(config)

        # Exhaust the limit
        await limiter.try_acquire()

        # acquire() should block - test with timeout
        async def acquire_with_timeout() -> bool:
            try:
                await asyncio.wait_for(limiter.acquire(), timeout=0.3)
                return True
            except TimeoutError:
                return False

        # Should timeout because we're rate limited
        result = await acquire_with_timeout()
        assert result is False

    @pytest.mark.asyncio
    async def test_burst_multiplier(self) -> None:
        """Test burst multiplier allows temporary overage."""
        config = ModelRateLimiterConfig(
            provider="test",
            model="test",
            requests_per_minute=2,
            burst_multiplier=2.0,  # Allow 2x burst
        )
        limiter = ProviderRateLimiter(config)

        # Should allow 4 requests (2 * 2.0 burst)
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is False

    @pytest.mark.asyncio
    async def test_try_acquire_negative_tokens_raises(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test that negative token count raises ValueError."""
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            await limiter.try_acquire(tokens=-1)

    @pytest.mark.asyncio
    async def test_acquire_negative_tokens_raises(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test that negative token count raises ValueError in acquire."""
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            await limiter.acquire(tokens=-5)

    @pytest.mark.asyncio
    async def test_acquire_tokens_exceed_max_raises(
        self,
        config_with_tokens: ModelRateLimiterConfig,
    ) -> None:
        """Test that tokens exceeding max raises ValueError to prevent infinite wait."""
        limiter = ProviderRateLimiter(config_with_tokens)

        # config_with_tokens has tokens_per_minute=1000
        # Requesting 1001 tokens should raise immediately
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            await limiter.acquire(tokens=1001)

    @pytest.mark.asyncio
    async def test_acquire_tokens_at_max_succeeds(
        self,
        config_with_tokens: ModelRateLimiterConfig,
    ) -> None:
        """Test that tokens at exactly max_tokens can be acquired."""
        limiter = ProviderRateLimiter(config_with_tokens)

        # config_with_tokens has tokens_per_minute=1000
        # Requesting exactly 1000 tokens should succeed
        result = await limiter.try_acquire(tokens=1000)
        assert result is True

    @pytest.mark.asyncio
    async def test_try_acquire_tokens_exceed_max_returns_false(
        self,
        config_with_tokens: ModelRateLimiterConfig,
    ) -> None:
        """Test try_acquire returns False when tokens exceed max (no ValueError).

        Note: try_acquire does NOT raise for exceeding max tokens because it's
        a non-blocking check. The ValueError is only raised in acquire() to
        prevent infinite waits.
        """
        limiter = ProviderRateLimiter(config_with_tokens)

        # config_with_tokens has tokens_per_minute=1000
        # try_acquire should just return False (not raise)
        result = await limiter.try_acquire(tokens=1001)
        assert result is False


# =============================================================================
# Registry Tests
# =============================================================================


class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry functionality."""

    @pytest.mark.asyncio
    async def test_get_or_create(self, registry: RateLimiterRegistry) -> None:
        """Test get_or_create creates new limiter."""
        limiter = await registry.get_or_create(
            provider="openai",
            model="text-embedding-3-small",
        )
        assert limiter is not None
        assert registry.count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(
        self,
        registry: RateLimiterRegistry,
    ) -> None:
        """Test get_or_create returns existing limiter."""
        limiter1 = await registry.get_or_create(
            provider="openai",
            model="text-embedding-3-small",
        )
        limiter2 = await registry.get_or_create(
            provider="openai",
            model="text-embedding-3-small",
        )
        assert limiter1 is limiter2
        assert registry.count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_normalizes_keys(
        self,
        registry: RateLimiterRegistry,
    ) -> None:
        """Test keys are normalized for consistent lookup."""
        limiter1 = await registry.get_or_create(
            provider="OpenAI",
            model="Text-Embedding-3-Small",
        )
        limiter2 = await registry.get_or_create(
            provider="openai",
            model="text-embedding-3-small",
        )
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(
        self,
        registry: RateLimiterRegistry,
    ) -> None:
        """Test get returns None for non-existent limiter."""
        result = await registry.get("unknown", "unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_existing(
        self,
        registry: RateLimiterRegistry,
    ) -> None:
        """Test get returns existing limiter."""
        await registry.get_or_create(
            provider="openai",
            model="test",
        )
        result = await registry.get("openai", "test")
        assert result is not None

    @pytest.mark.asyncio
    async def test_remove(self, registry: RateLimiterRegistry) -> None:
        """Test remove deletes limiter."""
        await registry.get_or_create(provider="openai", model="test")
        assert registry.count == 1

        removed = await registry.remove("openai", "test")
        assert removed is True
        assert registry.count == 0

    @pytest.mark.asyncio
    async def test_remove_returns_false_for_missing(
        self,
        registry: RateLimiterRegistry,
    ) -> None:
        """Test remove returns False for non-existent limiter."""
        removed = await registry.remove("unknown", "unknown")
        assert removed is False

    @pytest.mark.asyncio
    async def test_clear(self, registry: RateLimiterRegistry) -> None:
        """Test clear removes all limiters."""
        await registry.get_or_create(provider="openai", model="test1")
        await registry.get_or_create(provider="openai", model="test2")
        assert registry.count == 2

        await registry.clear()
        assert registry.count == 0
