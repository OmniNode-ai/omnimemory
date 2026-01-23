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

    def test_backoff_defaults(self) -> None:
        """Test default backoff configuration values."""
        config = ModelRateLimiterConfig(
            provider="test",
            model="test-model",
        )
        assert config.initial_backoff_seconds == 0.1
        assert config.max_backoff_seconds == 5.0
        assert config.backoff_multiplier == 2.0

    def test_backoff_custom_values(self) -> None:
        """Test custom backoff configuration values."""
        config = ModelRateLimiterConfig(
            provider="test",
            model="test-model",
            initial_backoff_seconds=0.5,
            max_backoff_seconds=30.0,
            backoff_multiplier=3.0,
        )
        assert config.initial_backoff_seconds == 0.5
        assert config.max_backoff_seconds == 30.0
        assert config.backoff_multiplier == 3.0

    def test_validation_backoff_bounds(self) -> None:
        """Test backoff configuration validation bounds."""
        # initial_backoff_seconds too low
        with pytest.raises(ValueError):
            ModelRateLimiterConfig(
                provider="test",
                model="test",
                initial_backoff_seconds=0.001,  # Below 0.01 minimum
            )

        # max_backoff_seconds too high
        with pytest.raises(ValueError):
            ModelRateLimiterConfig(
                provider="test",
                model="test",
                max_backoff_seconds=100.0,  # Above 60.0 maximum
            )

        # backoff_multiplier below 1.0
        with pytest.raises(ValueError):
            ModelRateLimiterConfig(
                provider="test",
                model="test",
                backoff_multiplier=0.5,
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
        config: ModelRateLimiterConfig,
    ) -> None:
        """Test reset time after requests using mocked time.

        Uses mocked time.monotonic to make the test deterministic and
        avoid flakiness in CI environments where scheduling delays could
        cause timing-based assertions to fail.
        """
        from unittest.mock import patch

        # Create a fresh limiter for this test (not the fixture)
        # so we control the timing from the start
        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic"
        ) as mock_time:
            mock_time.return_value = 1000.0  # Fixed start time
            limiter = ProviderRateLimiter(config)

            # Acquire at t=1000.0
            await limiter.try_acquire()

            # Simulate 5 seconds passing
            mock_time.return_value = 1005.0
            reset_time = limiter.get_reset_time()

            # Should be 55 seconds until the oldest request expires (60 - 5 = 55)
            assert reset_time == pytest.approx(55.0, abs=0.1)

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
    async def test_try_acquire_tokens_exceed_max_raises(
        self,
        config_with_tokens: ModelRateLimiterConfig,
    ) -> None:
        """Test try_acquire raises ValueError when tokens exceed max.

        This ensures consistency with acquire() - both methods validate
        that the requested tokens don't exceed the maximum allowed.
        """
        limiter = ProviderRateLimiter(config_with_tokens)

        # config_with_tokens has tokens_per_minute=1000
        # try_acquire should raise ValueError for tokens > max
        with pytest.raises(ValueError, match="tokens.*exceeds maximum allowed"):
            await limiter.try_acquire(tokens=1001)

    @pytest.mark.asyncio
    async def test_concurrent_try_acquire(self) -> None:
        """Test multiple concurrent try_acquire calls respect the rate limit.

        This validates the async lock properly serializes concurrent access
        and ensures exactly the configured number of requests succeed.
        """
        config = ModelRateLimiterConfig(
            provider="test",
            model="concurrent-test",
            requests_per_minute=10,
        )
        limiter = ProviderRateLimiter(config)

        # Attempt 15 concurrent acquisitions (more than the 10 allowed)
        results = await asyncio.gather(*[limiter.try_acquire() for _ in range(15)])

        # Exactly 10 should succeed (the rate limit)
        successful = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is False)

        assert successful == 10, f"Expected 10 successful, got {successful}"
        assert failed == 5, f"Expected 5 failed, got {failed}"
        assert limiter.get_remaining() == 0

    @pytest.mark.asyncio
    async def test_custom_backoff_config_used(self) -> None:
        """Test that custom backoff config values are used during rate limiting.

        Uses mocked time and sleep to verify the limiter respects config values.
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="test",
            model="backoff-test",
            requests_per_minute=1,
            initial_backoff_seconds=0.25,  # Custom initial
            max_backoff_seconds=2.0,  # Custom max
            backoff_multiplier=3.0,  # Custom multiplier
        )
        limiter = ProviderRateLimiter(config)

        # Exhaust the limit
        await limiter.try_acquire()

        # Track sleep calls to verify backoff behavior
        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            # After a few sleeps, simulate time passing so window clears
            if len(sleep_calls) >= 3:
                # Manually clear window to end the loop
                limiter._request_window.clear()
            await original_sleep(0.001)  # Minimal actual sleep

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await limiter.acquire()

        # Verify backoff progression uses custom values:
        # - First sleep should be min(reset_time, 0.25) = 0.25 (initial)
        # - Second sleep should be min(reset_time, 0.75) = 0.75 (0.25 * 3)
        # - Third sleep would be min(reset_time, 2.0) = 2.0 (0.75 * 3 = 2.25, capped at 2.0)
        assert len(sleep_calls) >= 1
        # First backoff should be initial_backoff_seconds or less (limited by reset_time)
        assert sleep_calls[0] <= config.initial_backoff_seconds

    @pytest.mark.asyncio
    async def test_concurrent_try_acquire_with_tokens(self) -> None:
        """Test concurrent token-based rate limiting.

        Validates that concurrent token-based acquisitions are properly
        serialized and the total tokens consumed respects the limit.
        """
        config = ModelRateLimiterConfig(
            provider="test",
            model="concurrent-token-test",
            requests_per_minute=100,  # High RPM so tokens are the constraint
            tokens_per_minute=1000,
        )
        limiter = ProviderRateLimiter(config)

        # 5 concurrent requests each requesting 300 tokens
        # Only 3 should succeed (300*3=900 < 1000, 300*4=1200 > 1000)
        results = await asyncio.gather(
            *[limiter.try_acquire(tokens=300) for _ in range(5)]
        )

        successful = sum(1 for r in results if r is True)
        assert successful == 3, f"Expected 3 successful (900 tokens), got {successful}"


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

    @pytest.mark.asyncio
    async def test_empty_provider_raises(self, registry: RateLimiterRegistry) -> None:
        """Test that empty provider raises ValueError."""
        with pytest.raises(ValueError, match="provider cannot be empty"):
            await registry.get_or_create(provider="", model="test")

    @pytest.mark.asyncio
    async def test_whitespace_provider_raises(
        self, registry: RateLimiterRegistry
    ) -> None:
        """Test that whitespace-only provider raises ValueError."""
        with pytest.raises(ValueError, match="provider cannot be empty"):
            await registry.get_or_create(provider="   ", model="test")

    @pytest.mark.asyncio
    async def test_empty_model_raises(self, registry: RateLimiterRegistry) -> None:
        """Test that empty model raises ValueError."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            await registry.get_or_create(provider="openai", model="")

    @pytest.mark.asyncio
    async def test_invalid_provider_characters_raises(
        self, registry: RateLimiterRegistry
    ) -> None:
        """Test that provider with invalid characters raises ValueError."""
        with pytest.raises(ValueError, match="provider contains invalid characters"):
            await registry.get_or_create(provider="open ai", model="test")

    @pytest.mark.asyncio
    async def test_invalid_model_characters_raises(
        self, registry: RateLimiterRegistry
    ) -> None:
        """Test that model with invalid characters raises ValueError."""
        with pytest.raises(ValueError, match="model contains invalid characters"):
            await registry.get_or_create(provider="openai", model="test@model")

    @pytest.mark.asyncio
    async def test_special_characters_rejected(
        self, registry: RateLimiterRegistry
    ) -> None:
        """Test various special characters are rejected."""
        invalid_chars = ["@", "#", "$", "%", "^", "&", "*", "(", ")", " ", "!", "?"]
        for char in invalid_chars:
            with pytest.raises(ValueError, match="contains invalid characters"):
                await registry.get_or_create(
                    provider=f"test{char}provider", model="model"
                )

    @pytest.mark.asyncio
    async def test_valid_identifier_patterns(
        self, registry: RateLimiterRegistry
    ) -> None:
        """Test that valid identifier patterns are accepted."""
        # Test various valid patterns
        valid_pairs = [
            ("openai", "text-embedding-3-small"),
            ("local_provider", "model_v2"),
            ("provider.name", "model.version"),
            ("UPPERCASE", "MixedCase"),
            ("provider-with-dashes", "model-with-dashes"),
            ("provider_with_underscores", "model_with_underscores"),
            ("provider123", "model456"),
            ("local/provider", "models/gpt-4"),
        ]
        for provider, model in valid_pairs:
            limiter = await registry.get_or_create(provider=provider, model=model)
            assert limiter is not None
        await registry.clear()
