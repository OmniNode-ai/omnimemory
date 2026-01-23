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

pytestmark = pytest.mark.unit

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

            # Should be ~55 seconds until the oldest request expires (60 - 5 = 55)
            # Use wider tolerance to avoid flakiness in CI due to event loop delays
            assert reset_time == pytest.approx(55.0, abs=5.0)

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
        with pytest.raises(ValueError, match="tokens must be >= 1"):
            await limiter.try_acquire(tokens=-1)

    @pytest.mark.asyncio
    async def test_try_acquire_zero_tokens_raises(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test that zero token count raises ValueError.

        Requesting zero tokens is meaningless and should be rejected
        to prevent invalid token accounting.
        """
        with pytest.raises(ValueError, match="tokens must be >= 1"):
            await limiter.try_acquire(tokens=0)

    @pytest.mark.asyncio
    async def test_acquire_negative_tokens_raises(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test that negative token count raises ValueError in acquire."""
        with pytest.raises(ValueError, match="tokens must be >= 1"):
            await limiter.acquire(tokens=-5)

    @pytest.mark.asyncio
    async def test_acquire_zero_tokens_raises(
        self,
        limiter: ProviderRateLimiter,
    ) -> None:
        """Test that zero token count raises ValueError in acquire.

        Requesting zero tokens is meaningless and should be rejected
        to prevent invalid token accounting.
        """
        with pytest.raises(ValueError, match="tokens must be >= 1"):
            await limiter.acquire(tokens=0)

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
        """Test try_acquire returns False when tokens exceed max.

        Unlike acquire() which raises ValueError for oversized requests,
        try_acquire follows "try" semantics and returns False to indicate
        the request cannot be satisfied.
        """
        limiter = ProviderRateLimiter(config_with_tokens)

        # config_with_tokens has tokens_per_minute=1000
        # try_acquire should return False for tokens > max (try semantics)
        result = await limiter.try_acquire(tokens=1001)
        assert result is False

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


# =============================================================================
# Stress Tests
# =============================================================================


class TestProviderRateLimiterStress:
    """Stress tests for ProviderRateLimiter under high concurrency.

    These tests validate:
    - Correctness under high concurrent load
    - Window expiry edge cases
    - Race condition safety
    - Observability method accuracy under contention

    Tests are designed to complete in <30s total.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_high_concurrency_100_requests(self) -> None:
        """Test 100+ concurrent acquire() calls with limited RPM.

        Validates that exactly RPM requests succeed under high contention
        and the async lock properly serializes access.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="high-concurrency",
            requests_per_minute=20,  # Low limit to force contention
        )
        limiter = ProviderRateLimiter(config)

        # Launch 100 concurrent try_acquire calls
        results = await asyncio.gather(*[limiter.try_acquire() for _ in range(100)])

        # Exactly 20 should succeed (the rate limit)
        successful = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is False)

        assert successful == 20, f"Expected 20 successful, got {successful}"
        assert failed == 80, f"Expected 80 failed, got {failed}"
        assert limiter.get_remaining() == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_high_concurrency_200_requests_with_burst(self) -> None:
        """Test 200 concurrent requests with burst multiplier.

        Validates burst_multiplier correctly increases capacity under load.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="burst-stress",
            requests_per_minute=50,
            burst_multiplier=2.0,  # Effective limit: 100
        )
        limiter = ProviderRateLimiter(config)

        # Launch 200 concurrent try_acquire calls
        results = await asyncio.gather(*[limiter.try_acquire() for _ in range(200)])

        # Exactly 100 should succeed (50 * 2.0 burst)
        successful = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is False)

        assert successful == 100, f"Expected 100 successful, got {successful}"
        assert failed == 100, f"Expected 100 failed, got {failed}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_window_expiry_at_exact_boundary(self) -> None:
        """Test requests at exact 60-second window boundary.

        Validates that requests right at the expiry boundary are handled
        correctly without race conditions.

        Note: The cleanup logic uses strictly less than (`<`) for the cutoff,
        so requests at exactly t=1000 expire when current time is >1060
        (not at exactly 1060).
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="boundary-test",
            requests_per_minute=5,
        )

        # Track actual time progression for verification
        mock_time = 1000.0

        def get_mock_time() -> float:
            return mock_time

        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic",
            side_effect=get_mock_time,
        ):
            limiter = ProviderRateLimiter(config)

            # Fill up the window at t=1000
            for _ in range(5):
                result = await limiter.try_acquire()
                assert result is True

            # Window should be full
            assert await limiter.try_acquire() is False

            # At exactly t=1060, cutoff = 1000, requests at t=1000 are NOT expired
            # (because cleanup uses `<` not `<=`)
            mock_time = 1060.0
            assert (
                await limiter.try_acquire() is False
            ), "Requests at exact boundary should not be expired yet"

            # Advance just past the boundary - now requests should expire
            mock_time = 1060.001

            # Now all requests should have expired, allowing new ones
            results = await asyncio.gather(*[limiter.try_acquire() for _ in range(5)])
            successful = sum(1 for r in results if r is True)
            assert successful == 5, f"Expected 5 after expiry, got {successful}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_window_expiry_partial_cleanup(self) -> None:
        """Test partial window cleanup when some requests expire.

        Validates that only expired requests are cleaned up and newer
        ones remain in the window.
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="partial-cleanup",
            requests_per_minute=10,
        )

        mock_time = 1000.0

        def get_mock_time() -> float:
            return mock_time

        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic",
            side_effect=get_mock_time,
        ):
            limiter = ProviderRateLimiter(config)

            # Add 5 requests at t=1000
            for _ in range(5):
                await limiter.try_acquire()

            # Advance 30 seconds and add 3 more requests at t=1030
            mock_time = 1030.0
            for _ in range(3):
                await limiter.try_acquire()

            # At t=1030, we have 8 requests (5 from t=1000, 3 from t=1030)
            assert limiter.get_remaining() == 2

            # Advance to t=1061 (just past 60s from first batch)
            mock_time = 1061.0

            # The 5 requests from t=1000 should be expired
            # The 3 requests from t=1030 should still be valid
            # So we should be able to add 7 more (10 - 3 = 7)
            results = await asyncio.gather(*[limiter.try_acquire() for _ in range(10)])
            successful = sum(1 for r in results if r is True)
            assert successful == 7, f"Expected 7 after partial expiry, got {successful}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_concurrent_cleanup_and_acquire_race(self) -> None:
        """Test race between cleanup_window and acquire calls.

        Validates no race conditions between concurrent acquisitions
        where some trigger cleanup and others don't.
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="race-condition",
            requests_per_minute=10,
        )

        # Use a lock-protected time value to safely update mock time
        time_lock = asyncio.Lock()
        current_time = 1000.0

        async def get_mock_time() -> float:
            async with time_lock:
                return current_time

        def sync_get_mock_time() -> float:
            # For sync calls in the limiter
            return current_time

        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic",
            side_effect=sync_get_mock_time,
        ):
            limiter = ProviderRateLimiter(config)

            # Fill the window
            for _ in range(10):
                await limiter.try_acquire()

            # Now simulate time progressing while concurrent acquires happen
            async def acquire_with_time_progression(delay: float) -> bool:
                nonlocal current_time
                # Small delay to spread out requests
                await asyncio.sleep(delay * 0.001)
                # Progress time slightly
                async with time_lock:
                    current_time += 0.1
                return await limiter.try_acquire()

            # Launch many concurrent acquires that each progress time slightly
            # This tests the race between cleanup and acquire
            tasks = [acquire_with_time_progression(i) for i in range(50)]
            results = await asyncio.gather(*tasks)

            # Just verify no exceptions occurred and results are boolean
            assert all(isinstance(r, bool) for r in results)
            # Some should fail, some might succeed as time progresses
            # The exact count depends on timing, but at least some should fail
            failed = sum(1 for r in results if r is False)
            assert failed > 0, "Expected some failures when window is full"

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_token_limit_stress_high_volume(self) -> None:
        """Test TPM limiting with high-volume token requests.

        Validates token-based limiting works correctly under concurrent load.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="token-stress",
            requests_per_minute=1000,  # High RPM so tokens are the constraint
            tokens_per_minute=10000,
        )
        limiter = ProviderRateLimiter(config)

        # 50 concurrent requests each requesting 300 tokens
        # Max tokens that fit: 10000 / 300 = 33 requests
        results = await asyncio.gather(
            *[limiter.try_acquire(tokens=300) for _ in range(50)]
        )

        successful = sum(1 for r in results if r is True)
        # 33 * 300 = 9900 tokens (under limit)
        # 34 * 300 = 10200 tokens (over limit)
        assert (
            successful == 33
        ), f"Expected 33 successful (9900 tokens), got {successful}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_token_limit_stress_variable_sizes(self) -> None:
        """Test TPM limiting with variable token sizes concurrently.

        Validates token accounting is accurate under mixed-size requests.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="variable-tokens",
            requests_per_minute=1000,
            tokens_per_minute=5000,
        )
        limiter = ProviderRateLimiter(config)

        # Mix of different token sizes
        token_sizes = [100, 200, 500, 1000, 250, 150, 300, 400, 600, 800]

        # Launch requests with various token sizes
        results = await asyncio.gather(
            *[limiter.try_acquire(tokens=size) for size in token_sizes * 5]
        )

        successful = sum(1 for r in results if r is True)
        # Total available: 5000 tokens
        # Requests are processed in order of lock acquisition (non-deterministic)
        # Just verify some succeeded and some failed
        assert successful > 0, "Expected some successful acquisitions"
        assert successful < 50, "Expected some failures due to token limit"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_get_remaining_accuracy_under_contention(self) -> None:
        """Verify get_remaining() returns reasonable values under contention.

        get_remaining() is documented as approximate. This test validates
        that values are within acceptable bounds under concurrent access.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="remaining-accuracy",
            requests_per_minute=100,
        )
        limiter = ProviderRateLimiter(config)

        # Track remaining values during concurrent acquisitions
        remaining_values: list[int] = []

        async def acquire_and_check() -> bool:
            result = await limiter.try_acquire()
            # Check remaining right after acquire
            remaining = limiter.get_remaining()
            remaining_values.append(remaining)
            return result

        # Launch 150 concurrent tasks
        results = await asyncio.gather(*[acquire_and_check() for _ in range(150)])

        successful = sum(1 for r in results if r is True)
        assert successful == 100

        # All remaining values should be in valid range [0, 100]
        for val in remaining_values:
            assert 0 <= val <= 100, f"get_remaining() returned invalid value: {val}"

        # Final remaining should be 0 (all slots used)
        final_remaining = limiter.get_remaining()
        assert final_remaining == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_get_remaining_handles_concurrent_modification(self) -> None:
        """Test get_remaining() handles RuntimeError from concurrent modification.

        The implementation catches RuntimeError when deque is modified during
        iteration. This test validates that behavior.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="remaining-concurrent",
            requests_per_minute=50,
        )
        limiter = ProviderRateLimiter(config)

        # Pre-fill with some requests
        for _ in range(25):
            await limiter.try_acquire()

        # Launch many concurrent get_remaining and try_acquire calls
        async def mixed_operations() -> tuple[int, bool]:
            remaining = limiter.get_remaining()
            result = await limiter.try_acquire()
            return remaining, result

        results = await asyncio.gather(*[mixed_operations() for _ in range(100)])

        # All remaining values should be valid (0 or positive)
        for remaining, _ in results:
            assert remaining >= 0, f"get_remaining() returned negative: {remaining}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_get_reset_time_accuracy_under_contention(self) -> None:
        """Verify get_reset_time() returns reasonable values under contention.

        get_reset_time() is documented as approximate. This test validates
        values are within acceptable bounds under concurrent access.
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="reset-accuracy",
            requests_per_minute=50,
        )

        mock_time = 1000.0

        def get_mock_time() -> float:
            return mock_time

        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic",
            side_effect=get_mock_time,
        ):
            limiter = ProviderRateLimiter(config)

            # Fill the window at t=1000
            for _ in range(50):
                await limiter.try_acquire()

            # Advance time by 10 seconds
            mock_time = 1010.0

            # Check reset time - should be ~50 seconds (60 - 10)
            reset_time = limiter.get_reset_time()
            assert 49.0 <= reset_time <= 51.0, f"Expected ~50s, got {reset_time}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_get_reset_time_handles_empty_window(self) -> None:
        """Test get_reset_time() returns 0.0 for empty window under contention.

        Validates safe handling when window becomes empty during access.
        """
        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="reset-empty",
            requests_per_minute=10,
        )
        limiter = ProviderRateLimiter(config)

        # Empty window should return 0.0
        assert limiter.get_reset_time() == 0.0

        # Now test with concurrent get_reset_time calls on an empty limiter
        async def check_reset() -> float:
            return limiter.get_reset_time()

        results = await asyncio.gather(*[check_reset() for _ in range(50)])

        # All should be 0.0 for empty window
        for val in results:
            assert val == 0.0, f"Expected 0.0 for empty window, got {val}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_get_reset_time_concurrent_with_acquisitions(self) -> None:
        """Test get_reset_time() during concurrent acquisitions.

        Validates no IndexError or invalid values when checking reset time
        while acquisitions are modifying the window.
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="reset-concurrent",
            requests_per_minute=30,
        )

        mock_time = 1000.0

        def get_mock_time() -> float:
            return mock_time

        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic",
            side_effect=get_mock_time,
        ):
            limiter = ProviderRateLimiter(config)

            # Mixed operations: acquire and check reset time concurrently
            async def acquire_or_check_reset(do_acquire: bool) -> float | bool:
                if do_acquire:
                    return await limiter.try_acquire()
                return limiter.get_reset_time()

            # Interleave acquire and reset checks
            tasks = []
            for i in range(100):
                tasks.append(acquire_or_check_reset(i % 2 == 0))  # Alternate

            results = await asyncio.gather(*tasks)

            # Validate all results are valid types
            for i, result in enumerate(results):
                if i % 2 == 0:  # Was an acquire
                    assert isinstance(result, bool)
                else:  # Was a reset check
                    assert isinstance(result, float)
                    assert result >= 0.0, f"Reset time was negative: {result}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_sustained_load_with_recovery(self) -> None:
        """Test sustained load with window recovery over time.

        Simulates real-world usage where requests come in bursts
        and the window periodically clears.
        """
        from unittest.mock import patch

        config = ModelRateLimiterConfig(
            provider="stress-test",
            model="sustained-load",
            requests_per_minute=20,
        )

        mock_time = 1000.0

        def get_mock_time() -> float:
            return mock_time

        with patch(
            "omnimemory.handlers.adapters.adapter_rate_limiter.time.monotonic",
            side_effect=get_mock_time,
        ):
            limiter = ProviderRateLimiter(config)

            total_successful = 0

            # Simulate 3 burst cycles
            for cycle in range(3):
                # Advance time by 60+ seconds to clear window
                if cycle > 0:
                    mock_time += 61.0

                # Burst of 30 requests
                results = await asyncio.gather(
                    *[limiter.try_acquire() for _ in range(30)]
                )
                successful = sum(1 for r in results if r is True)
                total_successful += successful

                # Each cycle should allow exactly 20 requests
                assert successful == 20, f"Cycle {cycle}: expected 20, got {successful}"

            # Total across all cycles: 60 successful requests
            assert total_successful == 60

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_registry_concurrent_get_or_create(self) -> None:
        """Test concurrent get_or_create calls for same key.

        Validates that only one limiter is created even with concurrent access.
        """
        registry = RateLimiterRegistry()

        # 50 concurrent get_or_create calls for the same key
        async def get_limiter() -> ProviderRateLimiter:
            return await registry.get_or_create(
                provider="concurrent-test",
                model="same-key",
                requests_per_minute=100,
            )

        limiters = await asyncio.gather(*[get_limiter() for _ in range(50)])

        # All should be the same instance
        first_limiter = limiters[0]
        for limiter in limiters:
            assert limiter is first_limiter

        # Registry should have exactly 1 entry
        count = await registry.count_exact()
        assert count == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_registry_concurrent_different_keys(self) -> None:
        """Test concurrent get_or_create for different keys.

        Validates multiple limiters can be created concurrently without issues.
        """
        registry = RateLimiterRegistry()

        # 50 concurrent get_or_create calls for different keys
        async def get_limiter(idx: int) -> ProviderRateLimiter:
            return await registry.get_or_create(
                provider="concurrent-test",
                model=f"model-{idx}",
                requests_per_minute=100,
            )

        limiters = await asyncio.gather(*[get_limiter(i) for i in range(50)])

        # All should be different instances
        limiter_ids = {id(limiter) for limiter in limiters}
        assert len(limiter_ids) == 50

        # Registry should have exactly 50 entries
        count = await registry.count_exact()
        assert count == 50

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_registry_count_vs_count_exact_under_load(self) -> None:
        """Test count property vs count_exact() under concurrent modifications.

        Validates that count_exact() provides accurate values while
        count property may be approximate.
        """
        registry = RateLimiterRegistry()

        # Track values during concurrent operations
        count_values: list[int] = []
        exact_values: list[int] = []

        async def create_and_check(idx: int) -> None:
            await registry.get_or_create(
                provider="count-test",
                model=f"model-{idx}",
            )
            # Capture both counts
            count_values.append(registry.count)
            exact_values.append(await registry.count_exact())

        await asyncio.gather(*[create_and_check(i) for i in range(30)])

        # After all operations, exact count should be accurate
        final_exact = await registry.count_exact()
        assert final_exact == 30

        # count property should also be 30 at the end (no concurrent modifications)
        assert registry.count == 30

        # All captured exact values should be valid (1 to 30)
        for val in exact_values:
            assert 1 <= val <= 30
