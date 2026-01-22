# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Provider-scoped rate limiter for external API calls.

This module provides a rate limiter keyed by (provider, model) to support
different rate limits for different endpoints. Local endpoints typically
have no rate limits while cloud providers (OpenAI) have strict limits.

The rate limiter uses a sliding window algorithm with async-safe locking
to prevent API throttling across concurrent requests.

Example::

    import asyncio
    from omnimemory.handlers.adapters import (
        ProviderRateLimiter,
        ModelRateLimiterConfig,
    )

    async def example():
        config = ModelRateLimiterConfig(
            provider="openai",
            model="text-embedding-3-small",
            requests_per_minute=60,
        )
        limiter = ProviderRateLimiter(config)

        # Acquire permission before making request
        await limiter.acquire()
        # Make API call here...

        # Check remaining capacity
        remaining = limiter.get_remaining()
        print(f"Remaining requests: {remaining}")

    asyncio.run(example())

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from omnimemory.models.config import (
    DEFAULT_REQUESTS_PER_MINUTE,
    ModelRateLimiterConfig,
)

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

__all__ = [
    "ProviderRateLimiter",
    "ModelRateLimiterConfig",
    "RateLimiterRegistry",
]

# Constants for rate limiting
SECONDS_PER_MINUTE = 60.0


class ProviderRateLimiter:
    """Async-safe rate limiter with sliding window algorithm.

    Uses a sliding window to track requests and enforce rate limits.
    Supports both RPM (requests per minute) and TPM (tokens per minute)
    limiting.

    Thread-safe via asyncio.Lock for concurrent access.

    Attributes:
        config: The rate limiter configuration.
    """

    def __init__(self, config: ModelRateLimiterConfig) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limiter configuration.
        """
        self._config = config
        self._lock = asyncio.Lock()

        # Sliding window: deque of (timestamp, tokens) tuples
        self._request_window: deque[tuple[float, int]] = deque()

        # Pre-calculate limits
        self._max_requests = int(config.requests_per_minute * config.burst_multiplier)
        self._max_tokens = (
            int(config.tokens_per_minute * config.burst_multiplier)
            if config.tokens_per_minute > 0
            else 0
        )

        logger.debug(
            "Rate limiter initialized for %s/%s: %d RPM, %d TPM",
            config.provider,
            config.model,
            config.requests_per_minute,
            config.tokens_per_minute,
        )

    @property
    def config(self) -> ModelRateLimiterConfig:
        """Get the rate limiter configuration."""
        return self._config

    def _cleanup_window(self, now: float) -> None:
        """Remove expired entries from the sliding window.

        Args:
            now: Current timestamp.
        """
        cutoff = now - SECONDS_PER_MINUTE
        while self._request_window and self._request_window[0][0] < cutoff:
            self._request_window.popleft()

    def _get_current_usage(self) -> tuple[int, int]:
        """Get current request and token counts in the window.

        Returns:
            Tuple of (request_count, token_count).
        """
        request_count = len(self._request_window)
        token_count = sum(tokens for _, tokens in self._request_window)
        return request_count, token_count

    async def acquire(
        self,
        tokens: int = 1,
        correlation_id: UUID | None = None,
    ) -> None:
        """Acquire permission to make a request, blocking if rate limited.

        Blocks until a request slot is available. Uses exponential backoff
        to avoid busy-waiting.

        Args:
            tokens: Number of tokens for this request (for TPM limiting).
            correlation_id: Optional correlation ID for logging.
        """
        backoff = 0.1  # Start with 100ms
        max_backoff = 5.0  # Cap at 5 seconds

        while True:
            acquired = await self.try_acquire(tokens, correlation_id)
            if acquired:
                return

            # Calculate wait time
            reset_time = self.get_reset_time()
            wait_time = min(reset_time, backoff)

            if correlation_id:
                logger.debug(
                    "Rate limited for %s/%s (correlation_id=%s), waiting %.2fs",
                    self._config.provider,
                    self._config.model,
                    correlation_id,
                    wait_time,
                )

            await asyncio.sleep(wait_time)

            # Exponential backoff with cap
            backoff = min(backoff * 2, max_backoff)

    async def try_acquire(
        self,
        tokens: int = 1,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Try to acquire permission without blocking.

        Args:
            tokens: Number of tokens for this request.
            correlation_id: Optional correlation ID for logging.

        Returns:
            True if permission was granted, False if rate limited.
        """
        async with self._lock:
            now = time.monotonic()
            self._cleanup_window(now)

            request_count, token_count = self._get_current_usage()

            # Check request limit
            if request_count >= self._max_requests:
                logger.debug(
                    "Rate limit reached for %s/%s: %d/%d requests",
                    self._config.provider,
                    self._config.model,
                    request_count,
                    self._max_requests,
                )
                return False

            # Check token limit (if enabled)
            if self._max_tokens > 0 and token_count + tokens > self._max_tokens:
                logger.debug(
                    "Token limit reached for %s/%s: %d+%d/%d tokens",
                    self._config.provider,
                    self._config.model,
                    token_count,
                    tokens,
                    self._max_tokens,
                )
                return False

            # Record this request
            self._request_window.append((now, tokens))

            if correlation_id:
                logger.debug(
                    "Rate limit acquired for %s/%s (correlation_id=%s): %d/%d requests",
                    self._config.provider,
                    self._config.model,
                    correlation_id,
                    request_count + 1,
                    self._max_requests,
                )

            return True

    def get_remaining(self) -> int:
        """Get approximate remaining requests in current window.

        This is a best-effort observability method that provides an approximate
        count without acquiring the async lock. The value may be slightly stale
        under high contention, as it does not clean up expired window entries.

        Note:
            This method is intentionally non-modifying to be safe for concurrent
            reads. For accurate counts, the next ``try_acquire()`` call will
            perform cleanup and provide precise limiting.

        Returns:
            Approximate number of requests remaining before rate limit.
        """
        # Intentionally skip _cleanup_window() to avoid modifying shared state
        # without the lock. This may return a slightly conservative estimate
        # (fewer remaining than actual) if expired entries haven't been cleaned.
        request_count, _ = self._get_current_usage()
        return max(0, self._max_requests - request_count)

    def get_reset_time(self) -> float:
        """Get approximate seconds until rate limit resets.

        This is a best-effort observability method that provides an approximate
        reset time without acquiring the async lock. The value may be slightly
        inaccurate under high contention due to concurrent modifications.

        Note:
            This method safely handles the case where the window becomes empty
            between the check and access by catching IndexError. Under high
            contention, this may occasionally return 0.0 even when requests
            are pending cleanup.

        Returns:
            Approximate seconds until the oldest request in the window expires.
            Returns 0.0 if the window is empty or becomes empty during access.
        """
        # Safely access the deque without locking. The deque may be modified
        # concurrently, so we catch IndexError if it becomes empty between
        # the check and access.
        try:
            # Capture reference to avoid issues if deque is cleared
            window = self._request_window
            if not window:
                return 0.0

            now = time.monotonic()
            oldest_timestamp = window[0][0]
            time_until_reset = (oldest_timestamp + SECONDS_PER_MINUTE) - now
            return max(0.0, time_until_reset)
        except IndexError:
            # Window was emptied between check and access
            return 0.0


class RateLimiterRegistry:
    """Registry for managing rate limiters by (provider, model) key.

    Provides a centralized way to get or create rate limiters for
    different provider/model combinations. Ensures only one rate
    limiter exists per (provider, model) pair.

    Example::

        registry = RateLimiterRegistry()

        # Get or create a rate limiter
        limiter = registry.get_or_create(
            provider="openai",
            model="text-embedding-3-small",
            requests_per_minute=60,
        )

        await limiter.acquire()
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._limiters: dict[tuple[str, str], ProviderRateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        provider: str,
        model: str,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = 0,
        burst_multiplier: float = 1.0,
    ) -> ProviderRateLimiter:
        """Get existing rate limiter or create a new one.

        Args:
            provider: Provider identifier.
            model: Model identifier.
            requests_per_minute: Maximum RPM for new limiters.
            tokens_per_minute: Maximum TPM for new limiters (0 to disable).
            burst_multiplier: Burst allowance for new limiters.

        Returns:
            The rate limiter for the (provider, model) combination.
        """
        key = (provider.lower().strip(), model.lower().strip())

        async with self._lock:
            if key not in self._limiters:
                config = ModelRateLimiterConfig(
                    provider=provider,
                    model=model,
                    requests_per_minute=requests_per_minute,
                    tokens_per_minute=tokens_per_minute,
                    burst_multiplier=burst_multiplier,
                )
                self._limiters[key] = ProviderRateLimiter(config)
                logger.info(
                    "Created rate limiter for %s/%s: %d RPM",
                    provider,
                    model,
                    requests_per_minute,
                )

            return self._limiters[key]

    def get(self, provider: str, model: str) -> ProviderRateLimiter | None:
        """Get existing rate limiter without creating.

        Args:
            provider: Provider identifier.
            model: Model identifier.

        Returns:
            The rate limiter if it exists, None otherwise.
        """
        key = (provider.lower().strip(), model.lower().strip())
        return self._limiters.get(key)

    async def remove(self, provider: str, model: str) -> bool:
        """Remove a rate limiter from the registry.

        Args:
            provider: The provider name.
            model: The model name.

        Returns:
            True if a limiter was removed, False if not found.
        """
        key = (provider.lower().strip(), model.lower().strip())
        async with self._lock:
            if key in self._limiters:
                del self._limiters[key]
                logger.info("Removed rate limiter for %s/%s", provider, model)
                return True
            return False

    async def clear(self) -> None:
        """Remove all rate limiters from the registry."""
        async with self._lock:
            self._limiters.clear()
        logger.info("Cleared all rate limiters from registry")

    @property
    def count(self) -> int:
        """Get the number of registered rate limiters."""
        return len(self._limiters)
