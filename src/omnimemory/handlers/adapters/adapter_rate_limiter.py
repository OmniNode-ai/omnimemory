# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Provider-scoped rate limiter for external API calls.

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
import math
import re
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

# Safety ceiling multiplier for deque maxlen to prevent runaway memory in edge cases.
# The deque maxlen is set to max_requests * this multiplier.
DEQUE_SAFETY_CEILING_MULTIPLIER = 2

# Pattern for validating provider/model identifiers
# Allows alphanumeric characters, hyphens, underscores, periods, and forward slashes
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9._/-]+$")


class ProviderRateLimiter:
    """Async-safe rate limiter with sliding window algorithm.

    Uses a sliding window to track requests and enforce rate limits.
    Supports both RPM (requests per minute) and TPM (tokens per minute)
    limiting.

    Thread-safe via asyncio.Lock for concurrent access.

    Attributes:
        config: The rate limiter configuration.
    """

    __slots__ = ("_config", "_lock", "_max_requests", "_max_tokens", "_request_window")

    def __init__(self, config: ModelRateLimiterConfig) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limiter configuration.
        """
        self._config = config
        self._lock = asyncio.Lock()

        # Pre-calculate limits (needed for maxlen calculation)
        # Use math.ceil to preserve burst headroom (e.g., 1 RPM * 1.5x = 2 not 1)
        self._max_requests = math.ceil(
            config.requests_per_minute * config.burst_multiplier
        )
        self._max_tokens = (
            math.ceil(config.tokens_per_minute * config.burst_multiplier)
            if config.tokens_per_minute > 0
            else 0
        )

        # Sliding window: deque of (timestamp, tokens) tuples
        self._request_window: deque[tuple[float, int]] = deque(
            maxlen=self._max_requests * DEQUE_SAFETY_CEILING_MULTIPLIER
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

        Raises:
            ValueError: If tokens is negative or exceeds maximum allowed.

        Note:
            Exponential backoff is controlled by config fields:
            ``initial_backoff_seconds`` (default 0.1), ``backoff_multiplier``
            (default 2.0), and ``max_backoff_seconds`` (default 5.0). The actual
            wait time is the minimum of the backoff value and the time until
            the oldest request in the window expires.
        """
        # Validate token count to prevent infinite waits
        # tokens=0 is rejected because requesting zero tokens is meaningless
        if tokens < 1:
            raise ValueError(f"tokens must be >= 1, got {tokens}")

        # If token limiting is enabled and request exceeds max, it can never succeed
        if self._max_tokens > 0 and tokens > self._max_tokens:
            raise ValueError(
                f"tokens ({tokens}) exceeds maximum allowed ({self._max_tokens}) "
                f"for {self._config.provider}/{self._config.model}"
            )

        backoff = self._config.initial_backoff_seconds
        max_backoff = self._config.max_backoff_seconds
        multiplier = self._config.backoff_multiplier
        waited = False

        while True:
            acquired = await self.try_acquire(tokens, correlation_id)
            if acquired:
                if not waited and correlation_id:
                    logger.debug(
                        "Rate limit acquired immediately for %s/%s (correlation_id=%s)",
                        self._config.provider,
                        self._config.model,
                        correlation_id,
                    )
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
            waited = True

            # Exponential backoff with cap
            backoff = min(backoff * multiplier, max_backoff)

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
            True if permission was granted, False if rate limited or if
            the request exceeds the maximum token limit.

        Raises:
            ValueError: If tokens is less than 1.
        """
        # Validate token count - zero or negative tokens are invalid
        # tokens=0 is rejected because requesting zero tokens is meaningless
        if tokens < 1:
            raise ValueError(f"tokens must be >= 1, got {tokens}")

        # If token limiting is enabled and request exceeds max, return False
        # (try semantics: report failure rather than throw for "can't succeed" cases)
        if self._max_tokens > 0 and tokens > self._max_tokens:
            logger.debug(
                "Token request exceeds maximum for %s/%s: %d > %d",
                self._config.provider,
                self._config.model,
                tokens,
                self._max_tokens,
            )
            return False

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

        Staleness Tolerance:
            Values may be stale by up to a few hundred milliseconds under high
            contention due to concurrent modifications and lack of window cleanup.
            This is acceptable for monitoring, logging, and display purposes.

        Concurrent Modification:
            If concurrent modification of the internal deque is detected during
            iteration, this method returns 0 as a conservative fallback. This
            ensures thread-safety without blocking but may occasionally underreport
            available capacity. This behavior is by design to prioritize safety
            over accuracy in edge cases.

        Note:
            This method is intentionally non-modifying to be safe for concurrent
            reads. For accurate counts, the next ``try_acquire()`` call will
            perform cleanup and provide precise limiting.

        Returns:
            Approximate number of requests remaining before rate limit.
            Returns 0 if concurrent modification is detected during iteration.
        """
        # Snapshot to avoid iteration issues during concurrent modification.
        # Intentionally skip _cleanup_window() to avoid modifying shared state
        # without the lock. This may return a slightly conservative estimate
        # (fewer remaining than actual) if expired entries haven't been cleaned.
        try:
            window_snapshot = list(self._request_window)
            return max(0, self._max_requests - len(window_snapshot))
        except RuntimeError:
            # Deque was modified during iteration - return conservative estimate
            logger.debug(
                "Concurrent modification detected in get_remaining() for %s/%s, "
                "returning conservative estimate",
                self._config.provider,
                self._config.model,
            )
            return 0

    def get_reset_time(self) -> float:
        """Get approximate seconds until rate limit resets.

        This is a best-effort observability method that provides an approximate
        reset time without acquiring the async lock. The value may be slightly
        inaccurate under high contention due to concurrent modifications.

        Staleness Tolerance:
            This method intentionally skips locking for non-blocking reads.
            The returned value may be stale by a few milliseconds under
            concurrent access. This is acceptable for monitoring, logging,
            and display purposes where exact precision is not required.

            For precise rate limit timing decisions, rely on the return
            values and blocking behavior of :meth:`acquire` or
            :meth:`try_acquire` instead.

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

    __slots__ = ("_limiters", "_lock")

    def __init__(self) -> None:
        """Initialize the registry."""
        self._limiters: dict[tuple[str, str], ProviderRateLimiter] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> None:
        """Validate that an identifier contains only allowed characters.

        Args:
            value: The identifier value to validate.
            field_name: Name of the field for error messages.

        Raises:
            ValueError: If the identifier is empty or contains invalid characters.
        """
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        if not _IDENTIFIER_PATTERN.match(value.strip()):
            raise ValueError(
                f"{field_name} contains invalid characters: {value!r}. "
                "Only alphanumeric characters, hyphens, underscores, periods, "
                "and forward slashes are allowed."
            )

    @staticmethod
    def _normalize_key(provider: str, model: str) -> tuple[str, str]:
        """Normalize provider and model to lowercase for consistent keying.

        Args:
            provider: Provider identifier.
            model: Model identifier.

        Returns:
            Tuple of (normalized_provider, normalized_model).

        Raises:
            ValueError: If provider or model are empty or contain invalid characters.
        """
        RateLimiterRegistry._validate_identifier(provider, "provider")
        RateLimiterRegistry._validate_identifier(model, "model")
        return (provider.lower().strip(), model.lower().strip())

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
        key = self._normalize_key(provider, model)

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

    async def get(self, provider: str, model: str) -> ProviderRateLimiter | None:
        """Get existing rate limiter without creating.

        This is an async method for consistency with other registry methods
        that access shared state under the async lock.

        Args:
            provider: Provider identifier.
            model: Model identifier.

        Returns:
            The rate limiter if it exists, None otherwise.
        """
        key = self._normalize_key(provider, model)
        async with self._lock:
            return self._limiters.get(key)

    async def remove(self, provider: str, model: str) -> bool:
        """Remove a rate limiter from the registry.

        Args:
            provider: The provider name.
            model: The model name.

        Returns:
            True if a limiter was removed, False if not found.
        """
        key = self._normalize_key(provider, model)
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
        """Get the approximate number of registered rate limiters.

        This is a best-effort observability method that provides an approximate
        count without acquiring the async lock. The value may be slightly stale
        under high contention due to concurrent get_or_create() or remove() calls.

        Use Cases:
            This property is ideal for non-critical observability scenarios:

            - **Metrics dashboards**: Displaying current limiter count
            - **Debug logging**: Recording registry state in log messages
            - **Health checks**: Quick status checks where approximate is acceptable
            - **Monitoring**: Tracking trends over time (minor variance is tolerable)

        When to Use :meth:`count_exact` Instead:
            If your use case requires guaranteed accuracy (e.g., capacity planning,
            alert threshold evaluation, test assertions, or audit logging), use
            the async :meth:`count_exact` method which acquires the lock.

        Note:
            Under typical usage patterns with low contention, this property
            returns accurate values. Staleness only occurs when concurrent
            modifications happen during the dict length read.

        Returns:
            Approximate number of registered rate limiters.

        See Also:
            :meth:`count_exact`: Async method for exact count with locking.
        """
        return len(self._limiters)

    async def count_exact(self) -> int:
        """Get the exact number of registered rate limiters.

        Unlike the :attr:`count` property which returns an approximate value
        without locking, this method acquires the lock to provide an exact count.

        When Precision Matters:
            Use this method when accuracy is critical:

            - **Capacity planning**: Deciding whether to create new limiters
            - **Alert thresholds**: Triggering alerts based on exact limiter count
            - **Audit logging**: Recording precise state for compliance/forensics
            - **Test assertions**: Verifying exact registry state in unit tests
            - **Conditional logic**: Making decisions based on limiter count

        When :attr:`count` Is Sufficient:
            For most observability scenarios, the non-locking :attr:`count`
            property is preferred as it avoids lock contention:

            - Metrics dashboards (trend visualization)
            - Debug logging (informational output)
            - Health check endpoints (approximate status)
            - Monitoring systems (variance is tolerable)

        Performance Note:
            This method acquires the registry's async lock, which may cause
            brief contention if called frequently alongside get_or_create()
            or remove() operations. For high-frequency polling, prefer
            :attr:`count` unless exact accuracy is required.

        Returns:
            Exact number of registered rate limiters.

        See Also:
            :attr:`count`: Non-locking property for approximate count.
        """
        async with self._lock:
            return len(self._limiters)
