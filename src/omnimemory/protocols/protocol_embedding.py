# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for embedding clients and rate limiting.

This module defines the contract boundary protocols for external HTTP operations.
All embedding and rate limiting implementations must conform to these protocols.

IMPORTANT: These protocols define the ONLY allowed exit hatch for external HTTP
calls in OmniMemory. Direct use of httpx/requests is forbidden in business logic.

Example::

    from omnimemory.protocols.protocol_embedding import (
        ProtocolEmbeddingClient,
        ProtocolRateLimiter,
    )

    class MyEmbeddingClient:
        '''Concrete implementation conforming to ProtocolEmbeddingClient.'''

        async def get_embedding(
            self,
            text: str,
            correlation_id: UUID | None = None,
        ) -> list[float]:
            # Implementation here
            ...

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

__all__ = [
    "ProtocolEmbeddingClient",
    "ProtocolRateLimiter",
]


@runtime_checkable
class ProtocolEmbeddingClient(Protocol):
    """Protocol for embedding client implementations.

    Defines the contract for all embedding clients in OmniMemory.
    Implementations must wrap HandlerHttp and support:
    - Correlation ID tracking for distributed tracing
    - Provider-agnostic interface (OpenAI, local vLLM, etc.)
    - Async context manager lifecycle

    All implementations are expected to:
    - Use HandlerHttp for HTTP operations (no direct httpx/requests)
    - Pass correlation_id through to all HTTP calls
    - Validate embedding dimensions
    - Handle provider-specific response formats internally
    """

    async def get_embedding(
        self,
        text: str,
        correlation_id: UUID | None = None,
    ) -> list[float]:
        """Generate embedding vector for the given text.

        Args:
            text: The text to embed. Must be non-empty.
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, implementations should generate one.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingClientError: If text is empty or response is invalid.
            EmbeddingConnectionError: If connection to server fails.
            EmbeddingTimeoutError: If request times out.
        """
        ...

    async def get_embeddings_batch(
        self,
        texts: list[str],
        correlation_id: UUID | None = None,
        max_concurrency: int = 5,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            correlation_id: Optional correlation ID for distributed tracing.
            max_concurrency: Maximum concurrent requests.

        Returns:
            List of embedding vectors in the same order as input texts.

        Raises:
            EmbeddingClientError: If any text fails to embed.
        """
        ...

    async def health_check(self, correlation_id: UUID | None = None) -> bool:
        """Check if the embedding server is reachable and responding.

        Args:
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            True if server is healthy, False otherwise.
        """
        ...

    async def initialize(self) -> None:
        """Initialize the client and establish connections.

        Safe to call multiple times - subsequent calls are no-ops.
        """
        ...

    async def shutdown(self) -> None:
        """Close connections and release resources.

        Safe to call multiple times - subsequent calls are no-ops.
        """
        ...


@runtime_checkable
class ProtocolRateLimiter(Protocol):
    """Protocol for rate limiter implementations.

    Defines the contract for rate limiting external API calls.
    Rate limiters are keyed by (provider, model) to support different
    limits for different endpoints.

    Implementations must support:
    - Requests per minute (RPM) limiting
    - Tokens per minute (TPM) limiting (optional)
    - Async-safe acquisition with proper blocking
    - Graceful handling of limit exhaustion
    """

    async def acquire(
        self,
        tokens: int = 1,
        correlation_id: UUID | None = None,
    ) -> None:
        """Acquire permission to make a request.

        Blocks until a request slot is available. Does not raise on
        rate limit - instead waits until the limit resets.

        Args:
            tokens: Number of tokens to acquire (for TPM limiting).
                Defaults to 1 for simple RPM limiting.
            correlation_id: Optional correlation ID for logging.

        Note:
            This method blocks (via asyncio.sleep) until the rate limit
            allows the request. For non-blocking behavior, use try_acquire().
        """
        ...

    async def try_acquire(
        self,
        tokens: int = 1,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Try to acquire permission without blocking.

        Args:
            tokens: Number of tokens to acquire.
            correlation_id: Optional correlation ID for logging.

        Returns:
            True if permission was granted, False if rate limited.
        """
        ...

    def get_remaining(self) -> int:
        """Get remaining requests in current window.

        Returns:
            Number of requests remaining before rate limit.
        """
        ...

    def get_reset_time(self) -> float:
        """Get seconds until rate limit resets.

        Returns:
            Seconds until the rate limit window resets.
        """
        ...
