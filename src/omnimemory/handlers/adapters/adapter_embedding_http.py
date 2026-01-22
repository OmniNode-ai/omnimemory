# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP-based embedding client adapter wrapping HandlerHttp.

This module provides an embedding client that wraps `HandlerHttp` from
omnibase_infra to enforce the contract boundary principle: all external
HTTP calls must go through this adapter layer.

The client supports multiple embedding providers (OpenAI, local vLLM, etc.)
with automatic provider selection (local-first) and integrated rate limiting.

IMPORTANT: This is the ONLY allowed exit hatch for embedding HTTP calls.
Direct use of httpx/requests is forbidden in OmniMemory business logic.

Example::

    import asyncio
    from omnimemory.handlers.adapters import (
        EmbeddingHttpClient,
        ModelEmbeddingHttpClientConfig,
    )

    async def example():
        config = ModelEmbeddingHttpClientConfig(
            provider="local",
            base_url="http://192.168.86.201:8002",
            model="gte-qwen2",
        )
        client = EmbeddingHttpClient(config)

        async with client:
            embedding = await client.get_embedding("Hello world")
            print(f"Embedding dimension: {len(embedding)}")

    asyncio.run(example())

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.handlers.adapters.adapter_rate_limiter import (
    ModelRateLimiterConfig,
    ProviderRateLimiter,
)

if TYPE_CHECKING:
    from types import TracebackType
    from uuid import UUID

# omnibase_infra is a dev dependency - make imports conditional
_OMNIBASE_INFRA_AVAILABLE = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

try:
    from omnibase_infra.errors import (
        InfraConnectionError,
        InfraTimeoutError,
    )
    from omnibase_infra.handlers.handler_http import HandlerHttpRest

    _OMNIBASE_INFRA_AVAILABLE = True
except ImportError as e:
    _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

    # Provide stub types for type checking
    class InfraConnectionError(Exception):  # type: ignore[no-redef]
        """Stub for InfraConnectionError when omnibase_infra is not installed."""

    class InfraTimeoutError(Exception):  # type: ignore[no-redef]
        """Stub for InfraTimeoutError when omnibase_infra is not installed."""

    class HandlerHttpRest:  # type: ignore[no-redef]
        """Stub for HandlerHttpRest when omnibase_infra is not installed."""

        async def initialize(self, config: dict[str, object]) -> None:
            raise ImportError(
                f"omnibase_infra is required for EmbeddingHttpClient. "
                f"Install it with: poetry install --with dev. "
                f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
            )


logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingHttpClient",
    "ModelEmbeddingHttpClientConfig",
    "EmbeddingClientError",
    "EmbeddingConnectionError",
    "EmbeddingTimeoutError",
    "EmbeddingProviderType",
]


# =============================================================================
# Exceptions
# =============================================================================


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""


class EmbeddingConnectionError(EmbeddingClientError):
    """Raised when connection to embedding server fails."""


class EmbeddingTimeoutError(EmbeddingClientError):
    """Raised when embedding request times out."""


# =============================================================================
# Enums
# =============================================================================


class EmbeddingProviderType(str, Enum):
    """Supported embedding provider types."""

    LOCAL = "local"
    OPENAI = "openai"
    VLLM = "vllm"

    @classmethod
    def from_string(cls, value: str) -> EmbeddingProviderType:
        """Convert string to provider type."""
        normalized = value.lower().strip()
        for provider in cls:
            if provider.value == normalized:
                return provider
        raise ValueError(f"Unknown provider type: {value}")


# =============================================================================
# Configuration
# =============================================================================


class ModelEmbeddingHttpClientConfig(BaseModel):
    """Configuration for the HTTP embedding client.

    Attributes:
        provider: Provider type (local, openai, vllm).
        base_url: Base URL of the embedding server. Required.
        model: Model identifier for the embedding model.
        timeout_seconds: Request timeout in seconds.
        embedding_dimension: Expected dimension of embedding vectors.
        rate_limit_rpm: Requests per minute limit (0 for no limit).
        rate_limit_tpm: Tokens per minute limit (0 for no limit).
        auth_header: Optional authorization header value (e.g., "Bearer <token>").
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    provider: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.LOCAL,
        description="Embedding provider type",
    )
    base_url: str = Field(
        ...,
        min_length=1,
        description="Base URL of the embedding server (REQUIRED)",
    )
    model: str = Field(
        default="gte-qwen2",
        min_length=1,
        description="Model identifier",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300.0,
        description="Request timeout in seconds",
    )
    embedding_dimension: int = Field(
        default=1024,
        gt=0,
        le=8192,
        description="Expected embedding vector dimension",
    )
    rate_limit_rpm: int = Field(
        default=0,
        ge=0,
        le=10_000,
        description="Requests per minute limit (0 for no limit)",
    )
    rate_limit_tpm: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
        description="Tokens per minute limit (0 for no limit)",
    )
    auth_header: str | None = Field(
        default=None,
        description="Authorization header value (e.g., 'Bearer <token>')",
    )

    @field_validator("base_url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        """Normalize URL by stripping trailing slashes."""
        return v.rstrip("/")

    @property
    def embed_endpoint(self) -> str:
        """Get the full URL for the embed endpoint."""
        # Different providers have different endpoint patterns
        if self.provider == EmbeddingProviderType.OPENAI:
            return f"{self.base_url}/v1/embeddings"
        # Local and vLLM use /embed
        return f"{self.base_url}/embed"


# =============================================================================
# Client Implementation
# =============================================================================


class EmbeddingHttpClient:
    """HTTP embedding client that wraps HandlerHttp.

    This client is the contract boundary for all embedding HTTP operations.
    It wraps HandlerHttp from omnibase_infra and integrates:
    - Correlation ID tracking for distributed tracing
    - Provider-scoped rate limiting
    - Provider-specific request/response formatting
    - Proper error transformation

    IMPORTANT: No retry logic is implemented here. Retries are orchestrator-owned
    per the ONEX handler architecture.

    Attributes:
        config: The client configuration.
    """

    def __init__(
        self,
        config: ModelEmbeddingHttpClientConfig,
        rate_limiter: ProviderRateLimiter | None = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            config: Client configuration.
            rate_limiter: Optional pre-configured rate limiter. If not provided
                and rate limiting is configured, one will be created.
        """
        self._config = config
        self._handler: HandlerHttpRest | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._rate_limiter: ProviderRateLimiter | None = None

        # Set up rate limiter if configured
        if rate_limiter is not None:
            self._rate_limiter = rate_limiter
        elif config.rate_limit_rpm > 0:
            limiter_config = ModelRateLimiterConfig(
                provider=config.provider.value,
                model=config.model,
                requests_per_minute=config.rate_limit_rpm,
                tokens_per_minute=config.rate_limit_tpm,
            )
            self._rate_limiter = ProviderRateLimiter(limiter_config)

    @property
    def config(self) -> ModelEmbeddingHttpClientConfig:
        """Get the client configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the HTTP handler.

        Safe to call multiple times - subsequent calls are no-ops.

        Raises:
            ImportError: If omnibase_infra is not installed.
        """
        async with self._init_lock:
            if self._initialized:
                return

            if not _OMNIBASE_INFRA_AVAILABLE:
                raise ImportError(
                    f"omnibase_infra is required for EmbeddingHttpClient. "
                    f"Install it with: poetry install --with dev. "
                    f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
                )

            self._handler = HandlerHttpRest()
            await self._handler.initialize(
                {
                    "timeout_seconds": self._config.timeout_seconds,
                }
            )
            self._initialized = True

            logger.info(
                "EmbeddingHttpClient initialized for %s/%s at %s",
                self._config.provider.value,
                self._config.model,
                self._config.base_url,
            )

    async def shutdown(self) -> None:
        """Close the HTTP handler and release resources.

        Safe to call multiple times - subsequent calls are no-ops.
        """
        if self._handler is not None:
            await self._handler.shutdown()
            self._handler = None
        self._initialized = False
        logger.debug("EmbeddingHttpClient shutdown complete")

    async def __aenter__(self) -> EmbeddingHttpClient:
        """Enter async context manager."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager."""
        await self.shutdown()

    async def get_embedding(
        self,
        text: str,
        correlation_id: UUID | None = None,
    ) -> list[float]:
        """Generate embedding vector for the given text.

        Args:
            text: The text to embed. Must be non-empty.
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingClientError: If text is empty or response is invalid.
            EmbeddingConnectionError: If connection to server fails.
            EmbeddingTimeoutError: If request times out.
        """
        if not text or not text.strip():
            raise EmbeddingClientError("Text cannot be empty")

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Generate correlation ID if not provided
        cid = correlation_id or uuid4()

        # Acquire rate limit permission
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(tokens=1, correlation_id=cid)

        # Build the envelope for HandlerHttp
        envelope = self._build_envelope(text, cid)

        try:
            result = await self._handler.execute(envelope)  # type: ignore[union-attr]
            return self._parse_response(result, cid)

        except InfraConnectionError as e:
            raise EmbeddingConnectionError(
                f"Connection failed to {self._config.base_url}: {e}"
            ) from e

        except InfraTimeoutError as e:
            raise EmbeddingTimeoutError(
                f"Timeout after {self._config.timeout_seconds}s: {e}"
            ) from e

    def _build_envelope(self, text: str, correlation_id: UUID) -> dict[str, object]:
        """Build the HTTP request envelope for HandlerHttp.

        Args:
            text: The text to embed.
            correlation_id: Correlation ID for the request.

        Returns:
            Envelope dictionary for HandlerHttp.execute().
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._config.auth_header:
            headers["Authorization"] = self._config.auth_header

        # Build provider-specific request body
        if self._config.provider == EmbeddingProviderType.OPENAI:
            body = {
                "input": text,
                "model": self._config.model,
            }
        else:
            # Local and vLLM format
            body = {"text": text}

        return {
            "operation": "http.post",
            "payload": {
                "url": self._config.embed_endpoint,
                "headers": headers,
                "body": body,
            },
            "correlation_id": correlation_id,
            "envelope_id": uuid4(),
        }

    def _parse_response(
        self,
        result: object,
        correlation_id: UUID,
    ) -> list[float]:
        """Parse the HTTP response to extract embedding vector.

        Args:
            result: The ModelHandlerOutput from HandlerHttp.
            correlation_id: Correlation ID for logging.

        Returns:
            The embedding vector.

        Raises:
            EmbeddingClientError: If response format is invalid.
        """
        # Extract the result payload
        if not hasattr(result, "result"):
            raise EmbeddingClientError(
                f"Invalid response structure: missing 'result' attribute "
                f"(correlation_id={correlation_id})"
            )

        result_dict = result.result
        if not isinstance(result_dict, dict):
            raise EmbeddingClientError(
                f"Invalid response: result is not a dict "
                f"(correlation_id={correlation_id})"
            )

        # Check status
        status = result_dict.get("status")
        if status != "success":
            raise EmbeddingClientError(
                f"Request failed with status={status} "
                f"(correlation_id={correlation_id})"
            )

        payload = result_dict.get("payload", {})
        if not isinstance(payload, dict):
            raise EmbeddingClientError(
                f"Invalid payload structure (correlation_id={correlation_id})"
            )

        # Check HTTP status code
        status_code = payload.get("status_code", 0)
        if status_code >= 400:
            body = payload.get("body", {})
            error_msg = (
                body.get("error", str(body)) if isinstance(body, dict) else str(body)
            )
            raise EmbeddingClientError(
                f"HTTP {status_code}: {error_msg} (correlation_id={correlation_id})"
            )

        # Extract embedding from response body
        body = payload.get("body")
        return self._extract_embedding(body, correlation_id)

    def _extract_embedding(
        self,
        body: object,
        correlation_id: UUID,
    ) -> list[float]:
        """Extract embedding vector from response body.

        Handles different provider response formats:
        - OpenAI: {"data": [{"embedding": [...]}]}
        - Local/vLLM: {"embedding": [...]} or [...]

        Args:
            body: The response body (dict or list).
            correlation_id: Correlation ID for logging.

        Returns:
            The embedding vector.

        Raises:
            EmbeddingClientError: If embedding cannot be extracted.
        """
        embedding: list[float] | None = None

        if isinstance(body, list):
            # Direct list response
            embedding = body

        elif isinstance(body, dict):
            # OpenAI format: {"data": [{"embedding": [...]}]}
            if "data" in body:
                data = body["data"]
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if isinstance(first_item, dict) and "embedding" in first_item:
                        embedding = first_item["embedding"]

            # Local format: {"embedding": [...]}
            elif "embedding" in body:
                embedding = body["embedding"]

        if embedding is None:
            raise EmbeddingClientError(
                f"Could not extract embedding from response "
                f"(correlation_id={correlation_id})"
            )

        if not isinstance(embedding, list):
            raise EmbeddingClientError(
                f"Expected list for embedding, got {type(embedding)} "
                f"(correlation_id={correlation_id})"
            )

        # Validate dimension
        if len(embedding) != self._config.embedding_dimension:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d "
                "(correlation_id=%s)",
                self._config.embedding_dimension,
                len(embedding),
                correlation_id,
            )

        return embedding

    async def get_embeddings_batch(
        self,
        texts: list[str],
        correlation_id: UUID | None = None,
        max_concurrency: int = 5,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Processes texts with controlled concurrency.

        Args:
            texts: List of texts to embed.
            correlation_id: Optional correlation ID.
            max_concurrency: Maximum concurrent requests.

        Returns:
            List of embedding vectors in same order as input.

        Raises:
            EmbeddingClientError: If any text fails to embed.
        """
        if not texts:
            return []

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        cid = correlation_id or uuid4()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_with_semaphore(text: str) -> list[float]:
            async with semaphore:
                return await self.get_embedding(text, cid)

        tasks = [embed_with_semaphore(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def health_check(self, correlation_id: UUID | None = None) -> bool:
        """Check if the embedding server is reachable.

        Args:
            correlation_id: Optional correlation ID.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            await self.get_embedding("health check", correlation_id)
            return True
        except EmbeddingClientError:
            return False
