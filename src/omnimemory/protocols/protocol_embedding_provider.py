# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for embedding generation providers.

This protocol abstracts the embedding generation I/O, allowing the
HandlerSemanticAnalyzerCompute to remain pure and testable.

Implementations can be:
- HTTP-backed (calling LLM_EMBEDDING_URL)
- Local model-backed
- Mock/fake for testing
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ProtocolEmbeddingProvider(Protocol):
    """Protocol for embedding generation providers.

    Provides the abstraction layer between the semantic compute handler
    and the actual embedding service. Implementations handle the I/O;
    the handler remains pure compute.

    Example:
        >>> class FakeEmbeddingProvider:
        ...     async def generate_embedding(self, text: str, **kwargs) -> list[float]:
        ...         return [0.1] * 1024  # Fake embedding
        ...
        >>> provider: ProtocolEmbeddingProvider = FakeEmbeddingProvider()
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this embedding provider."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model being used."""
        ...

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is currently available."""
        ...

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> list[float]:
        """Generate embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            model: Optional model override (uses default if not specified).
            correlation_id: Optional correlation ID for tracing.
            timeout_seconds: Optional timeout override.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingProviderError: If embedding generation fails.
            EmbeddingProviderTimeoutError: If the operation times out.
        """
        ...

    @abstractmethod
    async def generate_embeddings_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Args:
            texts: The texts to generate embeddings for.
            model: Optional model override.
            correlation_id: Optional correlation ID for tracing.
            timeout_seconds: Optional timeout override.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            EmbeddingProviderError: If embedding generation fails.
            EmbeddingProviderTimeoutError: If the operation times out.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the embedding provider is healthy.

        Returns:
            True if the provider is healthy and ready to serve requests.
        """
        ...


@runtime_checkable
class ProtocolLLMProvider(Protocol):
    """Protocol for LLM providers (used for entity extraction, analysis).

    Provides the abstraction layer for LLM-backed operations like
    entity extraction and semantic analysis that require language models.

    Example:
        >>> class FakeLLMProvider:
        ...     async def complete(self, prompt: str, **kwargs) -> str:
        ...         return '{"entities": []}'  # Fake response
        ...
        >>> provider: ProtocolLLMProvider = FakeLLMProvider()
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the LLM model being used."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is currently available."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        seed: int | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete.
            model: Optional model override.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Maximum tokens to generate.
            seed: Optional seed for reproducibility.
            correlation_id: Optional correlation ID for tracing.
            timeout_seconds: Optional timeout override.

        Returns:
            The completion text.

        Raises:
            LLMProviderError: If completion fails.
            LLMProviderTimeoutError: If the operation times out.
        """
        ...

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        output_schema: dict[str, object],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        """Generate a structured completion matching the schema.

        Args:
            prompt: The prompt to complete.
            output_schema: JSON schema for the expected output.
            model: Optional model override.
            temperature: Sampling temperature.
            seed: Optional seed for reproducibility.
            correlation_id: Optional correlation ID for tracing.
            timeout_seconds: Optional timeout override.

        Returns:
            A dictionary matching the output schema.

        Raises:
            LLMProviderError: If completion fails.
            LLMProviderSchemaError: If output doesn't match schema.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy.

        Returns:
            True if the provider is healthy and ready to serve requests.
        """
        ...
