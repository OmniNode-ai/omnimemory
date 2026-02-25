# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for EmbeddingHttpClient with real HandlerHttp.

This module tests the embedding HTTP client adapter against a real embedding
server to verify end-to-end functionality with actual HTTP calls through
HandlerHttpRest (no mocks).

Test Categories:
    - Connection: Real HTTP connection lifecycle
    - Embedding: Single embedding generation with real server
    - Batch: Batch embedding operations with real concurrency
    - Health: Health check with live server
    - Envelope: Verify envelope format correctness for real HTTP calls

Prerequisites:
    - Embedding server running at EMBEDDING_SERVER_URL (default: http://localhost:8100)
    - omnibase_infra installed (dev dependency)

Usage:
    # Run only integration tests
    pytest tests/handlers/adapters/test_adapter_embedding_http_integration.py -v

    # Run with specific markers
    pytest -m "integration and embedding" -v

    # Skip if embedding server unavailable (automatic)
    pytest -m integration -v

Environment Variables:
    EMBEDDING_SERVER_URL: Embedding server base URL (default: http://localhost:8100)
    EMBEDDING_MODEL: Model name (default: gte-qwen2)
    EMBEDDING_DIMENSION: Expected dimension (default: 1024)

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

# Check if dependencies are available
_DEPENDENCIES_AVAILABLE = False
_SKIP_REASON = "omnibase_infra not installed"

try:
    from omnibase_infra.handlers.handler_http import HandlerHttpRest

    _DEPENDENCIES_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError:
    _SKIP_REASON = "omnibase_infra not installed (required for HandlerHttpRest)"

try:
    from omnimemory.handlers.adapters.adapter_embedding_http import (
        EmbeddingClientError,
        EmbeddingHttpClient,
        EnumEmbeddingProviderType,
        ModelEmbeddingHttpClientConfig,
    )
except ImportError:
    _DEPENDENCIES_AVAILABLE = False
    _SKIP_REASON = "EmbeddingHttpClient not available"

if TYPE_CHECKING:
    from uuid import UUID


# =============================================================================
# Configuration
# =============================================================================

# Default embedding server settings (configure via EMBEDDING_SERVER_URL or LLM_EMBEDDING_URL env var)
DEFAULT_EMBEDDING_SERVER_URL = "http://localhost:8100"
DEFAULT_EMBEDDING_MODEL = "gte-qwen2"
DEFAULT_EMBEDDING_DIMENSION = 1024

# Test timeout for server availability check
SERVER_CHECK_TIMEOUT_SECONDS = 5.0


def get_embedding_server_url() -> str:
    """Get embedding server URL from environment or default."""
    return os.environ.get("EMBEDDING_SERVER_URL", DEFAULT_EMBEDDING_SERVER_URL)


def get_embedding_model() -> str:
    """Get embedding model name from environment or default."""
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_dimension() -> int:
    """Get expected embedding dimension from environment or default."""
    dim_str = os.environ.get("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
    return int(dim_str)


async def check_embedding_server_available() -> bool:
    """Check if the embedding server is available and responding.

    Performs a lightweight check by attempting to generate an embedding
    for a simple test phrase. This verifies both connectivity and basic
    API functionality.

    Returns:
        True if server responds with a valid embedding, False otherwise.
    """
    if not _DEPENDENCIES_AVAILABLE:
        return False

    url = get_embedding_server_url()
    model = get_embedding_model()
    dimension = get_embedding_dimension()

    config = ModelEmbeddingHttpClientConfig(
        provider=EnumEmbeddingProviderType.LOCAL,
        base_url=url,
        model=model,
        embedding_dimension=dimension,
        timeout_seconds=SERVER_CHECK_TIMEOUT_SECONDS,
    )

    try:
        # Use asyncio.wait_for to enforce timeout at the coroutine level
        async def _check() -> bool:
            client = EmbeddingHttpClient(config)
            async with client:
                # Try to get an embedding for a simple test phrase
                embedding = await client.get_embedding("connectivity test")
                # Verify we got a valid response
                return isinstance(embedding, list) and len(embedding) > 0

        return await asyncio.wait_for(_check(), timeout=SERVER_CHECK_TIMEOUT_SECONDS)
    except (TimeoutError, Exception):
        return False


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip all tests if dependencies are not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.embedding,
    pytest.mark.skipif(
        not _DEPENDENCIES_AVAILABLE,
        reason=_SKIP_REASON,
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def embedding_server_available() -> bool:
    """Check if embedding server is available for tests.

    Returns:
        True if embedding server is reachable, False otherwise.
    """
    return await check_embedding_server_available()


@pytest.fixture
def local_config() -> ModelEmbeddingHttpClientConfig:
    """Create a local provider configuration for integration tests.

    Returns:
        Configuration pointing to the local embedding server.
    """
    return ModelEmbeddingHttpClientConfig(
        provider=EnumEmbeddingProviderType.LOCAL,
        base_url=get_embedding_server_url(),
        model=get_embedding_model(),
        embedding_dimension=get_embedding_dimension(),
        timeout_seconds=30.0,
    )


@pytest.fixture
async def initialized_client(
    embedding_server_available: bool,
    local_config: ModelEmbeddingHttpClientConfig,
) -> AsyncGenerator[EmbeddingHttpClient, None]:
    """Create and initialize an embedding client for testing.

    Yields:
        Initialized EmbeddingHttpClient instance using real HandlerHttpRest.
    """
    if not embedding_server_available:
        pytest.skip("Embedding server is not available")

    client = EmbeddingHttpClient(local_config)
    await client.initialize()

    yield client

    await client.shutdown()


@pytest.fixture
def correlation_id() -> UUID:
    """Generate a unique correlation ID for test requests.

    Returns:
        UUID for distributed tracing.
    """
    return uuid4()


# =============================================================================
# Connection Tests
# =============================================================================


class TestRealConnection:
    """Tests for real HTTP connection lifecycle with HandlerHttpRest."""

    @pytest.mark.asyncio
    async def test_initialize_real_connection(
        self,
        embedding_server_available: bool,
        local_config: ModelEmbeddingHttpClientConfig,
    ) -> None:
        """Test actual HTTP handler initialization.

        Verifies that EmbeddingHttpClient correctly initializes the real
        HandlerHttpRest without mocking.
        """
        if not embedding_server_available:
            pytest.skip("Embedding server is not available")

        client = EmbeddingHttpClient(local_config)

        # Should not be initialized yet
        assert not client.is_initialized

        # Initialize connection
        await client.initialize()

        # Should be initialized
        assert client.is_initialized
        assert client._handler is not None
        assert isinstance(client._handler, HandlerHttpRest)

        # Cleanup
        await client.shutdown()
        assert not client.is_initialized

    @pytest.mark.asyncio
    async def test_context_manager_real_connection(
        self,
        embedding_server_available: bool,
        local_config: ModelEmbeddingHttpClientConfig,
    ) -> None:
        """Test async context manager with real connection.

        Verifies that the context manager properly initializes and
        cleans up the real HTTP handler.
        """
        if not embedding_server_available:
            pytest.skip("Embedding server is not available")

        async with EmbeddingHttpClient(local_config) as client:
            assert client.is_initialized
            assert client._handler is not None

        # After exit, should be cleaned up
        assert not client.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_idempotent_real(
        self,
        embedding_server_available: bool,
        local_config: ModelEmbeddingHttpClientConfig,
    ) -> None:
        """Test that multiple initialize calls are safe with real handler."""
        if not embedding_server_available:
            pytest.skip("Embedding server is not available")

        client = EmbeddingHttpClient(local_config)

        # Initialize twice
        await client.initialize()
        await client.initialize()

        # Should still be initialized
        assert client.is_initialized

        await client.shutdown()


# =============================================================================
# Embedding Generation Tests
# =============================================================================


class TestRealEmbedding:
    """Tests for embedding generation with real server."""

    @pytest.mark.asyncio
    async def test_get_embedding_simple_text(
        self,
        initialized_client: EmbeddingHttpClient,
        correlation_id: UUID,
    ) -> None:
        """Test single embedding generation with simple text.

        Verifies that the real HTTP call returns a valid embedding vector.
        """
        embedding = await initialized_client.get_embedding(
            "Hello, world!",
            correlation_id=correlation_id,
        )

        # Should return a list of floats
        assert isinstance(embedding, list)
        assert len(embedding) == get_embedding_dimension()
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_get_embedding_longer_text(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test embedding generation with longer text content.

        Verifies that longer text is properly handled by the real server.
        """
        long_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer text to test embedding generation "
            "with more substantial content that exercises the "
            "full embedding pipeline including tokenization."
        )

        embedding = await initialized_client.get_embedding(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) == get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_get_embedding_unicode_text(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test embedding generation with unicode characters.

        Verifies that the real HTTP call handles non-ASCII text correctly.
        """
        unicode_text = "Hello from different languages: Bonjour, Hola, Guten Tag"

        embedding = await initialized_client.get_embedding(unicode_text)

        assert isinstance(embedding, list)
        assert len(embedding) == get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_get_embedding_deterministic(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that same text produces same embedding.

        Verifies embedding consistency from the real server.
        """
        text = "consistent test phrase"

        embedding1 = await initialized_client.get_embedding(text)
        embedding2 = await initialized_client.get_embedding(text)

        # Embeddings should be identical (or very close due to floating point)
        assert len(embedding1) == len(embedding2)
        for v1, v2 in zip(embedding1, embedding2, strict=True):
            assert abs(v1 - v2) < 1e-6, "Embeddings should be deterministic"

    @pytest.mark.asyncio
    async def test_get_embedding_different_texts_different_vectors(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that different texts produce different embeddings.

        Verifies that the server actually processes text semantically.
        """
        embedding_cat = await initialized_client.get_embedding("cat")
        embedding_dog = await initialized_client.get_embedding("dog")

        # Embeddings should be different
        differences = [
            abs(v1 - v2) for v1, v2 in zip(embedding_cat, embedding_dog, strict=True)
        ]
        total_diff = sum(differences)

        assert total_diff > 0.1, "Different texts should produce different embeddings"


# =============================================================================
# Batch Embedding Tests
# =============================================================================


class TestRealBatchEmbedding:
    """Tests for batch embedding operations with real server."""

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_small(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test batch embedding with small number of texts.

        Verifies that batch processing works correctly with real concurrency.
        """
        texts = ["apple", "banana", "cherry"]

        embeddings = await initialized_client.get_embeddings_batch(
            texts,
            max_concurrency=2,
        )

        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_preserves_order(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that batch embedding preserves input order.

        Verifies that concurrent processing returns results in the
        same order as the input texts.
        """
        texts = ["first", "second", "third", "fourth", "fifth"]

        # Get batch embeddings
        batch_embeddings = await initialized_client.get_embeddings_batch(
            texts,
            max_concurrency=5,
        )

        # Get individual embeddings in order
        individual_embeddings = []
        for text in texts:
            emb = await initialized_client.get_embedding(text)
            individual_embeddings.append(emb)

        # Results should match
        assert len(batch_embeddings) == len(individual_embeddings)
        for batch_emb, ind_emb in zip(
            batch_embeddings, individual_embeddings, strict=True
        ):
            for v1, v2 in zip(batch_emb, ind_emb, strict=True):
                assert abs(v1 - v2) < 1e-6, "Batch should match individual embeddings"

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_concurrency(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that batch processing respects concurrency limits.

        Verifies that max_concurrency controls parallel requests.
        This is verified indirectly by ensuring the operation completes
        successfully with different concurrency levels.
        """
        texts = ["text1", "text2", "text3", "text4"]

        # Test with concurrency=1 (sequential)
        embeddings_seq = await initialized_client.get_embeddings_batch(
            texts,
            max_concurrency=1,
        )

        # Test with concurrency=4 (parallel)
        embeddings_par = await initialized_client.get_embeddings_batch(
            texts,
            max_concurrency=4,
        )

        # Both should produce same results
        assert len(embeddings_seq) == len(embeddings_par)
        for seq_emb, par_emb in zip(embeddings_seq, embeddings_par, strict=True):
            for v1, v2 in zip(seq_emb, par_emb, strict=True):
                assert abs(v1 - v2) < 1e-6, (
                    "Sequential and parallel should produce same results"
                )


# =============================================================================
# Health Check Tests
# =============================================================================


class TestRealHealthCheck:
    """Tests for health check with real server."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test health check returns True for healthy server.

        Verifies that health_check correctly determines server health.
        """
        is_healthy = await initialized_client.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_with_correlation_id(
        self,
        initialized_client: EmbeddingHttpClient,
        correlation_id: UUID,
    ) -> None:
        """Test health check accepts correlation ID."""
        is_healthy = await initialized_client.health_check(
            correlation_id=correlation_id,
        )

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_invalid_endpoint(
        self,
        embedding_server_available: bool,
    ) -> None:
        """Test health check returns False for invalid endpoint.

        Verifies that health_check correctly reports unhealthy status
        when the server cannot be reached.
        """
        if not embedding_server_available:
            pytest.skip("Embedding server is not available")

        # Config pointing to non-existent server
        config = ModelEmbeddingHttpClientConfig(
            provider=EnumEmbeddingProviderType.LOCAL,
            base_url="http://localhost:9999",  # Unlikely to be running
            model="test-model",
            embedding_dimension=1024,
            timeout_seconds=2.0,  # Short timeout
        )

        async with EmbeddingHttpClient(config) as client:
            is_healthy = await client.health_check()

        assert is_healthy is False


# =============================================================================
# Envelope Format Tests
# =============================================================================


class TestEnvelopeFormat:
    """Tests to verify envelope format correctness for real HTTP calls.

    These tests verify that the envelope structure passed to HandlerHttpRest
    is correct and results in successful HTTP operations.
    """

    @pytest.mark.asyncio
    async def test_envelope_produces_successful_response(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that envelope format results in successful embedding.

        The fact that we get a valid embedding back confirms the envelope
        format is correct for the real HandlerHttpRest.
        """
        embedding = await initialized_client.get_embedding("envelope test")

        # Success means envelope format is correct
        assert isinstance(embedding, list)
        assert len(embedding) == get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_envelope_with_various_text_lengths(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test envelope handles various text lengths correctly.

        Verifies the envelope correctly transmits text of different sizes.
        """
        test_cases = [
            "a",  # Single character
            "short",  # Short word
            "This is a medium length sentence.",  # Medium
            "x" * 1000,  # Long text (1000 chars)
        ]

        for text in test_cases:
            embedding = await initialized_client.get_embedding(text)
            assert len(embedding) == get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_correlation_id_passed_through(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that correlation ID is properly included in envelope.

        While we cannot directly inspect the envelope sent to the server,
        we verify that providing a correlation_id does not break the request
        and the response is still valid.
        """
        custom_id = uuid4()

        embedding = await initialized_client.get_embedding(
            "correlation test",
            correlation_id=custom_id,
        )

        assert isinstance(embedding, list)
        assert len(embedding) == get_embedding_dimension()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestRealErrorHandling:
    """Tests for error handling with real server."""

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that empty text raises EmbeddingClientError.

        This validation happens client-side before the HTTP call.
        """
        with pytest.raises(EmbeddingClientError, match="Text cannot be empty"):
            await initialized_client.get_embedding("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises_error(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that whitespace-only text raises EmbeddingClientError."""
        with pytest.raises(EmbeddingClientError, match="Text cannot be empty"):
            await initialized_client.get_embedding("   ")

    @pytest.mark.asyncio
    async def test_operations_auto_initialize(
        self,
        embedding_server_available: bool,
        local_config: ModelEmbeddingHttpClientConfig,
    ) -> None:
        """Test that operations auto-initialize if not already initialized.

        Verifies the client can handle being used without explicit initialize().
        """
        if not embedding_server_available:
            pytest.skip("Embedding server is not available")

        client = EmbeddingHttpClient(local_config)

        # Not initialized
        assert not client.is_initialized

        # Call get_embedding without initialize - should auto-initialize
        embedding = await client.get_embedding("auto init test")

        # Should now be initialized
        assert client.is_initialized
        assert isinstance(embedding, list)

        # Cleanup
        await client.shutdown()


# =============================================================================
# Performance Tests
# =============================================================================


class TestRealPerformance:
    """Tests for performance characteristics with real server.

    These tests verify that the integration performs within acceptable
    bounds and can handle realistic workloads.
    """

    @pytest.mark.asyncio
    async def test_sequential_requests_complete(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that sequential requests complete without timeout.

        Verifies the connection remains healthy across multiple requests.
        """
        for i in range(5):
            embedding = await initialized_client.get_embedding(f"sequential test {i}")
            assert len(embedding) == get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_concurrent_requests_complete(
        self,
        initialized_client: EmbeddingHttpClient,
    ) -> None:
        """Test that concurrent requests complete successfully.

        Verifies the client handles concurrent operations correctly.
        """
        texts = [f"concurrent test {i}" for i in range(10)]

        # Run all requests concurrently
        tasks = [initialized_client.get_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)

        assert len(embeddings) == 10
        for emb in embeddings:
            assert len(emb) == get_embedding_dimension()
