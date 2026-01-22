# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EmbeddingHttpClient.

This module tests the embedding HTTP client adapter that wraps HandlerHttp
for embedding API calls with rate limiting and correlation ID tracking.

Test Categories:
    - Configuration: Config validation and defaults
    - Lifecycle: Initialize, shutdown, context manager
    - Embedding: get_embedding with various providers
    - Batch: get_embeddings_batch concurrent processing
    - Error Handling: Connection and timeout errors
    - Rate Limiting: Integration with ProviderRateLimiter

Usage:
    pytest tests/handlers/adapters/test_adapter_embedding_http.py -v
    pytest tests/handlers/adapters/ -v -k "embedding_http"

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Skip all tests if omnibase_infra is not installed
pytest.importorskip(
    "omnibase_infra", reason="omnibase_infra required for adapter tests"
)

from omnimemory.handlers.adapters.adapter_embedding_http import (
    EmbeddingClientError,
    EmbeddingConnectionError,
    EmbeddingHttpClient,
    EmbeddingProviderType,
    EmbeddingTimeoutError,
    ModelEmbeddingHttpClientConfig,
)
from omnimemory.handlers.adapters.adapter_rate_limiter import (
    ModelRateLimiterConfig,
    ProviderRateLimiter,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def local_config() -> ModelEmbeddingHttpClientConfig:
    """Create a local provider configuration."""
    return ModelEmbeddingHttpClientConfig(
        provider=EmbeddingProviderType.LOCAL,
        base_url="http://192.168.86.201:8002",
        model="gte-qwen2",
        embedding_dimension=1024,
    )


@pytest.fixture
def openai_config() -> ModelEmbeddingHttpClientConfig:
    """Create an OpenAI provider configuration."""
    return ModelEmbeddingHttpClientConfig(
        provider=EmbeddingProviderType.OPENAI,
        base_url="https://api.openai.com",
        model="text-embedding-3-small",
        embedding_dimension=1536,
        auth_header="Bearer test-key",
        rate_limit_rpm=60,
    )


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create a mock HandlerHttpRest.

    Returns:
        MagicMock configured with async methods matching HandlerHttpRest interface.
    """
    handler = MagicMock()
    handler.initialize = AsyncMock()
    handler.shutdown = AsyncMock()
    handler.execute = AsyncMock()
    return handler


@pytest.fixture
def mock_handler_result() -> MagicMock:
    """Create a mock handler result with embedding response."""
    result = MagicMock()
    result.result = {
        "status": "success",
        "payload": {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body": {"embedding": [0.1] * 1024},
        },
    }
    return result


@pytest.fixture
def mock_openai_result() -> MagicMock:
    """Create a mock handler result with OpenAI embedding response."""
    result = MagicMock()
    result.result = {
        "status": "success",
        "payload": {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body": {
                "data": [{"embedding": [0.1] * 1536, "index": 0}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        },
    }
    return result


# =============================================================================
# Configuration Tests
# =============================================================================


class TestModelEmbeddingHttpClientConfig:
    """Tests for ModelEmbeddingHttpClientConfig validation."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8000",
        )
        assert config.provider == EmbeddingProviderType.LOCAL
        assert config.model == "gte-qwen2"
        assert config.timeout_seconds == 30.0
        assert config.embedding_dimension == 1024
        assert config.strict_dimension_validation is False
        assert config.rate_limit_rpm == 0
        assert config.auth_header is None

    def test_url_normalization(self) -> None:
        """Test URL trailing slash is stripped."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8000/",
        )
        assert config.base_url == "http://localhost:8000"

    def test_embed_endpoint_local(
        self, local_config: ModelEmbeddingHttpClientConfig
    ) -> None:
        """Test embed endpoint for local provider."""
        assert local_config.embed_endpoint == "http://192.168.86.201:8002/embed"

    def test_embed_endpoint_openai(
        self, openai_config: ModelEmbeddingHttpClientConfig
    ) -> None:
        """Test embed endpoint for OpenAI provider."""
        assert openai_config.embed_endpoint == "https://api.openai.com/v1/embeddings"

    def test_validation_timeout_bounds(self) -> None:
        """Test timeout validation bounds."""
        with pytest.raises(ValueError):
            ModelEmbeddingHttpClientConfig(
                base_url="http://localhost",
                timeout_seconds=0,
            )

        with pytest.raises(ValueError):
            ModelEmbeddingHttpClientConfig(
                base_url="http://localhost",
                timeout_seconds=500,
            )


class TestEmbeddingProviderType:
    """Tests for EmbeddingProviderType enum."""

    def test_from_string(self) -> None:
        """Test string to enum conversion."""
        assert EmbeddingProviderType.from_string("local") == EmbeddingProviderType.LOCAL
        assert (
            EmbeddingProviderType.from_string("OPENAI") == EmbeddingProviderType.OPENAI
        )
        assert EmbeddingProviderType.from_string("vllm") == EmbeddingProviderType.VLLM

    def test_from_string_invalid(self) -> None:
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            EmbeddingProviderType.from_string("unknown")


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestEmbeddingHttpClientLifecycle:
    """Tests for EmbeddingHttpClient lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialize(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test client initialization."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(local_config)
            assert not client.is_initialized

            await client.initialize()
            assert client.is_initialized
            mock_handler.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test initialize is idempotent."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(local_config)
            await client.initialize()
            await client.initialize()
            # Should only be called once
            assert mock_handler.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test client shutdown."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(local_config)
            await client.initialize()
            await client.shutdown()

            assert not client.is_initialized
            mock_handler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test async context manager."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                assert client.is_initialized

            assert not client.is_initialized


# =============================================================================
# Embedding Tests
# =============================================================================


class TestEmbeddingHttpClientEmbedding:
    """Tests for get_embedding functionality."""

    @pytest.mark.asyncio
    async def test_get_embedding_local(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test get_embedding with local provider."""
        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                embedding = await client.get_embedding("Hello world")

                assert len(embedding) == 1024
                mock_handler.execute.assert_called_once()

                # Verify envelope structure
                call_args = mock_handler.execute.call_args[0][0]
                assert call_args["operation"] == "http.post"
                assert call_args["payload"]["url"] == "http://192.168.86.201:8002/embed"
                assert call_args["payload"]["body"] == {"text": "Hello world"}

    @pytest.mark.asyncio
    async def test_get_embedding_openai(
        self,
        openai_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_openai_result: MagicMock,
    ) -> None:
        """Test get_embedding with OpenAI provider."""
        mock_handler.execute.return_value = mock_openai_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(openai_config) as client:
                embedding = await client.get_embedding("Hello world")

                assert len(embedding) == 1536

                # Verify OpenAI-specific envelope
                call_args = mock_handler.execute.call_args[0][0]
                assert (
                    call_args["payload"]["url"]
                    == "https://api.openai.com/v1/embeddings"
                )
                assert call_args["payload"]["body"]["model"] == "text-embedding-3-small"
                assert (
                    call_args["payload"]["headers"]["Authorization"]
                    == "Bearer test-key"
                )

    @pytest.mark.asyncio
    async def test_get_embedding_with_correlation_id(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test correlation ID is passed through."""
        mock_handler.execute.return_value = mock_handler_result
        cid = uuid4()

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                await client.get_embedding("test", correlation_id=cid)

                call_args = mock_handler.execute.call_args[0][0]
                assert call_args["correlation_id"] == cid

    @pytest.mark.asyncio
    async def test_get_embedding_empty_text_raises(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test empty text raises EmbeddingClientError."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(EmbeddingClientError, match="Text cannot be empty"):
                    await client.get_embedding("")

                with pytest.raises(EmbeddingClientError, match="Text cannot be empty"):
                    await client.get_embedding("   ")


# =============================================================================
# Batch Tests
# =============================================================================


class TestEmbeddingHttpClientBatch:
    """Tests for get_embeddings_batch functionality."""

    @pytest.mark.asyncio
    async def test_get_embeddings_batch(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test batch embedding generation."""
        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                texts = ["Hello", "World", "Test"]
                embeddings = await client.get_embeddings_batch(texts)

                assert len(embeddings) == 3
                assert all(len(e) == 1024 for e in embeddings)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_empty(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test batch with empty list returns empty list."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                embeddings = await client.get_embeddings_batch([])
                assert embeddings == []


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestEmbeddingHttpClientErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_connection_error(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test connection error is transformed."""
        from uuid import uuid4

        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.models.errors import ModelInfraErrorContext

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="http.post",
            target_name="http://localhost:8002/embed",
            correlation_id=uuid4(),
        )
        mock_handler.execute.side_effect = InfraConnectionError(
            "Connection refused", context=context
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(EmbeddingConnectionError, match="Connection failed"):
                    await client.get_embedding("test")

    @pytest.mark.asyncio
    async def test_timeout_error(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test timeout error is transformed."""
        from uuid import uuid4

        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraTimeoutError
        from omnibase_infra.models.errors import ModelTimeoutErrorContext

        context = ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="http.post",
            target_name="http://localhost:8002/embed",
            correlation_id=uuid4(),
            timeout_seconds=30.0,
        )
        mock_handler.execute.side_effect = InfraTimeoutError(
            "Request timed out", context=context
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(EmbeddingTimeoutError, match="Timeout after"):
                    await client.get_embedding("test")

    @pytest.mark.asyncio
    async def test_http_error_status(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test HTTP error status is handled."""
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 500,
                "body": {"error": "Internal server error"},
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(EmbeddingClientError, match="HTTP 500"):
                    await client.get_embedding("test")

    @pytest.mark.asyncio
    async def test_invalid_response_format(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test invalid response format raises error."""
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"unexpected": "format"},
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(
                    EmbeddingClientError,
                    match="Could not extract embedding",
                ):
                    await client.get_embedding("test")


# =============================================================================
# Dimension Validation Tests
# =============================================================================


class TestDimensionValidation:
    """Tests for strict_dimension_validation feature."""

    @pytest.mark.asyncio
    async def test_dimension_mismatch_warning_by_default(
        self,
        mock_handler: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test dimension mismatch logs warning when strict mode is disabled (default)."""
        # Config expects 1024 dimensions
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            embedding_dimension=1024,
            strict_dimension_validation=False,  # default
        )

        # But response returns 512 dimensions
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"embedding": [0.1] * 512},
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Should NOT raise, just warn
                embedding = await client.get_embedding("test")
                assert len(embedding) == 512  # Returns mismatched embedding

        # Verify warning was logged
        assert "dimension mismatch" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_when_strict(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test dimension mismatch raises error when strict mode is enabled."""
        # Config expects 1024 dimensions with strict validation
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            embedding_dimension=1024,
            strict_dimension_validation=True,
        )

        # But response returns 512 dimensions
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"embedding": [0.1] * 512},
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                with pytest.raises(
                    EmbeddingClientError,
                    match=r"dimension mismatch.*expected 1024.*got 512",
                ):
                    await client.get_embedding("test")

    @pytest.mark.asyncio
    async def test_correct_dimension_no_error_in_strict_mode(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test correct dimensions do not raise even in strict mode."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            embedding_dimension=1024,
            strict_dimension_validation=True,
        )

        # Response returns correct 1024 dimensions
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"embedding": [0.1] * 1024},
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                embedding = await client.get_embedding("test")
                assert len(embedding) == 1024

    def test_strict_dimension_validation_default_is_false(self) -> None:
        """Test strict_dimension_validation defaults to False."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
        )
        assert config.strict_dimension_validation is False


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestEmbeddingHttpClientRateLimiting:
    """Tests for rate limiting integration."""

    @pytest.mark.asyncio
    async def test_rate_limiter_created_from_config(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test rate limiter is created from config."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost",
            rate_limit_rpm=10,
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(config)
            assert client._rate_limiter is not None
            assert client._rate_limiter.config.requests_per_minute == 10

    @pytest.mark.asyncio
    async def test_rate_limiter_not_created_when_disabled(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test rate limiter is not created when disabled."""
        # local_config has rate_limit_rpm=0 by default
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(local_config)
            assert client._rate_limiter is None

    @pytest.mark.asyncio
    async def test_custom_rate_limiter(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test custom rate limiter can be provided."""
        limiter_config = ModelRateLimiterConfig(
            provider="custom",
            model="custom-model",
            requests_per_minute=5,
        )
        custom_limiter = ProviderRateLimiter(limiter_config)

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(local_config, rate_limiter=custom_limiter)
            assert client._rate_limiter is custom_limiter


# =============================================================================
# Health Check Tests
# =============================================================================


class TestEmbeddingHttpClientHealthCheck:
    """Tests for health_check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health check returns True on success."""
        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                result = await client.health_check()
                assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check returns False on error."""
        from uuid import uuid4

        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.models.errors import ModelInfraErrorContext

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="http.post",
            target_name="http://localhost:8002/embed",
            correlation_id=uuid4(),
        )
        mock_handler.execute.side_effect = InfraConnectionError(
            "Connection failed", context=context
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                result = await client.health_check()
                assert result is False

    @pytest.mark.asyncio
    async def test_health_check_does_not_consume_rate_limit_tokens(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health check bypasses rate limiter and does not consume tokens.

        This is critical for infrastructure health checks (e.g., Kubernetes
        liveness probes) that should not impact the rate limit budget.
        """
        # Config with rate limiting enabled
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            rate_limit_rpm=60,
        )

        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(config)
            await client.initialize()

            # Verify rate limiter was created
            assert client._rate_limiter is not None

            # Mock the rate limiter's acquire method
            client._rate_limiter.acquire = AsyncMock()

            # Perform health check
            result = await client.health_check()
            assert result is True

            # Verify rate limiter was NOT called
            client._rate_limiter.acquire.assert_not_called()

            # Verify the actual HTTP request was still made
            mock_handler.execute.assert_called_once()

            await client.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_uses_minimal_test_text(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health check uses a minimal test phrase."""
        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                await client.health_check()

                # Verify the request used "health" as the test text
                call_args = mock_handler.execute.call_args[0][0]
                assert call_args["payload"]["body"] == {"text": "health"}
