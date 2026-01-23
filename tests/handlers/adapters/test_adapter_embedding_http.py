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

from omnimemory.handlers.adapters.adapter_embedding_http import (
    EmbeddingClientError,
    EmbeddingConnectionError,
    EmbeddingHttpClient,
    EmbeddingTimeoutError,
    EnumEmbeddingProviderType,
    ModelEmbeddingHttpClientConfig,
)
from omnimemory.handlers.adapters.adapter_rate_limiter import (
    ModelRateLimiterConfig,
    ProviderRateLimiter,
)

pytestmark = pytest.mark.unit

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def local_config() -> ModelEmbeddingHttpClientConfig:
    """Create a local provider configuration."""
    return ModelEmbeddingHttpClientConfig(
        provider=EnumEmbeddingProviderType.LOCAL,
        base_url="http://192.168.86.201:8002",
        model="gte-qwen2",
        embedding_dimension=1024,
    )


@pytest.fixture
def openai_config() -> ModelEmbeddingHttpClientConfig:
    """Create an OpenAI provider configuration."""
    return ModelEmbeddingHttpClientConfig(
        provider=EnumEmbeddingProviderType.OPENAI,
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
# Protocol Conformance Tests
# =============================================================================


class TestProtocolConformance:
    """Tests verifying protocol conformance for embedding and rate limiting."""

    def test_embedding_http_client_implements_protocol(self) -> None:
        """Verify EmbeddingHttpClient conforms to ProtocolEmbeddingClient.

        Uses runtime_checkable Protocol to verify structural conformance.
        This ensures the adapter properly implements the contract boundary.
        """
        from omnimemory.protocols import ProtocolEmbeddingClient

        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8000",
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
        ):
            client = EmbeddingHttpClient(config)

            assert isinstance(
                client, ProtocolEmbeddingClient
            ), "EmbeddingHttpClient must implement ProtocolEmbeddingClient protocol"

    def test_provider_rate_limiter_implements_protocol(self) -> None:
        """Verify ProviderRateLimiter conforms to ProtocolRateLimiter.

        Uses runtime_checkable Protocol to verify structural conformance.
        This ensures the rate limiter properly implements the contract boundary.
        """
        from omnimemory.protocols import ProtocolRateLimiter

        config = ModelRateLimiterConfig(
            provider="test",
            model="test-model",
            requests_per_minute=60,
        )
        limiter = ProviderRateLimiter(config)

        assert isinstance(
            limiter, ProtocolRateLimiter
        ), "ProviderRateLimiter must implement ProtocolRateLimiter protocol"


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
        assert config.provider == EnumEmbeddingProviderType.LOCAL
        assert config.model == "gte-qwen2"
        assert config.timeout_seconds == 30.0
        assert config.embedding_dimension == 1024
        assert config.strict_dimension_validation is False
        assert config.rate_limit_rpm == 0
        assert config.auth_header is None
        assert config.health_check_text == "health"

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

    def test_embed_endpoint_path_normalized(self) -> None:
        """Test embed_endpoint_path without leading slash is normalized."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8000",
            embed_endpoint_path="custom/embed",
        )
        # Validator should prepend /
        assert config.embed_endpoint_path == "/custom/embed"
        # Full endpoint should be correct
        assert config.embed_endpoint == "http://localhost:8000/custom/embed"

    def test_embed_endpoint_path_just_slash(self) -> None:
        """Test embed_endpoint_path handles just '/' correctly."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8000",
            embed_endpoint_path="/",
        )
        assert config.embed_endpoint_path == "/"
        assert config.embed_endpoint == "http://localhost:8000/"

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

    def test_health_check_text_validation(self) -> None:
        """Test health_check_text validation constraints."""
        # Valid custom text
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost",
            health_check_text="custom health check phrase",
        )
        assert config.health_check_text == "custom health check phrase"

        # Empty text should fail (min_length=1)
        with pytest.raises(ValueError):
            ModelEmbeddingHttpClientConfig(
                base_url="http://localhost",
                health_check_text="",
            )

        # Text exceeding max_length=100 should fail
        with pytest.raises(ValueError):
            ModelEmbeddingHttpClientConfig(
                base_url="http://localhost",
                health_check_text="x" * 101,
            )


class TestEnumEmbeddingProviderType:
    """Tests for EnumEmbeddingProviderType enum."""

    def test_from_string(self) -> None:
        """Test string to enum conversion."""
        assert (
            EnumEmbeddingProviderType.from_string("local")
            == EnumEmbeddingProviderType.LOCAL
        )
        assert (
            EnumEmbeddingProviderType.from_string("OPENAI")
            == EnumEmbeddingProviderType.OPENAI
        )
        assert (
            EnumEmbeddingProviderType.from_string("vllm")
            == EnumEmbeddingProviderType.VLLM
        )

    def test_from_string_invalid(self) -> None:
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            EnumEmbeddingProviderType.from_string("unknown")


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

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_max_concurrency_zero_raises(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test batch with max_concurrency=0 raises ValueError."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(
                    ValueError,
                    match=r"max_concurrency must be positive",
                ):
                    await client.get_embeddings_batch(["test"], max_concurrency=0)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_max_concurrency_negative_raises(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
    ) -> None:
        """Test batch with negative max_concurrency raises ValueError."""
        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                with pytest.raises(
                    ValueError,
                    match=r"max_concurrency must be positive.*got -5",
                ):
                    await client.get_embeddings_batch(["test"], max_concurrency=-5)


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

    def test_tpm_only_config_raises_validation_error(
        self,
    ) -> None:
        """Test that TPM-only configuration (rpm=0, tpm>0) is rejected.

        TPM-only rate limiting is not supported because the underlying
        ProviderRateLimiter requires a positive RPM to function correctly.
        The config should reject this invalid combination at validation time.
        """
        with pytest.raises(
            ValueError,
            match=r"TPM-only rate limiting is not supported.*rate_limit_rpm must also be > 0",
        ):
            ModelEmbeddingHttpClientConfig(
                base_url="http://localhost",
                rate_limit_rpm=0,  # No RPM limit
                rate_limit_tpm=100_000,  # Only TPM limit - INVALID
            )

    @pytest.mark.asyncio
    async def test_rate_limiter_not_created_when_both_rpm_and_tpm_zero(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test rate limiter is NOT created when both RPM and TPM are 0."""
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost",
            rate_limit_rpm=0,
            rate_limit_tpm=0,
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            client = EmbeddingHttpClient(config)
            assert client._rate_limiter is None


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

        # Create a mock rate limiter that we can verify was not called
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            # Inject the mock rate limiter via constructor
            # (rate_limiter param takes precedence over config-based creation)
            client = EmbeddingHttpClient(config, rate_limiter=mock_rate_limiter)
            await client.initialize()

            # Verify our mock rate limiter was used
            assert client._rate_limiter is mock_rate_limiter

            # Perform health check
            result = await client.health_check()
            assert result is True

            # Verify rate limiter was NOT called (health check bypasses it)
            mock_rate_limiter.acquire.assert_not_called()

            # Verify the actual HTTP request was still made
            mock_handler.execute.assert_called_once()

            await client.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_uses_default_test_text(
        self,
        local_config: ModelEmbeddingHttpClientConfig,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health check uses default test phrase when not configured."""
        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(local_config) as client:
                await client.health_check()

                # Verify the request used "health" as the default test text
                call_args = mock_handler.execute.call_args[0][0]
                assert call_args["payload"]["body"] == {"text": "health"}

    @pytest.mark.asyncio
    async def test_health_check_uses_custom_configured_text(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health check uses custom text from config.

        Some embedding providers may reject very short text like "health".
        This test verifies that the health_check_text config option allows
        users to customize the test phrase.
        """
        custom_text = "ping test for server health verification"
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            embedding_dimension=1024,
            health_check_text=custom_text,
        )
        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                await client.health_check()

                # Verify the request used the custom configured text
                call_args = mock_handler.execute.call_args[0][0]
                assert call_args["payload"]["body"] == {"text": custom_text}

    @pytest.mark.asyncio
    async def test_health_check_skips_dimension_validation(
        self,
        mock_handler: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test health check skips dimension validation even with mismatch.

        Health checks use arbitrary test text ("health") which may return
        embeddings with different dimensions than configured. The health
        check should NOT log warnings or raise errors for dimension mismatch.
        """
        # Config expects 1024 dimensions
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            embedding_dimension=1024,
            strict_dimension_validation=False,
        )

        # But health check response returns 512 dimensions (mismatch)
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"embedding": [0.1] * 512},  # Different dimension!
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Health check should succeed despite dimension mismatch
                is_healthy = await client.health_check()
                assert is_healthy is True

        # No dimension mismatch warning should be logged
        assert "dimension mismatch" not in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_health_check_skips_dimension_validation_strict_mode(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test health check skips dimension validation even in strict mode.

        Even when strict_dimension_validation is enabled, health checks should
        NOT raise errors for dimension mismatch since the test text may return
        different dimensions than real content.
        """
        # Config expects 1024 dimensions with STRICT validation
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            embedding_dimension=1024,
            strict_dimension_validation=True,  # Strict mode enabled
        )

        # But health check response returns 512 dimensions (mismatch)
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"embedding": [0.1] * 512},  # Different dimension!
            },
        }
        mock_handler.execute.return_value = result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Health check should succeed despite strict mode + mismatch
                is_healthy = await client.health_check()
                assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_works_when_rate_limit_exhausted(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health_check succeeds even when rate limit is fully exhausted.

        This is critical for Kubernetes liveness probes and load balancer health
        checks that must work regardless of the application's rate limit state.
        If health checks blocked when rate limited, infrastructure monitoring
        would incorrectly report the service as unhealthy.
        """
        # Config with very restrictive rate limit (1 RPM)
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            rate_limit_rpm=1,  # Only 1 request per minute allowed
        )

        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Verify rate limiter was created
                assert client._rate_limiter is not None
                rate_limiter = client._rate_limiter

                # Exhaust the rate limit by making one regular embedding call
                await client.get_embedding("exhaust the rate limit")

                # Verify rate limit is exhausted (0 remaining)
                remaining = rate_limiter.get_remaining()
                assert remaining == 0, f"Expected 0 remaining, got {remaining}"

                # Now verify health_check still works immediately
                # (if it consumed tokens, it would block waiting for rate limit reset)
                is_healthy = await client.health_check()
                assert is_healthy is True

                # Verify rate limit is still exhausted after health check
                # (health check didn't consume any tokens)
                remaining_after = rate_limiter.get_remaining()
                assert (
                    remaining_after == 0
                ), f"Expected 0 remaining after health check, got {remaining_after}"

    @pytest.mark.asyncio
    async def test_health_check_does_not_accumulate_in_rate_window(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test multiple health_check calls don't accumulate in rate window.

        The rate limiter's sliding window should remain unchanged after
        multiple health checks. This ensures health monitoring doesn't
        gradually consume rate budget over time.
        """
        # Config with rate limiting
        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
            rate_limit_rpm=10,  # 10 RPM to make counting clear
        )

        mock_handler.execute.return_value = mock_handler_result

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                assert client._rate_limiter is not None
                rate_limiter = client._rate_limiter

                # Check initial state - should have full capacity
                initial_remaining = rate_limiter.get_remaining()
                assert initial_remaining == 10

                # Perform multiple health checks
                for _ in range(5):
                    result = await client.health_check()
                    assert result is True

                # Verify rate window is still at full capacity
                # (none of the health checks consumed tokens)
                final_remaining = rate_limiter.get_remaining()
                assert final_remaining == initial_remaining, (
                    f"Rate window changed from {initial_remaining} to {final_remaining} "
                    "after health checks - health checks should not consume tokens"
                )

                # Also verify the internal window is empty (no entries added)
                window_length = len(rate_limiter._request_window)
                assert (
                    window_length == 0
                ), f"Expected empty rate window, but found {window_length} entries"

    @pytest.mark.asyncio
    async def test_health_check_with_correlation_id_for_tracing(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health_check properly passes correlation_id for distributed tracing.

        Infrastructure monitoring tools often need to correlate health check
        requests with their responses across distributed systems. This test
        verifies the correlation_id is properly propagated to the HTTP handler.
        """
        mock_handler.execute.return_value = mock_handler_result
        custom_cid = uuid4()

        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Call health_check with explicit correlation_id
                result = await client.health_check(correlation_id=custom_cid)
                assert result is True

                # Verify correlation_id was passed to the HTTP handler
                call_args = mock_handler.execute.call_args[0][0]
                assert call_args["correlation_id"] == custom_cid

    @pytest.mark.asyncio
    async def test_health_check_generates_correlation_id_when_not_provided(
        self,
        mock_handler: MagicMock,
        mock_handler_result: MagicMock,
    ) -> None:
        """Test health_check generates a correlation_id when none is provided.

        Even when the caller doesn't provide a correlation_id, the health check
        should generate one for internal tracing purposes.
        """
        from uuid import UUID

        mock_handler.execute.return_value = mock_handler_result

        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Call health_check without correlation_id
                result = await client.health_check()
                assert result is True

                # Verify a correlation_id was generated and passed
                call_args = mock_handler.execute.call_args[0][0]
                assert "correlation_id" in call_args
                assert isinstance(call_args["correlation_id"], UUID)

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_http_error_status(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test health_check returns False (not raises) on HTTP error status.

        When the embedding server returns an error status code (e.g., 500, 503),
        health_check should return False rather than raising an exception.
        This allows infrastructure monitoring to handle the unhealthy state gracefully.
        """
        # Simulate server returning 503 Service Unavailable
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 503,
                "body": {"error": "Service temporarily unavailable"},
            },
        }
        mock_handler.execute.return_value = result

        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Should return False, not raise
                is_healthy = await client.health_check()
                assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_invalid_response_format(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test health_check returns False on malformed server response.

        If the embedding server returns a response that cannot be parsed
        (e.g., missing embedding field), health_check should return False
        rather than raising an exception.
        """
        # Simulate server returning unexpected format
        result = MagicMock()
        result.result = {
            "status": "success",
            "payload": {
                "status_code": 200,
                "body": {"unexpected": "format", "no_embedding_here": True},
            },
        }
        mock_handler.execute.return_value = result

        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Should return False, not raise
                is_healthy = await client.health_check()
                assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_timeout(
        self,
        mock_handler: MagicMock,
    ) -> None:
        """Test health_check returns False on timeout error.

        When the embedding server times out, health_check should return False
        rather than raising an exception, allowing callers to handle the
        unhealthy state appropriately.
        """
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

        config = ModelEmbeddingHttpClientConfig(
            base_url="http://localhost:8002",
        )

        with patch(
            "omnimemory.handlers.adapters.adapter_embedding_http.HandlerHttpRest",
            return_value=mock_handler,
        ):
            async with EmbeddingHttpClient(config) as client:
                # Should return False, not raise
                is_healthy = await client.health_check()
                assert is_healthy is False
