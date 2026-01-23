# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Embedding HTTP client configuration model following ONEX standards.

This module defines the configuration model for the HTTP-based embedding client,
supporting multiple providers (OpenAI, local vLLM, etc.) with integrated rate limiting.

Example::

    from omnimemory.models.config import (
        ModelEmbeddingHttpClientConfig,
        EnumEmbeddingProviderType,
    )

    config = ModelEmbeddingHttpClientConfig(
        provider=EnumEmbeddingProviderType.LOCAL,
        base_url="http://192.168.86.201:8002",
        model="gte-qwen2",
        embedding_dimension=1024,
    )

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "EnumEmbeddingProviderType",
    "ModelEmbeddingHttpClientConfig",
]


class EnumEmbeddingProviderType(str, Enum):
    """Supported embedding provider types.

    Attributes:
        LOCAL: Local embedding server (e.g., vLLM, text-embeddings-inference).
        OPENAI: OpenAI embeddings API.
        VLLM: Explicit vLLM server (uses same format as LOCAL).
    """

    LOCAL = "local"
    OPENAI = "openai"
    VLLM = "vllm"

    @classmethod
    def from_string(cls, value: str) -> EnumEmbeddingProviderType:
        """Convert string to provider type.

        Args:
            value: String representation of the provider type.

        Returns:
            The corresponding EnumEmbeddingProviderType enum value.

        Raises:
            ValueError: If the value does not match any provider type.
        """
        normalized = value.lower().strip()
        for provider in cls:
            if provider.value == normalized:
                return provider
        raise ValueError(f"Unknown provider type: {value}")


class ModelEmbeddingHttpClientConfig(BaseModel):
    """Configuration for the HTTP embedding client.

    This config defines connection and behavior parameters for embedding
    API calls, following the ONEX pattern of focused, single-responsibility
    config objects.

    Attributes:
        provider: Provider type (local, openai, vllm).
        base_url: Base URL of the embedding server. Required.
        model: Model identifier for the embedding model.
        timeout_seconds: Request timeout in seconds.
        embedding_dimension: Expected dimension of embedding vectors.
        strict_dimension_validation: If True, raise error on dimension mismatch;
            if False (default), log warning only.
        rate_limit_rpm: Requests per minute limit (0 for no limit).
        rate_limit_tpm: Tokens per minute limit (0 for no limit).
        auth_header: Optional authorization header value (e.g., "Bearer <token>").
        embed_endpoint_path: Custom embed endpoint path. If None, auto-detects
            based on provider (OpenAI: /v1/embeddings, Local/vLLM: /embed).

    Note:
        **Rate limiting configuration interaction:**

        - ``rpm=0, tpm=0``: No rate limiting applied.
        - ``rpm>0, tpm=0``: RPM-only limiting.
        - ``rpm>0, tpm>0``: Both RPM and TPM limiting.
        - ``rpm=0, tpm>0``: **NOT SUPPORTED** - raises validation error.
          You must set rate_limit_rpm > 0 when using TPM limiting.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        strict=True,
        from_attributes=True,
    )

    provider: EnumEmbeddingProviderType = Field(
        default=EnumEmbeddingProviderType.LOCAL,
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
    strict_dimension_validation: bool = Field(
        default=False,
        description="If True, raise error on dimension mismatch; if False, log warning only",
    )
    rate_limit_rpm: int = Field(
        default=0,
        ge=0,
        le=10_000,
        description=(
            "Requests per minute limit. Set to 0 to disable rate limiting entirely. "
            "Must be > 0 if rate_limit_tpm is set (TPM-only not supported)."
        ),
    )
    rate_limit_tpm: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
        description=(
            "Tokens per minute limit. Set to 0 to disable TPM limiting. "
            "Requires rate_limit_rpm > 0 (TPM-only not supported)."
        ),
    )
    auth_header: str | None = Field(
        default=None,
        description="Authorization header value (e.g., 'Bearer <token>')",
    )
    embed_endpoint_path: str | None = Field(
        default=None,
        description=(
            "Custom embed endpoint path (e.g., '/v1/embeddings'). "
            "If None, auto-detects based on provider: OpenAI uses /v1/embeddings, "
            "Local/vLLM uses /embed."
        ),
    )
    health_check_text: str = Field(
        default="health",
        min_length=1,
        max_length=100,
        description=(
            "Text to send for health check requests. Short text recommended "
            "to minimize token usage. Default 'health' (6 chars) chosen to work "
            "with most providers; some have minimum length requirements (e.g., "
            "certain providers reject text under 3-5 characters)."
        ),
    )

    @field_validator("base_url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        """Normalize URL by stripping trailing slashes."""
        return v.rstrip("/")

    @field_validator("embed_endpoint_path")
    @classmethod
    def validate_embed_endpoint_path(cls, v: str | None) -> str | None:
        """Ensure embed_endpoint_path starts with '/' to prevent malformed URLs."""
        if v is not None and not v.startswith("/"):
            return "/" + v
        return v

    @model_validator(mode="after")
    def validate_rate_limit_configuration(self) -> ModelEmbeddingHttpClientConfig:
        """Validate rate limit configuration is not TPM-only.

        TPM-only configurations (rate_limit_tpm > 0 with rate_limit_rpm == 0) are
        rejected because the underlying ProviderRateLimiter requires a positive RPM
        to function correctly. Users must explicitly set rate_limit_rpm > 0 when
        using token-based rate limiting.

        Raises:
            ValueError: If rate_limit_tpm > 0 but rate_limit_rpm == 0.
        """
        if self.rate_limit_tpm > 0 and self.rate_limit_rpm == 0:
            raise ValueError(
                f"TPM-only rate limiting is not supported. "
                f"When rate_limit_tpm > 0 (got {self.rate_limit_tpm}), "
                f"rate_limit_rpm must also be > 0. "
                f"Set rate_limit_rpm to a positive value or set rate_limit_tpm to 0."
            )
        return self

    @property
    def embed_endpoint(self) -> str:
        """Get the full URL for the embed endpoint.

        If ``embed_endpoint_path`` is set, uses that path directly.
        Otherwise auto-detects based on provider:

        - OpenAI: /v1/embeddings
        - Local/vLLM: /embed

        Returns:
            The full URL for the embedding endpoint.
        """
        if self.embed_endpoint_path is not None:
            return f"{self.base_url}{self.embed_endpoint_path}"
        if self.provider == EnumEmbeddingProviderType.OPENAI:
            return f"{self.base_url}/v1/embeddings"
        # Local and vLLM use /embed
        return f"{self.base_url}/embed"
