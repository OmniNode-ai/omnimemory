# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Rate limiter configuration model following ONEX standards.

of external API calls. Used by ProviderRateLimiter in the adapters layer.

Example::

    from omnimemory.models.config import ModelRateLimiterConfig

    config = ModelRateLimiterConfig(
        provider="openai",
        model="text-embedding-3-small",
        requests_per_minute=60,
    )

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "DEFAULT_REQUESTS_PER_MINUTE",
    "ModelRateLimiterConfig",
]

# Constants for rate limiting
DEFAULT_REQUESTS_PER_MINUTE = 60


class ModelRateLimiterConfig(BaseModel):
    """Configuration for provider-scoped rate limiting.

    This config is used by ProviderRateLimiter instances which require at least
    1 request per minute to function. The ``requests_per_minute`` field enforces
    ``ge=1`` to ensure valid rate limiter operation.

    Note:
        **Constraint interaction with EmbeddingHttpClientConfig:**

        ModelEmbeddingHttpClientConfig allows ``rate_limit_rpm=0`` to indicate
        "no rate limiting." However, TPM-only configurations (rpm=0, tpm>0) are
        rejected at validation time because ProviderRateLimiter requires a
        positive RPM to function. Users must set ``rate_limit_rpm > 0`` when
        using token-based rate limiting.

    Attributes:
        provider: Provider identifier (e.g., "openai", "local", "vllm").
        model: Model identifier (e.g., "text-embedding-3-small").
        requests_per_minute: Maximum requests per minute (RPM). Must be >= 1.
        tokens_per_minute: Maximum tokens per minute (TPM). Set to 0 to disable.
        burst_multiplier: Allow burst up to this multiple of the rate limit.
            A multiplier of 1.0 means strict rate limiting.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        strict=True,
        from_attributes=True,
    )

    provider: str = Field(
        ...,
        min_length=1,
        description="Provider identifier (e.g., 'openai', 'local')",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier",
    )
    requests_per_minute: int = Field(
        default=DEFAULT_REQUESTS_PER_MINUTE,
        ge=1,
        le=10_000,
        description="Maximum requests per minute",
    )
    tokens_per_minute: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
        description="Maximum tokens per minute (0 to disable)",
    )
    burst_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description="Burst allowance multiplier",
    )
    initial_backoff_seconds: float = Field(
        default=0.1,
        ge=0.01,
        le=10.0,
        description="Initial backoff delay when rate limited (seconds)",
    )
    max_backoff_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Maximum backoff delay between retry attempts (seconds)",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Multiplier for exponential backoff (e.g., 2.0 = double each retry)",
    )

    @field_validator("provider", "model")
    @classmethod
    def normalize_identifier(cls, v: str) -> str:
        """Normalize identifiers to lowercase for consistent keying."""
        v = v.lower().strip()
        if v == "":
            raise ValueError("identifier must not be empty after normalization")
        return v

    @property
    def key(self) -> tuple[str, str]:
        """Get the (provider, model) key for this config."""
        return (self.provider, self.model)
