# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Rate limiter configuration model following ONEX standards.

This module provides the configuration model for provider-scoped rate limiting
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

# Constants for rate limiting
DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_TOKENS_PER_MINUTE = 150_000  # OpenAI default for embeddings


class ModelRateLimiterConfig(BaseModel):
    """Configuration for provider-scoped rate limiting.

    Attributes:
        provider: Provider identifier (e.g., "openai", "local", "vllm").
        model: Model identifier (e.g., "text-embedding-3-small").
        requests_per_minute: Maximum requests per minute (RPM).
        tokens_per_minute: Maximum tokens per minute (TPM). Set to 0 to disable.
        burst_multiplier: Allow burst up to this multiple of the rate limit.
            A multiplier of 1.0 means strict rate limiting.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
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

    @field_validator("provider", "model")
    @classmethod
    def normalize_identifier(cls, v: str) -> str:
        """Normalize identifiers to lowercase for consistent keying."""
        return v.lower().strip()

    @property
    def key(self) -> tuple[str, str]:
        """Get the (provider, model) key for this config."""
        return (self.provider, self.model)
