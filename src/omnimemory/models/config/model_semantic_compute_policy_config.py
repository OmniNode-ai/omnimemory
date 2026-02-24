# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Semantic compute policy configuration model following ONEX standards.

Defines configuration knobs for semantic analysis, embedding generation,
and entity extraction operations. Used by SemanticComputePolicy for decision logic.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...enums.enum_entity_extraction_mode import EnumEntityExtractionMode


class ModelSemanticComputePolicyConfig(BaseModel):
    """Configuration for semantic compute policy decisions.

    This config defines the operational parameters for semantic analysis.
    The SemanticComputePolicy class uses this config to make runtime decisions.

    Example:
        >>> config = ModelSemanticComputePolicyConfig(
        ...     max_tokens_per_request=8000,
        ...     cache_embeddings=True,
        ... )
        >>> config.entity_extraction_mode
        <EnumEntityExtractionMode.DETERMINISTIC: 'deterministic'>
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    # =========================================================================
    # Token and Size Limits
    # =========================================================================

    max_tokens_per_request: int = Field(
        default=4000,
        ge=100,
        le=128000,
        description="Maximum tokens per LLM request for semantic analysis",
    )

    max_content_length: int = Field(
        default=100000,
        ge=100,
        le=10000000,
        description="Maximum content length in characters before chunking",
    )

    chunk_size: int = Field(
        default=2000,
        ge=100,
        le=16000,
        description="Size of content chunks for processing large content",
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks to preserve context at boundaries",
    )

    # =========================================================================
    # Timeout and Retry Configuration
    # =========================================================================

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Hard timeout for semantic operations in seconds",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient failures",
    )

    retry_base_delay_ms: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Base delay for exponential backoff in milliseconds",
    )

    retry_max_delay_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Maximum delay between retries in milliseconds",
    )

    # =========================================================================
    # Fallback and Resilience
    # =========================================================================

    fallback_on_timeout: bool = Field(
        default=True,
        description="Return partial/cached results on timeout instead of failing",
    )

    fallback_on_provider_error: bool = Field(
        default=True,
        description="Use fallback provider chain on primary provider error",
    )

    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Consecutive failures before opening circuit breaker",
    )

    circuit_breaker_reset_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Time before attempting to close circuit breaker",
    )

    # =========================================================================
    # Caching Configuration
    # =========================================================================

    cache_embeddings: bool = Field(
        default=True,
        description="Cache embedding results for identical content",
    )

    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds (not yet implemented; reserved for future TTL-based cache expiration)",
    )

    cache_key_include_model: bool = Field(
        default=True,
        description="Include model name in cache key (different models = different cache)",
    )

    # =========================================================================
    # Entity Extraction Configuration
    # =========================================================================

    entity_extraction_mode: EnumEntityExtractionMode = Field(
        default=EnumEntityExtractionMode.DETERMINISTIC,
        description="Mode for entity extraction (deterministic vs best_effort)",
    )

    entity_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for entity extraction",
    )

    heuristic_entity_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence score assigned to heuristic-extracted entities",
    )

    max_entities_per_request: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum entities to extract per request",
    )

    # =========================================================================
    # LLM Configuration (for LLM-backed operations)
    # =========================================================================

    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0 for deterministic, higher for creative)",
    )

    llm_seed: int | None = Field(
        default=42,
        description="LLM seed for reproducibility (None for non-deterministic)",
    )

    # =========================================================================
    # Model/Provider Constraints
    # =========================================================================

    allowed_embedding_models: list[str] = Field(
        default_factory=lambda: ["text-embedding-3-small", "gte-qwen2"],
        description="List of allowed embedding model identifiers",
    )

    default_embedding_model: str = Field(
        default="gte-qwen2",
        description="Default embedding model to use",
    )

    allowed_llm_models: list[str] = Field(
        default_factory=lambda: ["qwen2.5-72b", "qwen2.5-14b"],
        description="List of allowed LLM model identifiers",
    )

    default_llm_model: str = Field(
        default="qwen2.5-14b",
        description="Default LLM model for semantic analysis",
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> Self:
        """Validate constraints that depend on multiple fields.

        Validates:
        - chunk_overlap must be less than chunk_size
        - retry_max_delay_ms must be >= retry_base_delay_ms
        - default_embedding_model must be in allowed_embedding_models
        - default_llm_model must be in allowed_llm_models
        """
        # Validate chunk_overlap < chunk_size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        # Validate retry_max_delay_ms >= retry_base_delay_ms
        if self.retry_max_delay_ms < self.retry_base_delay_ms:
            raise ValueError(
                f"retry_max_delay_ms ({self.retry_max_delay_ms}) must be >= "
                f"retry_base_delay_ms ({self.retry_base_delay_ms})"
            )

        # Validate default_embedding_model in allowed_embedding_models
        if self.default_embedding_model not in self.allowed_embedding_models:
            raise ValueError(
                f"default_embedding_model '{self.default_embedding_model}' is not in "
                f"allowed_embedding_models: {self.allowed_embedding_models}"
            )

        # Validate default_llm_model in allowed_llm_models
        if self.default_llm_model not in self.allowed_llm_models:
            raise ValueError(
                f"default_llm_model '{self.default_llm_model}' is not in "
                f"allowed_llm_models: {self.allowed_llm_models}"
            )

        return self

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def is_deterministic(self) -> bool:
        """Return True if all operations are configured for determinism."""
        return (
            self.entity_extraction_mode == EnumEntityExtractionMode.DETERMINISTIC
            and self.llm_temperature == 0.0
            and self.llm_seed is not None
        )

    @property
    def effective_llm_temperature(self) -> float:
        """Return effective LLM temperature based on extraction mode."""
        if self.entity_extraction_mode == EnumEntityExtractionMode.DETERMINISTIC:
            return 0.0
        return self.llm_temperature

    @property
    def effective_llm_seed(self) -> int | None:
        """Return effective LLM seed based on extraction mode."""
        if self.entity_extraction_mode == EnumEntityExtractionMode.DETERMINISTIC:
            return self.llm_seed if self.llm_seed is not None else 42
        return None
