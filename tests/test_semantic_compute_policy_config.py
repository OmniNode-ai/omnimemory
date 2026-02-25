# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for semantic compute policy configuration model.

These tests validate that the ModelSemanticComputePolicyConfig:
- Validates cross-field constraints correctly
- Rejects invalid configurations with clear error messages
- Accepts valid configurations
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnimemory.models.config.model_semantic_compute_policy_config import (
    ModelSemanticComputePolicyConfig,
)


class TestModelSemanticComputePolicyConfigDefaults:
    """Tests for default values and basic construction."""

    def test_default_construction(self) -> None:
        """Test that default values are valid and construct successfully."""
        config = ModelSemanticComputePolicyConfig()
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 200
        assert config.retry_base_delay_ms == 100
        assert config.retry_max_delay_ms == 5000
        assert config.default_embedding_model == "gte-qwen2"
        assert config.default_llm_model == "qwen2.5-14b"

    def test_default_embedding_model_in_allowed_list(self) -> None:
        """Test that default embedding model is in the allowed list."""
        config = ModelSemanticComputePolicyConfig()
        assert config.default_embedding_model in config.allowed_embedding_models

    def test_default_llm_model_in_allowed_list(self) -> None:
        """Test that default LLM model is in the allowed list."""
        config = ModelSemanticComputePolicyConfig()
        assert config.default_llm_model in config.allowed_llm_models


class TestChunkOverlapValidation:
    """Tests for chunk_overlap < chunk_size validation."""

    def test_valid_chunk_overlap(self) -> None:
        """Test that valid chunk_overlap is accepted."""
        config = ModelSemanticComputePolicyConfig(
            chunk_size=2000,
            chunk_overlap=500,
        )
        assert config.chunk_overlap == 500
        assert config.chunk_size == 2000

    def test_chunk_overlap_zero_is_valid(self) -> None:
        """Test that zero chunk_overlap is valid."""
        config = ModelSemanticComputePolicyConfig(
            chunk_size=2000,
            chunk_overlap=0,
        )
        assert config.chunk_overlap == 0

    def test_chunk_overlap_equal_to_chunk_size_rejected(self) -> None:
        """Test that chunk_overlap equal to chunk_size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                chunk_size=1000,
                chunk_overlap=1000,
            )
        error_str = str(exc_info.value)
        assert "chunk_overlap" in error_str
        assert "chunk_size" in error_str
        assert "less than" in error_str

    def test_chunk_overlap_greater_than_chunk_size_rejected(self) -> None:
        """Test that chunk_overlap greater than chunk_size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                chunk_size=500,
                chunk_overlap=600,
            )
        error_str = str(exc_info.value)
        assert "chunk_overlap (600)" in error_str
        assert "chunk_size (500)" in error_str

    def test_chunk_overlap_one_less_than_chunk_size_valid(self) -> None:
        """Test that chunk_overlap one less than chunk_size is valid."""
        config = ModelSemanticComputePolicyConfig(
            chunk_size=1000,
            chunk_overlap=999,
        )
        assert config.chunk_overlap == 999


class TestRetryDelayValidation:
    """Tests for retry_max_delay_ms >= retry_base_delay_ms validation."""

    def test_valid_retry_delays(self) -> None:
        """Test that valid retry delays are accepted."""
        config = ModelSemanticComputePolicyConfig(
            retry_base_delay_ms=100,
            retry_max_delay_ms=5000,
        )
        assert config.retry_base_delay_ms == 100
        assert config.retry_max_delay_ms == 5000

    def test_equal_retry_delays_valid(self) -> None:
        """Test that equal retry delays are valid (max >= base)."""
        config = ModelSemanticComputePolicyConfig(
            retry_base_delay_ms=1000,
            retry_max_delay_ms=1000,
        )
        assert config.retry_base_delay_ms == config.retry_max_delay_ms

    def test_max_delay_less_than_base_delay_rejected(self) -> None:
        """Test that max_delay less than base_delay is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                retry_base_delay_ms=5000,
                retry_max_delay_ms=1000,
            )
        error_str = str(exc_info.value)
        assert "retry_max_delay_ms (1000)" in error_str
        assert "retry_base_delay_ms (5000)" in error_str
        assert ">=" in error_str


class TestEmbeddingModelValidation:
    """Tests for default_embedding_model in allowed_embedding_models validation."""

    def test_default_model_in_allowed_list(self) -> None:
        """Test that default model in allowed list is accepted."""
        config = ModelSemanticComputePolicyConfig(
            allowed_embedding_models=["model-a", "model-b", "model-c"],
            default_embedding_model="model-b",
        )
        assert config.default_embedding_model == "model-b"

    def test_default_model_not_in_allowed_list_rejected(self) -> None:
        """Test that default model not in allowed list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                allowed_embedding_models=["model-a", "model-b"],
                default_embedding_model="model-x",
            )
        error_str = str(exc_info.value)
        assert "default_embedding_model 'model-x'" in error_str
        assert "allowed_embedding_models" in error_str

    def test_empty_allowed_list_with_default_rejected(self) -> None:
        """Test that empty allowed list with any default is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                allowed_embedding_models=[],
                default_embedding_model="any-model",
            )
        error_str = str(exc_info.value)
        assert "default_embedding_model" in error_str

    def test_custom_embedding_models_accepted(self) -> None:
        """Test that custom embedding models are accepted when consistent."""
        config = ModelSemanticComputePolicyConfig(
            allowed_embedding_models=["custom-embed-v1", "custom-embed-v2"],
            default_embedding_model="custom-embed-v1",
        )
        assert config.default_embedding_model == "custom-embed-v1"
        assert len(config.allowed_embedding_models) == 2


class TestLlmModelValidation:
    """Tests for default_llm_model in allowed_llm_models validation."""

    def test_default_llm_in_allowed_list(self) -> None:
        """Test that default LLM in allowed list is accepted."""
        config = ModelSemanticComputePolicyConfig(
            allowed_llm_models=["gpt-4", "claude-3", "llama-3"],
            default_llm_model="claude-3",
        )
        assert config.default_llm_model == "claude-3"

    def test_default_llm_not_in_allowed_list_rejected(self) -> None:
        """Test that default LLM not in allowed list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                allowed_llm_models=["gpt-4", "claude-3"],
                default_llm_model="gemini-pro",
            )
        error_str = str(exc_info.value)
        assert "default_llm_model 'gemini-pro'" in error_str
        assert "allowed_llm_models" in error_str

    def test_empty_allowed_llm_list_rejected(self) -> None:
        """Test that empty allowed LLM list with any default is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSemanticComputePolicyConfig(
                allowed_llm_models=[],
                default_llm_model="any-llm",
            )
        error_str = str(exc_info.value)
        assert "default_llm_model" in error_str


class TestMultipleValidationErrors:
    """Tests for scenarios with multiple validation errors."""

    def test_multiple_errors_all_reported(self) -> None:
        """Test that configs with multiple errors raise validation error."""
        # This config has multiple issues - Pydantic will catch the first one
        with pytest.raises(ValidationError):
            ModelSemanticComputePolicyConfig(
                chunk_size=100,
                chunk_overlap=200,  # Invalid: >= chunk_size
                retry_base_delay_ms=5000,
                retry_max_delay_ms=100,  # Invalid: < base
            )


class TestValidComplexConfigurations:
    """Tests for valid complex configurations."""

    def test_minimal_valid_config(self) -> None:
        """Test minimal valid configuration with all constraints satisfied."""
        config = ModelSemanticComputePolicyConfig(
            chunk_size=100,  # minimum
            chunk_overlap=0,  # minimum, and < chunk_size
            retry_base_delay_ms=10,  # minimum
            retry_max_delay_ms=100,  # minimum, and >= base
            allowed_embedding_models=["embed-1"],
            default_embedding_model="embed-1",
            allowed_llm_models=["llm-1"],
            default_llm_model="llm-1",
        )
        assert config.chunk_overlap < config.chunk_size
        assert config.retry_max_delay_ms >= config.retry_base_delay_ms

    def test_maximal_valid_config(self) -> None:
        """Test configuration with maximum values that are still valid."""
        config = ModelSemanticComputePolicyConfig(
            chunk_size=16000,  # maximum
            chunk_overlap=1000,  # maximum, but still < chunk_size
            retry_base_delay_ms=10000,  # maximum
            retry_max_delay_ms=60000,  # maximum, and >= base
            allowed_embedding_models=["embed-1", "embed-2", "embed-3"],
            default_embedding_model="embed-2",
            allowed_llm_models=["llm-1", "llm-2"],
            default_llm_model="llm-1",
        )
        assert config.chunk_size == 16000
        assert config.chunk_overlap == 1000

    def test_config_is_frozen(self) -> None:
        """Test that configuration is immutable (frozen)."""
        config = ModelSemanticComputePolicyConfig()
        with pytest.raises(ValidationError):
            config.chunk_size = 5000  # type: ignore[misc]
