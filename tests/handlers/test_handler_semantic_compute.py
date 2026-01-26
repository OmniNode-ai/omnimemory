# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerSemanticCompute with mocked providers.

This module provides comprehensive testing for the semantic compute handler,
using fake provider implementations to test the handler logic in isolation.

Test Categories:
    1. TestFakeProviders: Verify fake providers work correctly
    2. TestEmbed: Embedding generation tests
    3. TestExtractEntities: Entity extraction tests
    4. TestAnalyze: Full semantic analysis tests
    5. TestPolicy: Policy decision logic tests
    6. TestCaching: Embedding cache behavior tests
    7. TestValidation: Input validation tests
    8. TestConfig: Configuration model tests

Usage:
    pytest tests/handlers/test_handler_semantic_compute.py -v
    pytest tests/handlers/test_handler_semantic_compute.py -v -k "embed"
    pytest tests/handlers/test_handler_semantic_compute.py -v -k "policy"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1390.

.. versionchanged:: 0.2.0
    Updated for container-driven pattern (OMN-1577).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from omnibase_core.container import ModelONEXContainer

from omnimemory.enums import EnumEntityExtractionMode, EnumSemanticEntityType
from omnimemory.handlers import (
    HandlerSemanticCompute,
    HandlerSemanticComputePolicy,
    ModelHandlerSemanticComputeConfig,
)
from omnimemory.models.config import ModelSemanticComputePolicyConfig
from omnimemory.models.intelligence import (
    ModelSemanticAnalysisResult,
    ModelSemanticEntity,
    ModelSemanticEntityList,
)

# =============================================================================
# Fake Provider Implementations
# =============================================================================


class FakeEmbeddingProvider:
    """Fake embedding provider for testing.

    Generates deterministic embeddings based on content hash.
    """

    def __init__(
        self,
        *,
        dimension: int = 1024,
        fail_on: set[str] | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        self._dimension = dimension
        self._fail_on = fail_on or set()
        self._latency_ms = latency_ms
        self._call_count = 0
        self._last_content: str | None = None

    @property
    def provider_name(self) -> str:
        return "fake-embedding"

    @property
    def model_name(self) -> str:
        return "fake-model-v1"

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_content(self) -> str | None:
        return self._last_content

    async def generate_embedding(
        self,
        text: str,
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> list[float]:
        self._call_count += 1
        self._last_content = text

        if text in self._fail_on:
            raise RuntimeError(f"Simulated failure for: {text}")

        # Generate deterministic embedding based on content
        hash_val = hash(text)
        return [((hash_val + i) % 1000) / 1000.0 for i in range(self._dimension)]

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        return [
            await self.generate_embedding(
                text, model=model, correlation_id=correlation_id
            )
            for text in texts
        ]

    async def health_check(self) -> bool:
        return True


class FakeLLMProvider:
    """Fake LLM provider for testing.

    Returns predefined responses for entity extraction.
    """

    def __init__(
        self,
        *,
        entities_response: dict | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self._entities_response = entities_response or {"entities": []}
        self._fail_on = fail_on or set()
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake-llm"

    @property
    def model_name(self) -> str:
        return "fake-llm-v1"

    @property
    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count

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
        self._call_count += 1
        if any(fail_text in prompt for fail_text in self._fail_on):
            raise RuntimeError("Simulated LLM failure")
        return "{}"

    async def complete_structured(
        self,
        prompt: str,
        output_schema: dict,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        self._call_count += 1
        if any(fail_text in prompt for fail_text in self._fail_on):
            raise RuntimeError("Simulated LLM failure")
        return self._entities_response

    async def health_check(self) -> bool:
        return True


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create an ONEX container for testing."""
    return ModelONEXContainer()


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    """Create a fake embedding provider for testing."""
    return FakeEmbeddingProvider(dimension=1024)


@pytest.fixture
def fake_llm_provider() -> FakeLLMProvider:
    """Create a fake LLM provider for testing."""
    return FakeLLMProvider(
        entities_response={
            "entities": [
                {
                    "type": "person",
                    "text": "John",
                    "confidence": 0.95,
                    "start": 0,
                    "end": 4,
                },
                {
                    "type": "organization",
                    "text": "Google",
                    "confidence": 0.92,
                    "start": 14,
                    "end": 20,
                },
            ]
        }
    )


@pytest.fixture
def policy_config() -> ModelSemanticComputePolicyConfig:
    """Create a default policy configuration."""
    return ModelSemanticComputePolicyConfig()


@pytest.fixture
def handler_config(
    policy_config: ModelSemanticComputePolicyConfig,
) -> ModelHandlerSemanticComputeConfig:
    """Create a default handler configuration."""
    return ModelHandlerSemanticComputeConfig(policy_config=policy_config)


@pytest.fixture
async def handler(
    container: ModelONEXContainer,
    handler_config: ModelHandlerSemanticComputeConfig,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> HandlerSemanticCompute:
    """Create an initialized handler with fake embedding provider."""
    h = HandlerSemanticCompute(container=container)
    await h.initialize(
        config=handler_config,
        embedding_provider=fake_embedding_provider,
    )
    return h


@pytest.fixture
async def handler_with_llm(
    container: ModelONEXContainer,
    handler_config: ModelHandlerSemanticComputeConfig,
    fake_embedding_provider: FakeEmbeddingProvider,
    fake_llm_provider: FakeLLMProvider,
) -> HandlerSemanticCompute:
    """Create an initialized handler with both embedding and LLM providers."""
    h = HandlerSemanticCompute(container=container)
    await h.initialize(
        config=handler_config,
        embedding_provider=fake_embedding_provider,
        llm_provider=fake_llm_provider,
    )
    return h


@pytest.fixture
def policy(
    policy_config: ModelSemanticComputePolicyConfig,
) -> HandlerSemanticComputePolicy:
    """Create a policy instance for testing."""
    return HandlerSemanticComputePolicy(policy_config)


# =============================================================================
# Fake Provider Tests
# =============================================================================


class TestFakeProviders:
    """Tests to verify fake providers work correctly."""

    @pytest.mark.asyncio
    async def test_fake_embedding_provider_generates_deterministic_embeddings(
        self,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Same content should produce same embedding."""
        content = "Hello, world!"

        embedding1 = await fake_embedding_provider.generate_embedding(content)
        embedding2 = await fake_embedding_provider.generate_embedding(content)

        assert embedding1 == embedding2
        assert len(embedding1) == 1024

    @pytest.mark.asyncio
    async def test_fake_embedding_provider_different_content_different_embedding(
        self,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Different content should produce different embeddings."""
        embedding1 = await fake_embedding_provider.generate_embedding("Hello")
        embedding2 = await fake_embedding_provider.generate_embedding("Goodbye")

        assert embedding1 != embedding2

    @pytest.mark.asyncio
    async def test_fake_embedding_provider_tracks_calls(
        self,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Provider should track call count."""
        assert fake_embedding_provider.call_count == 0

        await fake_embedding_provider.generate_embedding("test")
        assert fake_embedding_provider.call_count == 1

        await fake_embedding_provider.generate_embedding("test2")
        assert fake_embedding_provider.call_count == 2

    @pytest.mark.asyncio
    async def test_fake_llm_provider_returns_entities(
        self,
        fake_llm_provider: FakeLLMProvider,
    ) -> None:
        """LLM provider should return configured entities."""
        result = await fake_llm_provider.complete_structured(
            "Extract entities", output_schema={}
        )

        assert "entities" in result
        assert len(result["entities"]) == 2


# =============================================================================
# Embed Operation Tests
# =============================================================================


class TestEmbed:
    """Tests for embed() operation."""

    @pytest.mark.asyncio
    async def test_embed_generates_vector(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Embed should generate a vector of correct dimension."""
        embedding = await handler.embed("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_embed_deterministic(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Same content should produce same embedding."""
        content = "Test content for embedding"

        embedding1 = await handler.embed(content)
        embedding2 = await handler.embed(content)

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_embed_with_correlation_id(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Embed should accept correlation_id."""
        correlation_id = uuid4()

        embedding = await handler.embed(
            "Test content",
            correlation_id=correlation_id,
        )

        assert embedding is not None

    @pytest.mark.asyncio
    async def test_embed_empty_content_raises_error(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            await handler.embed("")

    @pytest.mark.asyncio
    async def test_embed_whitespace_only_raises_error(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Whitespace-only content should raise ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            await handler.embed("   \n\t  ")

    @pytest.mark.asyncio
    async def test_embed_content_too_long_raises_error(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Content exceeding max length should raise ValueError."""
        long_content = "x" * 200000  # Exceeds default 100000

        with pytest.raises(ValueError, match="[Ll]ength|[Ee]xceed"):
            await handler.embed(long_content)


# =============================================================================
# Extract Entities Tests
# =============================================================================


class TestExtractEntities:
    """Tests for extract_entities() operation."""

    @pytest.mark.asyncio
    async def test_extract_entities_returns_entity_list(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Extract entities should return ModelSemanticEntityList."""
        result = await handler.extract_entities("John works at Google in NYC.")

        assert isinstance(result, ModelSemanticEntityList)
        assert result.source_text_length == len("John works at Google in NYC.")

    @pytest.mark.asyncio
    async def test_extract_entities_heuristic_finds_capitalized_words(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Heuristic extraction should find capitalized words."""
        result = await handler.extract_entities("John works at Google Inc.")

        # Should find John, Google, Inc
        entity_texts = [e.text for e in result.entities]
        assert "John" in entity_texts
        assert "Google" in entity_texts

    @pytest.mark.asyncio
    async def test_extract_entities_empty_content_raises_error(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            await handler.extract_entities("")

    @pytest.mark.asyncio
    async def test_extract_entities_deterministic_mode_is_flagged(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Deterministic mode should be reflected in result."""
        result = await handler.extract_entities("Test content.")

        assert result.is_deterministic is True

    @pytest.mark.asyncio
    async def test_extract_entities_with_best_effort_uses_llm(
        self,
        container: ModelONEXContainer,
        fake_embedding_provider: FakeEmbeddingProvider,
        fake_llm_provider: FakeLLMProvider,
    ) -> None:
        """Best effort mode should use LLM provider."""
        policy_config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
        )
        handler_config = ModelHandlerSemanticComputeConfig(policy_config=policy_config)
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=fake_embedding_provider,
            llm_provider=fake_llm_provider,
        )

        result = await handler.extract_entities("John works at Google.")

        # LLM provider should have been called
        assert fake_llm_provider.call_count > 0
        # Should have entities from fake LLM response
        assert len(result.entities) >= 1

    @pytest.mark.asyncio
    async def test_extract_entities_llm_failure_propagates(
        self,
        container: ModelONEXContainer,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """LLM failures should propagate as exceptions (fail-fast)."""
        # Create LLM provider that fails on specific content
        failing_llm = FakeLLMProvider(fail_on={"Google"})
        policy_config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
        )
        handler_config = ModelHandlerSemanticComputeConfig(policy_config=policy_config)
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=fake_embedding_provider,
            llm_provider=failing_llm,
        )

        # LLM failure should propagate, not silently fall back to heuristic
        with pytest.raises(RuntimeError, match="Simulated LLM failure"):
            await handler.extract_entities("John works at Google.")

    @pytest.mark.asyncio
    async def test_extract_entities_best_effort_without_llm_raises(
        self,
        container: ModelONEXContainer,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Best effort mode without LLM provider should raise RuntimeError."""
        policy_config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
        )
        handler_config = ModelHandlerSemanticComputeConfig(policy_config=policy_config)
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=fake_embedding_provider,
            llm_provider=None,  # No LLM provider
        )

        # Should raise RuntimeError because LLM is required for best_effort mode
        with pytest.raises(RuntimeError, match="LLM provider not configured"):
            await handler.extract_entities("John works at Google.")


# =============================================================================
# Analyze Tests
# =============================================================================


class TestAnalyze:
    """Tests for analyze() operation."""

    @pytest.mark.asyncio
    async def test_analyze_full_returns_complete_result(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Full analysis should return complete result with all fields."""
        result = await handler.analyze("John works at Google in NYC.")

        assert isinstance(result, ModelSemanticAnalysisResult)
        assert result.analysis_type == "full"
        assert len(result.semantic_vector) == 1024
        assert isinstance(result.entities, list)
        assert isinstance(result.topics, list)
        assert 0 <= result.confidence_score <= 1

    @pytest.mark.asyncio
    async def test_analyze_embedding_only_skips_entities(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Embedding-only analysis should skip entity extraction."""
        result = await handler.analyze(
            "Test content.",
            analysis_type="embedding_only",
        )

        assert result.analysis_type == "embedding_only"
        assert len(result.semantic_vector) > 0
        assert result.entities == []

    @pytest.mark.asyncio
    async def test_analyze_entities_only_skips_embedding(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Entities-only analysis should skip embedding."""
        result = await handler.analyze(
            "John works at Google.",
            analysis_type="entities_only",
        )

        assert result.analysis_type == "entities_only"
        assert result.semantic_vector == []
        # Should have entities - John and Google should be extracted
        # result.entities is list[str] containing entity names
        assert len(result.entities) > 0, "entities_only mode should extract entities"
        assert (
            "John" in result.entities
        ), f"Expected 'John' in entities, got: {result.entities}"
        assert (
            "Google" in result.entities
        ), f"Expected 'Google' in entities, got: {result.entities}"

    @pytest.mark.asyncio
    async def test_analyze_invalid_type_raises_error(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Invalid analysis type should raise ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            await handler.analyze("Test", analysis_type="invalid_type")

    @pytest.mark.asyncio
    async def test_analyze_empty_content_raises_error(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            await handler.analyze("")

    @pytest.mark.asyncio
    async def test_analyze_tracks_processing_time(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Analysis should track processing time."""
        result = await handler.analyze("Test content for timing.")

        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_analyze_includes_model_info(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Analysis result should include model information."""

        result = await handler.analyze("Test content.")

        assert result.model_name == "fake-model-v1"
        # ModelSemVer is used directly per NO BACKWARDS COMPATIBILITY policy
        assert (
            result.model_version.major,
            result.model_version.minor,
            result.model_version.patch,
        ) == (1, 0, 0)


# =============================================================================
# Policy Tests
# =============================================================================


class TestPolicy:
    """Tests for HandlerSemanticComputePolicy decision logic."""

    def test_policy_should_cache_embedding_default(
        self,
        policy: HandlerSemanticComputePolicy,
    ) -> None:
        """Policy should cache embeddings by default."""
        assert policy.should_cache_embedding("Normal length content") is True

    def test_policy_should_not_cache_very_short_content(
        self,
        policy: HandlerSemanticComputePolicy,
    ) -> None:
        """Policy should not cache very short content."""
        assert policy.should_cache_embedding("hi") is False

    def test_policy_should_retry_on_timeout(
        self,
        policy: HandlerSemanticComputePolicy,
    ) -> None:
        """Policy should retry on timeout errors."""
        assert policy.should_retry(1, TimeoutError()) is True
        assert policy.should_retry(2, TimeoutError()) is True

    def test_policy_should_not_retry_after_max_attempts(
        self,
        policy: HandlerSemanticComputePolicy,
    ) -> None:
        """Policy should not retry after max attempts."""
        # Default max_retries is 3
        assert policy.should_retry(4, TimeoutError()) is False

    def test_policy_retry_delay_exponential(
        self,
        policy: HandlerSemanticComputePolicy,
    ) -> None:
        """Retry delay should follow exponential backoff with jitter.

        Jitter is applied as: base_delay * 2^(attempt-1) * uniform(0.5, 1.0)
        So delays should be within 50%-100% of the base exponential value.
        Default base is 100ms.
        """
        # Sample multiple times to verify jitter range
        for _ in range(10):
            delay1 = policy.get_retry_delay_ms(1)
            delay2 = policy.get_retry_delay_ms(2)
            delay3 = policy.get_retry_delay_ms(3)

            # Attempt 1: base * 1, jittered to 50-100%
            assert 50 <= delay1 <= 100, f"delay1={delay1} not in [50, 100]"
            # Attempt 2: base * 2, jittered to 50-100%
            assert 100 <= delay2 <= 200, f"delay2={delay2} not in [100, 200]"
            # Attempt 3: base * 4, jittered to 50-100%
            assert 200 <= delay3 <= 400, f"delay3={delay3} not in [200, 400]"

    def test_policy_effective_llm_params_deterministic(
        self,
    ) -> None:
        """Deterministic mode should use temperature=0 and seed."""
        config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.DETERMINISTIC,
        )
        policy = HandlerSemanticComputePolicy(config)

        params = policy.get_effective_llm_params()

        assert params["temperature"] == 0.0
        assert params["seed"] is not None

    def test_policy_effective_llm_params_best_effort(
        self,
    ) -> None:
        """Best effort mode should allow higher temperature."""
        config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
            llm_temperature=0.7,
        )
        policy = HandlerSemanticComputePolicy(config)

        params = policy.get_effective_llm_params()

        assert params["temperature"] == 0.7
        assert params["seed"] is None

    def test_policy_filter_entities_by_confidence(
        self,
        policy: HandlerSemanticComputePolicy,
    ) -> None:
        """Policy should filter entities below confidence threshold."""
        entities = [
            ModelSemanticEntity(
                entity_type=EnumSemanticEntityType.PERSON,
                text="John",
                confidence=0.9,
                span_start=0,
                span_end=4,
            ),
            ModelSemanticEntity(
                entity_type=EnumSemanticEntityType.PERSON,
                text="Maybe",
                confidence=0.3,  # Below default threshold of 0.7
                span_start=5,
                span_end=10,
            ),
        ]

        filtered = policy.filter_entities_by_confidence(entities)

        assert len(filtered) == 1
        assert filtered[0].text == "John"

    def test_policy_should_use_llm_deterministic_false(
        self,
    ) -> None:
        """Deterministic mode should prefer heuristic extraction."""
        config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.DETERMINISTIC,
        )
        policy = HandlerSemanticComputePolicy(config)

        assert policy.should_use_llm_for_entities() is False

    def test_policy_should_use_llm_best_effort_true(
        self,
    ) -> None:
        """Best effort mode should prefer LLM extraction."""
        config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
        )
        policy = HandlerSemanticComputePolicy(config)

        assert policy.should_use_llm_for_entities() is True


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for embedding caching behavior."""

    @pytest.mark.asyncio
    async def test_caching_same_content_reuses_cache(
        self,
        handler: HandlerSemanticCompute,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Same content should hit cache on second call."""
        content = "Test content for caching."

        await handler.embed(content)
        assert fake_embedding_provider.call_count == 1

        await handler.embed(content)
        # Should use cache, not call provider again
        assert fake_embedding_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_caching_different_content_misses_cache(
        self,
        handler: HandlerSemanticCompute,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Different content should miss cache."""
        await handler.embed("Content 1")
        assert fake_embedding_provider.call_count == 1

        await handler.embed("Content 2")
        assert fake_embedding_provider.call_count == 2

    @pytest.mark.asyncio
    async def test_caching_disabled_always_calls_provider(
        self,
        container: ModelONEXContainer,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """With caching disabled, should always call provider."""
        config = ModelHandlerSemanticComputeConfig(enable_caching=False)
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=config,
            embedding_provider=fake_embedding_provider,
        )

        content = "Same content."
        await handler.embed(content)
        await handler.embed(content)

        assert fake_embedding_provider.call_count == 2


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_embed_validates_content_not_empty(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Embed should reject empty content."""
        with pytest.raises(ValueError):
            await handler.embed("")

    @pytest.mark.asyncio
    async def test_analyze_validates_analysis_type(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Analyze should reject invalid analysis types."""
        with pytest.raises(ValueError):
            await handler.analyze("Content", analysis_type="bad_type")


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for configuration models."""

    def test_handler_config_default_values(self) -> None:
        """Handler config should have sensible defaults."""
        config = ModelHandlerSemanticComputeConfig()

        assert config.handler_name == "semantic-compute"
        # ModelSemVer is used directly per NO BACKWARDS COMPATIBILITY policy
        assert (
            config.handler_version.major,
            config.handler_version.minor,
            config.handler_version.patch,
        ) == (1, 0, 0)
        assert config.enable_caching is True
        assert config.max_cache_size == 1000

    def test_handler_config_custom_values(self) -> None:
        """Handler config should accept custom values."""
        config = ModelHandlerSemanticComputeConfig(
            handler_name="custom-handler",
            enable_caching=False,
            max_cache_size=500,
        )

        assert config.handler_name == "custom-handler"
        assert config.enable_caching is False
        assert config.max_cache_size == 500

    def test_policy_config_default_deterministic(self) -> None:
        """Policy config should default to deterministic mode."""
        config = ModelSemanticComputePolicyConfig()

        assert config.entity_extraction_mode == EnumEntityExtractionMode.DETERMINISTIC
        assert config.llm_temperature == 0.0
        assert config.llm_seed == 42

    def test_policy_config_is_deterministic_property(self) -> None:
        """Policy config should report determinism correctly."""
        deterministic = ModelSemanticComputePolicyConfig()
        assert deterministic.is_deterministic is True

        nondeterministic = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
            llm_temperature=0.7,
            llm_seed=None,
        )
        assert nondeterministic.is_deterministic is False

    def test_policy_config_forbids_extra_fields(self) -> None:
        """Policy config should forbid extra fields."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelSemanticComputePolicyConfig(
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# Entity Type Classification Tests
# =============================================================================


class TestEntityClassification:
    """Tests for heuristic entity type classification."""

    @pytest.mark.asyncio
    async def test_organization_suffix_classification(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Organization suffixes should be classified correctly."""
        result = await handler.extract_entities("Google Inc is a company.")

        # Check if any entity is classified as organization
        org_entities = [
            e
            for e in result.entities
            if e.entity_type == EnumSemanticEntityType.ORGANIZATION
        ]
        # Inc should trigger org classification (matches "inc" suffix in heuristic)
        assert len(org_entities) >= 1, (
            f"Expected 'Inc' to be classified as ORGANIZATION. "
            f"Got entities: {[(e.text, e.entity_type) for e in result.entities]}"
        )

    @pytest.mark.asyncio
    async def test_misc_classification_for_capitalized_words(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Capitalized words should be classified as MISC by default."""
        result = await handler.extract_entities("Foobar is a made up word.")

        # Foobar should be found as MISC
        misc_entities = [
            e for e in result.entities if e.entity_type == EnumSemanticEntityType.MISC
        ]
        foobar_found = any(e.text == "Foobar" for e in misc_entities)
        assert foobar_found


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests using all components together."""

    @pytest.mark.asyncio
    async def test_full_workflow_embed_and_analyze(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Test embedding followed by analysis."""
        content = "John Smith works at Google in San Francisco."

        # First embed
        embedding = await handler.embed(content)
        assert len(embedding) == 1024

        # Then analyze
        result = await handler.analyze(content)
        assert result.semantic_vector == embedding  # Should use cached
        assert len(result.entities) >= 0

    @pytest.mark.asyncio
    async def test_handler_provider_access(
        self,
        handler: HandlerSemanticCompute,
        fake_embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        """Handler should provide access to its providers."""
        assert handler.embedding_provider is fake_embedding_provider
        assert handler.llm_provider is None  # Not configured in this fixture

    @pytest.mark.asyncio
    async def test_handler_config_access(
        self,
        handler: HandlerSemanticCompute,
        handler_config: ModelHandlerSemanticComputeConfig,
    ) -> None:
        """Handler should provide access to its config."""
        assert handler.config is handler_config
        assert handler.policy is not None


# =============================================================================
# Sentence-Starting Word Filtering Tests
# =============================================================================


class TestSentenceStartingWordFiltering:
    """Tests for filtering out sentence-starting words from entity extraction."""

    @pytest.mark.asyncio
    async def test_the_at_sentence_start_not_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """'The' at sentence start should not be extracted as entity."""
        result = await handler.extract_entities("The quick brown fox jumps.")

        entity_texts = [e.text for e in result.entities]
        assert "The" not in entity_texts

    @pytest.mark.asyncio
    async def test_this_at_sentence_start_not_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """'This' at sentence start should not be extracted as entity."""
        result = await handler.extract_entities("This is a test sentence.")

        entity_texts = [e.text for e in result.entities]
        assert "This" not in entity_texts

    @pytest.mark.asyncio
    async def test_however_at_sentence_start_not_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """'However' at sentence start should not be extracted as entity."""
        result = await handler.extract_entities(
            "The plan failed. However, we tried again."
        )

        entity_texts = [e.text for e in result.entities]
        assert "However" not in entity_texts
        assert "The" not in entity_texts

    @pytest.mark.asyncio
    async def test_therefore_at_sentence_start_not_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """'Therefore' at sentence start should not be extracted as entity."""
        result = await handler.extract_entities(
            "The data is clear. Therefore, we proceed."
        )

        entity_texts = [e.text for e in result.entities]
        assert "Therefore" not in entity_texts

    @pytest.mark.asyncio
    async def test_furthermore_at_sentence_start_not_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """'Furthermore' at sentence start should not be extracted as entity."""
        result = await handler.extract_entities(
            "We have evidence. Furthermore, the results are clear."
        )

        entity_texts = [e.text for e in result.entities]
        assert "Furthermore" not in entity_texts

    @pytest.mark.asyncio
    async def test_it_at_sentence_start_not_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """'It' at sentence start should not be extracted as entity."""
        result = await handler.extract_entities("It was a sunny day.")

        entity_texts = [e.text for e in result.entities]
        assert "It" not in entity_texts

    @pytest.mark.asyncio
    async def test_proper_noun_at_sentence_start_is_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Proper nouns at sentence start should still be extracted."""
        result = await handler.extract_entities("Google announced new features.")

        entity_texts = [e.text for e in result.entities]
        assert "Google" in entity_texts

    @pytest.mark.asyncio
    async def test_proper_noun_mid_sentence_is_extracted(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Proper nouns in middle of sentence should be extracted."""
        result = await handler.extract_entities("The company Google is large.")

        entity_texts = [e.text for e in result.entities]
        assert "Google" in entity_texts
        assert "The" not in entity_texts

    @pytest.mark.asyncio
    async def test_multiple_sentences_filters_correctly(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Multiple sentences should each have their starters filtered."""
        content = "The dog barked. However, the cat slept. John watched."

        result = await handler.extract_entities(content)

        entity_texts = [e.text for e in result.entities]
        assert "The" not in entity_texts
        assert "However" not in entity_texts
        assert "John" in entity_texts

    @pytest.mark.asyncio
    async def test_exclamation_ends_sentence(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Exclamation marks should end sentences for filtering."""
        result = await handler.extract_entities("Wow! The day was great.")

        entity_texts = [e.text for e in result.entities]
        assert "The" not in entity_texts
        # "Wow" might or might not be captured depending on implementation

    @pytest.mark.asyncio
    async def test_question_mark_ends_sentence(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Question marks should end sentences for filtering."""
        result = await handler.extract_entities("Why? The answer is clear.")

        entity_texts = [e.text for e in result.entities]
        assert "The" not in entity_texts
        # "Why" is a stopword, should also not be captured
        assert "Why" not in entity_texts

    @pytest.mark.asyncio
    async def test_first_word_always_checked_as_sentence_start(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """First word is always treated as sentence start."""
        # Content starting with a stopword
        result = await handler.extract_entities("These results are important.")

        entity_texts = [e.text for e in result.entities]
        assert "These" not in entity_texts

    @pytest.mark.asyncio
    async def test_compound_proper_nouns_second_word_captured(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """In 'The Beatles', 'The' is skipped but 'Beatles' is captured."""
        result = await handler.extract_entities("The Beatles performed well.")

        entity_texts = [e.text for e in result.entities]
        assert "The" not in entity_texts
        assert "Beatles" in entity_texts

    @pytest.mark.asyncio
    async def test_mixed_content_extracts_correct_entities(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Mixed content with stopwords and proper nouns works correctly."""
        content = (
            "The CEO of Apple, Tim Cook, announced new products. "
            "However, Microsoft responded quickly. "
            "This competition benefits consumers."
        )

        result = await handler.extract_entities(content)

        entity_texts = [e.text for e in result.entities]

        # Should NOT be extracted (sentence starters)
        assert "The" not in entity_texts
        assert "However" not in entity_texts
        assert "This" not in entity_texts

        # Should be extracted (proper nouns)
        assert "Apple" in entity_texts
        assert "Tim" in entity_texts
        assert "Cook" in entity_texts
        assert "Microsoft" in entity_texts

    @pytest.mark.asyncio
    async def test_all_common_stopwords_filtered_at_sentence_start(
        self,
        handler: HandlerSemanticCompute,
    ) -> None:
        """Various common stopwords should all be filtered at sentence start."""
        stopwords_to_test = [
            "The",
            "This",
            "That",
            "These",
            "Those",
            "However",
            "Therefore",
            "Furthermore",
            "Moreover",
            "Meanwhile",
            "Additionally",
            "Nevertheless",
        ]

        for stopword in stopwords_to_test:
            content = f"{stopword} test content here."
            result = await handler.extract_entities(content)

            entity_texts = [e.text for e in result.entities]
            assert (
                stopword not in entity_texts
            ), f"'{stopword}' should not be extracted as entity"


# =============================================================================
# Retry Behavior Tests
# =============================================================================


class TransientFailingEmbeddingProvider:
    """Embedding provider that fails with transient errors then succeeds.

    Used to test retry behavior with TimeoutError (a transient error).
    """

    def __init__(
        self,
        *,
        dimension: int = 1024,
        fail_times: int = 2,
        error_type: type[Exception] = TimeoutError,
    ) -> None:
        self._dimension = dimension
        self._fail_times = fail_times
        self._error_type = error_type
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "transient-failing-embedding"

    @property
    def model_name(self) -> str:
        return "fake-model-v1"

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count

    async def generate_embedding(
        self,
        text: str,
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> list[float]:
        self._call_count += 1

        if self._call_count <= self._fail_times:
            raise self._error_type(f"Simulated transient failure #{self._call_count}")

        # Generate deterministic embedding based on content
        hash_val = hash(text)
        return [((hash_val + i) % 1000) / 1000.0 for i in range(self._dimension)]

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        return [
            await self.generate_embedding(
                text, model=model, correlation_id=correlation_id
            )
            for text in texts
        ]

    async def health_check(self) -> bool:
        return True


class TransientFailingLLMProvider:
    """LLM provider that fails with transient errors then succeeds.

    Used to test retry behavior with ConnectionError (a transient error).
    """

    def __init__(
        self,
        *,
        fail_times: int = 2,
        error_type: type[Exception] = ConnectionError,
        entities_response: dict | None = None,
    ) -> None:
        self._fail_times = fail_times
        self._error_type = error_type
        self._entities_response = entities_response or {"entities": []}
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "transient-failing-llm"

    @property
    def model_name(self) -> str:
        return "fake-llm-v1"

    @property
    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count

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
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise self._error_type(f"Simulated transient failure #{self._call_count}")
        return "{}"

    async def complete_structured(
        self,
        prompt: str,
        output_schema: dict,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        correlation_id: UUID | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise self._error_type(f"Simulated transient failure #{self._call_count}")
        return self._entities_response

    async def health_check(self) -> bool:
        return True


class TestRetryBehavior:
    """Tests for retry behavior with transient failures."""

    @pytest.mark.asyncio
    async def test_embed_retries_on_timeout_error(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Embed should retry on TimeoutError and succeed after retries."""
        # Provider fails 2 times then succeeds
        provider = TransientFailingEmbeddingProvider(fail_times=2)

        # Configure handler with up to 3 retries
        policy_config = ModelSemanticComputePolicyConfig(
            max_retries=3,
            retry_base_delay_ms=10,  # Fast for testing
        )
        handler_config = ModelHandlerSemanticComputeConfig(
            policy_config=policy_config,
            enable_caching=False,
        )
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=provider,
        )

        # Should succeed after retries
        embedding = await handler.embed("Test content for retry")

        assert embedding is not None
        assert len(embedding) == 1024
        # Provider should have been called 3 times (2 failures + 1 success)
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_retries_on_connection_error(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Embed should retry on ConnectionError and succeed after retries."""
        # Provider fails 1 time then succeeds
        provider = TransientFailingEmbeddingProvider(
            fail_times=1, error_type=ConnectionError
        )

        policy_config = ModelSemanticComputePolicyConfig(
            max_retries=3,
            retry_base_delay_ms=10,
        )
        handler_config = ModelHandlerSemanticComputeConfig(
            policy_config=policy_config,
            enable_caching=False,
        )
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=provider,
        )

        embedding = await handler.embed("Test content")

        assert embedding is not None
        assert provider.call_count == 2  # 1 failure + 1 success

    @pytest.mark.asyncio
    async def test_embed_fails_when_retries_exhausted(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Embed should fail when all retries are exhausted."""
        # Provider fails more times than retries allow
        provider = TransientFailingEmbeddingProvider(fail_times=10)

        policy_config = ModelSemanticComputePolicyConfig(
            max_retries=2,  # Only 2 retries allowed
            retry_base_delay_ms=10,
        )
        handler_config = ModelHandlerSemanticComputeConfig(
            policy_config=policy_config,
            enable_caching=False,
        )
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=provider,
        )

        with pytest.raises(TimeoutError):
            await handler.embed("Test content")

        # Should have tried: 1 initial + 2 retries = 3 attempts
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_does_not_retry_non_transient_error(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Embed should not retry on non-transient errors like ValueError."""
        # Using ValueError which is not in transient indicators
        provider = TransientFailingEmbeddingProvider(
            fail_times=10, error_type=ValueError
        )

        policy_config = ModelSemanticComputePolicyConfig(
            max_retries=3,
            retry_base_delay_ms=10,
        )
        handler_config = ModelHandlerSemanticComputeConfig(
            policy_config=policy_config,
            enable_caching=False,
        )
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=provider,
        )

        with pytest.raises(ValueError):
            await handler.embed("Test content")

        # Should have only tried once (no retries for non-transient errors)
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_extraction_retries_on_connection_error(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """LLM entity extraction should retry on ConnectionError."""
        embedding_provider = FakeEmbeddingProvider()
        llm_provider = TransientFailingLLMProvider(
            fail_times=2,
            error_type=ConnectionError,
            entities_response={
                "entities": [
                    {
                        "type": "person",
                        "text": "John",
                        "confidence": 0.9,
                        "start": 0,
                        "end": 4,
                    }
                ]
            },
        )

        policy_config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
            max_retries=3,
            retry_base_delay_ms=10,
        )
        handler_config = ModelHandlerSemanticComputeConfig(policy_config=policy_config)
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

        result = await handler.extract_entities("John works at Google.")

        assert result is not None
        assert len(result.entities) >= 1
        # LLM should have been called 3 times (2 failures + 1 success)
        assert llm_provider.call_count == 3

    @pytest.mark.asyncio
    async def test_llm_extraction_fails_when_retries_exhausted(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """LLM entity extraction should fail when retries exhausted."""
        embedding_provider = FakeEmbeddingProvider()
        llm_provider = TransientFailingLLMProvider(
            fail_times=10,
            error_type=TimeoutError,
        )

        policy_config = ModelSemanticComputePolicyConfig(
            entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
            max_retries=2,
            retry_base_delay_ms=10,
        )
        handler_config = ModelHandlerSemanticComputeConfig(policy_config=policy_config)
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

        with pytest.raises(TimeoutError):
            await handler.extract_entities("John works at Google.")

        # Should have tried: 1 initial + 2 retries = 3 attempts
        assert llm_provider.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_zero_retries_fails_immediately(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """With max_retries=0, should fail on first transient error."""
        provider = TransientFailingEmbeddingProvider(fail_times=1)

        policy_config = ModelSemanticComputePolicyConfig(
            max_retries=0,  # No retries
            retry_base_delay_ms=10,
        )
        handler_config = ModelHandlerSemanticComputeConfig(
            policy_config=policy_config,
            enable_caching=False,
        )
        handler = HandlerSemanticCompute(container=container)
        await handler.initialize(
            config=handler_config,
            embedding_provider=provider,
        )

        with pytest.raises(TimeoutError):
            await handler.embed("Test content")

        # Should have only tried once
        assert provider.call_count == 1
