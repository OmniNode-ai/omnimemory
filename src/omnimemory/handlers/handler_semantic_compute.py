# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic compute handler for semantic analysis operations.

This handler provides pure compute operations for semantic analysis,
embedding generation, and entity extraction. It depends on provider
protocols for I/O abstraction, keeping the handler testable and the
architecture clean.

Key Design Principles:
    - **Pure compute**: Handler contains orchestration and transformation logic
    - **Protocol dependencies**: I/O is abstracted via ProtocolEmbeddingProvider
      and ProtocolLLMProvider
    - **Policy-driven**: SemanticComputePolicy makes runtime decisions
    - **Deterministic by default**: Reproducible results for testing

Operations:
    - **analyze**: Full semantic analysis (embedding + entities + topics)
    - **embed**: Generate embedding vector for content
    - **extract_entities**: Extract named entities from content

Example::

    from omnimemory.handlers import (
        HandlerSemanticCompute,
        ModelHandlerSemanticComputeConfig,
    )

    config = ModelHandlerSemanticComputeConfig()
    handler = HandlerSemanticCompute(
        config=config,
        embedding_provider=my_embedding_provider,
        llm_provider=my_llm_provider,  # optional
    )

    # Generate embedding
    embedding = await handler.embed("Hello, world!")

    # Extract entities
    entities = await handler.extract_entities("John works at Google in NYC.")

    # Full analysis
    result = await handler.analyze("Complex content to analyze...")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1390.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import UUID, uuid4

from cachetools import LRUCache  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from ..enums import EnumEntityExtractionMode, EnumSemanticEntityType
from ..models.config import ModelSemanticComputePolicyConfig
from ..models.intelligence import (
    ModelSemanticAnalysisResult,
    ModelSemanticEntity,
    ModelSemanticEntityList,
)

if TYPE_CHECKING:
    from ..protocols import ProtocolEmbeddingProvider, ProtocolLLMProvider

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerSemanticCompute",
    "HandlerSemanticComputePolicy",
    "ModelHandlerSemanticComputeConfig",
]

# TypeVar for generic retry helper
_T = TypeVar("_T")

# =============================================================================
# Module-Level Constants
# =============================================================================

# Common words that start sentences but aren't named entities.
# Used by _extract_entities_heuristic() to filter false positives.
_SENTENCE_STARTING_STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles and determiners
        "The",
        "A",
        "An",
        # Demonstratives
        "This",
        "That",
        "These",
        "Those",
        # Pronouns
        "It",
        "He",
        "She",
        "We",
        "They",
        "I",
        "You",
        # Common sentence starters
        "However",
        "Therefore",
        "Furthermore",
        "Moreover",
        "Nevertheless",
        "Meanwhile",
        "Additionally",
        "Consequently",
        "Subsequently",
        "Otherwise",
        "Accordingly",
        "Similarly",
        "Likewise",
        "Indeed",
        "Hence",
        "Thus",
        "Regardless",
        "Nonetheless",
        "Notwithstanding",
        # Question words
        "What",
        "When",
        "Where",
        "Who",
        "Why",
        "How",
        "Which",
        # Other common starters
        "There",
        "Here",
        "If",
        "As",
        "So",
        "But",
        "And",
        "Or",
        "Yet",
        "For",
        "Nor",
        "After",
        "Before",
        "Because",
        "Although",
        "While",
        "Since",
        "Until",
        "Unless",
        "Once",
        "Now",
        "Then",
        "Also",
        "First",
        "Second",
        "Third",
        "Finally",
        "Next",
        "Last",
        "Many",
        "Most",
        "Some",
        "All",
        "Any",
        "Each",
        "Every",
        "Both",
        "Few",
        "Several",
        "Such",
        "No",
        "Not",
        "Only",
        "Just",
        "Even",
        "Still",
        "Already",
        # Adverbial starters
        "Recently",
        "Currently",
        "Actually",
        "Basically",
        "Essentially",
        "Generally",
        "Usually",
        "Obviously",
        "Clearly",
        "Certainly",
        "Probably",
        "Perhaps",
        "Maybe",
        "Possibly",
        # Temporal starters (common in sentence-start context, not entity names)
        "Today",
        "Tomorrow",
        "Yesterday",
    }
)


# =============================================================================
# Configuration Model
# =============================================================================


class ModelHandlerSemanticComputeConfig(BaseModel):
    """Configuration for the semantic compute handler.

    This model configures the handler's behavior and wraps the policy config.
    The handler uses this config to initialize and the policy uses the
    nested policy_config for runtime decisions.

    Example::

        config = ModelHandlerSemanticComputeConfig(
            handler_name="my-semantic-handler",
            policy_config=ModelSemanticComputePolicyConfig(
                cache_embeddings=True,
                entity_extraction_mode=EnumEntityExtractionMode.DETERMINISTIC,
            ),
        )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    handler_name: str = Field(
        default="semantic-compute",
        min_length=1,
        max_length=100,
        description="Name identifier for this handler instance",
    )

    handler_version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version of the handler",
    )

    policy_config: ModelSemanticComputePolicyConfig = Field(
        default_factory=ModelSemanticComputePolicyConfig,
        description="Policy configuration for runtime decisions",
    )

    enable_caching: bool = Field(
        default=True,
        description="Enable in-memory caching of results",
    )

    max_cache_size: int = Field(
        default=1000,
        ge=0,
        le=100000,
        description="Maximum number of cached items (0 to disable)",
    )


# =============================================================================
# Policy Class
# =============================================================================


class HandlerSemanticComputePolicy:
    """Policy class for semantic compute handler decisions.

    This class encapsulates decision logic based on the policy configuration.
    It determines retry behavior, fallback chains, caching strategies, and
    model selection.

    The policy is separate from configuration because:
    - Config is serializable data (knobs, thresholds)
    - Policy is executable logic (decisions, fallback chains)

    Example::

        policy = HandlerSemanticComputePolicy(config.policy_config)

        if policy.should_cache_embedding("hello world"):
            cache[key] = embedding

        if policy.should_retry(attempt=2, error=TimeoutError()):
            # retry the operation
    """

    def __init__(self, config: ModelSemanticComputePolicyConfig) -> None:
        """Initialize the policy with configuration.

        Args:
            config: The policy configuration containing thresholds and settings.
        """
        self._config = config

    @property
    def config(self) -> ModelSemanticComputePolicyConfig:
        """Get the policy configuration."""
        return self._config

    def should_cache_embedding(self, content: str) -> bool:
        """Determine if an embedding should be cached.

        Args:
            content: The content that was embedded.

        Returns:
            True if the embedding should be cached.
        """
        if not self._config.cache_embeddings:
            return False
        # Don't cache very short or very long content
        content_len = len(content)
        return 10 <= content_len <= self._config.max_content_length

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if an operation should be retried.

        Args:
            attempt: Current attempt number (1-indexed).
            error: The error that occurred.

        Returns:
            True if the operation should be retried.
        """
        if attempt > self._config.max_retries:
            return False

        # Retry on transient errors
        error_name = type(error).__name__.lower()
        transient_indicators = ["timeout", "connection", "temporary", "retry"]
        return any(indicator in error_name for indicator in transient_indicators)

    def get_retry_delay_ms(self, attempt: int) -> int:
        """Calculate the retry delay for an attempt.

        Uses exponential backoff with jitter.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in milliseconds before the next retry.
        """
        base = self._config.retry_base_delay_ms
        max_delay = self._config.retry_max_delay_ms

        # Exponential backoff: base * 2^(attempt-1)
        delay = min(base * (2 ** (attempt - 1)), max_delay)
        return int(delay)

    def get_effective_llm_params(self) -> dict[str, float | int | None]:
        """Get effective LLM parameters based on extraction mode.

        Returns:
            Dictionary with temperature, seed, and max_tokens.
        """
        return {
            "temperature": self._config.effective_llm_temperature,
            "seed": self._config.effective_llm_seed,
            "max_tokens": self._config.max_tokens_per_request,
        }

    def filter_entities_by_confidence(
        self, entities: list[ModelSemanticEntity]
    ) -> list[ModelSemanticEntity]:
        """Filter entities by confidence threshold.

        Args:
            entities: List of extracted entities.

        Returns:
            Entities meeting the confidence threshold.
        """
        threshold = self._config.entity_confidence_threshold
        return [e for e in entities if e.confidence >= threshold]

    def should_use_llm_for_entities(self) -> bool:
        """Determine if LLM should be used for entity extraction.

        Returns:
            True if LLM-based extraction is preferred.
        """
        # For deterministic mode, prefer heuristic extraction
        # For best_effort mode, use LLM if available
        return (
            self._config.entity_extraction_mode == EnumEntityExtractionMode.BEST_EFFORT
        )


# =============================================================================
# Handler Implementation
# =============================================================================


class HandlerSemanticCompute:
    """Pure compute handler for semantic analysis operations.

    This handler provides semantic analysis capabilities including:
    - Embedding generation via ProtocolEmbeddingProvider
    - Entity extraction (heuristic or LLM-backed)
    - Full semantic analysis combining embeddings, entities, and topics

    The handler is "pure compute" in the ONEX sense: it orchestrates
    transformations and delegates I/O to injected provider protocols.

    Attributes:
        config: Handler configuration.
        policy: Policy for runtime decisions.

    Example::

        handler = HandlerSemanticCompute(
            config=ModelHandlerSemanticComputeConfig(),
            embedding_provider=http_embedding_provider,
        )

        # Generate embedding
        embedding = await handler.embed("Hello world")

        # Full analysis
        result = await handler.analyze("Analyze this text for insights.")
    """

    def __init__(
        self,
        config: ModelHandlerSemanticComputeConfig,
        embedding_provider: ProtocolEmbeddingProvider,
        llm_provider: ProtocolLLMProvider | None = None,
    ) -> None:
        """Initialize the semantic compute handler.

        Args:
            config: Handler configuration.
            embedding_provider: Provider for embedding generation.
            llm_provider: Optional provider for LLM-backed operations.
        """
        self._config = config
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._policy = HandlerSemanticComputePolicy(config.policy_config)

        # LRU cache for embeddings with automatic eviction
        self._embedding_cache: LRUCache[str, list[float]] = LRUCache(
            maxsize=config.max_cache_size
        )

    @property
    def config(self) -> ModelHandlerSemanticComputeConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def policy(self) -> HandlerSemanticComputePolicy:
        """Get the policy instance."""
        return self._policy

    @property
    def embedding_provider(self) -> ProtocolEmbeddingProvider:
        """Get the embedding provider."""
        return self._embedding_provider

    @property
    def llm_provider(self) -> ProtocolLLMProvider | None:
        """Get the LLM provider (may be None)."""
        return self._llm_provider

    # =========================================================================
    # Retry Logic
    # =========================================================================

    async def _execute_with_retry(
        self,
        operation: Callable[[], Awaitable[_T]],
        operation_name: str,
    ) -> _T:
        """Execute an operation with retry logic based on policy.

        Uses the policy's should_retry() and get_retry_delay_ms() methods
        to determine retry behavior. Retries are only attempted for transient
        errors (timeout, connection issues).

        Args:
            operation: Async callable to execute (no arguments).
            operation_name: Name for logging purposes.

        Returns:
            The operation result.

        Raises:
            The last exception if all retries are exhausted, or immediately
            for non-transient errors.
        """
        last_error: Exception | None = None
        max_attempts = self._config.policy_config.max_retries + 1  # initial + retries

        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e

                # Check if we should retry
                if not self._policy.should_retry(attempt, e):
                    logger.debug(
                        "Not retrying %s after attempt %d: %s (non-transient error)",
                        operation_name,
                        attempt,
                        type(e).__name__,
                    )
                    raise

                # Check if we have more attempts left
                if attempt >= max_attempts:
                    logger.warning(
                        "All %d retry attempts exhausted for %s: %s",
                        max_attempts,
                        operation_name,
                        str(e),
                    )
                    raise

                delay_ms = self._policy.get_retry_delay_ms(attempt)
                logger.warning(
                    "Retry %d/%d for %s after %dms: %s",
                    attempt,
                    self._config.policy_config.max_retries,
                    operation_name,
                    delay_ms,
                    str(e),
                )
                await asyncio.sleep(delay_ms / 1000.0)

        # Should not reach here, but satisfy type checker
        if last_error:
            raise last_error
        raise RuntimeError(f"Unexpected state in retry logic for {operation_name}")

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def embed(
        self,
        content: str,
        *,
        model: str | None = None,
        correlation_id: UUID | None = None,
    ) -> list[float]:
        """Generate embedding vector for content.

        Args:
            content: The text content to embed.
            model: Optional model override.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Embedding vector as list of floats.

        Raises:
            ValueError: If content is empty or too long.
            EmbeddingProviderError: If embedding generation fails.
        """
        # Validate content
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        content_len = len(content)
        max_len = self._config.policy_config.max_content_length
        if content_len > max_len:
            raise ValueError(
                f"Content length ({content_len}) exceeds maximum ({max_len})"
            )

        # Check cache
        cache_key = self._compute_cache_key(content, model)
        if self._config.enable_caching and cache_key in self._embedding_cache:
            return cast(list[float], self._embedding_cache[cache_key])

        # Generate embedding via provider with retry logic for transient failures
        timeout = self._config.policy_config.timeout_seconds
        effective_model = model or self._config.policy_config.default_embedding_model

        async def _do_embed() -> list[float]:
            async with asyncio.timeout(timeout):
                return await self._embedding_provider.generate_embedding(
                    text=content,
                    model=effective_model,
                    correlation_id=correlation_id,
                    timeout_seconds=timeout,
                )

        embedding = await self._execute_with_retry(_do_embed, "embed")

        # Cache if appropriate (LRUCache handles eviction automatically)
        if self._config.enable_caching and self._policy.should_cache_embedding(content):
            self._embedding_cache[cache_key] = embedding

        return embedding

    async def extract_entities(
        self,
        content: str,
        *,
        correlation_id: UUID | None = None,
    ) -> ModelSemanticEntityList:
        """Extract named entities from content.

        The extraction strategy is determined by ``entity_extraction_mode`` in
        the policy configuration:

        **DETERMINISTIC mode** (default):
            Uses ``_extract_entities_heuristic()`` for simple, reproducible
            extraction. Best for testing and when external dependencies are
            not desired.

            Strengths:
                - No external API calls (fast, offline-capable)
                - Deterministic results (same input = same output)
                - No LLM provider required

            Limitations:
                - Only detects single capitalized words
                - Cannot extract multi-word entities ("New York" split)
                - Cannot extract dates, times, or numbers
                - Acronyms classified as MISC (not semantic type)
                - No context awareness

        **BEST_EFFORT mode**:
            Uses LLM-backed extraction via ``_extract_entities_llm()`` for
            higher accuracy. Requires an LLM provider to be configured.

            Strengths:
                - Multi-word entity detection ("New York City" as one entity)
                - Date, time, and numeric extraction
                - Context-aware classification ("Apple" as ORG vs fruit)
                - Better acronym handling ("NYC" as LOCATION)

            Limitations:
                - Requires LLM provider (external dependency)
                - Non-deterministic (results may vary)
                - Higher latency (API calls)
                - Potential cost implications

        Example::

            # DETERMINISTIC mode (default) - fast, limited accuracy
            handler = HandlerSemanticCompute(
                config=ModelHandlerSemanticComputeConfig(
                    policy_config=ModelSemanticComputePolicyConfig(
                        entity_extraction_mode=EnumEntityExtractionMode.DETERMINISTIC,
                    )
                ),
                embedding_provider=provider,
            )
            # "NYC" -> MISC, "New York City" -> "New", "York", "City" separately

            # BEST_EFFORT mode - higher accuracy, requires LLM
            handler = HandlerSemanticCompute(
                config=ModelHandlerSemanticComputeConfig(
                    policy_config=ModelSemanticComputePolicyConfig(
                        entity_extraction_mode=EnumEntityExtractionMode.BEST_EFFORT,
                    )
                ),
                embedding_provider=provider,
                llm_provider=llm_provider,  # Required for BEST_EFFORT
            )
            # "NYC" -> LOCATION, "New York City" -> single LOCATION entity

        Args:
            content: The text content to analyze.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelSemanticEntityList with extracted entities, including metadata
            about the extraction method used (``extraction_model`` field).

        Raises:
            ValueError: If content is empty.
            RuntimeError: If ``entity_extraction_mode=BEST_EFFORT`` but no
                LLM provider is configured.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Determine extraction strategy
        policy_wants_llm = self._policy.should_use_llm_for_entities()

        if policy_wants_llm:
            # Fail fast if LLM is required but not configured
            if not self._llm_provider:
                raise RuntimeError(
                    "LLM provider not configured but LLM entity extraction requested "
                    "(entity_extraction_mode=BEST_EFFORT requires an LLM provider)"
                )
            entities = await self._extract_entities_llm(content, correlation_id)
        else:
            logger.debug(
                "Using heuristic entity extraction (deterministic mode). "
                "Note: Limited to single capitalized words. For multi-word entities, "
                "dates, and context-aware classification, use entity_extraction_mode=BEST_EFFORT."
            )
            start_time = time.perf_counter()
            entities = self._extract_entities_heuristic(content)
            elapsed = time.perf_counter() - start_time

            # Warn if heuristic took unexpectedly long (shouldn't happen)
            if elapsed > self._config.policy_config.timeout_seconds:
                logger.warning(
                    "Heuristic entity extraction took %.2fs, exceeding timeout of %.2fs",
                    elapsed,
                    self._config.policy_config.timeout_seconds,
                )

        # Filter by confidence
        filtered_entities = self._policy.filter_entities_by_confidence(entities)

        # Limit number of entities
        max_entities = self._config.policy_config.max_entities_per_request
        if len(filtered_entities) > max_entities:
            filtered_entities = filtered_entities[:max_entities]

        return ModelSemanticEntityList(
            entities=filtered_entities,
            source_text_length=len(content),
            extraction_model="llm" if policy_wants_llm else "heuristic",
            is_deterministic=self._config.policy_config.is_deterministic,
        )

    async def analyze(
        self,
        content: str,
        *,
        analysis_type: str = "full",
        correlation_id: UUID | None = None,
    ) -> ModelSemanticAnalysisResult:
        """Perform full semantic analysis on content.

        Combines embedding generation, entity extraction, and topic analysis
        into a comprehensive semantic analysis result.

        Args:
            content: The text content to analyze.
            analysis_type: Type of analysis ("full", "embedding_only", "entities_only").
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelSemanticAnalysisResult with analysis data.

        Raises:
            ValueError: If content is empty or analysis_type is invalid.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        valid_types = {"full", "embedding_only", "entities_only"}
        if analysis_type not in valid_types:
            raise ValueError(
                f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}"
            )

        correlation_id = correlation_id or uuid4()
        start_time = time.perf_counter()

        # Initialize result components
        embedding: list[float] = []
        entities: list[str] = []
        topics: list[str] = []
        key_concepts: list[str] = []

        # Generate embedding
        if analysis_type in {"full", "embedding_only"}:
            embedding = await self.embed(content, correlation_id=correlation_id)

        # Extract entities
        if analysis_type in {"full", "entities_only"}:
            entity_result = await self.extract_entities(
                content, correlation_id=correlation_id
            )
            entities = [e.text for e in entity_result.entities]

            # Extract key concepts from entities
            key_concepts = self._extract_key_concepts(entity_result.entities)

        # Extract topics (simple heuristic for now)
        if analysis_type == "full":
            topics = self._extract_topics_heuristic(content)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        return ModelSemanticAnalysisResult(
            result_id=correlation_id,
            analysis_type=analysis_type,
            analyzed_content=content[:1000],  # Truncate for storage
            content_language=None,  # Language detection not implemented
            semantic_vector=embedding,
            key_concepts=key_concepts,
            entities=entities,
            topics=topics,
            sentiment_score=None,  # Sentiment analysis not implemented
            complexity_score=self._compute_complexity_score(content),
            readability_score=self._compute_readability_score(content),
            coherence_score=None,  # Coherence analysis not implemented
            relevance_score=None,  # Relevance analysis not implemented
            confidence_score=0.9 if embedding else 0.7,
            model_name=self._embedding_provider.model_name,
            model_version=self._config.handler_version,
            processing_time_ms=processing_time_ms,
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _compute_cache_key(self, content: str, model: str | None) -> str:
        """Compute a cache key for content and model."""
        model_name = model or self._config.policy_config.default_embedding_model
        key_input = f"{model_name}:{content}"

        if self._config.policy_config.cache_key_include_model:
            return hashlib.sha256(key_input.encode()).hexdigest()[:32]
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _extract_entities_heuristic(self, content: str) -> list[ModelSemanticEntity]:
        """Extract entities using simple capitalization-based heuristics.

        This method provides deterministic entity extraction without external
        dependencies. It identifies capitalized words as potential named entities
        and filters common sentence-starting words to reduce false positives.

        Capabilities:
            - Detects single capitalized words (e.g., "John", "Google", "Paris")
            - Filters common sentence-starting stopwords ("The", "However", etc.)
            - Classifies entities by simple suffix/pattern matching:
                - Organization: words ending in Inc, Corp, LLC, Ltd, etc.
                - Location: words like Street, Avenue, City, etc.
                - Money: words starting with $ or ending with USD/EUR/GBP
                - Percent: words containing %
                - MISC: other capitalized words (default for proper nouns)

        Limitations:
            - Does NOT detect multi-word entities:
                - "New York City" -> extracts "New", "York", "City" separately
                - "United States" -> extracts "United", "States" separately
            - Does NOT detect dates, times, or numeric values:
                - "January 15, 2024" -> only "January" extracted
                - "3:30 PM" -> nothing extracted
            - Acronyms classified as MISC, not their semantic type:
                - "NYC" -> MISC (not LOCATION)
                - "FBI" -> MISC (not ORGANIZATION)
            - No context awareness:
                - Cannot distinguish "Apple" (company) vs "Apple" (fruit)
                - Cannot distinguish "Jordan" (person) vs "Jordan" (country)
            - Misses lowercase named entities:
                - "iPhone", "eBay" -> not detected (starts lowercase)
            - Limited organization detection:
                - "Google" -> MISC (no suffix like Inc/Corp)
                - "Microsoft Corporation" -> only "Microsoft" as MISC, "Corporation" skipped

        Use Cases:
            - Testing and development (deterministic, reproducible results)
            - Offline processing (no external API calls needed)
            - Quick extraction where precision is less critical

        For higher accuracy with multi-word entities, dates, numbers, and
        context-aware classification, use ``entity_extraction_mode=BEST_EFFORT``
        with an LLM provider configured.

        Args:
            content: Text to extract entities from.

        Returns:
            List of extracted entities with type, text, confidence (0.7), and spans.
        """
        entities: list[ModelSemanticEntity] = []
        words = content.split()

        # Track sentence boundaries
        sentence_end_chars = ".!?"

        i = 0
        is_sentence_start = True  # First word is always a sentence start

        for word_idx, word in enumerate(words):
            # Find position in original content
            try:
                span_start = content.index(word, i)
                span_end = span_start + len(word)
                i = span_end
            except ValueError:
                continue

            # Strip punctuation for analysis
            clean_word = word.strip(".,!?;:\"'()[]{}").strip()

            if not clean_word:
                # Check if this word ends a sentence for next iteration
                if any(c in word for c in sentence_end_chars):
                    is_sentence_start = True
                continue

            # Check if this word is at the start of a sentence
            word_is_sentence_start = is_sentence_start

            # Update sentence start tracker for next word
            is_sentence_start = any(c in word for c in sentence_end_chars)

            # Simple heuristic: capitalized words
            if clean_word[0].isupper() and len(clean_word) > 1:
                # Check if this is a sentence-starting stopword
                if (
                    word_is_sentence_start
                    and clean_word in _SENTENCE_STARTING_STOPWORDS
                ):
                    # Skip common stopwords at sentence start
                    # Note: proper nouns like "The Beatles" - "The" is skipped,
                    # but "Beatles" will be captured on its next iteration
                    continue

                # Determine entity type based on simple patterns
                entity_type = self._classify_entity_heuristic(clean_word)

                if entity_type != EnumSemanticEntityType.UNKNOWN:
                    entities.append(
                        ModelSemanticEntity(
                            entity_type=entity_type,
                            text=clean_word,
                            confidence=0.7,  # Lower confidence for heuristic
                            span_start=span_start,
                            span_end=span_end,
                        )
                    )

        return entities

    def _classify_entity_heuristic(self, word: str) -> EnumSemanticEntityType:
        """Classify an entity using simple heuristics.

        Args:
            word: The word to classify.

        Returns:
            The entity type classification.
        """
        word_lower = word.lower()

        # Organization indicators
        org_suffixes = {"inc", "corp", "llc", "ltd", "company", "co", "group"}
        if any(word_lower.endswith(suffix) for suffix in org_suffixes):
            return EnumSemanticEntityType.ORGANIZATION

        # Location indicators (very simplified)
        location_words = {
            "street",
            "avenue",
            "road",
            "city",
            "state",
            "country",
            "park",
            "building",
        }
        if word_lower in location_words:
            return EnumSemanticEntityType.LOCATION

        # Money indicators
        if word.startswith("$") or word.endswith(("USD", "EUR", "GBP")):
            return EnumSemanticEntityType.MONEY

        # Percent indicators
        if "%" in word:
            return EnumSemanticEntityType.PERCENT

        # Default to MISC for capitalized words (likely proper nouns)
        if word[0].isupper():
            return EnumSemanticEntityType.MISC

        return EnumSemanticEntityType.UNKNOWN

    async def _extract_entities_llm(
        self,
        content: str,
        correlation_id: UUID | None,
    ) -> list[ModelSemanticEntity]:
        """Extract entities using LLM provider.

        Args:
            content: Text to extract entities from.
            correlation_id: Optional correlation ID.

        Returns:
            List of extracted entities.

        Raises:
            RuntimeError: If LLM provider is not configured.
            Exception: If LLM provider fails (propagated from provider).
        """
        if not self._llm_provider:
            raise RuntimeError(
                "LLM provider not configured but LLM entity extraction requested"
            )

        # Build extraction prompt
        prompt = self._build_entity_extraction_prompt(content)
        llm_params = self._policy.get_effective_llm_params()

        # Extract typed values
        temperature = float(llm_params.get("temperature", 0.0) or 0.0)
        seed_val = llm_params.get("seed")
        seed = int(seed_val) if seed_val is not None else None
        timeout = self._config.policy_config.timeout_seconds

        # Define the LLM call to be retried
        async def _do_llm_extract() -> dict[str, object]:
            # self._llm_provider is checked at method entry, assert for type checker
            assert self._llm_provider is not None
            async with asyncio.timeout(timeout):
                return await self._llm_provider.complete_structured(
                    prompt=prompt,
                    output_schema=self._get_entity_extraction_schema(),
                    model=self._config.policy_config.default_llm_model,
                    temperature=temperature,
                    seed=seed,
                    correlation_id=correlation_id,
                    timeout_seconds=timeout,
                )

        try:
            response = await self._execute_with_retry(
                _do_llm_extract, "llm_entity_extraction"
            )
            return self._parse_llm_entity_response(response, content)

        except Exception:
            logger.exception("LLM entity extraction failed after retries")
            raise

    def _build_entity_extraction_prompt(self, content: str) -> str:
        """Build the prompt for LLM-based entity extraction."""
        return f"""Extract named entities from the following text.
Identify: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, PRODUCT, EVENT.

Text: {content}

Return a JSON array of entities with: type, text, confidence (0-1), start, end."""

    def _get_entity_extraction_schema(self) -> dict[str, object]:
        """Get JSON schema for entity extraction output."""
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "text": {"type": "string"},
                            "confidence": {"type": "number"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                        },
                        "required": ["type", "text", "confidence", "start", "end"],
                    },
                }
            },
            "required": ["entities"],
        }

    def _parse_llm_entity_response(
        self, response: dict[str, object], content: str
    ) -> list[ModelSemanticEntity]:
        """Parse LLM response into entity models."""
        entities: list[ModelSemanticEntity] = []

        # Extract and validate entities list from response
        raw_entities = response.get("entities", [])
        if not isinstance(raw_entities, list):
            return entities

        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            entity_data: dict[str, object] = item

            try:
                type_value = entity_data.get("type", "misc")
                entity_type_str = str(type_value).lower() if type_value else "misc"
                entity_type = EnumSemanticEntityType(entity_type_str)
            except ValueError:
                entity_type = EnumSemanticEntityType.MISC

            text_value = entity_data.get("text", "")
            confidence_value = entity_data.get("confidence", 0.8)
            start_value = entity_data.get("start", 0)
            end_value = entity_data.get("end", 0)

            entities.append(
                ModelSemanticEntity(
                    entity_type=entity_type,
                    text=str(text_value) if text_value else "",
                    confidence=min(
                        1.0,
                        max(
                            0.0,
                            float(confidence_value)
                            if isinstance(confidence_value, int | float)
                            else 0.8,
                        ),
                    ),
                    span_start=int(start_value)
                    if isinstance(start_value, int | float)
                    else 0,
                    span_end=int(end_value)
                    if isinstance(end_value, int | float)
                    else 0,
                )
            )

        return entities

    def _extract_key_concepts(self, entities: list[ModelSemanticEntity]) -> list[str]:
        """Extract key concepts from entities."""
        # Use high-confidence named entities as key concepts
        concepts = [
            e.text
            for e in entities
            if e.confidence >= 0.8
            and e.entity_type
            in {
                EnumSemanticEntityType.ORGANIZATION,
                EnumSemanticEntityType.PERSON,
                EnumSemanticEntityType.PRODUCT,
                EnumSemanticEntityType.EVENT,
            }
        ]
        return list(dict.fromkeys(concepts))[:10]  # Dedupe and limit

    def _extract_topics_heuristic(self, content: str) -> list[str]:
        """Extract topics using simple heuristics.

        Args:
            content: Text to analyze.

        Returns:
            List of topic strings.
        """
        # Simple word frequency approach
        words = content.lower().split()

        # Filter stop words and short words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        filtered_words = [
            w.strip(".,!?;:\"'()[]{}").lower()
            for w in words
            if len(w) > 3 and w.lower() not in stop_words
        ]

        # Count frequencies
        freq: dict[str, int] = {}
        for w in filtered_words:
            freq[w] = freq.get(w, 0) + 1

        # Return top words as topics
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:5]]

    def _compute_complexity_score(self, content: str) -> float:
        """Compute a simple complexity score.

        Based on average word length and sentence length.

        Args:
            content: Text to analyze.

        Returns:
            Complexity score between 0 and 1.
        """
        words = content.split()
        if not words:
            return 0.0

        avg_word_len = sum(len(w) for w in words) / len(words)

        # Normalize: 4 chars = 0.3, 8+ chars = 0.8
        word_complexity = min(1.0, max(0.0, (avg_word_len - 2) / 8))

        # Sentence length factor
        sentences = content.count(".") + content.count("!") + content.count("?")
        sentences = max(1, sentences)
        avg_sentence_len = len(words) / sentences

        # Normalize: 10 words = 0.3, 30+ words = 0.8
        sentence_complexity = min(1.0, max(0.0, (avg_sentence_len - 5) / 30))

        return (word_complexity + sentence_complexity) / 2

    def _compute_readability_score(self, content: str) -> float:
        """Compute a simple readability score.

        Higher score = more readable.

        Args:
            content: Text to analyze.

        Returns:
            Readability score between 0 and 1.
        """
        # Simple inverse of complexity
        complexity = self._compute_complexity_score(content)
        return 1.0 - complexity * 0.5  # Bias toward readable
