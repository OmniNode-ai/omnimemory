# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock Database Handler for full-text search operations.

This module provides a mock handler that simulates `HandlerDb` behavior
for full-text SQL search. It allows development and testing of the
memory_retrieval_effect node without requiring a running PostgreSQL instance.

The mock uses simple substring/word matching to simulate SQL LIKE and
full-text search behavior, making it suitable for unit tests and local
development.

Example::

    import asyncio
    from omnimemory.nodes.memory_retrieval_effect.handlers import (
        HandlerDbMock,
        ModelHandlerDbMockConfig,
    )

    async def example():
        config = ModelHandlerDbMockConfig()
        handler = HandlerDbMock(config)
        await handler.initialize()

        # Seed with test data
        handler.seed_snapshots([snapshot1, snapshot2])

        # Search
        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="authentication error",
            limit=10,
        )
        response = await handler.execute(request)

    asyncio.run(example())

Security:
    This mock handler uses in-memory data structures with no SQL execution,
    eliminating SQL injection risks. Production handlers (HandlerDb) prevent
    SQL injection through parameterized queries - user-provided query_text
    is never interpolated directly into SQL statements.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from omnibase_core.models.omnimemory import (
    ModelMemorySnapshot,  # noqa: TC002 - Pydantic needs runtime access
)
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

from ..models import (
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
)

logger = logging.getLogger(__name__)

# Minimum token length for text tokenization (filters out short words)
MIN_TOKEN_LENGTH = 2

__all__ = [
    "HandlerDbMock",
    "ModelHandlerDbMockConfig",
]


class ModelHandlerDbMockConfig(BaseModel):
    """Configuration for the mock database handler.

    Attributes:
        simulate_latency_ms: Simulated latency for operations in milliseconds.
            Set to 0 for instant responses.
        case_sensitive: Whether text search is case-sensitive. Defaults to False.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    simulate_latency_ms: int = Field(
        default=0,
        ge=0,
        description="Simulated latency in milliseconds",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether text search is case-sensitive",
    )


class HandlerDbMock:
    """Mock handler that simulates HandlerDb for full-text search.

    This handler provides a development-friendly interface for testing
    full-text search functionality without requiring a real PostgreSQL instance.
    It uses substring and word matching to simulate SQL LIKE and FTS behavior.

    The handler maintains an in-memory store of snapshots and can be seeded
    with test data for reproducible testing.

    Attributes:
        config: The handler configuration.

    Example::

        async def example():
            handler = HandlerDbMock(ModelHandlerDbMockConfig())
            await handler.initialize()

            # Seed test data
            handler.seed_snapshots([snapshot1, snapshot2])

            # Execute search
            request = ModelMemoryRetrievalRequest(
                operation="search_text",
                query_text="login error",
            )
            response = await handler.execute(request)
    """

    def __init__(self, config: ModelHandlerDbMockConfig) -> None:
        """Initialize the mock handler with configuration.

        Args:
            config: The handler configuration.
        """
        self._config = config
        self._snapshots: dict[str, ModelMemorySnapshot] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelHandlerDbMockConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    @property
    def snapshot_count(self) -> int:
        """Get the number of stored snapshots."""
        return len(self._snapshots)

    async def initialize(self) -> None:
        """Initialize the mock handler.

        Thread-safe: Uses asyncio.Lock to prevent concurrent initialization.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Mock DB handler initialized")
            self._initialized = True

    def seed_snapshots(self, snapshots: Sequence[ModelMemorySnapshot]) -> None:
        """Seed the mock store with test snapshots.

        Args:
            snapshots: List of snapshots to add to the mock store.

        Note:
            Snapshots with invalid or empty IDs are skipped with a warning.
        """
        valid_count = 0
        for snapshot in snapshots:
            # Validate snapshot ID is non-empty
            if not snapshot.snapshot_id or not str(snapshot.snapshot_id).strip():
                logger.warning(
                    "Skipping snapshot with invalid/empty ID: %r",
                    snapshot.snapshot_id,
                )
                continue

            snapshot_id = str(snapshot.snapshot_id)
            self._snapshots[snapshot_id] = snapshot
            valid_count += 1

        logger.debug("Seeded %d snapshots into mock DB store", valid_count)

    def clear(self) -> None:
        """Clear all snapshots from the mock store."""
        self._snapshots.clear()

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Execute a full-text search operation.

        Args:
            request: The retrieval request (must have operation="search_text").

        Returns:
            Response with search results ordered by relevance.

        Raises:
            ValueError: If operation is not "search_text".
        """
        if not self._initialized:
            await self.initialize()

        if request.operation != "search_text":
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: Only supports 'search_text', "
                    f"got '{request.operation}'"
                ),
            )

        if request.query_text is None:
            return ModelMemoryRetrievalResponse(
                status="error",
                error_message=(
                    f"{self.__class__.__name__}: query_text is required "
                    f"for operation '{request.operation}'"
                ),
            )

        # Simulate latency if configured
        if self._config.simulate_latency_ms > 0:
            await asyncio.sleep(self._config.simulate_latency_ms / 1000)

        # Parse query into search terms
        query = request.query_text
        if not self._config.case_sensitive:
            query = query.lower()

        search_terms = self._tokenize(query)

        # Score all snapshots by text match
        scored_results: list[tuple[str, float]] = []
        for snapshot_id, snapshot in self._snapshots.items():
            score = self._compute_text_score(snapshot, search_terms)
            if score > 0:
                # Apply metadata and tag filters if specified
                if self._matches_filters(snapshot, request):
                    scored_results.append((snapshot_id, score))

        # Sort by score (highest first) and limit
        scored_results.sort(key=lambda x: x[1], reverse=True)
        scored_results = scored_results[: request.limit]

        # Normalize scores to [0, 1] range
        if scored_results:
            max_score = scored_results[0][1]
            if max_score > 0:
                scored_results = [
                    (sid, score / max_score) for sid, score in scored_results
                ]

        # Build response
        if not scored_results:
            return ModelMemoryRetrievalResponse(
                status="no_results",
                results=[],
                total_count=0,
            )

        results = [
            ModelSearchResult(
                snapshot=self._snapshots[snapshot_id],
                score=score,
            )
            for snapshot_id, score in scored_results
        ]

        return ModelMemoryRetrievalResponse(
            status="success",
            results=results,
            total_count=len(results),
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into searchable terms.

        Args:
            text: The text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        # Split on non-alphanumeric characters
        tokens = re.split(r"[^\w]+", text)
        # Filter empty tokens and short words
        return [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]

    def _get_snapshot_text(self, snapshot: ModelMemorySnapshot) -> str:
        """Extract searchable text from a snapshot.

        Args:
            snapshot: The snapshot to extract text from.

        Returns:
            Combined searchable text from the snapshot.
        """
        text_parts = []

        # Subject - extract subject_key which contains the searchable text
        if snapshot.subject:
            if snapshot.subject.subject_key:
                text_parts.append(snapshot.subject.subject_key)
            # Also include subject_type as text
            if snapshot.subject.subject_type:
                text_parts.append(str(snapshot.subject.subject_type))

        # Tags
        if snapshot.tags:
            text_parts.extend(snapshot.tags)

        # Snapshot ID (for exact matches)
        text_parts.append(str(snapshot.snapshot_id))

        combined = " ".join(text_parts)
        if not self._config.case_sensitive:
            combined = combined.lower()

        return combined

    def _compute_text_score(
        self, snapshot: ModelMemorySnapshot, search_terms: list[str]
    ) -> float:
        """Compute a relevance score for a snapshot against search terms.

        The scoring algorithm simulates PostgreSQL full-text search behavior:
        - Each matching term contributes to the score
        - Exact word matches score higher than partial matches
        - Multiple matches of the same term don't double-count

        Args:
            snapshot: The snapshot to score.
            search_terms: List of search terms to match.

        Returns:
            Relevance score (0.0 if no matches).
        """
        if not search_terms:
            return 0.0

        text = self._get_snapshot_text(snapshot)
        text_tokens = set(self._tokenize(text))

        score = 0.0
        matched_terms = 0

        for term in search_terms:
            # Exact token match (highest weight)
            if term in text_tokens:
                score += 2.0
                matched_terms += 1
            # Substring match in any token (lower weight)
            elif any(term in token for token in text_tokens):
                score += 1.0
                matched_terms += 1
            # Substring match anywhere in text (lowest weight)
            elif term in text:
                score += 0.5
                matched_terms += 1

        # Boost score based on percentage of terms matched
        if len(search_terms) > 0:
            coverage = matched_terms / len(search_terms)
            score *= 1 + coverage  # Up to 2x boost for full coverage

        return score

    def _matches_filters(
        self, snapshot: ModelMemorySnapshot, request: ModelMemoryRetrievalRequest
    ) -> bool:
        """Check if a snapshot matches the request filters.

        Args:
            snapshot: The snapshot to check.
            request: The request containing filter criteria.

        Returns:
            True if the snapshot matches all filters.
        """
        # Tag filter
        if request.tags:
            snapshot_tags = set(snapshot.tags or [])
            if not any(tag in snapshot_tags for tag in request.tags):
                return False

        # Metadata filter (not implemented in mock - would need metadata field)
        # For now, always passes

        return True

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        if self._initialized:
            self._snapshots.clear()
            self._initialized = False
            logger.debug("Mock DB handler shutdown complete")
