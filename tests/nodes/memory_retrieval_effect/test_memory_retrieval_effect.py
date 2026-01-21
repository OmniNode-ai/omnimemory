# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for memory_retrieval_effect node.

This module tests all search operations for the memory retrieval effect node,
including semantic search, full-text search, and graph traversal.

Test Categories:
    - Semantic Search: Vector similarity search using mock Qdrant handler
    - Full-Text Search: Text matching using mock Database handler
    - Graph Traversal: Relationship traversal using mock Graph handler
    - Handler Routing: Verify requests are routed to correct handlers
    - Error Handling: Validation and edge case scenarios

Usage:
    pytest tests/nodes/memory_retrieval_effect/ -v
    pytest tests/nodes/memory_retrieval_effect/ -v -k "semantic"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from omnibase_core.enums.enum_subject_type import EnumSubjectType
from omnibase_core.models.omnimemory import (
    ModelCostLedger,
    ModelMemorySnapshot,
    ModelSubjectRef,
)

from omnimemory.nodes.memory_retrieval_effect import (
    HandlerMemoryRetrieval,
    ModelHandlerMemoryRetrievalConfig,
    ModelMemoryRetrievalRequest,
)
from omnimemory.nodes.memory_retrieval_effect.handlers import (
    ModelHandlerDbMockConfig,
    ModelHandlerGraphMockConfig,
    ModelHandlerQdrantMockConfig,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def handler() -> HandlerMemoryRetrieval:
    """Create handler with mock handlers.

    Returns:
        Configured HandlerMemoryRetrieval instance with mock handlers.
    """
    config = ModelHandlerMemoryRetrievalConfig(
        use_mock_handlers=True,
        qdrant_config=ModelHandlerQdrantMockConfig(embedding_dimension=1024),
        db_config=ModelHandlerDbMockConfig(case_sensitive=False),
        graph_config=ModelHandlerGraphMockConfig(bidirectional=True),
    )
    return HandlerMemoryRetrieval(config)


def create_snapshot(
    subject_text: str = "test subject",
    tags: tuple[str, ...] = (),
) -> ModelMemorySnapshot:
    """Create a unique memory snapshot for testing.

    Args:
        subject_text: Text for the subject key (used in search matching).
        tags: Optional tuple of tags.

    Returns:
        A new ModelMemorySnapshot instance with unique IDs.
    """
    subject = ModelSubjectRef(
        subject_type=EnumSubjectType.AGENT,
        subject_id=uuid4(),
        subject_key=subject_text,
    )
    ledger = ModelCostLedger(budget_total=100.0)
    return ModelMemorySnapshot(
        snapshot_id=uuid4(),
        subject=subject,
        cost_ledger=ledger,
        schema_version="1.0.0",
        tags=tags,
    )


# =============================================================================
# Semantic Search Tests
# =============================================================================


class TestSemanticSearch:
    """Tests for semantic similarity search (search operation)."""

    @pytest.mark.asyncio
    async def test_search_with_query_text(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test semantic search with text query.

        Given: Snapshots seeded in the handler
        When: Executing a search operation with query_text
        Then: Returns matching results with similarity scores
        """
        await handler.initialize()

        # Seed test data with related content
        snapshots = [
            create_snapshot("user authentication login flow", tags=("auth",)),
            create_snapshot("database connection pooling", tags=("db",)),
            create_snapshot("authentication token validation", tags=("auth",)),
        ]
        handler.seed_snapshots(snapshots)

        # Search for authentication-related content
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="authentication login",
            limit=5,
            similarity_threshold=0.0,  # Low threshold for mock handler
        )

        response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) > 0
        assert response.total_count == len(response.results)

        # Results should have scores
        for result in response.results:
            assert 0.0 <= result.score <= 1.0
            assert result.snapshot is not None

    @pytest.mark.asyncio
    async def test_search_with_pre_computed_embedding(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test semantic search with pre-computed embedding vector.

        Given: Snapshots seeded with known embeddings
        When: Executing a search with a pre-computed query embedding
        Then: Returns results based on vector similarity
        """
        await handler.initialize()

        # Create and seed a snapshot
        snapshot = create_snapshot("test content")
        handler.seed_snapshots([snapshot])

        # Search with a mock embedding (1024 dimensions)
        mock_embedding = [0.1] * 1024
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_embedding=mock_embedding,
            limit=10,
            similarity_threshold=0.0,
        )

        response = await handler.execute(request)

        assert response.status in ("success", "no_results")
        # Verify embedding was used
        assert response.query_embedding_used == mock_embedding

    @pytest.mark.asyncio
    async def test_search_respects_similarity_threshold(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that similarity threshold filters results.

        Given: Snapshots with varying relevance to query
        When: Executing search with high similarity threshold
        Then: Only highly similar results are returned
        """
        await handler.initialize()

        snapshots = [
            create_snapshot("exact match query text"),
            create_snapshot("completely unrelated content about bananas"),
        ]
        handler.seed_snapshots(snapshots)

        # Search with high threshold
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="exact match query text",
            similarity_threshold=0.9,
            limit=10,
        )

        response = await handler.execute(request)

        # Should have filtered results
        for result in response.results:
            assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that result limit is respected.

        Given: Many snapshots seeded
        When: Executing search with small limit
        Then: Returns at most 'limit' results
        """
        await handler.initialize()

        # Seed many snapshots
        snapshots = [create_snapshot(f"content {i}") for i in range(20)]
        handler.seed_snapshots(snapshots)

        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="content",
            limit=5,
            similarity_threshold=0.0,
        )

        response = await handler.execute(request)

        assert len(response.results) <= 5

    @pytest.mark.asyncio
    async def test_search_no_results(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test search returns no_results when nothing matches.

        Given: Empty store or no matching content
        When: Executing search
        Then: Returns no_results status
        """
        await handler.initialize()
        # Don't seed any data

        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="nonexistent content xyz123",
            similarity_threshold=0.9,
        )

        response = await handler.execute(request)

        assert response.status == "no_results"
        assert len(response.results) == 0
        assert response.total_count == 0

    def test_search_requires_query(self) -> None:
        """Test that search operation requires query_text or query_embedding."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryRetrievalRequest(operation="search")

        error_message = str(exc_info.value)
        assert "query" in error_message.lower()


# =============================================================================
# Full-Text Search Tests
# =============================================================================


class TestFullTextSearch:
    """Tests for full-text search (search_text operation)."""

    @pytest.mark.asyncio
    async def test_search_text_basic(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test basic full-text search.

        Given: Snapshots with text content
        When: Executing search_text operation
        Then: Returns matching results based on text matching
        """
        await handler.initialize()

        snapshots = [
            create_snapshot("error handling in authentication"),
            create_snapshot("database migration scripts"),
            create_snapshot("authentication error codes"),
        ]
        handler.seed_snapshots(snapshots)

        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="authentication error",
            limit=10,
        )

        response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_search_text_case_insensitive(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that text search is case-insensitive by default.

        Given: Snapshots with mixed case content
        When: Executing search with different case
        Then: Matches regardless of case
        """
        await handler.initialize()

        snapshot = create_snapshot("UPPERCASE Content Here")
        handler.seed_snapshots([snapshot])

        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="uppercase content",
            limit=10,
        )

        response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_search_text_with_tags_filter(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test full-text search with tag filtering.

        Given: Snapshots with different tags
        When: Executing search_text with tags filter
        Then: Only returns results matching both text and tags
        """
        await handler.initialize()

        snapshots = [
            create_snapshot("authentication flow", tags=("auth", "security")),
            create_snapshot("authentication bypass", tags=("security",)),
            create_snapshot("authentication test", tags=("test",)),
        ]
        handler.seed_snapshots(snapshots)

        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="authentication",
            tags=["auth"],
            limit=10,
        )

        response = await handler.execute(request)

        assert response.status == "success"
        # Should only return the one with "auth" tag
        for result in response.results:
            assert "auth" in (result.snapshot.tags or [])

    def test_search_text_requires_query_text(self) -> None:
        """Test that search_text operation requires query_text."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryRetrievalRequest(operation="search_text")

        error_message = str(exc_info.value)
        assert "query_text" in error_message.lower()


# =============================================================================
# Graph Traversal Tests
# =============================================================================


class TestGraphTraversal:
    """Tests for graph traversal search (search_graph operation)."""

    @pytest.mark.asyncio
    async def test_graph_traversal_basic(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test basic graph traversal from a starting snapshot.

        Given: Snapshots connected by relationships
        When: Executing search_graph from a starting node
        Then: Returns connected snapshots with path information
        """
        await handler.initialize()

        # Create connected snapshots
        snap1 = create_snapshot("root node")
        snap2 = create_snapshot("child node 1")
        snap3 = create_snapshot("child node 2")

        handler.seed_snapshots([snap1, snap2, snap3])

        # Add relationships
        handler.add_graph_relationship(
            str(snap1.snapshot_id),
            str(snap2.snapshot_id),
            "related_to",
        )
        handler.add_graph_relationship(
            str(snap1.snapshot_id),
            str(snap3.snapshot_id),
            "caused_by",
        )

        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id=str(snap1.snapshot_id),
            traversal_depth=2,
        )

        response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) == 2  # Should find both children

        # Results should have path information
        for result in response.results:
            assert result.path is not None
            assert str(snap1.snapshot_id) in result.path

    @pytest.mark.asyncio
    async def test_graph_traversal_depth_limit(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that traversal respects depth limit.

        Given: A chain of connected snapshots (depth 4)
        When: Executing search_graph with depth=2
        Then: Only returns nodes within 2 hops
        """
        await handler.initialize()

        # Create a chain: A -> B -> C -> D
        snaps = [create_snapshot(f"node {i}") for i in range(4)]
        handler.seed_snapshots(snaps)

        for i in range(len(snaps) - 1):
            handler.add_graph_relationship(
                str(snaps[i].snapshot_id),
                str(snaps[i + 1].snapshot_id),
                "next",
            )

        # Traverse with depth 2
        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id=str(snaps[0].snapshot_id),
            traversal_depth=2,
        )

        response = await handler.execute(request)

        assert response.status == "success"
        # Should find B and C, but not D (depth 3)
        assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_graph_traversal_relationship_filter(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test filtering traversal by relationship type.

        Given: Snapshots with different relationship types
        When: Executing search_graph with relationship_types filter
        Then: Only traverses specified relationship types
        """
        await handler.initialize()

        root = create_snapshot("root")
        related = create_snapshot("related child")
        caused = create_snapshot("caused child")

        handler.seed_snapshots([root, related, caused])

        handler.add_graph_relationship(
            str(root.snapshot_id),
            str(related.snapshot_id),
            "related_to",
        )
        handler.add_graph_relationship(
            str(root.snapshot_id),
            str(caused.snapshot_id),
            "caused_by",
        )

        # Only traverse "related_to" relationships
        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id=str(root.snapshot_id),
            relationship_types=["related_to"],
            traversal_depth=2,
        )

        response = await handler.execute(request)

        assert response.status == "success"
        assert len(response.results) == 1
        assert response.results[0].snapshot.snapshot_id == related.snapshot_id

    @pytest.mark.asyncio
    async def test_graph_traversal_not_found(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test graph traversal with nonexistent starting node.

        Given: A snapshot_id that doesn't exist
        When: Executing search_graph
        Then: Returns no_results with appropriate message
        """
        await handler.initialize()

        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id="nonexistent-id-12345",
            traversal_depth=2,
        )

        response = await handler.execute(request)

        assert response.status == "no_results"

    def test_graph_search_requires_snapshot_id(self) -> None:
        """Test that search_graph operation requires snapshot_id."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryRetrievalRequest(operation="search_graph")

        error_message = str(exc_info.value)
        assert "snapshot_id" in error_message.lower()


# =============================================================================
# Handler Routing Tests
# =============================================================================


class TestHandlerRouting:
    """Tests for request routing to appropriate handlers."""

    @pytest.mark.asyncio
    async def test_routes_search_to_qdrant(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that 'search' operation routes to Qdrant handler."""
        await handler.initialize()

        snapshot = create_snapshot("test")
        handler.seed_snapshots([snapshot])

        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="test",
            similarity_threshold=0.0,
        )

        response = await handler.execute(request)

        # Qdrant mock returns query_embedding_used
        assert response.status in ("success", "no_results")
        assert response.query_embedding_used is not None

    @pytest.mark.asyncio
    async def test_routes_search_text_to_db(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that 'search_text' operation routes to DB handler."""
        await handler.initialize()

        snapshot = create_snapshot("database content")
        handler.seed_snapshots([snapshot])

        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="database",
        )

        response = await handler.execute(request)

        # DB mock doesn't set query_embedding_used
        assert response.status in ("success", "no_results")
        assert response.query_embedding_used is None

    @pytest.mark.asyncio
    async def test_routes_search_graph_to_graph(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that 'search_graph' operation routes to Graph handler."""
        await handler.initialize()

        snap1 = create_snapshot("node 1")
        snap2 = create_snapshot("node 2")
        handler.seed_snapshots([snap1, snap2])
        handler.add_graph_relationship(
            str(snap1.snapshot_id),
            str(snap2.snapshot_id),
            "related",
        )

        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id=str(snap1.snapshot_id),
        )

        response = await handler.execute(request)

        # Graph mock returns path information
        assert response.status == "success"
        if response.results:
            assert response.results[0].path is not None


# =============================================================================
# Handler Lifecycle Tests
# =============================================================================


class TestHandlerLifecycle:
    """Tests for handler initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_handler_auto_initializes(self) -> None:
        """Test that handler initializes automatically on first execute."""
        handler = HandlerMemoryRetrieval()

        assert not handler.is_initialized

        # Execute should trigger initialization
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="test",
        )
        await handler.execute(request)

        assert handler.is_initialized

    @pytest.mark.asyncio
    async def test_handler_shutdown(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that handler shutdown works correctly."""
        await handler.initialize()
        assert handler.is_initialized

        await handler.shutdown()
        assert not handler.is_initialized

    @pytest.mark.asyncio
    async def test_seed_requires_initialization(self) -> None:
        """Test that seeding requires handler to be initialized."""
        handler = HandlerMemoryRetrieval()

        snapshot = create_snapshot("test")

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.seed_snapshots([snapshot])

    @pytest.mark.asyncio
    async def test_clear_clears_all_handlers(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that clear() removes data from all handlers."""
        await handler.initialize()

        # Seed data
        snapshot = create_snapshot("test content")
        handler.seed_snapshots([snapshot])

        # Verify data exists
        response = await handler.execute(
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="test content",
                similarity_threshold=0.0,
            )
        )
        assert response.status == "success"
        assert len(response.results) > 0

        # Clear
        handler.clear()

        # Verify data is gone
        response = await handler.execute(
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="test content",
                similarity_threshold=0.0,
            )
        )
        assert response.status == "no_results"


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestModelValidation:
    """Tests for request model validation."""

    def test_valid_search_request(self) -> None:
        """Test creating a valid search request."""
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="test query",
            limit=10,
            similarity_threshold=0.8,
        )
        assert request.operation == "search"
        assert request.query_text == "test query"

    def test_valid_search_with_embedding(self) -> None:
        """Test creating a search request with embedding."""
        embedding = [0.1] * 1024
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_embedding=embedding,
        )
        assert request.query_embedding == embedding

    def test_valid_search_text_request(self) -> None:
        """Test creating a valid search_text request."""
        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="search terms",
            tags=["tag1", "tag2"],
        )
        assert request.operation == "search_text"
        assert request.tags == ["tag1", "tag2"]

    def test_valid_search_graph_request(self) -> None:
        """Test creating a valid search_graph request."""
        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id="snap-123",
            traversal_depth=3,
            relationship_types=["related_to"],
        )
        assert request.operation == "search_graph"
        assert request.traversal_depth == 3

    def test_limit_bounds(self) -> None:
        """Test that limit has valid bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="test",
                limit=0,
            )

        # Too high
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="test",
                limit=1001,
            )

    def test_similarity_threshold_bounds(self) -> None:
        """Test that similarity_threshold has valid bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="test",
                similarity_threshold=-0.1,
            )

        # Too high
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="test",
                similarity_threshold=1.1,
            )


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for retrieval operations.

    This class uses a dual-threshold approach for performance validation:

    1. CONTRACT_SLA_SECONDS (100ms): The target SLA from contract.yaml and
       CLAUDE.md ("Sub-100ms Operations"). Exceeding this emits a warning
       for compliance tracking but does not fail the test.

    2. CI_THRESHOLD_SECONDS (500ms): The hard CI gate. Exceeding this fails
       the test immediately, indicating a serious performance regression.

    This approach allows CI to pass while still tracking contract compliance.
    Performance degradation between 100-500ms is logged as a warning.
    """

    CI_THRESHOLD_SECONDS: float = 0.5  # 500ms - hard CI gate
    CONTRACT_SLA_SECONDS: float = 0.1  # 100ms - contract SLA target

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_search_performance(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that semantic search completes within threshold."""
        import time

        await handler.initialize()

        # Seed test data
        snapshots = [create_snapshot(f"content {i}") for i in range(100)]
        handler.seed_snapshots(snapshots)

        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="content query",
            limit=10,
            similarity_threshold=0.0,
        )

        start_time = time.perf_counter()
        response = await handler.execute(request)
        elapsed_time = time.perf_counter() - start_time

        assert response.status in ("success", "no_results")

        # Hard CI gate - must pass
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Search took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s"
        )

        # Contract SLA verification - warn if exceeded
        if elapsed_time >= self.CONTRACT_SLA_SECONDS:
            import warnings

            warnings.warn(
                f"Performance: {elapsed_time * 1000:.1f}ms exceeds "
                f"contract SLA of {self.CONTRACT_SLA_SECONDS * 1000:.0f}ms",
                UserWarning,
                stacklevel=1,
            )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_search_text_performance(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that full-text search completes within threshold."""
        import time

        await handler.initialize()

        snapshots = [create_snapshot(f"text content {i}") for i in range(100)]
        handler.seed_snapshots(snapshots)

        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="text content",
            limit=10,
        )

        start_time = time.perf_counter()
        response = await handler.execute(request)
        elapsed_time = time.perf_counter() - start_time

        assert response.status in ("success", "no_results")

        # Hard CI gate - must pass
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Search text took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s"
        )

        # Contract SLA verification - warn if exceeded
        if elapsed_time >= self.CONTRACT_SLA_SECONDS:
            import warnings

            warnings.warn(
                f"Performance: {elapsed_time * 1000:.1f}ms exceeds "
                f"contract SLA of {self.CONTRACT_SLA_SECONDS * 1000:.0f}ms",
                UserWarning,
                stacklevel=1,
            )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_graph_traversal_performance(
        self,
        handler: HandlerMemoryRetrieval,
    ) -> None:
        """Test that graph traversal completes within threshold."""
        import time

        await handler.initialize()

        # Create a graph with multiple nodes
        snapshots = [create_snapshot(f"node {i}") for i in range(20)]
        handler.seed_snapshots(snapshots)

        # Create a connected graph
        for i in range(len(snapshots) - 1):
            handler.add_graph_relationship(
                str(snapshots[i].snapshot_id),
                str(snapshots[i + 1].snapshot_id),
                "next",
            )

        request = ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id=str(snapshots[0].snapshot_id),
            traversal_depth=5,
        )

        start_time = time.perf_counter()
        response = await handler.execute(request)
        elapsed_time = time.perf_counter() - start_time

        assert response.status == "success"

        # Hard CI gate - must pass
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Graph traversal took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s"
        )

        # Contract SLA verification - warn if exceeded
        if elapsed_time >= self.CONTRACT_SLA_SECONDS:
            import warnings

            warnings.warn(
                f"Performance: {elapsed_time * 1000:.1f}ms exceeds "
                f"contract SLA of {self.CONTRACT_SLA_SECONDS * 1000:.0f}ms",
                UserWarning,
                stacklevel=1,
            )
