# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for the similarity_compute handler and node.

Covers all operations (cosine_distance, euclidean_distance, compare),
validation edge cases, performance, and numerical stability.

Test Categories:
    1. TestCosineDistance: Cosine distance calculation tests
    2. TestEuclideanDistance: Euclidean distance calculation tests
    3. TestCompare: Full comparison with threshold tests
    4. TestValidation: Strict input validation tests
    5. TestNodeSimilarityCompute: Node wrapper integration tests
    6. TestPerformance: Performance regression tests
    7. TestNumericalStability: Numerical precision tests

Usage:
    pytest tests/nodes/similarity_compute/ -v
    pytest tests/nodes/similarity_compute/test_similarity_compute.py -v -k "cosine"
    pytest tests/nodes/similarity_compute/test_similarity_compute.py -v -k "validation"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1388.
"""

from __future__ import annotations

import math
import os
import time

import pytest
from omnibase_core.container import ModelONEXContainer

from omnimemory.nodes.similarity_compute import (
    HandlerSimilarityCompute,
    ModelHandlerSimilarityComputeConfig,
    ModelSimilarityComputeRequest,
    ModelSimilarityComputeResponse,
    NodeSimilarityCompute,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> ModelHandlerSimilarityComputeConfig:
    """Create a default handler configuration.

    Returns:
        ModelHandlerSimilarityComputeConfig with default settings.
    """
    return ModelHandlerSimilarityComputeConfig()


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create an ONEX container for node testing.

    Returns:
        ModelONEXContainer instance.
    """
    return ModelONEXContainer()


@pytest.fixture
async def handler(container: ModelONEXContainer) -> HandlerSimilarityCompute:
    """Create an initialized handler for testing.

    Uses the container-driven pattern. The handler must be initialized
    before use (fail-fast behavior).

    Args:
        container: ONEX container fixture.

    Returns:
        Initialized HandlerSimilarityCompute instance with container injection.
    """
    h = HandlerSimilarityCompute(container)
    await h.initialize()
    return h


@pytest.fixture
async def node(container: ModelONEXContainer) -> NodeSimilarityCompute:
    """Create an initialized node for testing.

    Args:
        container: ONEX container fixture.

    Returns:
        Initialized NodeSimilarityCompute instance.
    """
    n = NodeSimilarityCompute(container)
    await n.initialize()
    return n


@pytest.fixture
def high_dim_vectors() -> tuple[list[float], list[float]]:
    """Create 1536-dimensional vectors (typical embedding size).

    Returns:
        Tuple of two 1536-dimensional vectors with slightly different values.
    """
    vec_a = [0.5 + (i * 0.0001) for i in range(1536)]
    vec_b = [0.5 + (i * 0.0001) + 0.001 for i in range(1536)]
    return vec_a, vec_b


# =============================================================================
# Cosine Distance Tests
# =============================================================================


class TestCosineDistance:
    """Tests for cosine_distance operation."""

    def test_identical_vectors_return_zero_distance(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Identical vectors should have distance 0.

        Given: Two identical vectors
        When: Computing cosine distance
        Then: Distance should be 0.0
        """
        vec = [1.0, 2.0, 3.0]

        distance = handler.cosine_distance(vec, vec)

        assert distance == 0.0, f"Expected distance 0.0, got {distance}"

    def test_orthogonal_vectors_return_one_distance(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Orthogonal vectors should have distance 1.

        Given: Two orthogonal unit vectors
        When: Computing cosine distance
        Then: Distance should be 1.0
        """
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]

        distance = handler.cosine_distance(vec_a, vec_b)

        assert distance == 1.0, f"Expected distance 1.0, got {distance}"

    def test_opposite_vectors_return_two_distance(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Opposite vectors should have distance 2.

        Given: Two vectors pointing in opposite directions
        When: Computing cosine distance
        Then: Distance should be 2.0
        """
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]

        distance = handler.cosine_distance(vec_a, vec_b)

        assert distance == 2.0, f"Expected distance 2.0, got {distance}"

    def test_known_angle_60_degrees(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Vectors at 60 degrees should have distance 0.5.

        Given: Two unit vectors at 60 degrees angle
        When: Computing cosine distance
        Then: Distance should be approximately 0.5 (since cos(60) = 0.5)
        """
        # cos(60 degrees) = 0.5, so cosine_distance = 1 - 0.5 = 0.5
        # Unit vector at 0 degrees: (1, 0)
        # Unit vector at 60 degrees: (cos(60), sin(60)) = (0.5, sqrt(3)/2)
        vec_a = [1.0, 0.0]
        vec_b = [0.5, math.sqrt(3) / 2]

        distance = handler.cosine_distance(vec_a, vec_b)

        assert abs(distance - 0.5) < 1e-10, f"Expected distance ~0.5, got {distance}"

    def test_known_angle_45_degrees(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Vectors at 45 degrees should have distance ~0.293.

        Given: Two unit vectors at 45 degrees angle
        When: Computing cosine distance
        Then: Distance should be 1 - cos(45) = 1 - sqrt(2)/2 ~ 0.293
        """
        vec_a = [1.0, 0.0]
        vec_b = [math.sqrt(2) / 2, math.sqrt(2) / 2]  # 45 degrees

        distance = handler.cosine_distance(vec_a, vec_b)

        expected = 1.0 - (math.sqrt(2) / 2)  # ~0.293
        assert abs(distance - expected) < 1e-10, (
            f"Expected distance ~{expected}, got {distance}"
        )

    def test_high_dimensional_vectors(
        self,
        handler: HandlerSimilarityCompute,
        high_dim_vectors: tuple[list[float], list[float]],
    ) -> None:
        """Test with 1536-dimensional vectors (typical embedding size).

        Given: Two 1536-dimensional vectors
        When: Computing cosine distance
        Then: Distance should be a valid value in [0, 2]
        """
        vec_a, vec_b = high_dim_vectors

        distance = handler.cosine_distance(vec_a, vec_b)

        assert 0.0 <= distance <= 2.0, f"Distance {distance} out of valid range [0, 2]"

    def test_scaled_vectors_same_distance(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Scaled vectors should have the same cosine distance.

        Given: Two vectors and their scaled versions
        When: Computing cosine distance for original and scaled
        Then: Both distances should be equal (cosine is magnitude-invariant)
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [2.0, 3.0, 4.0]

        # Scale both vectors by different factors
        vec_a_scaled = [x * 100.0 for x in vec_a]
        vec_b_scaled = [x * 0.001 for x in vec_b]

        distance_original = handler.cosine_distance(vec_a, vec_b)
        distance_scaled = handler.cosine_distance(vec_a_scaled, vec_b_scaled)

        assert abs(distance_original - distance_scaled) < 1e-10, (
            f"Scaled vectors should have same distance: "
            f"{distance_original} vs {distance_scaled}"
        )

    def test_negative_components(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test cosine distance with negative vector components.

        Given: Vectors with negative components
        When: Computing cosine distance
        Then: Distance should be computed correctly
        """
        vec_a = [-1.0, -2.0, -3.0]
        vec_b = [-1.0, -2.0, -3.0]

        distance = handler.cosine_distance(vec_a, vec_b)

        assert distance == 0.0, (
            f"Identical vectors should have distance 0, got {distance}"
        )

    def test_mixed_sign_components(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test cosine distance with mixed positive/negative components.

        Given: Vectors with mixed sign components
        When: Computing cosine distance
        Then: Distance should be computed correctly
        """
        vec_a = [1.0, -2.0, 3.0, -4.0]
        vec_b = [-1.0, 2.0, -3.0, 4.0]

        distance = handler.cosine_distance(vec_a, vec_b)

        # These are exactly opposite vectors
        assert distance == 2.0, (
            f"Opposite vectors should have distance 2, got {distance}"
        )


# =============================================================================
# Euclidean Distance Tests
# =============================================================================


class TestEuclideanDistance:
    """Tests for euclidean_distance operation."""

    def test_identical_vectors_return_zero(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Identical vectors should have distance 0.

        Given: Two identical vectors
        When: Computing euclidean distance
        Then: Distance should be 0.0
        """
        vec = [1.0, 2.0, 3.0]

        distance = handler.euclidean_distance(vec, vec)

        assert distance == 0.0, f"Expected distance 0.0, got {distance}"

    def test_unit_distance_vectors(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Vectors differing by 1 in one dimension.

        Given: Two vectors differing by 1 in one dimension
        When: Computing euclidean distance
        Then: Distance should be 1.0
        """
        vec_a = [0.0, 0.0]
        vec_b = [1.0, 0.0]

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance == 1.0, f"Expected distance 1.0, got {distance}"

    def test_pythagorean_triangle_3_4_5(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """3-4-5 triangle should give distance 5.

        Given: Origin and point (3, 4)
        When: Computing euclidean distance
        Then: Distance should be 5.0 (3-4-5 Pythagorean triple)
        """
        vec_a = [0.0, 0.0]
        vec_b = [3.0, 4.0]

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance == 5.0, f"Expected distance 5.0, got {distance}"

    def test_pythagorean_triangle_5_12_13(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """5-12-13 triangle should give distance 13.

        Given: Origin and point (5, 12)
        When: Computing euclidean distance
        Then: Distance should be 13.0 (5-12-13 Pythagorean triple)
        """
        vec_a = [0.0, 0.0]
        vec_b = [5.0, 12.0]

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance == 13.0, f"Expected distance 13.0, got {distance}"

    def test_3d_distance(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test 3D euclidean distance calculation.

        Given: Two points in 3D space
        When: Computing euclidean distance
        Then: Distance should follow sqrt(dx^2 + dy^2 + dz^2)
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 6.0, 3.0]  # dx=3, dy=4, dz=0 -> sqrt(9+16+0) = 5

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance == 5.0, f"Expected distance 5.0, got {distance}"

    def test_high_dimensional_vectors(
        self,
        handler: HandlerSimilarityCompute,
        high_dim_vectors: tuple[list[float], list[float]],
    ) -> None:
        """Test with high-dimensional vectors.

        Given: Two 1536-dimensional vectors
        When: Computing euclidean distance
        Then: Distance should be a valid non-negative value
        """
        vec_a, vec_b = high_dim_vectors

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance >= 0.0, f"Distance {distance} should be non-negative"

    def test_negative_components(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test euclidean distance with negative components.

        Given: Vectors with negative components
        When: Computing euclidean distance
        Then: Distance should be computed correctly
        """
        vec_a = [-3.0, -4.0]
        vec_b = [0.0, 0.0]

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance == 5.0, f"Expected distance 5.0, got {distance}"

    def test_symmetry(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Euclidean distance should be symmetric.

        Given: Two different vectors
        When: Computing distance in both directions
        Then: Both distances should be equal
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]

        distance_ab = handler.euclidean_distance(vec_a, vec_b)
        distance_ba = handler.euclidean_distance(vec_b, vec_a)

        assert distance_ab == distance_ba, (
            f"Distance not symmetric: {distance_ab} vs {distance_ba}"
        )

    def test_zero_vectors_valid_for_euclidean(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Zero magnitude vectors are valid for euclidean distance.

        Given: A zero vector and a non-zero vector
        When: Computing euclidean distance
        Then: Distance should be computed correctly (no error)
        """
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [3.0, 4.0, 0.0]

        distance = handler.euclidean_distance(vec_a, vec_b)

        assert distance == 5.0, f"Expected distance 5.0, got {distance}"


# =============================================================================
# Compare Operation Tests
# =============================================================================


class TestCompare:
    """Tests for compare operation with full result."""

    def test_compare_returns_similarity_result(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Compare should return ModelSimilarityResult.

        Given: Two vectors
        When: Calling compare()
        Then: Result should be a ModelSimilarityResult with expected fields
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.0, 2.0, 3.0]

        result = handler.compare(vec_a, vec_b)

        # Importing here to avoid circular import issues
        from omnimemory.models.memory.model_similarity_result import (
            ModelSimilarityResult,
        )

        assert isinstance(result, ModelSimilarityResult)
        assert result.metric in ("cosine", "euclidean")
        assert result.distance is not None
        assert result.dimensions == 3

    def test_compare_with_cosine_metric(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Compare with cosine metric populates similarity field.

        Given: Two vectors
        When: Calling compare() with metric="cosine"
        Then: Result should have similarity field populated
        """
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 0.0]

        result = handler.compare(vec_a, vec_b, metric="cosine")

        assert result.metric == "cosine"
        assert result.similarity is not None
        assert result.similarity == 1.0  # Identical vectors
        assert result.distance == 0.0

    def test_compare_with_euclidean_metric(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Compare with euclidean metric has None similarity.

        Given: Two vectors
        When: Calling compare() with metric="euclidean"
        Then: Result should have similarity=None
        """
        vec_a = [0.0, 0.0]
        vec_b = [3.0, 4.0]

        result = handler.compare(vec_a, vec_b, metric="euclidean")

        assert result.metric == "euclidean"
        assert result.similarity is None  # Not applicable for euclidean
        assert result.distance == 5.0

    def test_compare_with_threshold_match(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Vectors within threshold should have is_match=True.

        Given: Two similar vectors and a threshold
        When: Calling compare() where distance < threshold
        Then: is_match should be True
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.01, 2.01, 3.01]  # Very similar

        result = handler.compare(vec_a, vec_b, metric="cosine", threshold=0.1)

        assert result.is_match is True
        assert result.threshold == 0.1

    def test_compare_with_threshold_no_match(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Vectors outside threshold should have is_match=False.

        Given: Two dissimilar vectors and a threshold
        When: Calling compare() where distance > threshold
        Then: is_match should be False
        """
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]  # Orthogonal, distance = 1.0

        result = handler.compare(vec_a, vec_b, metric="cosine", threshold=0.5)

        assert result.is_match is False
        assert result.threshold == 0.5

    def test_compare_without_threshold(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Without threshold, is_match should be None.

        Given: Two vectors without a threshold
        When: Calling compare() without threshold parameter
        Then: is_match should be None
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]

        result = handler.compare(vec_a, vec_b, metric="cosine")

        assert result.is_match is None
        assert result.threshold is None

    def test_dimensions_field_populated(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Result should include correct dimensions count.

        Given: Vectors of specific dimensions
        When: Calling compare()
        Then: dimensions field should match vector length
        """
        vec_a = [1.0] * 512
        vec_b = [0.5] * 512

        result = handler.compare(vec_a, vec_b)

        assert result.dimensions == 512

    def test_compare_threshold_exact_boundary(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test is_match when distance equals threshold exactly.

        Given: Vectors with known distance equal to threshold
        When: Calling compare() with threshold == distance
        Then: is_match should be True (<=)
        """
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]  # Orthogonal, cosine distance = 1.0

        result = handler.compare(vec_a, vec_b, metric="cosine", threshold=1.0)

        assert result.is_match is True, "Boundary case should match (<=)"

    def test_compare_euclidean_with_threshold(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test euclidean compare with threshold.

        Given: Two vectors with known euclidean distance
        When: Calling compare() with euclidean metric and threshold
        Then: is_match should reflect threshold comparison
        """
        vec_a = [0.0, 0.0]
        vec_b = [3.0, 4.0]  # Distance = 5.0

        result = handler.compare(vec_a, vec_b, metric="euclidean", threshold=10.0)

        assert result.is_match is True
        assert result.distance == 5.0


# =============================================================================
# Validation Error Tests (CRITICAL)
# =============================================================================


class TestValidation:
    """Tests for strict input validation."""

    def test_empty_vector_a_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Empty first vector should raise ValueError.

        Given: An empty first vector
        When: Calling cosine_distance
        Then: ValueError should be raised with appropriate message
        """
        with pytest.raises(ValueError, match="empty"):
            handler.cosine_distance([], [1.0])

    def test_empty_vector_b_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Empty second vector should raise ValueError.

        Given: An empty second vector
        When: Calling cosine_distance
        Then: ValueError should be raised with appropriate message
        """
        with pytest.raises(ValueError, match="empty"):
            handler.cosine_distance([1.0], [])

    def test_both_vectors_empty_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Both vectors empty should raise ValueError.

        Given: Both vectors empty
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="empty"):
            handler.cosine_distance([], [])

    def test_dimension_mismatch_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Mismatched dimensions should raise ValueError.

        Given: Two vectors with different dimensions
        When: Calling cosine_distance
        Then: ValueError should be raised with mismatch message
        """
        with pytest.raises(ValueError, match="[Mm]ismatch|dimension"):
            handler.cosine_distance([1.0, 2.0], [1.0])

    def test_dimension_mismatch_euclidean_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Mismatched dimensions should raise ValueError for euclidean.

        Given: Two vectors with different dimensions
        When: Calling euclidean_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="[Mm]ismatch|dimension"):
            handler.euclidean_distance([1.0, 2.0, 3.0], [1.0])

    def test_nan_in_vector_a_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """NaN in first vector should raise ValueError.

        Given: First vector containing NaN
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="NaN"):
            handler.cosine_distance([float("nan"), 1.0], [1.0, 1.0])

    def test_nan_in_vector_b_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """NaN in second vector should raise ValueError.

        Given: Second vector containing NaN
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="NaN"):
            handler.cosine_distance([1.0, 1.0], [float("nan"), 1.0])

    def test_positive_inf_in_vector_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Positive infinity in vector should raise ValueError.

        Given: Vector containing positive infinity
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="[Ii]nf"):
            handler.cosine_distance([float("inf"), 1.0], [1.0, 1.0])

    def test_negative_inf_in_vector_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Negative infinity in vector should raise ValueError.

        Given: Vector containing negative infinity
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="[Ii]nf"):
            handler.cosine_distance([float("-inf"), 1.0], [1.0, 1.0])

    def test_inf_in_euclidean_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Infinity should also raise error for euclidean distance.

        Given: Vector containing infinity
        When: Calling euclidean_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="[Ii]nf"):
            handler.euclidean_distance([float("inf"), 1.0], [1.0, 1.0])

    def test_zero_magnitude_cosine_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Zero magnitude vector for cosine should raise ValueError.

        Given: A zero-magnitude vector
        When: Calling cosine_distance
        Then: ValueError should be raised with magnitude message
        """
        with pytest.raises(ValueError, match="[Zz]ero|[Mm]agnitude"):
            handler.cosine_distance([0.0, 0.0], [1.0, 1.0])

    def test_zero_magnitude_second_vector_cosine_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Zero magnitude in second vector for cosine should raise ValueError.

        Given: Second vector has zero magnitude
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        with pytest.raises(ValueError, match="[Zz]ero|[Mm]agnitude"):
            handler.cosine_distance([1.0, 1.0], [0.0, 0.0])

    def test_zero_magnitude_euclidean_succeeds(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Zero magnitude is valid for euclidean distance.

        Given: A zero-magnitude vector
        When: Calling euclidean_distance
        Then: Should NOT raise - euclidean can handle zero vectors
        """
        # This should NOT raise - euclidean distance is defined for zero vectors
        distance = handler.euclidean_distance([0.0, 0.0], [1.0, 1.0])

        expected = math.sqrt(2.0)  # sqrt(1^2 + 1^2)
        assert abs(distance - expected) < 1e-10

    def test_near_zero_magnitude_cosine_raises_error(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Near-zero magnitude vector for cosine should raise ValueError.

        Given: A very small magnitude vector (smaller than epsilon)
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        # Values much smaller than epsilon (1e-10)
        with pytest.raises(ValueError, match="[Zz]ero|[Mm]agnitude"):
            handler.cosine_distance([1e-20, 1e-20], [1.0, 1.0])

    def test_nan_at_various_indices(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """NaN at different indices should still raise error.

        Given: NaN at various positions in vector
        When: Calling cosine_distance
        Then: ValueError should be raised
        """
        # NaN at end
        with pytest.raises(ValueError, match="NaN"):
            handler.cosine_distance([1.0, 2.0, float("nan")], [1.0, 2.0, 3.0])

        # NaN in middle
        with pytest.raises(ValueError, match="NaN"):
            handler.cosine_distance([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])


# =============================================================================
# Node Integration Tests
# =============================================================================


class TestNodeSimilarityCompute:
    """Tests for the node wrapper."""

    def test_node_routes_cosine_distance(
        self,
        node: NodeSimilarityCompute,
    ) -> None:
        """Node correctly routes cosine_distance operation.

        Given: A cosine_distance request
        When: Executing through the node
        Then: Response contains correct distance
        """
        request = ModelSimilarityComputeRequest(
            operation="cosine_distance",
            vector_a=[1.0, 0.0],
            vector_b=[1.0, 0.0],
        )

        response = node.execute(request)

        assert response.status == "success"
        assert response.distance == 0.0
        assert response.dimensions == 2

    def test_node_routes_euclidean_distance(
        self,
        node: NodeSimilarityCompute,
    ) -> None:
        """Node correctly routes euclidean_distance operation.

        Given: A euclidean_distance request
        When: Executing through the node
        Then: Response contains correct distance
        """
        request = ModelSimilarityComputeRequest(
            operation="euclidean_distance",
            vector_a=[0.0, 0.0],
            vector_b=[3.0, 4.0],
        )

        response = node.execute(request)

        assert response.status == "success"
        assert response.distance == 5.0
        assert response.dimensions == 2

    def test_node_routes_compare(
        self,
        node: NodeSimilarityCompute,
    ) -> None:
        """Node correctly routes compare operation.

        Given: A compare request
        When: Executing through the node
        Then: Response contains full comparison results
        """
        request = ModelSimilarityComputeRequest(
            operation="compare",
            vector_a=[1.0, 0.0],
            vector_b=[0.0, 1.0],
            metric="cosine",
            threshold=0.5,
        )

        response = node.execute(request)

        assert response.status == "success"
        assert response.distance == 1.0  # Orthogonal vectors
        assert response.similarity == 0.0  # cos(90) = 0
        assert response.is_match is False  # 1.0 > 0.5

    def test_node_returns_error_response_on_validation_error(
        self,
        node: NodeSimilarityCompute,
    ) -> None:
        """Node converts ValueError to error response.

        Given: A request with invalid vectors (dimension mismatch)
        When: Executing through the node
        Then: Response has error status and message
        """
        # Note: Pydantic validation on the request model will catch dimension mismatch
        # So we need to test a different validation error - like NaN values
        request = ModelSimilarityComputeRequest(
            operation="cosine_distance",
            vector_a=[float("nan"), 1.0],
            vector_b=[1.0, 1.0],
        )

        response = node.execute(request)

        assert response.status == "error"
        assert response.error_message is not None
        assert "NaN" in response.error_message

    def test_node_returns_error_for_zero_magnitude(
        self,
        node: NodeSimilarityCompute,
    ) -> None:
        """Node returns error response for zero magnitude vectors.

        Given: A request with zero magnitude vector
        When: Executing cosine_distance through the node
        Then: Response has error status
        """
        request = ModelSimilarityComputeRequest(
            operation="cosine_distance",
            vector_a=[0.0, 0.0],
            vector_b=[1.0, 1.0],
        )

        response = node.execute(request)

        assert response.status == "error"
        assert response.error_message is not None

    def test_node_compare_euclidean_no_similarity(
        self,
        node: NodeSimilarityCompute,
    ) -> None:
        """Node compare with euclidean returns None similarity.

        Given: A compare request with euclidean metric
        When: Executing through the node
        Then: Response has None for similarity field
        """
        request = ModelSimilarityComputeRequest(
            operation="compare",
            vector_a=[0.0, 0.0],
            vector_b=[3.0, 4.0],
            metric="euclidean",
        )

        response = node.execute(request)

        assert response.status == "success"
        assert response.distance == 5.0
        assert response.similarity is None

    def test_node_container_access(
        self,
        node: NodeSimilarityCompute,
        container: ModelONEXContainer,
    ) -> None:
        """Node provides access to injected container.

        Given: A node created with a container
        When: Accessing the container property
        Then: Same container instance should be returned
        """
        assert node.container is container


# =============================================================================
# Performance Tests
# =============================================================================

# Check if running in CI environment (must be defined before PERF_THRESHOLD_MS)
IS_CI = os.getenv("CI") == "true"

# Performance threshold in milliseconds for batch operations
# Default: 100ms for local development, 1000ms for CI (shared runners have high variance)
# CI runners (GitHub Actions, etc.) are significantly slower than local machines
# and have unpredictable variance, so we use a 10x higher threshold.
# Can override via PERF_THRESHOLD_MS env var for custom tolerance.
_DEFAULT_PERF_THRESHOLD = "1000" if IS_CI else "100"
PERF_THRESHOLD_MS = int(os.getenv("PERF_THRESHOLD_MS", _DEFAULT_PERF_THRESHOLD))


class TestPerformance:
    """Performance regression tests.

    These tests verify that similarity operations complete within acceptable
    time thresholds. The handler documentation specifies <0.1ms for 1536-dim vectors.

    Note: Sub-millisecond tests are skipped in CI because shared runners have
    high timing variance that makes sub-ms thresholds unreliable.
    """

    # Performance threshold in milliseconds for local testing
    THRESHOLD_MS: float = 1.0  # 1ms threshold for local development

    @pytest.mark.skipif(IS_CI, reason="Sub-ms tests unreliable on shared CI runners")
    def test_cosine_distance_sub_millisecond(
        self,
        handler: HandlerSimilarityCompute,
        high_dim_vectors: tuple[list[float], list[float]],
    ) -> None:
        """Cosine distance should complete in <1ms for 1536-dim vectors.

        Given: Two 1536-dimensional vectors
        When: Computing cosine distance
        Then: Operation completes within 1ms
        """
        vec_a, vec_b = high_dim_vectors

        start = time.perf_counter()
        handler.cosine_distance(vec_a, vec_b)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Cosine distance took {elapsed_ms:.3f}ms, expected <{self.THRESHOLD_MS}ms"
        )

    @pytest.mark.skipif(IS_CI, reason="Sub-ms tests unreliable on shared CI runners")
    def test_euclidean_distance_sub_millisecond(
        self,
        handler: HandlerSimilarityCompute,
        high_dim_vectors: tuple[list[float], list[float]],
    ) -> None:
        """Euclidean distance should complete in <1ms for 1536-dim vectors.

        Given: Two 1536-dimensional vectors
        When: Computing euclidean distance
        Then: Operation completes within 1ms
        """
        vec_a, vec_b = high_dim_vectors

        start = time.perf_counter()
        handler.euclidean_distance(vec_a, vec_b)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Euclidean distance took {elapsed_ms:.3f}ms, "
            f"expected <{self.THRESHOLD_MS}ms"
        )

    @pytest.mark.skipif(IS_CI, reason="Sub-ms tests unreliable on shared CI runners")
    def test_compare_sub_millisecond(
        self,
        handler: HandlerSimilarityCompute,
        high_dim_vectors: tuple[list[float], list[float]],
    ) -> None:
        """Compare operation should complete in <1ms for 1536-dim vectors.

        Given: Two 1536-dimensional vectors
        When: Performing full comparison
        Then: Operation completes within 1ms
        """
        vec_a, vec_b = high_dim_vectors

        start = time.perf_counter()
        handler.compare(vec_a, vec_b, metric="cosine", threshold=0.5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Compare operation took {elapsed_ms:.3f}ms, expected <{self.THRESHOLD_MS}ms"
        )

    @pytest.mark.benchmark
    @pytest.mark.skipif(
        IS_CI,
        reason="Batch performance tests are unreliable on shared CI runners due to "
        "variable CPU allocation, resource contention, and timing variance. "
        "Run locally to catch performance regressions.",
    )
    def test_cosine_batch_performance(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Benchmark: ~900 cosine distance calculations.

        Given: 100 vectors of 512 dimensions, computing pairwise distances
        When: Computing cosine distance for ~900 pairs
        Then: Total time should be within threshold (<100ms local)

        Note: This test is skipped in CI environments because shared runners
        have unpredictable performance characteristics that make timing-based
        assertions unreliable. The test still provides value for local
        development to catch performance regressions.
        """
        # Create test vectors
        vectors = [[0.5 + (i * 0.001)] * 512 for i in range(100)]

        start = time.perf_counter()
        for i in range(100):
            for j in range(i + 1, min(i + 10, 100)):
                handler.cosine_distance(vectors[i], vectors[j])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete ~900 operations within 100ms on local development machines
        # This test is skipped in CI, so we use a strict local threshold
        local_threshold_ms = 100
        assert elapsed_ms < local_threshold_ms, (
            f"Batch operations took {elapsed_ms:.1f}ms, "
            f"expected <{local_threshold_ms}ms (performance regression detected)"
        )


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical precision and stability."""

    def test_large_magnitude_vectors(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Should handle large magnitude vectors without overflow.

        Given: Vectors with very large values
        When: Computing cosine distance
        Then: Should not raise or return inf
        """
        vec_a = [1e10] * 100
        vec_b = [1e10] * 100

        distance = handler.cosine_distance(vec_a, vec_b)

        assert not math.isinf(distance), "Distance should not be infinite"
        assert not math.isnan(distance), "Distance should not be NaN"
        assert distance == 0.0, "Identical vectors should have distance 0"

    def test_small_magnitude_vectors(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Should handle small magnitude vectors without underflow issues.

        Given: Vectors with very small but non-zero values
        When: Computing cosine distance
        Then: Should not raise or return incorrect values
        """
        # Note: Must be above epsilon (1e-10) to pass magnitude check
        vec_a = [1e-5] * 100
        vec_b = [1e-5] * 100

        distance = handler.cosine_distance(vec_a, vec_b)

        assert not math.isnan(distance), "Distance should not be NaN"
        assert distance == 0.0, "Identical vectors should have distance 0"

    def test_mixed_magnitude_vectors(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Should handle mixed magnitudes correctly.

        Given: Identical vectors with mixed large/small values
        When: Computing cosine distance
        Then: Should return ~0 for identical vectors
        """
        vec_a = [1e8, 1e-5, 1.0, 1e6, 1e-3]
        vec_b = [1e8, 1e-5, 1.0, 1e6, 1e-3]

        distance = handler.cosine_distance(vec_a, vec_b)

        assert abs(distance) < 1e-10, (
            f"Identical vectors should have distance ~0, got {distance}"
        )

    def test_euclidean_large_differences(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Euclidean distance with large value differences.

        Given: Vectors with large value differences
        When: Computing euclidean distance
        Then: Should compute correctly without overflow
        """
        vec_a = [0.0] * 10
        vec_b = [1e7] * 10  # Large difference

        distance = handler.euclidean_distance(vec_a, vec_b)

        expected = math.sqrt(10 * (1e7**2))
        assert not math.isinf(distance), "Distance should not be infinite"
        assert abs(distance - expected) < 1e-5, f"Expected ~{expected}, got {distance}"

    def test_floating_point_precision(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test that math.fsum provides accurate summation.

        Given: Vectors that would lose precision with naive summation
        When: Computing cosine distance
        Then: Result should be accurate
        """
        # Many small values that would accumulate error with naive sum
        vec_a = [0.1] * 1000
        vec_b = [0.1] * 1000

        distance = handler.cosine_distance(vec_a, vec_b)

        # Identical vectors should give exactly 0
        assert distance == 0.0, f"Expected exact 0, got {distance}"

    def test_cosine_similarity_clamping(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Test that cosine similarity is clamped to [-1, 1].

        Given: Vectors that might produce out-of-bounds similarity due to floating point
        When: Computing cosine distance
        Then: Distance should be in valid range [0, 2]
        """
        # Use values that stress floating point precision
        vec_a = [1.0 + 1e-15] * 100
        vec_b = [1.0 + 1e-15] * 100

        distance = handler.cosine_distance(vec_a, vec_b)

        assert 0.0 <= distance <= 2.0, f"Distance {distance} out of valid range [0, 2]"

    def test_euclidean_zero_distance_precision(
        self,
        handler: HandlerSimilarityCompute,
    ) -> None:
        """Euclidean distance of identical vectors should be exactly 0.

        Given: Identical vectors
        When: Computing euclidean distance
        Then: Result should be exactly 0
        """
        vec = [1.23456789012345] * 500

        distance = handler.euclidean_distance(vec, vec)

        assert distance == 0.0, f"Expected exact 0, got {distance}"


# =============================================================================
# Request Model Validation Tests
# =============================================================================


class TestRequestModelValidation:
    """Tests for ModelSimilarityComputeRequest validation."""

    def test_request_dimension_mismatch_rejected(self) -> None:
        """Request model rejects mismatched vector dimensions.

        Given: Attempting to create request with mismatched dimensions
        When: Constructing ModelSimilarityComputeRequest
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="[Mm]ismatch|dimension"):
            ModelSimilarityComputeRequest(
                operation="cosine_distance",
                vector_a=[1.0, 2.0, 3.0],
                vector_b=[1.0, 2.0],  # Different length
            )

    def test_request_empty_vector_a_rejected(self) -> None:
        """Request model rejects empty vector_a.

        Given: Attempting to create request with empty vector_a
        When: Constructing ModelSimilarityComputeRequest
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelSimilarityComputeRequest(
                operation="cosine_distance",
                vector_a=[],
                vector_b=[1.0],
            )

    def test_request_empty_vector_b_rejected(self) -> None:
        """Request model rejects empty vector_b.

        Given: Attempting to create request with empty vector_b
        When: Constructing ModelSimilarityComputeRequest
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelSimilarityComputeRequest(
                operation="cosine_distance",
                vector_a=[1.0],
                vector_b=[],
            )

    def test_request_invalid_operation_rejected(self) -> None:
        """Request model rejects invalid operation values.

        Given: Attempting to create request with invalid operation
        When: Constructing ModelSimilarityComputeRequest
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelSimilarityComputeRequest(
                operation="invalid_operation",  # type: ignore[arg-type]
                vector_a=[1.0],
                vector_b=[1.0],
            )

    def test_request_negative_threshold_rejected(self) -> None:
        """Request model rejects negative threshold values.

        Given: Attempting to create request with negative threshold
        When: Constructing ModelSimilarityComputeRequest
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelSimilarityComputeRequest(
                operation="compare",
                vector_a=[1.0],
                vector_b=[1.0],
                threshold=-0.5,  # Negative threshold
            )

    def test_request_valid_cosine_distance(self) -> None:
        """Valid cosine_distance request is accepted.

        Given: Valid parameters for cosine_distance
        When: Constructing ModelSimilarityComputeRequest
        Then: Request is created successfully
        """
        request = ModelSimilarityComputeRequest(
            operation="cosine_distance",
            vector_a=[1.0, 2.0, 3.0],
            vector_b=[4.0, 5.0, 6.0],
        )

        assert request.operation == "cosine_distance"
        assert len(request.vector_a) == 3
        assert len(request.vector_b) == 3

    def test_request_valid_compare_with_all_options(self) -> None:
        """Valid compare request with all options is accepted.

        Given: Valid parameters for compare with threshold and metric
        When: Constructing ModelSimilarityComputeRequest
        Then: Request is created successfully
        """
        request = ModelSimilarityComputeRequest(
            operation="compare",
            vector_a=[0.5] * 10,
            vector_b=[0.6] * 10,
            metric="euclidean",
            threshold=1.5,
        )

        assert request.operation == "compare"
        assert request.metric == "euclidean"
        assert request.threshold == 1.5


# =============================================================================
# Response Model Tests
# =============================================================================


class TestResponseModel:
    """Tests for ModelSimilarityComputeResponse structure."""

    def test_success_response_structure(self) -> None:
        """Success response has expected fields populated.

        Given: A successful operation response
        When: Creating ModelSimilarityComputeResponse
        Then: All relevant fields should be populated
        """
        response = ModelSimilarityComputeResponse(
            status="success",
            distance=0.5,
            similarity=0.5,
            is_match=True,
            dimensions=100,
        )

        assert response.status == "success"
        assert response.distance == 0.5
        assert response.similarity == 0.5
        assert response.is_match is True
        assert response.dimensions == 100
        assert response.error_message is None

    def test_error_response_structure(self) -> None:
        """Error response has expected fields populated.

        Given: A failed operation response
        When: Creating ModelSimilarityComputeResponse
        Then: Status should be error and error_message populated
        """
        response = ModelSimilarityComputeResponse(
            status="error",
            error_message="Vector dimension mismatch",
        )

        assert response.status == "error"
        assert response.error_message is not None
        assert "dimension" in response.error_message.lower()
        assert response.distance is None

    def test_response_forbids_extra_fields(self) -> None:
        """Response model forbids extra fields.

        Given: Attempting to create response with extra field
        When: Constructing ModelSimilarityComputeResponse
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelSimilarityComputeResponse(
                status="success",
                distance=0.5,
                extra_field="not allowed",  # type: ignore[call-arg]
            )


# =============================================================================
# Config Tests
# =============================================================================


class TestHandlerConfig:
    """Tests for ModelHandlerSimilarityComputeConfig."""

    def test_default_epsilon(self) -> None:
        """Default epsilon is 1e-10.

        Given: Default configuration
        When: Creating ModelHandlerSimilarityComputeConfig
        Then: Epsilon should be 1e-10
        """
        config = ModelHandlerSimilarityComputeConfig()

        assert config.epsilon == 1e-10

    def test_custom_epsilon(self) -> None:
        """Custom epsilon can be specified.

        Given: Custom epsilon value
        When: Creating ModelHandlerSimilarityComputeConfig
        Then: Custom value should be used
        """
        config = ModelHandlerSimilarityComputeConfig(epsilon=1e-8)

        assert config.epsilon == 1e-8

    def test_epsilon_must_be_positive(self) -> None:
        """Epsilon must be greater than 0.

        Given: Zero or negative epsilon
        When: Creating ModelHandlerSimilarityComputeConfig
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelHandlerSimilarityComputeConfig(epsilon=0.0)

        with pytest.raises(ValidationError):
            ModelHandlerSimilarityComputeConfig(epsilon=-1e-10)

    def test_config_forbids_extra_fields(self) -> None:
        """Config model forbids extra fields.

        Given: Attempting to create config with extra field
        When: Constructing ModelHandlerSimilarityComputeConfig
        Then: Pydantic raises ValidationError
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelHandlerSimilarityComputeConfig(
                epsilon=1e-10,
                unknown_param=True,  # type: ignore[call-arg]
            )

    @pytest.mark.asyncio
    async def test_handler_uses_config_epsilon(self) -> None:
        """Handler uses epsilon from config for magnitude checks.

        Given: Custom config with larger epsilon
        When: Checking vectors near zero
        Then: Magnitude check should use custom epsilon
        """
        # With larger epsilon, more vectors are considered "zero"
        config = ModelHandlerSimilarityComputeConfig(epsilon=1e-3)
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize(config)

        # This vector has magnitude ~1.4e-4 which is < 1e-3
        with pytest.raises(ValueError, match="[Zz]ero|[Mm]agnitude"):
            handler.cosine_distance([1e-4, 1e-4], [1.0, 1.0])


# =============================================================================
# Fail-Fast Behavior Tests
# =============================================================================


class TestFailFastBehavior:
    """Tests for handler fail-fast behavior before initialization."""

    def test_config_raises_before_initialization(self) -> None:
        """Accessing config before initialization raises RuntimeError.

        Given: An uninitialized handler
        When: Accessing the config property
        Then: RuntimeError should be raised with descriptive message
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = handler.config

    def test_cosine_distance_raises_before_initialization(self) -> None:
        """Calling cosine_distance before initialization raises RuntimeError.

        Given: An uninitialized handler
        When: Calling cosine_distance
        Then: RuntimeError should be raised
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.cosine_distance([1.0, 2.0], [3.0, 4.0])

    def test_euclidean_distance_raises_before_initialization(self) -> None:
        """Calling euclidean_distance before initialization raises RuntimeError.

        Given: An uninitialized handler
        When: Calling euclidean_distance
        Then: RuntimeError should be raised
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.euclidean_distance([1.0, 2.0], [3.0, 4.0])

    def test_compare_raises_before_initialization(self) -> None:
        """Calling compare before initialization raises RuntimeError.

        Given: An uninitialized handler
        When: Calling compare
        Then: RuntimeError should be raised
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.compare([1.0, 2.0], [3.0, 4.0])

    @pytest.mark.asyncio
    async def test_methods_work_after_initialization(self) -> None:
        """All methods work normally after initialization.

        Given: A properly initialized handler
        When: Calling compute methods
        Then: Methods should succeed without errors
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()

        # All these should succeed after initialization
        config = handler.config
        assert config is not None

        distance = handler.cosine_distance([1.0, 0.0], [1.0, 0.0])
        assert distance == 0.0

        distance = handler.euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert distance == 5.0

        result = handler.compare([1.0, 0.0], [0.0, 1.0])
        assert result.distance == 1.0
