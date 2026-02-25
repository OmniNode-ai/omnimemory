# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for the container-driven handler lifecycle pattern.

This module validates the full lifecycle of handlers that follow the
container-driven pattern established in PR #28 (OMN-1577). All handlers
refactored to this pattern share these lifecycle methods:

    - __init__(container: ModelONEXContainer): Constructor with DI container
    - initialize(): Async setup of dependencies
    - health_check(): Returns health status with initialized flag
    - describe(): Returns handler metadata and capabilities
    - shutdown(): Cleanup resources and reset state

The pattern enforces fail-fast behavior: operations raise RuntimeError
if called before initialize() or after shutdown().

This test uses HandlerSimilarityCompute as the reference implementation
because it is the simplest pure-compute handler with no external dependencies.

Usage:
    pytest tests/integration/test_handler_lifecycle_pattern.py -v
    pytest tests/integration/test_handler_lifecycle_pattern.py -v -k "lifecycle"

.. versionadded:: 0.2.0
    Added for OMN-1577 to validate handler lifecycle pattern.
"""

from __future__ import annotations

import pytest
from omnibase_core.container import ModelONEXContainer

from omnimemory.nodes.similarity_compute.handlers import (
    HandlerSimilarityCompute,
    ModelHandlerSimilarityComputeConfig,
)


class TestHandlerLifecyclePattern:
    """Integration tests for the container-driven handler lifecycle pattern.

    These tests validate the complete lifecycle of a handler from creation
    through initialization, operation, and shutdown. The pattern ensures
    predictable behavior and fail-fast semantics.
    """

    @pytest.mark.asyncio
    async def test_handler_creation_with_container(self) -> None:
        """Handler can be created with a container.

        Given: A valid ONEX container
        When: Creating a handler with the container
        Then: Handler is created successfully with container stored
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        assert handler is not None
        assert handler.container is container

    @pytest.mark.asyncio
    async def test_health_check_before_initialize_shows_not_initialized(self) -> None:
        """Health check before initialize reports initialized=False.

        Given: A handler that has not been initialized
        When: Calling health_check()
        Then: Health status shows initialized=False
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        health = await handler.health_check()

        assert health.initialized is False
        assert health.handler == "similarity_compute"
        # Uninitialized handler reports unhealthy (cannot perform computations)
        assert health.healthy is False

    @pytest.mark.asyncio
    async def test_initialize_succeeds(self) -> None:
        """Handler initialization completes successfully.

        Given: A newly created handler
        When: Calling initialize()
        Then: Handler is initialized without errors
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        # Should not raise
        await handler.initialize()

        # Verify internal state changed
        health = await handler.health_check()
        assert health.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(self) -> None:
        """Handler can be initialized with custom configuration.

        Given: A handler and custom configuration
        When: Calling initialize(config)
        Then: Handler uses the provided configuration
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        custom_config = ModelHandlerSimilarityComputeConfig(epsilon=1e-8)

        await handler.initialize(config=custom_config)

        assert handler.config.epsilon == 1e-8

    @pytest.mark.asyncio
    async def test_health_check_after_initialize_shows_healthy(self) -> None:
        """Health check after initialize reports healthy and initialized.

        Given: A handler that has been initialized
        When: Calling health_check()
        Then: Health status shows initialized=True and healthy=True
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()

        health = await handler.health_check()

        assert health.initialized is True
        assert health.healthy is True
        assert health.handler == "similarity_compute"

    @pytest.mark.asyncio
    async def test_describe_returns_expected_metadata(self) -> None:
        """Describe returns handler metadata and capabilities.

        Given: An initialized handler
        When: Calling describe()
        Then: Metadata includes handler_type, capabilities, and metrics
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()

        metadata = await handler.describe()

        assert metadata.handler_type == "similarity_compute"
        assert metadata.is_pure_compute is True
        assert metadata.initialized is True
        assert "cosine_distance" in metadata.capabilities
        assert "euclidean_distance" in metadata.capabilities
        assert "compare" in metadata.capabilities
        assert "cosine" in metadata.supported_metrics
        assert "euclidean" in metadata.supported_metrics

    @pytest.mark.asyncio
    async def test_describe_before_initialize_shows_not_initialized(self) -> None:
        """Describe before initialize shows initialized=False.

        Given: A handler that has not been initialized
        When: Calling describe()
        Then: Metadata shows initialized=False
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        metadata = await handler.describe()

        assert metadata.initialized is False
        # Other metadata should still be available
        assert metadata.handler_type == "similarity_compute"

    @pytest.mark.asyncio
    async def test_operations_work_after_initialize(self) -> None:
        """Compute operations work correctly after initialization.

        Given: An initialized handler
        When: Calling compute operations
        Then: Operations complete successfully with correct results
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()

        # Test cosine distance - orthogonal vectors have distance 1.0
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        cosine_dist = handler.cosine_distance(vec_a, vec_b)
        assert cosine_dist == 1.0

        # Test euclidean distance - 3-4-5 triangle
        vec_a = [0.0, 0.0]
        vec_b = [3.0, 4.0]
        euclidean_dist = handler.euclidean_distance(vec_a, vec_b)
        assert euclidean_dist == 5.0

        # Test compare with threshold
        result = handler.compare([1.0, 0.0], [0.0, 1.0], metric="cosine", threshold=0.5)
        assert result.distance == 1.0
        assert result.is_match is False  # 1.0 > 0.5

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self) -> None:
        """Shutdown resets handler to uninitialized state.

        Given: An initialized handler
        When: Calling shutdown()
        Then: Handler returns to uninitialized state
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()

        # Verify initialized
        health = await handler.health_check()
        assert health.initialized is True

        # Shutdown
        await handler.shutdown()

        # Verify reset to uninitialized
        health = await handler.health_check()
        assert health.initialized is False

    @pytest.mark.asyncio
    async def test_operations_fail_after_shutdown(self) -> None:
        """Operations raise RuntimeError after shutdown.

        Given: A handler that was initialized then shutdown
        When: Calling compute operations
        Then: RuntimeError is raised with descriptive message
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()
        await handler.shutdown()

        # All operations should fail with RuntimeError
        with pytest.raises(RuntimeError, match="not initialized"):
            handler.cosine_distance([1.0, 0.0], [0.0, 1.0])

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.euclidean_distance([1.0, 0.0], [0.0, 1.0])

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.compare([1.0, 0.0], [0.0, 1.0])

    @pytest.mark.asyncio
    async def test_operations_fail_before_initialize(self) -> None:
        """Operations raise RuntimeError before initialization.

        Given: A handler that has not been initialized
        When: Calling compute operations
        Then: RuntimeError is raised with descriptive message
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.cosine_distance([1.0, 0.0], [0.0, 1.0])

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.euclidean_distance([1.0, 0.0], [0.0, 1.0])

        with pytest.raises(RuntimeError, match="not initialized"):
            handler.compare([1.0, 0.0], [0.0, 1.0])

    @pytest.mark.asyncio
    async def test_config_access_fails_before_initialize(self) -> None:
        """Config property raises RuntimeError before initialization.

        Given: A handler that has not been initialized
        When: Accessing the config property
        Then: RuntimeError is raised
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = handler.config

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self) -> None:
        """Shutdown can be called multiple times safely.

        Given: An initialized handler
        When: Calling shutdown() multiple times
        Then: No error is raised and state remains uninitialized
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)
        await handler.initialize()

        # Multiple shutdowns should not raise
        await handler.shutdown()
        await handler.shutdown()
        await handler.shutdown()

        health = await handler.health_check()
        assert health.initialized is False

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(self) -> None:
        """Handler can be reinitialized after shutdown.

        Given: A handler that was initialized, used, and shutdown
        When: Calling initialize() again
        Then: Handler becomes operational again
        """
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        # First lifecycle
        await handler.initialize()
        result1 = handler.cosine_distance([1.0, 0.0], [1.0, 0.0])
        assert result1 == 0.0
        await handler.shutdown()

        # Verify shutdown
        with pytest.raises(RuntimeError):
            handler.cosine_distance([1.0, 0.0], [1.0, 0.0])

        # Second lifecycle - reinitialize
        await handler.initialize()
        result2 = handler.cosine_distance([1.0, 0.0], [1.0, 0.0])
        assert result2 == 0.0

        health = await handler.health_check()
        assert health.initialized is True


class TestHandlerLifecycleFullCycle:
    """End-to-end test of the complete handler lifecycle.

    This test validates the entire lifecycle in a single test case,
    mimicking real-world usage patterns.
    """

    @pytest.mark.asyncio
    async def test_complete_handler_lifecycle(self) -> None:
        """Complete lifecycle: create -> init -> use -> shutdown.

        This test validates the full lifecycle pattern that all
        container-driven handlers must follow.
        """
        # Phase 1: Creation
        container = ModelONEXContainer()
        handler = HandlerSimilarityCompute(container)

        # Verify: uninitialized state
        health = await handler.health_check()
        assert health.initialized is False
        assert health.handler == "similarity_compute"

        # Phase 2: Initialization
        await handler.initialize()

        # Verify: initialized state
        health = await handler.health_check()
        assert health.initialized is True
        assert health.healthy is True

        # Verify: metadata available
        metadata = await handler.describe()
        assert metadata.handler_type == "similarity_compute"
        assert metadata.initialized is True
        assert metadata.is_pure_compute is True

        # Phase 3: Operation
        # Test various operations work correctly
        identical_dist = handler.cosine_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert identical_dist == 0.0

        orthogonal_dist = handler.cosine_distance([1.0, 0.0], [0.0, 1.0])
        assert orthogonal_dist == 1.0

        euclidean_dist = handler.euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert euclidean_dist == 5.0

        compare_result = handler.compare(
            [1.0, 0.0], [0.7071, 0.7071], metric="cosine", threshold=0.5
        )
        assert compare_result.is_match is True  # ~0.29 distance < 0.5

        # Phase 4: Shutdown
        await handler.shutdown()

        # Verify: uninitialized state restored
        health = await handler.health_check()
        assert health.initialized is False

        # Verify: operations fail
        with pytest.raises(RuntimeError, match="not initialized"):
            handler.cosine_distance([1.0, 0.0], [0.0, 1.0])

        # Phase 5: Reinitialize (verify handler is reusable)
        await handler.initialize()
        health = await handler.health_check()
        assert health.initialized is True

        # Operations work again
        dist = handler.cosine_distance([1.0, 0.0], [1.0, 0.0])
        assert dist == 0.0
