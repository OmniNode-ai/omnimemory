# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for AdapterPostgresDeactivateMemory.

Tests the memory deactivation adapter that wraps HandlerMemoryExpire.

Test Categories:
    - Initialization: Adapter setup, initialization state
    - Not Initialized: RuntimeError on deactivate() before initialize()
    - Command Delegation: deactivate() passes correct command to handler
    - Health Check: health_check() aggregates handler health
    - Describe: describe() returns correct adapter metadata
    - Shutdown: shutdown() is idempotent and resets state
    - Model Validation: ModelDeactivateAdapterHealth structure

Related Tickets:
    - OMN-1603: Add adapter implementations for memory lifecycle orchestrator

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_adapter_postgres_deactivate_memory.py -v
"""

from __future__ import annotations

from uuid import UUID

import pytest
from omnibase_core.container import ModelONEXContainer

from omnimemory.nodes.memory_lifecycle_orchestrator.adapters import (
    AdapterPostgresDeactivateMemory,
    ModelDeactivateAdapterHealth,
    ModelDeactivateAdapterMetadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def container() -> ModelONEXContainer:
    """Provide an ONEX container for adapter initialization."""
    return ModelONEXContainer()


@pytest.fixture
def memory_id() -> UUID:
    """Provide a fixed memory ID for testing."""
    return UUID("12345678-abcd-1234-abcd-567812345678")


@pytest.fixture
def adapter(container: ModelONEXContainer) -> AdapterPostgresDeactivateMemory:
    """Provide an uninitialized adapter for testing."""
    return AdapterPostgresDeactivateMemory(container)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterInitialization:
    """Verify adapter initialization behavior."""

    def test_not_initialized_before_initialize(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """Adapter reports not initialized before initialize() is called."""
        assert adapter.initialized is False

    @pytest.mark.asyncio
    async def test_initialized_requires_db_pool(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """initialize() requires a db_pool argument (raises without it)."""
        with pytest.raises(TypeError):
            await adapter.initialize()  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_initialize_validates_max_retries(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """initialize() raises ValueError for max_retries < 1."""
        from unittest.mock import MagicMock

        fake_pool = MagicMock()
        with pytest.raises(ValueError, match="max_retries"):
            await adapter.initialize(db_pool=fake_pool, max_retries=0)


# ---------------------------------------------------------------------------
# Not Initialized Guard Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNotInitializedGuard:
    """Verify that deactivate() raises RuntimeError before initialize()."""

    @pytest.mark.asyncio
    async def test_deactivate_raises_if_not_initialized(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.deactivate(
                memory_id=memory_id,
                expected_revision=1,
            )

    @pytest.mark.asyncio
    async def test_deactivate_with_retry_raises_if_not_initialized(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate_with_retry() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.deactivate_with_retry(
                memory_id=memory_id,
                initial_revision=1,
            )


# ---------------------------------------------------------------------------
# Health Check Tests (without db_pool - tests model structure)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterHealthCheck:
    """Verify health_check() returns correct structure."""

    @pytest.mark.asyncio
    async def test_health_before_initialization(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """health_check() reflects uninitialized state."""
        health = await adapter.health_check()
        assert isinstance(health, ModelDeactivateAdapterHealth)
        assert health.initialized is False
        assert health.handler_health.initialized is False

    @pytest.mark.asyncio
    async def test_health_reports_no_db_pool_before_init(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """health_check() reports db pool not available before init."""
        health = await adapter.health_check()
        assert health.handler_health.db_pool_available is False


# ---------------------------------------------------------------------------
# Describe Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDescribe:
    """Verify describe() returns correct adapter metadata."""

    @pytest.mark.asyncio
    async def test_describe_returns_correct_name(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """describe() returns the correct adapter class name."""
        metadata = await adapter.describe()
        assert isinstance(metadata, ModelDeactivateAdapterMetadata)
        assert metadata.name == "AdapterPostgresDeactivateMemory"

    @pytest.mark.asyncio
    async def test_describe_includes_handler_metadata(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """describe() includes metadata from the underlying handler."""
        metadata = await adapter.describe()
        assert metadata.handler_metadata.name == "HandlerMemoryExpire"

    @pytest.mark.asyncio
    async def test_describe_reflects_initialization_state(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """describe() reflects not-initialized state."""
        metadata = await adapter.describe()
        assert metadata.initialized is False

    @pytest.mark.asyncio
    async def test_describe_contains_deactivation_in_description(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """describe() contains 'deactivation' in the description string."""
        metadata = await adapter.describe()
        assert (
            "deactivation" in metadata.description.lower()
            or "deactivate" in metadata.description.lower()
        )


# ---------------------------------------------------------------------------
# Shutdown Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterShutdown:
    """Verify shutdown() is idempotent and resets state."""

    @pytest.mark.asyncio
    async def test_shutdown_before_initialize_is_safe(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """shutdown() on uninitialized adapter is a no-op."""
        await adapter.shutdown()  # Should not raise
        assert adapter.initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """Multiple shutdown() calls do not raise."""
        await adapter.shutdown()
        await adapter.shutdown()
        assert adapter.initialized is False


# ---------------------------------------------------------------------------
# Model Structure Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelDeactivateAdapterHealth:
    """Verify ModelDeactivateAdapterHealth structure and immutability."""

    @pytest.mark.asyncio
    async def test_health_model_is_frozen(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """ModelDeactivateAdapterHealth is immutable."""
        from pydantic import ValidationError

        health = await adapter.health_check()
        with pytest.raises((ValidationError, TypeError)):
            health.initialized = True  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_health_model_has_expected_fields(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """ModelDeactivateAdapterHealth has initialized and handler_health fields."""
        health = await adapter.health_check()
        assert hasattr(health, "initialized")
        assert hasattr(health, "handler_health")
        # handler_health has its own sub-fields
        assert hasattr(health.handler_health, "initialized")
        assert hasattr(health.handler_health, "db_pool_available")
        assert hasattr(health.handler_health, "max_retries")
        assert hasattr(health.handler_health, "circuit_breaker_state")


@pytest.mark.unit
class TestModelDeactivateAdapterMetadata:
    """Verify ModelDeactivateAdapterMetadata structure."""

    @pytest.mark.asyncio
    async def test_metadata_is_frozen(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """ModelDeactivateAdapterMetadata is immutable."""
        from pydantic import ValidationError

        metadata = await adapter.describe()
        with pytest.raises((ValidationError, TypeError)):
            metadata.name = "Changed"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_handler_metadata_has_valid_from_states(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """Handler metadata includes valid_from_states for the expire handler."""
        metadata = await adapter.describe()
        assert hasattr(metadata.handler_metadata, "valid_from_states")
        assert "active" in metadata.handler_metadata.valid_from_states

    @pytest.mark.asyncio
    async def test_handler_metadata_target_state_is_expired(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """Handler metadata target_state is 'expired'."""
        metadata = await adapter.describe()
        assert metadata.handler_metadata.target_state == "expired"
