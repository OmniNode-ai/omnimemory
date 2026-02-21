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

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from omnibase_core.container import ModelONEXContainer

from omnimemory.nodes.memory_lifecycle_orchestrator.adapters import (
    AdapterPostgresDeactivateMemory,
    ModelDeactivateAdapterHealth,
    ModelDeactivateAdapterMetadata,
)
from omnimemory.nodes.memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
    ModelExpireMemoryCommand,
    ModelMemoryExpireResult,
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
    async def test_initialized_after_initialize(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """Adapter reports initialized after initialize() is called with a db_pool."""
        # MagicMock() is a valid substitute here because HandlerMemoryExpire.initialize()
        # only stores the pool reference (self._db_pool = db_pool) without type-checking
        # it. No pool methods are called during initialization, so any object works.
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)
        assert adapter.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_validates_max_retries(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """initialize() raises ValueError for max_retries < 1."""
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
# Command Delegation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCommandDelegation:
    """Verify that deactivate() and deactivate_with_retry() delegate correctly.

    These tests bypass the real HandlerMemoryExpire by replacing the adapter's
    internal _handler with an AsyncMock after initialization, so no database
    connection is needed. This isolates the adapter's delegation logic from
    the handler's database logic.
    """

    @pytest.mark.asyncio
    async def test_deactivate_constructs_command_with_correct_memory_id(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() builds a ModelExpireMemoryCommand with the given memory_id."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate(memory_id=memory_id, expected_revision=1)

        mock_handler.handle.assert_awaited_once()
        command: ModelExpireMemoryCommand = mock_handler.handle.call_args[0][0]
        assert command.memory_id == memory_id

    @pytest.mark.asyncio
    async def test_deactivate_constructs_command_with_correct_expected_revision(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() builds a ModelExpireMemoryCommand with the given expected_revision."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=6,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate(memory_id=memory_id, expected_revision=5)

        command: ModelExpireMemoryCommand = mock_handler.handle.call_args[0][0]
        assert command.expected_revision == 5

    @pytest.mark.asyncio
    async def test_deactivate_constructs_command_with_default_reason(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() uses 'ttl_expired' as the default reason in the command."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate(memory_id=memory_id, expected_revision=1)

        command: ModelExpireMemoryCommand = mock_handler.handle.call_args[0][0]
        assert command.reason == "ttl_expired"

    @pytest.mark.asyncio
    async def test_deactivate_constructs_command_with_custom_reason(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() passes a custom reason through to the command."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate(
            memory_id=memory_id,
            expected_revision=1,
            reason="manual_admin_expire",
        )

        command: ModelExpireMemoryCommand = mock_handler.handle.call_args[0][0]
        assert command.reason == "manual_admin_expire"

    @pytest.mark.asyncio
    async def test_deactivate_constructs_command_with_explicit_expired_at(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() passes an explicit expired_at timestamp through to the command."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        fixed_ts = datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)
        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate(
            memory_id=memory_id,
            expected_revision=1,
            expired_at=fixed_ts,
        )

        command: ModelExpireMemoryCommand = mock_handler.handle.call_args[0][0]
        assert command.expired_at == fixed_ts

    @pytest.mark.asyncio
    async def test_deactivate_passes_command_to_handler_handle(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() calls handler.handle() with a ModelExpireMemoryCommand."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate(memory_id=memory_id, expected_revision=1)

        mock_handler.handle.assert_awaited_once()
        command: object = mock_handler.handle.call_args[0][0]
        assert isinstance(command, ModelExpireMemoryCommand)

    @pytest.mark.asyncio
    async def test_deactivate_returns_handler_result(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() returns exactly the result produced by handler.handle()."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=7,
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        result = await adapter.deactivate(memory_id=memory_id, expected_revision=6)

        assert result is expected_result

    @pytest.mark.asyncio
    async def test_deactivate_returns_conflict_result_from_handler(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate() passes through a conflict result from handler unchanged."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        conflict_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=False,
            conflict=True,
            error_message="Revision conflict: expected 3, found 4",
        )
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=conflict_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        result = await adapter.deactivate(memory_id=memory_id, expected_revision=3)

        assert result is conflict_result
        assert result.success is False
        assert result.conflict is True

    @pytest.mark.asyncio
    async def test_deactivate_with_retry_delegates_to_handler_handle_with_retry(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate_with_retry() calls handler.handle_with_retry() with correct args."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=3,
        )
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate_with_retry(memory_id=memory_id, initial_revision=2)

        mock_handler.handle_with_retry.assert_awaited_once_with(
            memory_id=memory_id,
            initial_revision=2,
            reason="ttl_expired",
            expired_at=None,
        )

    @pytest.mark.asyncio
    async def test_deactivate_with_retry_returns_handler_result(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate_with_retry() returns exactly the result from handler.handle_with_retry()."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=5,
        )
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        result = await adapter.deactivate_with_retry(
            memory_id=memory_id, initial_revision=4
        )

        assert result is expected_result

    @pytest.mark.asyncio
    async def test_deactivate_with_retry_passes_custom_reason(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate_with_retry() forwards a custom reason to handler.handle_with_retry()."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate_with_retry(
            memory_id=memory_id,
            initial_revision=1,
            reason="scheduled_cleanup",
        )

        mock_handler.handle_with_retry.assert_awaited_once_with(
            memory_id=memory_id,
            initial_revision=1,
            reason="scheduled_cleanup",
            expired_at=None,
        )

    @pytest.mark.asyncio
    async def test_deactivate_with_retry_passes_explicit_expired_at(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate_with_retry() forwards an explicit expired_at to handler.handle_with_retry()."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        fixed_ts = datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)
        expected_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(return_value=expected_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        await adapter.deactivate_with_retry(
            memory_id=memory_id,
            initial_revision=1,
            expired_at=fixed_ts,
        )

        mock_handler.handle_with_retry.assert_awaited_once_with(
            memory_id=memory_id,
            initial_revision=1,
            reason="ttl_expired",
            expired_at=fixed_ts,
        )

    @pytest.mark.asyncio
    async def test_deactivate_with_retry_returns_conflict_on_max_retries_exceeded(
        self,
        adapter: AdapterPostgresDeactivateMemory,
        memory_id: UUID,
    ) -> None:
        """deactivate_with_retry() passes through a max-retries-exceeded conflict result."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)

        exhausted_result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=False,
            conflict=True,
            error_message="Max retries (3) exceeded due to contention",
        )
        mock_handler = MagicMock()
        mock_handler.handle_with_retry = AsyncMock(return_value=exhausted_result)
        adapter._handler = mock_handler  # type: ignore[assignment]

        result = await adapter.deactivate_with_retry(
            memory_id=memory_id, initial_revision=1
        )

        assert result is exhausted_result
        assert result.success is False
        assert result.conflict is True


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

    @pytest.mark.asyncio
    async def test_health_after_initialization(
        self, adapter: AdapterPostgresDeactivateMemory
    ) -> None:
        """health_check() reflects initialized state after initialize() is called."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)
        health = await adapter.health_check()
        assert health.initialized is True
        assert health.handler_health.initialized is True


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
        """shutdown() after initialize() is idempotent: second call does not raise."""
        fake_pool = MagicMock()
        await adapter.initialize(db_pool=fake_pool)
        assert adapter.initialized is True
        await adapter.shutdown()
        assert adapter.initialized is False
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
