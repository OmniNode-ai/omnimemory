# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerMemoryExpire.

Tests the memory expiration handler that performs ACTIVE -> EXPIRED state
transitions using optimistic locking. Tests cover error handling when db_pool
is not configured, model validation, and SQL pattern documentation.

Test Categories:
    - Initialization: Handler setup with max_retries validation
    - No Database Pool: RuntimeError behavior when db_pool not configured
    - Command Validation: ModelExpireMemoryCommand field validation
    - Result Model: ModelMemoryExpireResult representation
    - Current State Model: ModelMemoryCurrentState validation
    - Retry Logic: handle_with_retry() error behavior without db_pool
    - Edge Cases: Command boundary condition validation
    - SQL Pattern Documentation: SQL constant verification

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_handler_memory_expire.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from omnibase_core.container import ModelONEXContainer
from pydantic import ValidationError

from omnimemory.enums import EnumLifecycleState
from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers import (
    HandlerMemoryExpire,
    ModelExpireMemoryCommand,
    ModelMemoryCurrentState,
    ModelMemoryExpireResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def container() -> ModelONEXContainer:
    """Provide a container for testing.

    Returns:
        A ModelONEXContainer instance.
    """
    return ModelONEXContainer()


@pytest.fixture
def fixed_now() -> datetime:
    """Provide a fixed timestamp for deterministic testing.

    Returns:
        A fixed datetime in UTC timezone.
    """
    return datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def memory_id() -> UUID:
    """Provide a fixed memory ID for testing.

    Returns:
        A deterministic UUID for the memory.
    """
    return UUID("12345678-abcd-1234-abcd-567812345678")


@pytest.fixture
def expire_command(memory_id: UUID, fixed_now: datetime) -> ModelExpireMemoryCommand:
    """Create an expiration command for testing.

    Args:
        memory_id: The memory entity ID.
        fixed_now: Fixed timestamp for expiration.

    Returns:
        Configured ModelExpireMemoryCommand instance.
    """
    return ModelExpireMemoryCommand(
        memory_id=memory_id,
        expected_revision=1,
        reason="ttl_expired",
        expired_at=fixed_now,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHandlerMemoryExpireInitialization:
    """Tests for HandlerMemoryExpire initialization."""

    def test_handler_creates_with_container(
        self, container: ModelONEXContainer
    ) -> None:
        """Test handler can be created with container.

        Given: A ModelONEXContainer
        When: Creating HandlerMemoryExpire
        Then: Handler is created but not initialized
        """
        handler = HandlerMemoryExpire(container)
        assert handler is not None
        assert handler._db_pool is None
        assert handler.initialized is False

    @pytest.mark.asyncio
    async def test_handler_default_max_retries(
        self, container: ModelONEXContainer
    ) -> None:
        """Test handler uses default max_retries of 3 after initialization.

        Given: No max_retries provided to initialize()
        When: Calling initialize() with mock db_pool
        Then: Handler uses default max_retries of 3
        """
        handler = HandlerMemoryExpire(container)
        # Create a mock pool (we just need a non-None value for initialization)
        mock_pool = object()  # type: ignore[assignment]
        await handler.initialize(db_pool=mock_pool)  # type: ignore[arg-type]
        assert handler.max_retries == 3
        assert handler.initialized is True

    @pytest.mark.asyncio
    async def test_handler_custom_max_retries(
        self, container: ModelONEXContainer
    ) -> None:
        """Test handler can be initialized with custom max_retries.

        Given: Custom max_retries value
        When: Calling initialize()
        Then: Handler uses the custom value
        """
        handler = HandlerMemoryExpire(container)
        mock_pool = object()  # type: ignore[assignment]
        await handler.initialize(db_pool=mock_pool, max_retries=5)  # type: ignore[arg-type]
        assert handler.max_retries == 5

    @pytest.mark.asyncio
    async def test_handler_rejects_invalid_max_retries(
        self, container: ModelONEXContainer
    ) -> None:
        """Test handler rejects max_retries < 1.

        Given: max_retries = 0
        When: Calling initialize()
        Then: ValueError is raised
        """
        handler = HandlerMemoryExpire(container)
        mock_pool = object()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            await handler.initialize(db_pool=mock_pool, max_retries=0)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_handler_rejects_negative_max_retries(
        self, container: ModelONEXContainer
    ) -> None:
        """Test handler rejects negative max_retries.

        Given: max_retries = -1
        When: Calling initialize()
        Then: ValueError is raised
        """
        handler = HandlerMemoryExpire(container)
        mock_pool = object()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            await handler.initialize(db_pool=mock_pool, max_retries=-1)  # type: ignore[arg-type]


# =============================================================================
# Not Initialized Tests
# =============================================================================


class TestNotInitialized:
    """Tests for handler behavior when not initialized."""

    @pytest.mark.asyncio
    async def test_handle_without_initialization_raises_error(
        self,
        container: ModelONEXContainer,
        memory_id: UUID,
    ) -> None:
        """Test that handle() raises RuntimeError when handler not initialized.

        Given: Handler created but not initialized
        When: Calling handle()
        Then: RuntimeError is raised with appropriate message
        """
        handler = HandlerMemoryExpire(container)
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
        )

        with pytest.raises(RuntimeError, match="Handler not initialized"):
            await handler.handle(command)

    @pytest.mark.asyncio
    async def test_handle_without_initialization_raises_error_with_custom_params(
        self,
        container: ModelONEXContainer,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test that handle() raises RuntimeError regardless of command parameters.

        Given: Handler not initialized and command with custom parameters
        When: Calling handle()
        Then: RuntimeError is raised
        """
        handler = HandlerMemoryExpire(container)
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=5,
            expired_at=fixed_now,
        )

        with pytest.raises(RuntimeError, match="Handler not initialized"):
            await handler.handle(command)

    @pytest.mark.asyncio
    async def test_handle_without_initialization_raises_error_for_various_revisions(
        self,
        container: ModelONEXContainer,
        memory_id: UUID,
    ) -> None:
        """Test that handle() raises RuntimeError for any expected_revision.

        Given: Handler not initialized and various revision values
        When: Calling handle()
        Then: RuntimeError is raised for all cases
        """
        handler = HandlerMemoryExpire(container)
        test_cases = [0, 1, 5, 100, 999]

        for expected_revision in test_cases:
            command = ModelExpireMemoryCommand(
                memory_id=memory_id,
                expected_revision=expected_revision,
            )

            with pytest.raises(RuntimeError, match="Handler not initialized"):
                await handler.handle(command)


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestCommandValidation:
    """Tests for ModelExpireMemoryCommand validation."""

    def test_command_requires_memory_id(self) -> None:
        """Test command requires memory_id field.

        Given: No memory_id provided
        When: Creating ModelExpireMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelExpireMemoryCommand(
                expected_revision=1,
            )  # type: ignore[call-arg]

    def test_command_requires_expected_revision(self, memory_id: UUID) -> None:
        """Test command requires expected_revision field.

        Given: No expected_revision provided
        When: Creating ModelExpireMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelExpireMemoryCommand(
                memory_id=memory_id,
            )  # type: ignore[call-arg]

    def test_command_rejects_negative_revision(self, memory_id: UUID) -> None:
        """Test command rejects negative expected_revision.

        Given: negative expected_revision
        When: Creating ModelExpireMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelExpireMemoryCommand(
                memory_id=memory_id,
                expected_revision=-1,
            )

    def test_command_accepts_zero_revision(self, memory_id: UUID) -> None:
        """Test command accepts zero expected_revision.

        Given: expected_revision = 0
        When: Creating ModelExpireMemoryCommand
        Then: Command is created successfully
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=0,
        )
        assert command.expected_revision == 0

    def test_command_default_reason(self, memory_id: UUID) -> None:
        """Test command has default reason of 'ttl_expired'.

        Given: No reason provided
        When: Creating ModelExpireMemoryCommand
        Then: reason defaults to 'ttl_expired'
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
        )
        assert command.reason == "ttl_expired"

    def test_command_custom_reason(self, memory_id: UUID) -> None:
        """Test command accepts custom reason.

        Given: Custom reason provided
        When: Creating ModelExpireMemoryCommand
        Then: Command uses the custom reason
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
            reason="manual_expiration",
        )
        assert command.reason == "manual_expiration"

    def test_command_reason_max_length(self, memory_id: UUID) -> None:
        """Test command reason has max length of 256.

        Given: reason longer than 256 characters
        When: Creating ModelExpireMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelExpireMemoryCommand(
                memory_id=memory_id,
                expected_revision=1,
                reason="x" * 257,
            )

    def test_command_reason_min_length(self, memory_id: UUID) -> None:
        """Test command reason has min length of 1.

        Given: empty reason
        When: Creating ModelExpireMemoryCommand
        Then: ValidationError is raised
        """
        with pytest.raises(ValidationError):
            ModelExpireMemoryCommand(
                memory_id=memory_id,
                expected_revision=1,
                reason="",
            )

    def test_command_is_frozen(self, memory_id: UUID) -> None:
        """Test command model is immutable.

        Given: A ModelExpireMemoryCommand instance
        When: Attempting to modify a field
        Then: Error is raised
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
        )

        with pytest.raises(ValidationError):
            command.expected_revision = 5  # type: ignore[misc]


# =============================================================================
# Result Model Tests
# =============================================================================


class TestResultModel:
    """Tests for ModelMemoryExpireResult model."""

    def test_result_success_state(self, memory_id: UUID) -> None:
        """Test result model for successful expiration.

        Given: Successful expiration data
        When: Creating ModelMemoryExpireResult
        Then: Model represents success correctly
        """
        result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=6,
            conflict=False,
            previous_state=EnumLifecycleState.ACTIVE,
        )

        assert result.success is True
        assert result.new_revision == 6
        assert result.conflict is False
        assert result.error_message is None
        assert result.previous_state == EnumLifecycleState.ACTIVE

    def test_result_conflict_state(self, memory_id: UUID) -> None:
        """Test result model for conflict scenario.

        Given: Conflict expiration data
        When: Creating ModelMemoryExpireResult
        Then: Model represents conflict correctly
        """
        result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=False,
            conflict=True,
            error_message="Revision conflict: expected 5, found 7",
            previous_state=EnumLifecycleState.ACTIVE,
        )

        assert result.success is False
        assert result.conflict is True
        assert result.new_revision is None
        assert "Revision conflict" in result.error_message
        assert result.previous_state == EnumLifecycleState.ACTIVE

    def test_result_hard_failure_state(self, memory_id: UUID) -> None:
        """Test result model for hard failure (invalid state).

        Given: Invalid state failure data
        When: Creating ModelMemoryExpireResult
        Then: Model represents hard failure correctly
        """
        result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=False,
            conflict=False,
            error_message="Cannot expire memory in state archived",
            previous_state=EnumLifecycleState.ARCHIVED,
        )

        assert result.success is False
        assert result.conflict is False
        assert result.new_revision is None
        assert "Cannot expire" in result.error_message
        assert result.previous_state == EnumLifecycleState.ARCHIVED

    def test_result_is_frozen(self, memory_id: UUID) -> None:
        """Test result model is immutable.

        Given: A ModelMemoryExpireResult instance
        When: Attempting to modify a field
        Then: Error is raised
        """
        result = ModelMemoryExpireResult(
            memory_id=memory_id,
            success=True,
            new_revision=2,
        )

        with pytest.raises(ValidationError):
            result.success = False  # type: ignore[misc]


# =============================================================================
# Current State Model Tests
# =============================================================================


class TestCurrentStateModel:
    """Tests for ModelMemoryCurrentState model."""

    def test_current_state_creation(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test ModelMemoryCurrentState can be created with valid data.

        Given: Valid current state data
        When: Creating ModelMemoryCurrentState
        Then: Model is created successfully
        """
        state = ModelMemoryCurrentState(
            memory_id=memory_id,
            lifecycle_state=EnumLifecycleState.ACTIVE,
            lifecycle_revision=5,
            updated_at=fixed_now,
        )

        assert state.memory_id == memory_id
        assert state.lifecycle_state == EnumLifecycleState.ACTIVE
        assert state.lifecycle_revision == 5
        assert state.updated_at == fixed_now

    def test_current_state_is_frozen(
        self,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test ModelMemoryCurrentState is immutable.

        Given: A ModelMemoryCurrentState instance
        When: Attempting to modify a field
        Then: Error is raised
        """
        state = ModelMemoryCurrentState(
            memory_id=memory_id,
            lifecycle_state=EnumLifecycleState.ACTIVE,
            lifecycle_revision=5,
            updated_at=fixed_now,
        )

        with pytest.raises(ValidationError):
            state.lifecycle_revision = 10  # type: ignore[misc]


# =============================================================================
# Retry Logic Tests (Not Initialized)
# =============================================================================


class TestRetryLogic:
    """Tests for handle_with_retry() behavior when handler not initialized."""

    @pytest.mark.asyncio
    async def test_handle_with_retry_without_initialization_raises_error(
        self,
        container: ModelONEXContainer,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test handle_with_retry raises RuntimeError when handler not initialized.

        Given: Handler not initialized
        When: Calling handle_with_retry
        Then: RuntimeError is raised
        """
        handler = HandlerMemoryExpire(container)

        with pytest.raises(RuntimeError, match="Handler not initialized"):
            await handler.handle_with_retry(
                memory_id=memory_id,
                initial_revision=1,
                reason="test_retry",
                expired_at=fixed_now,
            )

    @pytest.mark.asyncio
    async def test_handle_with_retry_without_initialization_raises_error_with_params(
        self,
        container: ModelONEXContainer,
        memory_id: UUID,
    ) -> None:
        """Test handle_with_retry raises RuntimeError regardless of parameters.

        Given: Handler not initialized and specific parameters
        When: Calling handle_with_retry
        Then: RuntimeError is raised
        """
        handler = HandlerMemoryExpire(container)

        with pytest.raises(RuntimeError, match="Handler not initialized"):
            await handler.handle_with_retry(
                memory_id=memory_id,
                initial_revision=10,
                reason="custom_reason",
            )


# =============================================================================
# Edge Cases - Command Validation
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions in command creation."""

    def test_large_revision_number_accepted(
        self,
        memory_id: UUID,
    ) -> None:
        """Test command accepts large revision numbers.

        Given: Very large expected_revision
        When: Creating expire command
        Then: Command is created successfully
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=999999999,
        )

        assert command.expected_revision == 999999999

    def test_zero_revision_accepted(
        self,
        memory_id: UUID,
    ) -> None:
        """Test command accepts revision 0.

        Given: expected_revision = 0 (first revision)
        When: Creating expire command
        Then: Command is created successfully
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=0,
        )

        assert command.expected_revision == 0

    def test_custom_expired_at_timestamp_accepted(
        self,
        memory_id: UUID,
    ) -> None:
        """Test command accepts custom expired_at timestamp.

        Given: Custom expired_at timestamp
        When: Creating expire command
        Then: Command is created with the custom timestamp
        """
        custom_time = datetime(2020, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
            expired_at=custom_time,
        )

        assert command.expired_at == custom_time

    def test_none_expired_at_accepted(
        self,
        memory_id: UUID,
    ) -> None:
        """Test command accepts None for expired_at.

        Given: No expired_at provided
        When: Creating expire command
        Then: Command is created with expired_at=None
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
            expired_at=None,
        )

        assert command.expired_at is None


# =============================================================================
# SQL Pattern Documentation Tests
# =============================================================================


class TestSQLPatternDocumentation:
    """Tests verifying SQL patterns are properly documented.

    These tests verify the handler has the expected SQL constants
    documented for future database implementation (OMN-1524).
    """

    def test_expire_sql_constant_exists(self) -> None:
        """Test EXPIRE_SQL constant is defined.

        Given: HandlerMemoryExpire class
        When: Checking for _EXPIRE_SQL attribute
        Then: Attribute exists and contains expected SQL pattern
        """
        assert hasattr(HandlerMemoryExpire, "_EXPIRE_SQL")
        sql = HandlerMemoryExpire._EXPIRE_SQL

        # Verify SQL contains key elements
        assert "UPDATE memories" in sql
        assert "lifecycle_state" in sql
        assert "expired_at" in sql
        assert "lifecycle_revision" in sql
        assert "WHERE" in sql
        assert "RETURNING" in sql

    def test_read_state_sql_constant_exists(self) -> None:
        """Test READ_STATE_SQL constant is defined.

        Given: HandlerMemoryExpire class
        When: Checking for _READ_STATE_SQL attribute
        Then: Attribute exists and contains expected SQL pattern
        """
        assert hasattr(HandlerMemoryExpire, "_READ_STATE_SQL")
        sql = HandlerMemoryExpire._READ_STATE_SQL

        # Verify SQL contains key elements
        assert "SELECT" in sql
        assert "lifecycle_state" in sql
        assert "lifecycle_revision" in sql
        assert "FROM memories" in sql

    def test_valid_from_states_constant_exists(self) -> None:
        """Test VALID_FROM_STATES constant is defined.

        Given: HandlerMemoryExpire class
        When: Checking for _VALID_FROM_STATES attribute
        Then: Attribute exists and contains only ACTIVE state

        Note: Only ACTIVE memories can be expired. You cannot expire an
        already EXPIRED memory - that would be a no-op or error condition.
        """
        assert hasattr(HandlerMemoryExpire, "_VALID_FROM_STATES")
        valid_states = HandlerMemoryExpire._VALID_FROM_STATES

        # Only ACTIVE memories can be expired
        assert EnumLifecycleState.ACTIVE in valid_states
        # EXPIRED is NOT a valid source state - can't expire already expired
        assert EnumLifecycleState.EXPIRED not in valid_states
        assert EnumLifecycleState.ARCHIVED not in valid_states
        assert EnumLifecycleState.DELETED not in valid_states


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test health_check returns not initialized status.

        Given: Handler not initialized
        When: Calling health_check()
        Then: Returns typed health model indicating not initialized
        """
        handler = HandlerMemoryExpire(container)
        health = await handler.health_check()

        assert health.initialized is False
        assert health.db_pool_available is False
        assert health.circuit_breaker_state == "not_configured"
        assert health.max_retries == 3  # Default value

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test health_check returns healthy status after initialization.

        Given: Handler initialized with mock pool
        When: Calling health_check()
        Then: Returns typed health model indicating initialized
        """
        handler = HandlerMemoryExpire(container)
        mock_pool = object()  # type: ignore[assignment]
        await handler.initialize(db_pool=mock_pool, max_retries=5)  # type: ignore[arg-type]

        health = await handler.health_check()

        assert health.initialized is True
        assert health.db_pool_available is True
        assert health.circuit_breaker_state == "closed"
        assert health.max_retries == 5


# =============================================================================
# Describe Tests
# =============================================================================


class TestDescribe:
    """Tests for describe() method."""

    @pytest.mark.asyncio
    async def test_describe_returns_metadata(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test describe returns handler metadata.

        Given: Handler instance
        When: Calling describe()
        Then: Returns typed metadata model with handler information
        """
        handler = HandlerMemoryExpire(container)
        metadata = await handler.describe()

        assert metadata.name == "HandlerMemoryExpire"
        assert metadata.description  # Non-empty description
        assert metadata.version == "1.0.0"
        assert "memory_expiration" in metadata.capabilities
        assert "optimistic_locking" in metadata.capabilities
        assert metadata.target_state == "expired"
        assert "active" in metadata.valid_from_states

    @pytest.mark.asyncio
    async def test_describe_reflects_initialization_state(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Test describe reflects initialization state.

        Given: Handler before and after initialization
        When: Calling describe()
        Then: initialized field reflects actual state
        """
        handler = HandlerMemoryExpire(container)

        # Before initialization
        metadata_before = await handler.describe()
        assert metadata_before.initialized is False

        # After initialization
        mock_pool = object()  # type: ignore[assignment]
        await handler.initialize(db_pool=mock_pool)  # type: ignore[arg-type]

        metadata_after = await handler.describe()
        assert metadata_after.initialized is True
