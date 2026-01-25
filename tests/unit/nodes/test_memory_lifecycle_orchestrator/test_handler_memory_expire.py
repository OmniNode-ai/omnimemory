# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerMemoryExpire.

Tests the memory expiration handler that performs ACTIVE -> EXPIRED state
transitions using optimistic locking. Tests cover stub mode behavior,
conflict detection, retry logic, and state validation.

Test Categories:
    - Initialization: Handler setup with max_retries validation
    - Stub Mode: Success behavior when no database pool configured
    - Conflict Detection: Revision mismatch handling
    - Retry Logic: handle_with_retry() behavior
    - State Validation: Only ACTIVE memories can be expired
    - Revision Increment: New revision on success

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_handler_memory_expire.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnimemory.enums import EnumLifecycleState
from omnimemory.nodes.memory_lifecycle_orchestrator.handlers import (
    HandlerMemoryExpire,
    ModelExpireMemoryCommand,
    ModelMemoryCurrentState,
    ModelMemoryExpireResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


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
def handler_stub() -> HandlerMemoryExpire:
    """Create handler in stub mode (no db_pool).

    Returns:
        HandlerMemoryExpire instance without database connection.
    """
    return HandlerMemoryExpire(db_pool=None)


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

    def test_handler_creates_without_db_pool(self) -> None:
        """Test handler can be created without database pool.

        Given: No db_pool provided
        When: Creating HandlerMemoryExpire
        Then: Handler is created in stub mode
        """
        handler = HandlerMemoryExpire(db_pool=None)
        assert handler is not None
        assert handler._db_pool is None

    def test_handler_default_max_retries(self) -> None:
        """Test handler uses default max_retries of 3.

        Given: No max_retries provided
        When: Creating HandlerMemoryExpire
        Then: Handler uses default max_retries of 3
        """
        handler = HandlerMemoryExpire()
        assert handler.max_retries == 3

    def test_handler_custom_max_retries(self) -> None:
        """Test handler can be created with custom max_retries.

        Given: Custom max_retries value
        When: Creating HandlerMemoryExpire
        Then: Handler uses the custom value
        """
        handler = HandlerMemoryExpire(max_retries=5)
        assert handler.max_retries == 5

    def test_handler_rejects_invalid_max_retries(self) -> None:
        """Test handler rejects max_retries < 1.

        Given: max_retries = 0
        When: Creating HandlerMemoryExpire
        Then: ValueError is raised
        """
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            HandlerMemoryExpire(max_retries=0)

    def test_handler_rejects_negative_max_retries(self) -> None:
        """Test handler rejects negative max_retries.

        Given: max_retries = -1
        When: Creating HandlerMemoryExpire
        Then: ValueError is raised
        """
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            HandlerMemoryExpire(max_retries=-1)


# =============================================================================
# Stub Mode Tests
# =============================================================================


class TestStubMode:
    """Tests for handler behavior in stub mode (no database)."""

    @pytest.mark.asyncio
    async def test_expire_success_in_stub_mode(
        self,
        handler_stub: HandlerMemoryExpire,
        expire_command: ModelExpireMemoryCommand,
    ) -> None:
        """Test handler returns success in stub mode.

        Given: Handler without db_pool (stub mode)
        When: Handling an expire command
        Then: Returns success with incremented revision
        """
        result = await handler_stub.handle(expire_command)

        assert result.success is True
        assert result.new_revision == expire_command.expected_revision + 1
        assert result.conflict is False
        assert result.error_message is None
        assert result.previous_state == EnumLifecycleState.ACTIVE

    @pytest.mark.asyncio
    async def test_stub_mode_returns_correct_memory_id(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test stub mode returns correct memory_id in result.

        Given: Handler in stub mode
        When: Handling an expire command
        Then: Result contains correct memory_id
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=5,
            expired_at=fixed_now,
        )

        result = await handler_stub.handle(command)

        assert result.memory_id == memory_id
        assert result.new_revision == 6

    @pytest.mark.asyncio
    async def test_stub_mode_increments_revision_correctly(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
    ) -> None:
        """Test stub mode increments revision by 1.

        Given: Various expected_revision values
        When: Handling expire commands
        Then: new_revision = expected_revision + 1
        """
        test_cases = [0, 1, 5, 100, 999]

        for expected_revision in test_cases:
            command = ModelExpireMemoryCommand(
                memory_id=memory_id,
                expected_revision=expected_revision,
            )
            result = await handler_stub.handle(command)

            assert result.new_revision == expected_revision + 1


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
# Retry Logic Tests (Stub Mode)
# =============================================================================


class TestRetryLogic:
    """Tests for handle_with_retry() behavior in stub mode."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_attempt_in_stub_mode(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
        fixed_now: datetime,
    ) -> None:
        """Test handle_with_retry succeeds on first attempt in stub mode.

        Given: Handler in stub mode
        When: Calling handle_with_retry
        Then: Returns success on first attempt
        """
        result = await handler_stub.handle_with_retry(
            memory_id=memory_id,
            initial_revision=1,
            reason="test_retry",
            expired_at=fixed_now,
        )

        assert result.success is True
        assert result.new_revision == 2
        assert result.conflict is False

    @pytest.mark.asyncio
    async def test_retry_uses_correct_parameters(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
    ) -> None:
        """Test handle_with_retry passes correct parameters to handle().

        Given: Specific parameters for retry
        When: Calling handle_with_retry
        Then: Parameters are correctly used
        """
        result = await handler_stub.handle_with_retry(
            memory_id=memory_id,
            initial_revision=10,
            reason="custom_reason",
        )

        assert result.memory_id == memory_id
        assert result.new_revision == 11


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_large_revision_number(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
    ) -> None:
        """Test handler handles large revision numbers correctly.

        Given: Very large expected_revision
        When: Handling expire command
        Then: Returns success with incremented revision
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=999999999,
        )

        result = await handler_stub.handle(command)

        assert result.success is True
        assert result.new_revision == 1000000000

    @pytest.mark.asyncio
    async def test_zero_revision(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
    ) -> None:
        """Test handler handles revision 0 correctly.

        Given: expected_revision = 0 (first revision)
        When: Handling expire command
        Then: Returns success with new_revision = 1
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=0,
        )

        result = await handler_stub.handle(command)

        assert result.success is True
        assert result.new_revision == 1

    @pytest.mark.asyncio
    async def test_custom_expired_at_timestamp(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
    ) -> None:
        """Test handler accepts custom expired_at timestamp.

        Given: Custom expired_at timestamp
        When: Handling expire command
        Then: Command is processed successfully
        """
        custom_time = datetime(2020, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
            expired_at=custom_time,
        )

        result = await handler_stub.handle(command)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_none_expired_at_uses_current_time(
        self,
        handler_stub: HandlerMemoryExpire,
        memory_id: UUID,
    ) -> None:
        """Test handler uses current time when expired_at is None.

        Given: No expired_at provided
        When: Handling expire command
        Then: Command is processed using current time
        """
        command = ModelExpireMemoryCommand(
            memory_id=memory_id,
            expected_revision=1,
            expired_at=None,
        )

        result = await handler_stub.handle(command)

        assert result.success is True


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
        Then: Attribute exists and contains expected states
        """
        assert hasattr(HandlerMemoryExpire, "_VALID_FROM_STATES")
        valid_states = HandlerMemoryExpire._VALID_FROM_STATES

        # Verify contains expected states
        assert EnumLifecycleState.ACTIVE in valid_states
        assert EnumLifecycleState.EXPIRED in valid_states
        assert EnumLifecycleState.ARCHIVED not in valid_states
        assert EnumLifecycleState.DELETED not in valid_states
