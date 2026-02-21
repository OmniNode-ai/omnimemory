# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ValidatorLifecycleTransition.

Tests the state machine validator that enforces the 5-state memory
lifecycle transition rules.

Test Categories:
    - Valid transitions: All allowed state machine paths
    - Invalid transitions: All forbidden transitions
    - Self-transitions: Same-state transitions always invalid
    - Terminal state (DELETED): No outbound transitions
    - Convenience methods: is_valid(), get_valid_transitions()
    - Result model: ModelTransitionValidationResult structure

Related Tickets:
    - OMN-1603: Add adapter implementations for memory lifecycle orchestrator

Usage:
    pytest tests/unit/nodes/test_memory_lifecycle_orchestrator/test_validator_lifecycle_transition.py -v
"""

from __future__ import annotations

import pytest

from omnimemory.enums import EnumLifecycleState
from omnimemory.nodes.memory_lifecycle_orchestrator.validators import (
    VALID_TRANSITIONS,
    ModelTransitionValidationResult,
    ValidatorLifecycleTransition,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator() -> ValidatorLifecycleTransition:
    """Provide a ValidatorLifecycleTransition instance for testing."""
    return ValidatorLifecycleTransition()


# ---------------------------------------------------------------------------
# Valid Transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidTransitions:
    """Verify that all valid lifecycle transitions are accepted."""

    def test_active_to_stale(self, validator: ValidatorLifecycleTransition) -> None:
        """ACTIVE -> STALE is the first step in the lifecycle."""
        result = validator.validate(
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.STALE,
        )
        assert result.valid is True
        assert result.reason is None
        assert result.from_state == EnumLifecycleState.ACTIVE
        assert result.to_state == EnumLifecycleState.STALE

    def test_active_to_expired(self, validator: ValidatorLifecycleTransition) -> None:
        """ACTIVE -> EXPIRED is allowed for direct hard-TTL expiration."""
        result = validator.validate(
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.EXPIRED,
        )
        assert result.valid is True

    def test_active_to_deleted(self, validator: ValidatorLifecycleTransition) -> None:
        """ACTIVE -> DELETED is allowed for explicit deletion."""
        result = validator.validate(
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.DELETED,
        )
        assert result.valid is True

    def test_stale_to_expired(self, validator: ValidatorLifecycleTransition) -> None:
        """STALE -> EXPIRED is the normal progression after soft-TTL."""
        result = validator.validate(
            EnumLifecycleState.STALE,
            EnumLifecycleState.EXPIRED,
        )
        assert result.valid is True

    def test_stale_to_active(self, validator: ValidatorLifecycleTransition) -> None:
        """STALE -> ACTIVE is allowed when a memory is refreshed."""
        result = validator.validate(
            EnumLifecycleState.STALE,
            EnumLifecycleState.ACTIVE,
        )
        assert result.valid is True

    def test_stale_to_deleted(self, validator: ValidatorLifecycleTransition) -> None:
        """STALE -> DELETED is allowed for explicit deletion."""
        result = validator.validate(
            EnumLifecycleState.STALE,
            EnumLifecycleState.DELETED,
        )
        assert result.valid is True

    def test_expired_to_archived(self, validator: ValidatorLifecycleTransition) -> None:
        """EXPIRED -> ARCHIVED is the cold storage path."""
        result = validator.validate(
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.ARCHIVED,
        )
        assert result.valid is True

    def test_expired_to_deleted(self, validator: ValidatorLifecycleTransition) -> None:
        """EXPIRED -> DELETED is allowed for explicit deletion."""
        result = validator.validate(
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.DELETED,
        )
        assert result.valid is True

    def test_archived_to_active(self, validator: ValidatorLifecycleTransition) -> None:
        """ARCHIVED -> ACTIVE is the promote/restore path."""
        result = validator.validate(
            EnumLifecycleState.ARCHIVED,
            EnumLifecycleState.ACTIVE,
        )
        assert result.valid is True

    def test_archived_to_deleted(self, validator: ValidatorLifecycleTransition) -> None:
        """ARCHIVED -> DELETED is allowed for explicit deletion."""
        result = validator.validate(
            EnumLifecycleState.ARCHIVED,
            EnumLifecycleState.DELETED,
        )
        assert result.valid is True


# ---------------------------------------------------------------------------
# Invalid Transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInvalidTransitions:
    """Verify that all forbidden lifecycle transitions are rejected."""

    def test_deleted_to_active_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """DELETED is terminal - cannot transition to ACTIVE."""
        result = validator.validate(
            EnumLifecycleState.DELETED,
            EnumLifecycleState.ACTIVE,
        )
        assert result.valid is False
        assert result.reason is not None
        assert "terminal" in result.reason.lower()

    def test_deleted_to_stale_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """DELETED is terminal - cannot transition to STALE."""
        result = validator.validate(
            EnumLifecycleState.DELETED,
            EnumLifecycleState.STALE,
        )
        assert result.valid is False

    def test_deleted_to_expired_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """DELETED is terminal - cannot transition to EXPIRED."""
        result = validator.validate(
            EnumLifecycleState.DELETED,
            EnumLifecycleState.EXPIRED,
        )
        assert result.valid is False

    def test_deleted_to_archived_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """DELETED is terminal - cannot transition to ARCHIVED."""
        result = validator.validate(
            EnumLifecycleState.DELETED,
            EnumLifecycleState.ARCHIVED,
        )
        assert result.valid is False

    def test_archived_to_expired_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """ARCHIVED -> EXPIRED is invalid; must restore to ACTIVE first."""
        result = validator.validate(
            EnumLifecycleState.ARCHIVED,
            EnumLifecycleState.EXPIRED,
        )
        assert result.valid is False
        assert result.reason is not None

    def test_archived_to_stale_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """ARCHIVED -> STALE is invalid; must restore to ACTIVE first."""
        result = validator.validate(
            EnumLifecycleState.ARCHIVED,
            EnumLifecycleState.STALE,
        )
        assert result.valid is False

    def test_expired_to_active_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """EXPIRED -> ACTIVE is invalid; must go through ARCHIVED."""
        result = validator.validate(
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.ACTIVE,
        )
        assert result.valid is False

    def test_expired_to_stale_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """EXPIRED -> STALE is an invalid regression."""
        result = validator.validate(
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.STALE,
        )
        assert result.valid is False

    def test_active_to_archived_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """ACTIVE -> ARCHIVED is invalid; must expire first."""
        result = validator.validate(
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.ARCHIVED,
        )
        assert result.valid is False

    def test_stale_to_archived_is_invalid(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """STALE -> ARCHIVED is invalid; must expire first."""
        result = validator.validate(
            EnumLifecycleState.STALE,
            EnumLifecycleState.ARCHIVED,
        )
        assert result.valid is False


# ---------------------------------------------------------------------------
# Self-Transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelfTransitions:
    """Self-transitions (same state -> same state) are always invalid."""

    @pytest.mark.parametrize(
        "state",
        [
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.STALE,
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.ARCHIVED,
            EnumLifecycleState.DELETED,
        ],
    )
    def test_self_transition_is_invalid(
        self,
        validator: ValidatorLifecycleTransition,
        state: EnumLifecycleState,
    ) -> None:
        """Self-transition from any state to itself is always invalid."""
        result = validator.validate(state, state)
        assert result.valid is False
        assert result.reason is not None
        assert "self" in result.reason.lower() or "already" in result.reason.lower()
        assert result.from_state == state
        assert result.to_state == state


# ---------------------------------------------------------------------------
# Convenience Method: is_valid
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsValidConvenienceMethod:
    """is_valid() returns a boolean equivalent of validate().valid."""

    def test_is_valid_returns_true_for_valid_transition(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """is_valid() returns True for a valid transition."""
        assert (
            validator.is_valid(EnumLifecycleState.ACTIVE, EnumLifecycleState.STALE)
            is True
        )

    def test_is_valid_returns_false_for_invalid_transition(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """is_valid() returns False for an invalid transition."""
        assert (
            validator.is_valid(EnumLifecycleState.DELETED, EnumLifecycleState.ACTIVE)
            is False
        )

    def test_is_valid_returns_false_for_self_transition(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """is_valid() returns False for a self-transition."""
        assert (
            validator.is_valid(EnumLifecycleState.ACTIVE, EnumLifecycleState.ACTIVE)
            is False
        )


# ---------------------------------------------------------------------------
# Convenience Method: get_valid_transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetValidTransitions:
    """get_valid_transitions() returns the allowed destination states."""

    def test_active_has_three_valid_destinations(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """ACTIVE state has three valid destinations: STALE, EXPIRED, DELETED."""
        valid = validator.get_valid_transitions(EnumLifecycleState.ACTIVE)
        assert EnumLifecycleState.STALE in valid
        assert EnumLifecycleState.EXPIRED in valid
        assert EnumLifecycleState.DELETED in valid
        assert EnumLifecycleState.ARCHIVED not in valid
        assert EnumLifecycleState.ACTIVE not in valid

    def test_stale_has_three_valid_destinations(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """STALE state has three valid destinations: ACTIVE, EXPIRED, DELETED."""
        valid = validator.get_valid_transitions(EnumLifecycleState.STALE)
        assert EnumLifecycleState.ACTIVE in valid
        assert EnumLifecycleState.EXPIRED in valid
        assert EnumLifecycleState.DELETED in valid

    def test_expired_has_two_valid_destinations(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """EXPIRED state has two valid destinations: ARCHIVED, DELETED."""
        valid = validator.get_valid_transitions(EnumLifecycleState.EXPIRED)
        assert EnumLifecycleState.ARCHIVED in valid
        assert EnumLifecycleState.DELETED in valid
        assert len(valid) == 2

    def test_archived_has_two_valid_destinations(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """ARCHIVED state has two valid destinations: ACTIVE, DELETED."""
        valid = validator.get_valid_transitions(EnumLifecycleState.ARCHIVED)
        assert EnumLifecycleState.ACTIVE in valid
        assert EnumLifecycleState.DELETED in valid
        assert len(valid) == 2

    def test_deleted_has_no_valid_destinations(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """DELETED state is terminal - no valid destinations."""
        valid = validator.get_valid_transitions(EnumLifecycleState.DELETED)
        assert len(valid) == 0

    def test_returns_frozenset(self, validator: ValidatorLifecycleTransition) -> None:
        """get_valid_transitions() returns a frozenset (immutable)."""
        result = validator.get_valid_transitions(EnumLifecycleState.ACTIVE)
        assert isinstance(result, frozenset)


# ---------------------------------------------------------------------------
# Result Model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelTransitionValidationResult:
    """Verify ModelTransitionValidationResult structure and immutability."""

    def test_valid_result_has_no_reason(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """A valid transition result has reason=None."""
        result = validator.validate(
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.STALE,
        )
        assert isinstance(result, ModelTransitionValidationResult)
        assert result.valid is True
        assert result.reason is None

    def test_invalid_result_has_reason(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """An invalid transition result has a non-empty reason string."""
        result = validator.validate(
            EnumLifecycleState.EXPIRED,
            EnumLifecycleState.ACTIVE,
        )
        assert result.valid is False
        assert result.reason is not None
        assert len(result.reason) > 0

    def test_result_is_frozen(self, validator: ValidatorLifecycleTransition) -> None:
        """ModelTransitionValidationResult is immutable (frozen=True)."""
        from pydantic import ValidationError

        result = validator.validate(
            EnumLifecycleState.ACTIVE,
            EnumLifecycleState.STALE,
        )
        with pytest.raises((ValidationError, TypeError)):
            result.valid = False  # type: ignore[misc]

    def test_result_from_and_to_state_fields(
        self, validator: ValidatorLifecycleTransition
    ) -> None:
        """Result correctly records from_state and to_state."""
        result = validator.validate(
            EnumLifecycleState.STALE,
            EnumLifecycleState.EXPIRED,
        )
        assert result.from_state == EnumLifecycleState.STALE
        assert result.to_state == EnumLifecycleState.EXPIRED


# ---------------------------------------------------------------------------
# VALID_TRANSITIONS module constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidTransitionsConstant:
    """Verify the module-level VALID_TRANSITIONS constant is complete."""

    def test_all_states_present_as_keys(self) -> None:
        """VALID_TRANSITIONS has an entry for every lifecycle state."""
        for state in EnumLifecycleState:
            assert (
                state in VALID_TRANSITIONS
            ), f"State {state!r} missing from VALID_TRANSITIONS"

    def test_deleted_has_empty_transitions(self) -> None:
        """DELETED maps to an empty frozenset."""
        assert VALID_TRANSITIONS[EnumLifecycleState.DELETED] == frozenset()

    def test_all_values_are_frozensets(self) -> None:
        """All values in VALID_TRANSITIONS are frozensets."""
        for state, destinations in VALID_TRANSITIONS.items():
            assert isinstance(
                destinations, frozenset
            ), f"Expected frozenset for {state!r}, got {type(destinations).__name__}"
