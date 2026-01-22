"""Lifecycle transition validation for memory state machine.

This module implements the state machine validation for memory lifecycle transitions.
The valid state machine is:
    ACTIVE -> EXPIRED -> ARCHIVED -> DELETED

Any state (except DELETED) can be PROMOTED back to ACTIVE.
DELETED is a terminal state with no valid outgoing transitions.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.enum_memory_lifecycle_state import EnumMemoryLifecycleState

# Valid state transitions
# Key: current state, Value: set of valid target states
VALID_TRANSITIONS: dict[EnumMemoryLifecycleState, set[EnumMemoryLifecycleState]] = {
    EnumMemoryLifecycleState.ACTIVE: {
        EnumMemoryLifecycleState.EXPIRED,
        EnumMemoryLifecycleState.ACTIVE,  # promotion returns to ACTIVE (no-op but valid)
    },
    EnumMemoryLifecycleState.EXPIRED: {
        EnumMemoryLifecycleState.ARCHIVED,
        EnumMemoryLifecycleState.ACTIVE,  # can promote from expired
    },
    EnumMemoryLifecycleState.ARCHIVED: {
        EnumMemoryLifecycleState.DELETED,
        EnumMemoryLifecycleState.ACTIVE,  # can promote from archived (resurrect)
    },
    EnumMemoryLifecycleState.DELETED: set(),  # terminal state - no valid transitions
}


class ModelLifecycleTransitionResult(BaseModel):
    """Result of a lifecycle transition attempt.

    This model captures the outcome of attempting a state transition,
    including success/failure status, state information, and any
    timestamp fields that should be updated.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    success: bool = Field(description="Whether transition was valid and applied")
    from_state: EnumMemoryLifecycleState = Field(description="Original state")
    to_state: EnumMemoryLifecycleState = Field(description="Target state")
    new_revision: int = Field(description="New lifecycle_revision after transition")
    timestamp_field: str | None = Field(
        default=None,
        description="Which timestamp field was set (e.g., 'expired_at', 'archived_at')",
    )
    error: str | None = Field(
        default=None,
        description="Error message if transition failed",
    )


def validate_transition(
    current_state: EnumMemoryLifecycleState,
    target_state: EnumMemoryLifecycleState,
) -> bool:
    """Check if a state transition is valid according to the state machine.

    Args:
        current_state: The current lifecycle state of the memory item.
        target_state: The desired target state to transition to.

    Returns:
        True if the transition is allowed, False otherwise.

    Examples:
        >>> validate_transition(EnumMemoryLifecycleState.ACTIVE, EnumMemoryLifecycleState.EXPIRED)
        True
        >>> validate_transition(EnumMemoryLifecycleState.DELETED, EnumMemoryLifecycleState.ACTIVE)
        False
    """
    allowed = VALID_TRANSITIONS.get(current_state, set())
    return target_state in allowed


def apply_transition(
    current_state: EnumMemoryLifecycleState,
    target_state: EnumMemoryLifecycleState,
    current_revision: int,
    now: datetime,
    is_promotion: bool = False,
) -> ModelLifecycleTransitionResult:
    """Validate and compute the result of a lifecycle transition.

    This function validates the transition and computes all the values that
    should be updated on the memory item. It does NOT perform any database
    operations - that is the responsibility of the caller.

    Args:
        current_state: Current lifecycle state of the memory item.
        target_state: Desired target state to transition to.
        current_revision: Current lifecycle_revision value.
        now: Authoritative timestamp from the envelope (NOT datetime.now()).
        is_promotion: Whether this is a promotion operation (affects timestamp field).

    Returns:
        ModelLifecycleTransitionResult with success/failure and computed values.

    Note:
        The `now` parameter should come from the event envelope or request
        context to ensure consistent timestamps across distributed systems.
        Never call datetime.now() inside this function.

    Examples:
        >>> from datetime import datetime
        >>> result = apply_transition(
        ...     current_state=EnumMemoryLifecycleState.ACTIVE,
        ...     target_state=EnumMemoryLifecycleState.EXPIRED,
        ...     current_revision=1,
        ...     now=datetime(2025, 1, 22, 12, 0, 0),
        ... )
        >>> result.success
        True
        >>> result.new_revision
        2
        >>> result.timestamp_field
        'expired_at'
    """
    # Validate the transition against the state machine
    if not validate_transition(current_state, target_state):
        return ModelLifecycleTransitionResult(
            success=False,
            from_state=current_state,
            to_state=target_state,
            new_revision=current_revision,
            error=f"Invalid transition: {current_state.value} -> {target_state.value}",
        )

    # Determine which timestamp field to set based on the transition
    timestamp_field: str | None = None
    if is_promotion:
        timestamp_field = "last_promoted_at"
    elif target_state == EnumMemoryLifecycleState.EXPIRED:
        timestamp_field = "expired_at"
    elif target_state == EnumMemoryLifecycleState.ARCHIVED:
        timestamp_field = "archived_at"
    elif target_state == EnumMemoryLifecycleState.DELETED:
        timestamp_field = "deleted_at"
    # Note: ACTIVE state from non-promotion doesn't set a timestamp
    # (this handles the edge case of ACTIVE -> ACTIVE which is a no-op)

    return ModelLifecycleTransitionResult(
        success=True,
        from_state=current_state,
        to_state=target_state,
        new_revision=current_revision + 1,
        timestamp_field=timestamp_field,
    )
