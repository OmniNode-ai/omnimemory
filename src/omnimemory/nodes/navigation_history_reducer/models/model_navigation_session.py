# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Navigation session domain models for navigation_history_reducer.

These types are defined locally because the canonical sources (ContractGraph from
OMN-2540, PlanStep from OMN-2561) are in omnibase_core PRs not yet merged.
Once those PRs land, replace these local definitions with imports from
omnibase_core.contracts.navigation.

.. note::
    Dependency note (OMN-2584): Replace with omnibase_core imports when available:
      - ``PlanStep`` → ``omnibase_core.contracts.navigation.PlanStep``
      - ``NavigationOutcome`` → ``omnibase_core.contracts.navigation.NavigationOutcome``
      - ``NavigationSession`` → ``omnibase_core.contracts.navigation.NavigationSession``

.. versionadded:: 0.4.0
    Initial implementation for OMN-2584 Navigation History Storage.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single executed step in a navigation plan.

    Represents one state transition in a backward-chaining plan produced by the
    planner (OMN-2561). Once OMN-2561 lands, this type will be imported from
    omnibase_core.

    Attributes:
        step_index: Zero-based position of this step in the plan.
        from_state_id: Identifier of the source graph state.
        to_state_id: Identifier of the target graph state.
        action: Human-readable description of the action taken.
        executed_at: UTC timestamp when this step was executed.
        metadata: Optional typed metadata for the step (e.g., edge weights,
            transition preconditions). Only typed graph artifacts — no raw
            model outputs or user content.
    """

    step_index: int = Field(ge=0, description="Zero-based step position in the plan")
    from_state_id: str = Field(description="Source graph state identifier")
    to_state_id: str = Field(description="Target graph state identifier")
    action: str = Field(description="Description of the action taken")
    executed_at: datetime = Field(description="UTC timestamp of step execution")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional typed graph artifact metadata",
    )

    model_config = {"frozen": True}


class NavigationOutcomeTag(StrEnum):
    """Discriminator tag for navigation outcome union."""

    SUCCESS = "success"
    FAILURE = "failure"


class NavigationOutcomeSuccess(BaseModel):
    """Successful navigation outcome.

    Attributes:
        tag: Discriminator literal, always ``"success"``.
        reached_state_id: Final state ID that satisfied the goal condition.
    """

    tag: Literal["success"] = "success"
    reached_state_id: str = Field(description="Final state ID satisfying the goal")

    model_config = {"frozen": True}


class NavigationOutcomeFailure(BaseModel):
    """Failed navigation outcome.

    Attributes:
        tag: Discriminator literal, always ``"failure"``.
        reason: Structured reason for failure (e.g., "no_path_found",
            "max_steps_exceeded", "goal_unreachable").
        details: Optional human-readable details for debugging. Must not
            contain raw model outputs or user content.
    """

    tag: Literal["failure"] = "failure"
    reason: str = Field(description="Structured failure reason code")
    details: str | None = Field(
        default=None,
        description="Optional debug details (no user content)",
    )

    model_config = {"frozen": True}


# Union type for the outcome discriminator
NavigationOutcome = NavigationOutcomeSuccess | NavigationOutcomeFailure


class NavigationSession(BaseModel):
    """A completed navigation session with all execution metadata.

    Captures the full record of a backward-chaining navigation attempt:
    which goal was targeted, from which start state, what steps were taken,
    and whether the goal was reached.

    Attributes:
        session_id: Unique identifier for this navigation session.
        goal_condition: Human-readable or structured description of the goal
            state that the planner was targeting. Only typed graph artifacts.
        start_state_id: Identifier of the graph state where navigation began.
        end_state_id: Identifier of the final graph state (reached or last).
        executed_steps: Ordered list of steps taken during navigation.
        final_outcome: The terminal outcome — success or structured failure.
        graph_fingerprint: Content hash of the contract graph at navigation time.
            Used for cache invalidation and drift detection.
        created_at: UTC timestamp when this session record was created.
    """

    session_id: UUID = Field(description="Unique session identifier")
    goal_condition: str = Field(
        description="Goal state description (typed graph artifact only)"
    )
    start_state_id: str = Field(description="Initial graph state identifier")
    end_state_id: str = Field(description="Final graph state identifier")
    executed_steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered plan steps executed during navigation",
    )
    final_outcome: NavigationOutcome = Field(
        description="Terminal outcome: success or structured failure"
    )
    graph_fingerprint: str = Field(
        description="Content hash of the contract graph at navigation time"
    )
    created_at: datetime = Field(
        description="UTC timestamp when this session record was created"
    )

    model_config = {"frozen": True}

    @property
    def is_successful(self) -> bool:
        """Return True if the navigation session ended in success."""
        return self.final_outcome.tag == "success"

    @property
    def step_count(self) -> int:
        """Return the number of executed steps."""
        return len(self.executed_steps)
