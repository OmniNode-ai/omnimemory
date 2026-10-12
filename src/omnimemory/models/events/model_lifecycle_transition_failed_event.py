# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Lifecycle transition failure event for the memory lifecycle orchestrator.

Published when a lifecycle command (for example expire-memory) is refused or
fails, so a consumer reads an outcome instead of silence.

Kafka topic: ``onex.evt.omnimemory.lifecycle-transition-failed.v1``
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums import EnumLifecycleState


class ModelLifecycleTransitionFailedEvent(BaseModel):
    """A lifecycle transition was refused or failed."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    memory_id: UUID | None = Field(
        ...,
        description=(
            "Memory entity the command named; None when the command carried no "
            "readable memory id."
        ),
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Lifecycle operation that failed (for example 'expire').",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Why the transition was refused or failed.",
    )
    conflict: bool = Field(
        ...,
        description="True when a concurrent revision change caused the refusal.",
    )
    previous_state: EnumLifecycleState | None = Field(
        ...,
        description="Lifecycle state the entity was found in, when it was read.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID of the command, for tracing.",
    )
    emitted_at: datetime = Field(
        ...,
        description="When this event was emitted.",
    )


__all__ = ["ModelLifecycleTransitionFailedEvent"]
