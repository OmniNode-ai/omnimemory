# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Single observation about user behavior for persona inference."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelPersonaSignal(BaseModel):
    """A single observation about user behavior that contributes to persona inference.

    Emitted by session hooks at session end. Consumed by
    node_persona_builder_compute to incrementally update a persona snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this signal",
    )
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session that produced this signal")
    signal_type: str = Field(
        ...,
        description="Signal category (e.g. 'technical_level', 'vocabulary', 'tone')",
    )
    evidence: str = Field(
        ...,
        max_length=500,
        description="What was observed in the session",
    )
    inferred_value: str = Field(
        ...,
        description="The classification value derived from the evidence",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the inference (0.0-1.0)",
    )
    emitted_at: datetime = Field(..., description="When this signal was emitted")
