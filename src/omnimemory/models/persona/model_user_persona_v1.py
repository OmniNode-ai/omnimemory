# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deterministic persona snapshot model derived from observed behavior."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums import EnumPreferredTone, EnumTechnicalLevel


class ModelUserPersonaV1(BaseModel):
    """Deterministic persona snapshot derived from memory items.

    Not a configuration — this is INFERRED from observed behavior.
    Updated incrementally between sessions. Read-only during sessions.

    Consent enforcement is deferred to Phase B (OMN-3980). Phase 3 assumes
    trusted internal deployment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(..., description="User or agent identifier")
    agent_id: str | None = Field(
        default=None,
        description="Bound agent from Phase 2 (convenience binding, not partition key)",
    )
    technical_level: EnumTechnicalLevel = Field(
        default=EnumTechnicalLevel.INTERMEDIATE,
        description="Inferred technical capability",
    )
    vocabulary_complexity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0=simple, 1=advanced jargon (derived from prompt analysis)",
    )
    preferred_tone: EnumPreferredTone = Field(
        default=EnumPreferredTone.EXPLANATORY,
        description="Inferred interaction style preference",
    )
    domain_familiarity: dict[str, float] = Field(
        default_factory=dict,
        description="Repo/domain to proficiency score (0.0-1.0)",
    )
    session_count: int = Field(
        default=0,
        ge=0,
        description="Total sessions observed (for confidence weighting)",
    )
    persona_version: int = Field(
        default=1,
        ge=1,
        description="Monotonic version for append-only snapshots",
    )
    created_at: datetime = Field(..., description="When this snapshot was created")
    rebuilt_from_signals: int = Field(
        default=0,
        ge=0,
        description="Number of signals that contributed to this snapshot",
    )
