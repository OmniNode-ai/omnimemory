# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""
Promotion decision model for ContextItem tier transitions.

Output of the promotion evaluation engine (``PromotionReducer``).
Records the before/after tier, the triggering signal, and — for
document items — which threshold set was used and whether bootstrap
confidence was cleared in this decision.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §12
Ticket: OMN-2426
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.enum_promotion_tier import EnumPromotionTier


class ModelPromotionDecision(BaseModel):
    """Records the outcome of a single tier promotion/demotion evaluation.

    Persisted in PostgreSQL for auditability and replay verification.
    One row per ContextItem per evaluation pass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ------------------------------------------------------------------
    # Item identity
    # ------------------------------------------------------------------

    context_item_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the ContextItem being evaluated.",
    )
    evaluated_at_utc: datetime = Field(
        ...,
        description="UTC timestamp of this evaluation.",
    )

    # ------------------------------------------------------------------
    # Tier transition
    # ------------------------------------------------------------------

    tier_before: EnumPromotionTier = Field(
        ...,
        description="Promotion tier before this evaluation.",
    )
    tier_after: EnumPromotionTier = Field(
        ...,
        description="Promotion tier after this evaluation (may equal tier_before if no change).",
    )
    tier_changed: bool = Field(
        ...,
        description="True when tier_before != tier_after.",
    )
    trigger_signal: str | None = Field(
        default=None,
        description=(
            "Attribution signal type that triggered this evaluation, if any. "
            "None for scheduled batch evaluations."
        ),
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of the decision (for audit logs).",
    )

    # ------------------------------------------------------------------
    # Threshold context (OMN-2426)
    # ------------------------------------------------------------------

    source_type: EnumContextSourceType | None = Field(
        default=None,
        description=(
            "Source type of the ContextItem. Used to select the PromotionThresholdSet "
            "and included here for audit trail completeness."
        ),
    )
    threshold_set_used: str | None = Field(
        default=None,
        description=(
            "Identifier of the PromotionThresholdSet used in this decision, "
            "e.g. 'static_standards_v1'. Enables replay verification: "
            "if thresholds change, this field shows which version was active."
        ),
    )

    # Bootstrap tracking
    bootstrap_cleared: bool = Field(
        default=False,
        description=(
            "True when this decision cleared the bootstrap grant "
            "(bootstrap_runs_remaining reached zero). "
            "The item transitions from bootstrapped VALIDATED to earned VALIDATED."
        ),
    )

    # ------------------------------------------------------------------
    # Cross-field invariants
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_tier_changed_consistency(self) -> Self:
        """Ensure tier_changed is consistent with tier_before and tier_after."""
        expected = self.tier_before != self.tier_after
        if self.tier_changed != expected:
            raise ValueError(
                f"tier_changed={self.tier_changed!r} is inconsistent with "
                f"tier_before={self.tier_before!r} and tier_after={self.tier_after!r}; "
                f"expected tier_changed={expected!r}"
            )
        return self
