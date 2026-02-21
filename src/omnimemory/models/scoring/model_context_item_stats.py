# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
ContextItem statistics model for tier promotion tracking.

Accumulated per-item usage statistics persisted in PostgreSQL. The
promotion engine reads these stats to decide tier transitions. Fields
added in OMN-2426 support document-specific signals (citation count,
version hash staleness tracking, bootstrap confidence decay).

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §11-12
Ticket: OMN-2426
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType


class ModelContextItemStats(BaseModel):
    """Accumulated usage statistics for a single ContextItem.

    ``scored_runs`` is incremented each time the item is evaluated
    (regardless of injection). ``used_runs`` is incremented each time
    the item is actually injected into a session.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ------------------------------------------------------------------
    # Core run counters
    # ------------------------------------------------------------------

    context_item_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the ContextItem these stats belong to.",
    )
    scored_runs: int = Field(
        default=0,
        ge=0,
        description="Total number of sessions where this item was evaluated.",
    )
    used_runs: int = Field(
        default=0,
        ge=0,
        description="Number of sessions where this item was injected.",
    )
    positive_signals: int = Field(
        default=0,
        ge=0,
        description=(
            "Cumulative positive attribution signal count "
            "(RULE_FOLLOWED + STANDARD_CITED + DOC_SECTION_MATCHED + FAILURE_RESOLVED "
            "+ FILE_TOUCHED_MATCH + GATE_DELTA_IMPROVEMENT)."
        ),
    )
    negative_signals: int = Field(
        default=0,
        ge=0,
        description=(
            "Cumulative negative attribution signal count "
            "(PATTERN_VIOLATED + NEGATIVE_CONTRADICTION)."
        ),
    )

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    avg_fused_utility: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Rolling average of fused utility scores from sessions where the item "
            "was injected. Higher values indicate greater demonstrated usefulness."
        ),
    )
    hurt_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of scored runs where a negative signal was received. "
            "Exceeding the demotion threshold triggers tier demotion."
        ),
    )

    # ------------------------------------------------------------------
    # Document-specific fields (OMN-2426)
    # ------------------------------------------------------------------

    source_type: EnumContextSourceType | None = Field(
        default=None,
        description=(
            "Source type of the ContextItem. Used to look up the correct "
            "PromotionThresholdSet during promotion evaluation. None when the "
            "source type is not yet known or not applicable."
        ),
    )
    doc_version_hash: str | None = Field(
        default=None,
        description=(
            "The version_hash of the document version these stats were accumulated "
            "against. When the document is updated and a new ContextItem is created, "
            "stats may be partially carried (70%) if similarity >= 0.85."
        ),
    )
    citation_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Cumulative count of STANDARD_CITED signals received. "
            "Required to meet validated_to_shared_signal_floor for document items."
        ),
    )

    # Bootstrap fields
    bootstrap_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Bootstrap trust confidence assigned at item creation. "
            "None means no bootstrap grant was applied (standard scoring from the start). "
            "0.85 for CLAUDE.md, 0.75 for design docs. "
            "Decays toward 0 as bootstrap_runs_remaining reaches zero."
        ),
    )
    bootstrap_runs_remaining: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Remaining scored runs before bootstrap confidence expires. "
            "None means no bootstrap grant (standard scoring applies). "
            "Decremented each scored run; promotion to earned VALIDATED at zero."
        ),
    )
    bootstrap_cleared: bool = Field(
        default=False,
        description=(
            "True when bootstrap_runs_remaining has reached zero and the item "
            "has transitioned from bootstrapped to earned VALIDATED. "
            "Once True, bootstrap_confidence and bootstrap_runs_remaining are ignored."
        ),
    )

    @model_validator(mode="after")
    def _validate_bootstrap_cleared_consistency(self) -> Self:
        """Enforce that cleared bootstrap records have no residual bootstrap fields.

        When bootstrap_cleared is True, both bootstrap_confidence and
        bootstrap_runs_remaining must be None — the bootstrap phase is over
        and those values no longer apply.
        """
        if self.bootstrap_cleared:
            if self.bootstrap_confidence is not None:
                raise ValueError(
                    "bootstrap_confidence must be None when bootstrap_cleared is True; "
                    f"got bootstrap_confidence={self.bootstrap_confidence!r}"
                )
            if self.bootstrap_runs_remaining is not None:
                raise ValueError(
                    "bootstrap_runs_remaining must be None when bootstrap_cleared is True; "
                    f"got bootstrap_runs_remaining={self.bootstrap_runs_remaining!r}"
                )
        return self
