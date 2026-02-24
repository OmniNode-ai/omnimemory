# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""
Context policy configuration model for session context assembly.

Controls which ContextItems are eligible for injection, how many tokens
are allocated to each source type, and which similarity thresholds gate
item inclusion. ``doc_source_config = None`` disables document ingestion
and runs the engine in hook-only mode.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §13
Ticket: OMN-2426
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnimemory.models.config.model_doc_source_config import ModelDocSourceConfig


class ModelContextPolicyConfig(BaseModel):
    """Top-level policy configuration for session context assembly.

    Used by ``ContextSelectorNode`` to decide which items to retrieve,
    how to budget tokens, and whether to include document-sourced items.

    All fields have defaults; an empty ``ContextPolicyConfig()`` is a
    valid hook-only (pre-OMN-2426) configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ------------------------------------------------------------------
    # Base retrieval settings
    # ------------------------------------------------------------------

    max_total_items: int = Field(
        default=20,
        ge=1,
        description=(
            "Maximum total ContextItems injected per session, across all source types."
        ),
    )
    min_similarity: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description=(
            "Global minimum cosine similarity threshold. Items below this score are "
            "excluded before budget enforcement."
        ),
    )
    hook_token_budget_fraction: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the session token budget reserved for hook-derived items "
            "(MEMORY_HOOK, MEMORY_PATTERN). The remainder is available for "
            "document-sourced items when doc_source_config is set."
        ),
    )

    # ------------------------------------------------------------------
    # Tier eligibility
    # ------------------------------------------------------------------

    include_quarantine: bool = Field(
        default=False,
        description=(
            "When True, QUARANTINE items are eligible for injection. "
            "Intended for debugging only; not recommended for production."
        ),
    )
    include_bootstrapped_validated: bool = Field(
        default=True,
        description=(
            "When True, VALIDATED items that have not yet accumulated scored runs "
            "(bootstrap grant) are eligible. Set False to require earned promotion."
        ),
    )

    # ------------------------------------------------------------------
    # Document source integration (OMN-2426)
    # ------------------------------------------------------------------

    doc_source_config: ModelDocSourceConfig | None = Field(
        default=None,
        description=(
            "Document source policy configuration. "
            "None = document ingestion disabled; engine runs in hook-only mode."
        ),
    )

    @model_validator(mode="after")
    def _validate_token_budget_fractions(self) -> Self:
        """Validate that hook and doc token budget fractions do not exceed 1.0.

        When doc_source_config is set, the sum of hook_token_budget_fraction and
        doc_token_budget_fraction_default must be at most 1.0 to prevent
        silent over-allocation of the session token budget.
        """
        if self.doc_source_config is not None:
            h = self.hook_token_budget_fraction
            d = self.doc_source_config.doc_token_budget_fraction_default
            total = h + d
            if total > 1.0:
                raise ValueError(
                    f"hook_token_budget_fraction ({h}) + "
                    f"doc_token_budget_fraction_default ({d}) = {total:.2f} exceeds 1.0"
                )
        return self
