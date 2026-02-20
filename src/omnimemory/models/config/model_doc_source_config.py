"""
Document source configuration model for ContextPolicyConfig.

Controls adaptive token budget fractions, similarity thresholds,
and bootstrap behaviour for document-sourced ContextItems. Added to
``ContextPolicyConfig.doc_source_config``; ``None`` means doc ingestion
is disabled and the engine runs in hook-only mode.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §13
Ticket: OMN-2426
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


class ModelDocSourceConfig(BaseModel):
    """Policy configuration for document-sourced ContextItems.

    All fields have conservative defaults that match the design doc's
    initial values. Operators can override per deployment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    prefer_narrow_scope: bool = Field(
        default=True,
        description=(
            "When True, items from the current repo scope are preferred over "
            "org-wide items during context assembly (scope_boost multiplier applies)."
        ),
    )

    # ------------------------------------------------------------------
    # Adaptive token budget fractions
    # ------------------------------------------------------------------

    doc_token_budget_fraction_default: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Default fraction of the session token budget reserved for document-sourced "
            "items. Applied when the current intent has no explicit override."
        ),
    )
    doc_token_budget_fraction_overrides: Mapping[str, float] = Field(
        default_factory=lambda: {
            "architecture": 0.40,
            "refactoring": 0.40,
            "compliance": 0.40,
            "code_generation": 0.25,
            "debugging": 0.20,
        },
        description=(
            "Per-intent budget fraction overrides (read-only mapping). "
            "Keyed by intent_category string. "
            "Rationale: complex coding sessions must not starve hook-derived patterns; "
            "compliance/architecture sessions benefit from higher doc context."
        ),
    )

    max_doc_items: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum number of document-sourced items included in a single context "
            "assembly pass. Items are de-duplicated by content_fingerprint before counting."
        ),
    )

    doc_min_similarity: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity required for a document item to be included. "
            "Also gates DOC_SECTION_MATCHED signal emission — items below this "
            "threshold do not emit attribution signals."
        ),
    )

    # ------------------------------------------------------------------
    # Bootstrap behaviour
    # ------------------------------------------------------------------

    allow_bootstrap_validated: bool = Field(
        default=True,
        description=(
            "When True, STATIC_STANDARDS items (CLAUDE.md, design docs) are injected "
            "immediately at bootstrap VALIDATED tier without waiting for promotion. "
            "Set False to require earned promotion for all items."
        ),
    )
    allow_unscored_static_standards: bool = Field(
        default=True,
        description=(
            "When True, newly-indexed CLAUDE.md chunks are eligible for injection "
            "before any attribution signals have been collected. "
            "Rationale: policy documents are authoritative on day one."
        ),
    )
