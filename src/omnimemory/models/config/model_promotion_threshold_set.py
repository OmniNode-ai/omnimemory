"""
Promotion threshold set model for per-source-type tier transitions.

Encapsulates the run-count and signal floor thresholds that govern
ContextItem tier transitions. One set per ``ContextSourceType`` allows
document-sourced items to use lower run counts than hook-derived items
(documents are not naturally "used" on every agent run).

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §12
Ticket: OMN-2426
"""

import types
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType


class ModelPromotionThresholdSet(BaseModel):
    """Per-source-type tier transition thresholds.

    ``quarantine_to_validated_runs``: None means the source type starts
    directly at VALIDATED (e.g., STATIC_STANDARDS bootstrap grant).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source_type: EnumContextSourceType = Field(
        ...,
        description="Source type this threshold set applies to.",
    )

    # ------------------------------------------------------------------
    # QUARANTINE → VALIDATED
    # ------------------------------------------------------------------

    quarantine_to_validated_runs: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of scored runs required to promote from QUARANTINE to VALIDATED. "
            "None means the source type bypasses QUARANTINE (bootstrapped VALIDATED)."
        ),
    )

    # ------------------------------------------------------------------
    # VALIDATED → SHARED
    # ------------------------------------------------------------------

    validated_to_shared_runs: int = Field(
        ...,
        ge=1,
        description=(
            "Minimum number of scored runs before VALIDATED → SHARED promotion is "
            "considered. Must be combined with ``validated_to_shared_used_rate`` "
            "and ``validated_to_shared_signal_floor``."
        ),
    )
    validated_to_shared_used_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum used_rate (fraction of scored runs where item was injected) "
            "required for VALIDATED → SHARED. Lower for doc items (0.10) because "
            "rules are followed implicitly without explicit citation."
        ),
    )
    validated_to_shared_signal_floor: int = Field(
        default=0,
        ge=0,
        description=(
            "Minimum cumulative positive attribution signals required for V→S. "
            "For document items: sum of RULE_FOLLOWED + STANDARD_CITED + DOC_SECTION_MATCHED. "
            "Prevents small-sample noise from promoting items on lucky runs."
        ),
    )


# ---------------------------------------------------------------------------
# Default threshold sets matching design doc §12
# ---------------------------------------------------------------------------

PROMOTION_THRESHOLD_STATIC_STANDARDS = ModelPromotionThresholdSet(
    source_type=EnumContextSourceType.STATIC_STANDARDS,
    quarantine_to_validated_runs=None,  # starts VALIDATED (bootstrap grant)
    validated_to_shared_runs=10,
    validated_to_shared_used_rate=0.10,
    validated_to_shared_signal_floor=5,
)

PROMOTION_THRESHOLD_REPO_DERIVED = ModelPromotionThresholdSet(
    source_type=EnumContextSourceType.REPO_DERIVED,
    quarantine_to_validated_runs=5,
    validated_to_shared_runs=20,
    validated_to_shared_used_rate=0.15,
    validated_to_shared_signal_floor=5,
)

PROMOTION_THRESHOLD_MEMORY_HOOK = ModelPromotionThresholdSet(
    source_type=EnumContextSourceType.MEMORY_HOOK,
    quarantine_to_validated_runs=10,
    validated_to_shared_runs=30,
    validated_to_shared_used_rate=0.25,
    validated_to_shared_signal_floor=0,
)

PROMOTION_THRESHOLD_MEMORY_PATTERN = ModelPromotionThresholdSet(
    source_type=EnumContextSourceType.MEMORY_PATTERN,
    quarantine_to_validated_runs=5,
    validated_to_shared_runs=20,
    validated_to_shared_used_rate=0.15,
    validated_to_shared_signal_floor=5,
)

PROMOTION_THRESHOLD_LINEAR_TICKET = ModelPromotionThresholdSet(
    source_type=EnumContextSourceType.LINEAR_TICKET,
    quarantine_to_validated_runs=5,
    validated_to_shared_runs=20,
    validated_to_shared_used_rate=0.15,
    validated_to_shared_signal_floor=5,
)

DEFAULT_PROMOTION_THRESHOLDS: Mapping[
    EnumContextSourceType, ModelPromotionThresholdSet
] = types.MappingProxyType(
    {
        EnumContextSourceType.STATIC_STANDARDS: PROMOTION_THRESHOLD_STATIC_STANDARDS,
        EnumContextSourceType.REPO_DERIVED: PROMOTION_THRESHOLD_REPO_DERIVED,
        EnumContextSourceType.MEMORY_HOOK: PROMOTION_THRESHOLD_MEMORY_HOOK,
        EnumContextSourceType.MEMORY_PATTERN: PROMOTION_THRESHOLD_MEMORY_PATTERN,
        EnumContextSourceType.LINEAR_TICKET: PROMOTION_THRESHOLD_LINEAR_TICKET,
    }
)
