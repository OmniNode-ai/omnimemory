# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Attribution signal type enumeration for ContextItem scoring.

Signals are emitted by the attribution engine after each agent session
to update ``ContextItemStats`` for items that were injected. Positive
signals drive tier promotion; ``PATTERN_VIOLATED`` drives demotion.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §12
Ticket: OMN-2426
"""

from enum import Enum


class EnumAttributionSignalType(str, Enum):
    """Describes why a ContextItem's stats were updated after a session.

    Hook-derived signals (the first group) are unchanged from v0.
    Document-specific signals (the second group) are additive and
    required for Stream C tier integration.
    """

    # ------------------------------------------------------------------
    # Hook-derived signals (v0 — do not reorder or rename)
    # ------------------------------------------------------------------

    FILE_TOUCHED_MATCH = "file_touched_match"
    """An edited file path matched the item's ``source_ref`` or tag."""

    RULE_ID_CITED = "rule_id_cited"
    """Model output explicitly referenced a rule ID from the item."""

    FAILURE_SIGNATURE_MATCH = "failure_signature_match"
    """A known failure pattern from the item was detected in the session."""

    FAILURE_RESOLVED = "failure_resolved"
    """A failure that matched this item was resolved in the session."""

    DIFF_HUNK_MATCH = "diff_hunk_match"
    """A diff hunk in the session matched content from the item."""

    GATE_DELTA_IMPROVEMENT = "gate_delta_improvement"
    """Session quality gate score improved when this item was injected."""

    NEGATIVE_CONTRADICTION = "negative_contradiction"
    """Model output contradicted guidance from the item."""

    DUPLICATE_MATCH = "duplicate_match"
    """Item content is a near-duplicate of another higher-scoring item."""

    # ------------------------------------------------------------------
    # Document-specific signals (OMN-2426 — additive, do not reorder)
    # ------------------------------------------------------------------

    RULE_FOLLOWED = "rule_followed"
    """Model output complies with a rule stated in the injected document.

    Strength: 0.9 (explicit compliance), 0.5 (implicit compliance).
    """

    STANDARD_CITED = "standard_cited"
    """Model explicitly referenced the source document or a rule from it.

    Strength: 1.0 (exact source_ref), 0.8 (rule_id), 0.6 (title match).
    """

    PATTERN_VIOLATED = "pattern_violated"
    """Model output violated a rule stated in the injected document.

    Strength: 1.0. Primary driver of ``hurt_rate`` for document items.
    """

    DOC_SECTION_MATCHED = "doc_section_matched"
    """Retrieval similarity at injection time met the minimum threshold.

    Only emitted when similarity >= ``doc_min_similarity`` (default 0.65).
    Strength equals the raw similarity score.
    """
