"""
Promotion tier enumeration for ContextItem lifecycle management.

Tiers represent the trust/confidence level of a ContextItem based on
its source and accumulated usage history. Higher tiers receive greater
weight during session context assembly.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §11-12
Ticket: OMN-2426
"""

from enum import Enum


class EnumPromotionTier(str, Enum):
    """Trust tier of a ContextItem.

    Values are stored in PostgreSQL and Qdrant payload so must remain
    stable once deployed. Tier transitions are driven by ``ContextItemStats``
    thresholds defined in ``PromotionThresholdSet``.
    """

    QUARANTINE = "quarantine"
    """New or unvalidated items. Not included in context assembly by default."""

    VALIDATED = "validated"
    """Items that have met the promotion threshold from QUARANTINE.

    STATIC_STANDARDS items start here directly (bootstrap trust grant).
    Tier multiplier at retrieval: 1.00 (earned), 0.85 (bootstrapped).
    """

    SHARED = "shared"
    """Items promoted beyond VALIDATED via sustained positive signal history.

    Tier multiplier at retrieval: 1.15.
    """

    BLACKLISTED = "blacklisted"
    """Items permanently excluded: deleted source files or demoted by violation."""
