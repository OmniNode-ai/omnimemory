"""
Context source type enumeration for ContextItem provenance tracking.

Describes the origin category of a ContextItem — used to select
bootstrap tier assignments, promotion thresholds, and retrieval
scope boosts.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §11-12
Ticket: OMN-2426
"""

from enum import Enum


class EnumContextSourceType(str, Enum):
    """Provenance category of a ContextItem.

    Values are stored in PostgreSQL, Qdrant payload, and Kafka events
    so must remain stable once deployed.
    """

    STATIC_STANDARDS = "static_standards"
    """Explicitly curated policy documents: CLAUDE.md, design docs.

    Bootstrap tier: VALIDATED (confidence 0.75-0.85).
    Promotion thresholds: Q→V = n/a (starts VALIDATED), V→S at 10 runs.
    """

    REPO_DERIVED = "repo_derived"
    """Documents derived from a repository: READMEs, plans, handoffs.

    Bootstrap tier: QUARANTINE.
    Promotion thresholds: Q→V at 5 runs, V→S at 20 runs.
    """

    MEMORY_HOOK = "memory_hook"
    """Hook-derived patterns from agent execution history (v0 pipeline).

    Bootstrap tier: QUARANTINE.
    Promotion thresholds: Q→V at 10 runs, V→S at 30 runs (v0 rules).
    """

    MEMORY_PATTERN = "memory_pattern"
    """Aggregated patterns extracted from MEMORY_HOOK items."""

    LINEAR_DERIVED = "linear_derived"
    """Tickets and documents fetched from the Linear API."""
