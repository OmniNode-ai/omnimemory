# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Context source type enumeration for document ingestion pipeline."""

from enum import Enum


class EnumContextSourceType(str, Enum):
    """Classifies the authoritative source of a ContextItem.

    Used for bootstrap tier assignment and retrieval scoring. Documents
    from higher-trust sources receive higher initial promotion tiers and
    stricter promotion thresholds.

    Attributes:
        STATIC_STANDARDS: Manually curated authoritative documents such
            as CLAUDE.md files and design documents. These start at
            VALIDATED tier with bootstrap_confidence=0.85.
        REPO_DERIVED: Documents derived from git repository content
            (README.md, architecture docs, etc.). These start at
            QUARANTINE tier.
        MEMORY_HOOK: Items derived from agent execution hook events
            (existing v0 system). Unchanged from v0 promotion rules.
        LINEAR_TICKET: Issues and documents fetched from Linear.
    """

    STATIC_STANDARDS = "static_standards"
    REPO_DERIVED = "repo_derived"
    MEMORY_HOOK = "memory_hook"
    LINEAR_TICKET = "linear_ticket"
