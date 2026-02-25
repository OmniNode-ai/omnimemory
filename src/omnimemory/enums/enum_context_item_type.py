# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Context item type enumeration for ContextItem classification.

Assigned by ``ChunkClassifierCompute`` using frozen, deterministic
string-matching rules (v1 rule set). Type determines intent-type
retrieval weight during session context assembly.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §9
Ticket: OMN-2426
"""

from enum import Enum


class EnumContextItemType(str, Enum):
    """Structural/semantic type of a ContextItem chunk.

    Rule order is frozen per classifier version. A change to rule order
    or trigger strings requires a version bump (replay determinism).

    Values are stored in PostgreSQL and Qdrant payload so must remain
    stable once deployed.
    """

    API_CONSTRAINT = "api_constraint"
    """URL patterns, port numbers, env vars like KAFKA_BOOTSTRAP_SERVERS."""

    CONFIG_NOTE = "config_note"
    """Source .env patterns, POSTGRES_/KAFKA_ vars, Docker network config."""

    RULE = "rule"
    """Normative rules: "must", "never", CRITICAL, NON-NEGOTIABLE, PROHIBITED."""

    FAILURE_PATTERN = "failure_pattern"
    """Anti-patterns, pitfalls, common mistakes, gotchas."""

    EXAMPLE = "example"
    """Code fences or sections preceded by "Example:" or usage headings."""

    REPO_MAP = "repo_map"
    """Directory tree representations (``├──``, ``└──``, ``│``)."""

    DOC_EXCERPT = "doc_excerpt"
    """Default fallback for chunks that do not match a more specific type."""
