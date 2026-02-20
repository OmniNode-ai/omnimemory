# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Detected document type enumeration for document ingestion pipeline."""

from enum import Enum


class EnumDetectedDocType(str, Enum):
    """Classification of document type for chunking strategy selection.

    Rules are applied in priority order by DetectedDocTypeClassifier.
    Each type maps to a distinct chunking strategy in DocumentParserCompute.

    Attributes:
        CLAUDE_MD: Any file named CLAUDE.md (highest priority — section-split).
        DESIGN_DOC: Files under design/ directories.
        ARCHITECTURE_DOC: Files matching *ARCHITECTURE*.md or *OVERVIEW*.md.
        PLAN: Files under plans/ directories.
        HANDOFF: Files under handoffs/ directories.
        README: README.md at repo root level.
        TICKET: Linear issue fetched via MCP.
        LINEAR_DOCUMENT: Linear document fetched via MCP.
        DEEP_DIVE: Files matching *DEEP_DIVE*.md.
        UNKNOWN_MD: Any .md file that matches none of the above rules.
    """

    CLAUDE_MD = "claude_md"
    DESIGN_DOC = "design_doc"
    ARCHITECTURE_DOC = "architecture_doc"
    PLAN = "plan"
    HANDOFF = "handoff"
    README = "readme"
    TICKET = "ticket"
    LINEAR_DOCUMENT = "linear_document"
    DEEP_DIVE = "deep_dive"
    UNKNOWN_MD = "unknown_md"
