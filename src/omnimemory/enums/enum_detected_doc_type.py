"""
Detected document type enumeration for the document ingestion pipeline.

Assigned by ``ChunkClassifierCompute`` based on filename patterns and
path heuristics. Used to select the appropriate chunking strategy and
to compute initial priority hints for ContextItems.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §8
Ticket: OMN-2426
"""

from enum import Enum


class EnumDetectedDocType(str, Enum):
    """Classification of a document's structural role in the codebase.

    Values are stored in Kafka event payloads and Qdrant metadata so
    must remain stable once deployed. Version bumps are required for
    any change to classification rules (replay determinism invariant).
    """

    CLAUDE_MD = "claude_md"
    """Any file named ``CLAUDE.md`` — explicit policy/standards document."""

    DESIGN_DOC = "design_doc"
    """Files in a ``design/`` directory."""

    ARCHITECTURE_DOC = "architecture_doc"
    """Files matching ``*ARCHITECTURE*.md`` or ``*OVERVIEW*.md``."""

    PLAN = "plan"
    """Files in a ``plans/`` directory."""

    HANDOFF = "handoff"
    """Files in a ``handoffs/`` directory."""

    README = "readme"
    """``README.md`` at a repository root."""

    TICKET = "ticket"
    """Linear issue fetched via the Linear crawler."""

    LINEAR_DOCUMENT = "linear_document"
    """Linear document (not an issue) fetched via the Linear crawler."""

    DEEP_DIVE = "deep_dive"
    """Files matching ``*DEEP_DIVE*.md``."""

    UNKNOWN_MD = "unknown_md"
    """Any other ``.md`` file that does not match a more specific pattern."""
