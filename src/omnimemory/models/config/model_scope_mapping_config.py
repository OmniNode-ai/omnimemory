# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Scope mapping configuration for the document ingestion pipeline.

Provides path-to-scope longest-prefix-match, Linear team/project lookup,
and priority hint resolution. Used by all crawler Effects to assign
``scope_ref`` and ``priority_hint`` to discovered documents.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §7
Ticket: OMN-2426
"""

import types
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.config.model_linear_scope_mapping import (
    ModelLinearScopeMapping,
)
from omnimemory.models.config.model_path_scope_mapping import ModelPathScopeMapping


class ModelScopeMappingConfig(BaseModel):
    """Full scope mapping configuration for the document ingestion pipeline.

    Provides ``resolve_scope_for_path()`` (longest prefix match) and
    ``resolve_scope_for_linear()`` (exact lookup with team fallback).
    Both return ``None`` when no mapping is found (callers should fall
    back to a default or skip the document).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    path_mappings: tuple[ModelPathScopeMapping, ...] = Field(
        default=(),
        description=(
            "Ordered list of path-to-scope mappings. Longest prefix match wins. "
            "On equal-length prefixes, the first entry wins. "
            "Typically ends with a broad fallback entry."
        ),
    )
    linear_mappings: tuple[ModelLinearScopeMapping, ...] = Field(
        default=(),
        description=(
            "List of (team, project) to scope_ref mappings. "
            "Resolution order: exact (team, project) first, then (team, None) fallback."
        ),
    )
    priority_hints: Mapping[str, int] = Field(
        default_factory=dict,
        description=(
            "Static priority hint overrides keyed by source pattern or "
            "EnumDetectedDocType value (read-only mapping). "
            "Used by crawlers when no more specific hint is available. "
            "Values 0-100; higher is more important."
        ),
    )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_scope_for_path(self, absolute_path: str) -> str | None:
        """Return the scope_ref for an absolute filesystem path.

        Uses longest prefix match. On equal-length prefixes, the first
        entry in ``path_mappings`` wins (declaration order).

        Returns ``None`` if no mapping covers the given path.

        Args:
            absolute_path: Normalised absolute path (no trailing slash).

        Returns:
            Matching ``scope_ref`` string, or ``None``.
        """
        best_match: ModelPathScopeMapping | None = None
        best_length: int = -1

        for mapping in self.path_mappings:
            prefix = mapping.path_prefix
            # Require path-separator boundary to avoid '/Code/omni' matching
            # '/Code/omnimemory2'.
            if absolute_path == prefix or absolute_path.startswith(prefix + "/"):
                prefix_length = len(prefix)
                if prefix_length > best_length:
                    best_length = prefix_length
                    best_match = mapping

        return best_match.scope_ref if best_match is not None else None

    # ------------------------------------------------------------------
    # Linear resolution
    # ------------------------------------------------------------------

    def resolve_scope_for_linear(self, team: str, project: str | None) -> str | None:
        """Return the scope_ref for a Linear (team, project) pair.

        Resolution order:
        1. Exact match on (team, project).
        2. Fallback match on (team, None) for unassigned issues.

        Returns ``None`` if no mapping covers the given pair.

        Args:
            team:    Linear team name (case-sensitive).
            project: Linear project name, or None for unassigned issues.

        Returns:
            Matching ``scope_ref`` string, or ``None``.
        """
        fallback: str | None = None

        for mapping in self.linear_mappings:
            if mapping.team != team:
                continue
            if mapping.project == project:
                return mapping.scope_ref
            if mapping.project is None:
                fallback = mapping.scope_ref

        return fallback

    # ------------------------------------------------------------------
    # Priority hint resolution
    # ------------------------------------------------------------------

    def resolve_priority_hint(
        self,
        detected_doc_type: EnumDetectedDocType,
        absolute_path: str | None = None,
    ) -> int:
        """Return the priority hint (0-100) for a document.

        Lookup order:
        1. Path-specific hint (e.g., ``~/.claude/CLAUDE.md`` -> 95).
        2. EnumDetectedDocType-based hint.
        3. Default fallback of 35.

        Args:
            detected_doc_type: Classified document type.
            absolute_path:     Normalised absolute path, or None for
                               non-filesystem sources (Linear tickets).

        Returns:
            Integer priority hint in [0, 100].
        """
        if absolute_path is not None:
            path_hint = self.priority_hints.get(absolute_path)
            if path_hint is not None:
                return path_hint

        type_hint = self.priority_hints.get(detected_doc_type.value)
        if type_hint is not None:
            return type_hint

        return _DEFAULT_PRIORITY_HINTS.get(detected_doc_type, 35)


# ---------------------------------------------------------------------------
# Default priority hints matching design doc §7
# ---------------------------------------------------------------------------

_DEFAULT_PRIORITY_HINTS: Mapping[EnumDetectedDocType, int] = types.MappingProxyType(
    {
        EnumDetectedDocType.CLAUDE_MD: 85,
        EnumDetectedDocType.DESIGN_DOC: 70,
        EnumDetectedDocType.ARCHITECTURE_DOC: 80,
        EnumDetectedDocType.PLAN: 65,
        EnumDetectedDocType.HANDOFF: 60,
        EnumDetectedDocType.README: 55,
        EnumDetectedDocType.TICKET: 50,
        EnumDetectedDocType.LINEAR_DOCUMENT: 70,
        EnumDetectedDocType.DEEP_DIVE: 60,
        EnumDetectedDocType.UNKNOWN_MD: 35,
    }
)

# ---------------------------------------------------------------------------
# Default scope mapping config matching design doc §7 examples
# ---------------------------------------------------------------------------
#
# WARNING: LOCAL-DEV CONVENIENCE ONLY.
# The config returned by get_default_scope_mapping_config() contains
# machine-specific paths (/Volumes/PRO-G40/Code/..., /Users/jonah/.claude)
# that are only valid on the primary developer's machine.
# In CI and production, callers MUST supply a ModelScopeMappingConfig
# built from environment-appropriate paths (e.g., read from config file or
# env vars). Do NOT call get_default_scope_mapping_config() in any deployed
# or shared code path.


def get_default_scope_mapping_config() -> ModelScopeMappingConfig:
    """Return a local-dev convenience scope mapping config.

    Constructs and returns a ``ModelScopeMappingConfig`` pre-populated with
    machine-specific paths (``/Volumes/PRO-G40/Code/...``,
    ``/Users/jonah/.claude``) that are only valid on the primary developer's
    machine.

    This function is intentionally lazy: the config is only constructed when
    explicitly called, so importing this module in CI or production code does
    not silently embed wrong paths via a module-level side effect.

    Returns:
        A ``ModelScopeMappingConfig`` covering the local developer's
        filesystem layout and Linear team/project mappings.

    Warning:
        Do NOT call this function in any CI, staging, or production code path.
        Callers in those environments must build their own
        ``ModelScopeMappingConfig`` from environment-appropriate configuration
        (e.g., environment variables or a config file).
    """
    return ModelScopeMappingConfig(
        path_mappings=(
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code/omniintelligence",
                scope_ref="omninode/omniintelligence",
            ),
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code/omnimemory2",
                scope_ref="omninode/omnimemory",
            ),
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code/omnimemory",
                scope_ref="omninode/omnimemory",
            ),
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code/omnibase_core",
                scope_ref="omninode/omnibase_core",
            ),
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code/omni_save/design",
                scope_ref="omninode/shared/design",
            ),
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code/omni_save/plans",
                scope_ref="omninode/shared/plans",
            ),
            ModelPathScopeMapping(
                path_prefix="/Users/jonah/.claude",
                scope_ref="omninode/shared/global-standards",
            ),
            ModelPathScopeMapping(
                path_prefix="/Volumes/PRO-G40/Code",
                scope_ref="omninode/shared",
            ),
        ),
        linear_mappings=(
            ModelLinearScopeMapping(
                team="OmniNode",
                project="OmniIntelligence",
                scope_ref="omninode/omniintelligence",
            ),
            ModelLinearScopeMapping(
                team="OmniNode",
                project="OmniMemory",
                scope_ref="omninode/omnimemory",
            ),
            ModelLinearScopeMapping(
                team="OmniNode",
                project=None,
                scope_ref="omninode/shared",
            ),
        ),
        priority_hints={
            "/Users/jonah/.claude/CLAUDE.md": 95,
        },
    )
