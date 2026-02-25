# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Path scope mapping model for filesystem-based scope resolution.

Maps a filesystem path prefix to a scope_ref string. Used by
``ModelScopeMappingConfig.resolve_scope_for_path()`` for longest-prefix-match.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §7
Ticket: OMN-2426
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelPathScopeMapping(BaseModel):
    """Maps a path prefix to a scope_ref string.

    Entries are evaluated in the order they appear in
    ``ModelScopeMappingConfig.path_mappings``. Longest match wins; on equal
    length, the first entry in the list wins.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    path_prefix: str = Field(
        ...,
        min_length=1,
        description=(
            "Absolute path prefix (no trailing slash). "
            "Example: '/Volumes/PRO-G40/Code/omniintelligence'."
        ),
    )
    scope_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "Resolved scope string: org/repo or org/repo/subpath. "
            "Example: 'omninode/omniintelligence'."
        ),
    )
