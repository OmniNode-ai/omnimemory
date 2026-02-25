# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Linear scope mapping model for Linear-based scope resolution.

Maps a (team, project) pair from the Linear API to a scope_ref string.
Used by ``ModelScopeMappingConfig.resolve_scope_for_linear()``.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §7
Ticket: OMN-2426
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelLinearScopeMapping(BaseModel):
    """Maps a Linear (team, project) pair to a scope_ref string.

    ``project`` may be None to match all issues in a team that are not
    assigned to a specific project (team-level fallback mapping).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    team: str = Field(
        ...,
        min_length=1,
        description="Linear team name (case-sensitive).",
    )
    project: str | None = Field(
        default=None,
        description=(
            "Linear project name (case-sensitive), or None to match all "
            "unassigned issues in the team."
        ),
    )
    scope_ref: str = Field(
        ...,
        min_length=1,
        description="Resolved scope string: org/repo or org/repo/subpath.",
    )
