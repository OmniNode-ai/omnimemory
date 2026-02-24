# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelScopeMappingConfig longest-prefix-match logic.

Tests cover:
- Path-to-scope longest-prefix-match (basic, tie-break, no-match, path boundary)
- Linear scope resolution (exact, team fallback, no-match)
- Priority hint resolution (path override, doc-type lookup, default fallback)
- get_default_scope_mapping_config() sanity checks
- ModelScopeMappingConfig round-trip serialization

Ticket: OMN-2426
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.config.model_linear_scope_mapping import (
    ModelLinearScopeMapping,
)
from omnimemory.models.config.model_path_scope_mapping import ModelPathScopeMapping
from omnimemory.models.config.model_scope_mapping_config import (
    ModelScopeMappingConfig,
    get_default_scope_mapping_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(*path_mappings: tuple[str, str]) -> ModelScopeMappingConfig:
    """Build a ModelScopeMappingConfig from (prefix, scope_ref) pairs."""
    return ModelScopeMappingConfig(
        path_mappings=tuple(
            ModelPathScopeMapping(path_prefix=prefix, scope_ref=scope)
            for prefix, scope in path_mappings
        ),
    )


# ---------------------------------------------------------------------------
# Path resolution -- basic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveScopeForPath:
    """Longest-prefix-match path resolution."""

    def test_exact_match_returns_scope(self) -> None:
        cfg = make_config(("/Code/repo", "omninode/repo"))
        assert cfg.resolve_scope_for_path("/Code/repo") == "omninode/repo"

    def test_subpath_match_returns_scope(self) -> None:
        cfg = make_config(("/Code/repo", "omninode/repo"))
        assert cfg.resolve_scope_for_path("/Code/repo/src/foo.py") == "omninode/repo"

    def test_longer_prefix_wins_over_shorter(self) -> None:
        cfg = make_config(
            ("/Code", "omninode/shared"),
            ("/Code/repo", "omninode/repo"),
        )
        assert cfg.resolve_scope_for_path("/Code/repo/file.py") == "omninode/repo"

    def test_shorter_prefix_wins_when_longer_does_not_match(self) -> None:
        cfg = make_config(
            ("/Code", "omninode/shared"),
            ("/Code/other", "omninode/other"),
        )
        assert cfg.resolve_scope_for_path("/Code/repo/file.py") == "omninode/shared"

    def test_no_match_returns_none(self) -> None:
        cfg = make_config(("/Code/repo", "omninode/repo"))
        assert cfg.resolve_scope_for_path("/Users/jonah/file.py") is None

    def test_empty_mappings_returns_none(self) -> None:
        cfg = ModelScopeMappingConfig()
        assert cfg.resolve_scope_for_path("/Code/anything") is None

    # ------------------------------------------------------------------
    # Path boundary safety
    # ------------------------------------------------------------------

    def test_prefix_does_not_match_sibling_directory(self) -> None:
        """/Code/omni must NOT match /Code/omnimemory2."""
        cfg = make_config(("/Code/omni", "omninode/wrong"))
        assert cfg.resolve_scope_for_path("/Code/omnimemory2/src/file.py") is None

    def test_prefix_matches_same_name_with_slash(self) -> None:
        """/Code/omni should match /Code/omni/src/file.py."""
        cfg = make_config(("/Code/omni", "omninode/correct"))
        assert (
            cfg.resolve_scope_for_path("/Code/omni/src/file.py") == "omninode/correct"
        )

    def test_exact_path_match_with_no_trailing_slash(self) -> None:
        cfg = make_config(("/Code/repo", "omninode/repo"))
        assert cfg.resolve_scope_for_path("/Code/repo") == "omninode/repo"

    # ------------------------------------------------------------------
    # Tie-breaking: first declaration wins on equal-length prefixes
    # ------------------------------------------------------------------

    def test_equal_length_first_declaration_wins(self) -> None:
        cfg = ModelScopeMappingConfig(
            path_mappings=(
                ModelPathScopeMapping(path_prefix="/Code/abc", scope_ref="first"),
                ModelPathScopeMapping(path_prefix="/Code/abc", scope_ref="second"),
            )
        )
        assert cfg.resolve_scope_for_path("/Code/abc/file.py") == "first"

    # ------------------------------------------------------------------
    # Multiple competing prefixes
    # ------------------------------------------------------------------

    def test_three_level_nesting_selects_deepest(self) -> None:
        cfg = make_config(
            ("/a", "level1"),
            ("/a/b", "level2"),
            ("/a/b/c", "level3"),
        )
        assert cfg.resolve_scope_for_path("/a/b/c/file.txt") == "level3"

    def test_mid_level_selected_when_deepest_does_not_match(self) -> None:
        cfg = make_config(
            ("/a", "level1"),
            ("/a/b", "level2"),
            ("/a/b/c", "level3"),
        )
        assert cfg.resolve_scope_for_path("/a/b/other/file.txt") == "level2"


# ---------------------------------------------------------------------------
# Linear resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveScopeForLinear:
    """Linear (team, project) scope resolution."""

    def _make_linear_config(self) -> ModelScopeMappingConfig:
        return ModelScopeMappingConfig(
            linear_mappings=(
                ModelLinearScopeMapping(
                    team="OmniNode",
                    project="OmniMemory",
                    scope_ref="omninode/omnimemory",
                ),
                ModelLinearScopeMapping(
                    team="OmniNode",
                    project="OmniIntelligence",
                    scope_ref="omninode/omniintelligence",
                ),
                ModelLinearScopeMapping(
                    team="OmniNode",
                    project=None,
                    scope_ref="omninode/shared",
                ),
            ),
        )

    def test_exact_team_project_match(self) -> None:
        cfg = self._make_linear_config()
        assert (
            cfg.resolve_scope_for_linear("OmniNode", "OmniMemory")
            == "omninode/omnimemory"
        )

    def test_exact_match_takes_precedence_over_fallback(self) -> None:
        cfg = self._make_linear_config()
        assert (
            cfg.resolve_scope_for_linear("OmniNode", "OmniIntelligence")
            == "omninode/omniintelligence"
        )

    def test_team_fallback_for_unknown_project(self) -> None:
        cfg = self._make_linear_config()
        assert (
            cfg.resolve_scope_for_linear("OmniNode", "SomeOtherProject")
            == "omninode/shared"
        )

    def test_team_fallback_for_none_project(self) -> None:
        cfg = self._make_linear_config()
        assert cfg.resolve_scope_for_linear("OmniNode", None) == "omninode/shared"

    def test_unknown_team_returns_none(self) -> None:
        cfg = self._make_linear_config()
        assert cfg.resolve_scope_for_linear("UnknownTeam", "Project") is None

    def test_empty_linear_mappings_returns_none(self) -> None:
        cfg = ModelScopeMappingConfig()
        assert cfg.resolve_scope_for_linear("OmniNode", "OmniMemory") is None


# ---------------------------------------------------------------------------
# Priority hints
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolvePriorityHint:
    """Priority hint resolution: path override > doctype > default."""

    def test_path_override_wins(self) -> None:
        cfg = ModelScopeMappingConfig(
            priority_hints={"/Users/jonah/.claude/CLAUDE.md": 95},
        )
        hint = cfg.resolve_priority_hint(
            EnumDetectedDocType.CLAUDE_MD,
            absolute_path="/Users/jonah/.claude/CLAUDE.md",
        )
        assert hint == 95

    def test_doctype_lookup_without_path(self) -> None:
        cfg = ModelScopeMappingConfig(
            priority_hints={"claude_md": 90},
        )
        assert cfg.resolve_priority_hint(EnumDetectedDocType.CLAUDE_MD) == 90

    def test_default_fallback_for_unknown_md(self) -> None:
        cfg = ModelScopeMappingConfig()
        assert cfg.resolve_priority_hint(EnumDetectedDocType.UNKNOWN_MD) == 35

    def test_default_fallback_for_claude_md(self) -> None:
        """Default (no overrides) returns the built-in default."""
        cfg = ModelScopeMappingConfig()
        assert cfg.resolve_priority_hint(EnumDetectedDocType.CLAUDE_MD) == 85

    def test_default_fallback_for_architecture_doc(self) -> None:
        cfg = ModelScopeMappingConfig()
        assert cfg.resolve_priority_hint(EnumDetectedDocType.ARCHITECTURE_DOC) == 80

    def test_path_override_does_not_affect_other_paths(self) -> None:
        cfg = ModelScopeMappingConfig(
            priority_hints={"/Users/jonah/.claude/CLAUDE.md": 95},
        )
        hint = cfg.resolve_priority_hint(
            EnumDetectedDocType.CLAUDE_MD,
            absolute_path="/Volumes/PRO-G40/Code/repo/CLAUDE.md",
        )
        assert hint == 85


# ---------------------------------------------------------------------------
# get_default_scope_mapping_config() sanity checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultScopeMappingConfig:
    """Sanity-checks for the local-dev default config."""

    def test_global_claude_md_resolves_to_global_standards(self) -> None:
        result = get_default_scope_mapping_config().resolve_scope_for_path(
            "/Users/jonah/.claude/CLAUDE.md"
        )
        assert result == "omninode/shared/global-standards"

    def test_repo_claude_md_resolves_to_repo_scope(self) -> None:
        result = get_default_scope_mapping_config().resolve_scope_for_path(
            "/Volumes/PRO-G40/Code/omniintelligence/CLAUDE.md"
        )
        assert result == "omninode/omniintelligence"

    def test_omnimemory2_resolves_to_omnimemory(self) -> None:
        result = get_default_scope_mapping_config().resolve_scope_for_path(
            "/Volumes/PRO-G40/Code/omnimemory2/src/omnimemory/models/foo.py"
        )
        assert result == "omninode/omnimemory"

    def test_design_doc_resolves_to_shared_design(self) -> None:
        result = get_default_scope_mapping_config().resolve_scope_for_path(
            "/Volumes/PRO-G40/Code/omni_save/design/DESIGN_FOO.md"
        )
        assert result == "omninode/shared/design"

    def test_fallback_code_path_resolves_to_omninode_shared(self) -> None:
        result = get_default_scope_mapping_config().resolve_scope_for_path(
            "/Volumes/PRO-G40/Code/some_unknown_repo/README.md"
        )
        assert result == "omninode/shared"

    def test_linear_omnimemory_project_resolves_correctly(self) -> None:
        result = get_default_scope_mapping_config().resolve_scope_for_linear(
            "OmniNode", "OmniMemory"
        )
        assert result == "omninode/omnimemory"

    def test_global_claude_md_priority_hint(self) -> None:
        hint = get_default_scope_mapping_config().resolve_priority_hint(
            EnumDetectedDocType.CLAUDE_MD,
            absolute_path="/Users/jonah/.claude/CLAUDE.md",
        )
        assert hint == 95


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelScopeMappingConfigSerialization:
    """Verify ModelScopeMappingConfig serializes and deserializes correctly."""

    def test_empty_config_round_trip(self) -> None:
        cfg = ModelScopeMappingConfig()
        data = cfg.model_dump()
        restored = ModelScopeMappingConfig.model_validate(data)
        assert restored == cfg

    def test_full_config_round_trip(self) -> None:
        cfg = ModelScopeMappingConfig(
            path_mappings=(
                ModelPathScopeMapping(path_prefix="/a/b", scope_ref="s/a"),
                ModelPathScopeMapping(path_prefix="/a", scope_ref="s/fallback"),
            ),
            linear_mappings=(
                ModelLinearScopeMapping(team="T", project="P", scope_ref="t/p"),
                ModelLinearScopeMapping(team="T", project=None, scope_ref="t/shared"),
            ),
            priority_hints={"claude_md": 90, "/special/path": 99},
        )
        data = cfg.model_dump()
        restored = ModelScopeMappingConfig.model_validate(data)
        assert restored == cfg

    def test_model_is_frozen(self) -> None:
        cfg = ModelScopeMappingConfig()
        with pytest.raises(ValidationError):
            cfg.path_mappings = ()  # type: ignore[misc]

    def test_default_config_round_trip(self) -> None:
        default_cfg = get_default_scope_mapping_config()
        data = default_cfg.model_dump()
        restored = ModelScopeMappingConfig.model_validate(data)
        assert restored == default_cfg
