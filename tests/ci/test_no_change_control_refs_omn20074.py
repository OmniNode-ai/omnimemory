# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20074: live files and dead fixtures name no change-control repository.

The pattern is a character-class regex, so this file does not match itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGE_CONTROL = re.compile(r"onex[_-]change[_-]control")

# Live configuration and tooling that used to name the repository.
CLEAN_FILES = (
    "architecture-handshakes/validator-requirements.yaml",
    "scripts/check_dep_provenance.py",
    "tests/scripts/test_check_dep_provenance.py",
    ".github/workflows/dep-provenance-gate.yml",
)

# Fixtures that only exercised callers removed by the S6 cut-over; each one
# returned early once the caller was gone, so none still asserts anything.
DEAD_FIXTURES = (
    "tests/unit/scripts/test_occ_manual_replay_precheck.py",
    "tests/unit/test_occ_companion_effect_edited_trigger_omn_18334.py",
    "tests/unit/util_occ_retirement.py",
    "tests/workflows/test_occ_retirement.py",
)


@pytest.mark.parametrize("relative", CLEAN_FILES)
def test_live_file_names_no_change_control_repository(relative: str) -> None:
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert not CHANGE_CONTROL.findall(text), relative


@pytest.mark.parametrize("relative", DEAD_FIXTURES)
def test_dead_fixture_is_gone(relative: str) -> None:
    assert not (REPO_ROOT / relative).exists(), relative


def test_the_pattern_has_a_positive_control() -> None:
    for separator in ("_", "-"):
        name = separator.join(("onex", "change", "control"))
        assert CHANGE_CONTROL.search(f"pins {name} at a rev")
    assert not CHANGE_CONTROL.search("omnibase_core")


def test_no_test_selection_entry_points_at_a_missing_file() -> None:
    text = (REPO_ROOT / "scripts/ci/test_selection_adjacency.yaml").read_text(
        encoding="utf-8"
    )
    missing = [
        match
        for match in re.findall(r"^  (tests/\S+\.py):$", text, flags=re.MULTILINE)
        if not (REPO_ROOT / match).exists()
    ]
    assert not missing, missing
