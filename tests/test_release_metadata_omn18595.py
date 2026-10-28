# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Release metadata regression checks for OMN-18595, runnable without dependencies."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestReleaseMetadata(unittest.TestCase):
    """Keep the package, editable lock entry and newest release notes in sync."""

    def test_release_includes_omn18595_version(self) -> None:
        """The release must advance beyond the 0.18.6 merge-base version."""
        project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
        version = tuple(int(part) for part in project["version"].split("."))
        assert version >= (0, 18, 7), f"Release {version} predates 0.18.7"

    def test_lock_matches_project_version(self) -> None:
        """The editable package must carry the published project version."""
        project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
        lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
        package = next(
            item for item in lock["package"] if item["name"] == project["name"]
        )
        assert package["version"] == project["version"]

    def test_newest_changelog_matches_project_version(self) -> None:
        """The first changelog heading must describe the package being released."""
        project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
        heading = (_REPO_ROOT / "CHANGELOG.md").read_text().splitlines()[0]
        assert heading.startswith(f"## v{project['version']} ("), heading


if __name__ == "__main__":
    unittest.main()
