# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Documentation inventory checks for OMN-17427, runnable without dependencies."""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KNOWLEDGE_BASE_URL = "https://github.com/OmniNode-ai/knowledge_base"
_REMOVED = (".github/CODE_OF_CONDUCT.md", ".github/CONTRIBUTING.md", "SECURITY.md")
_REMOVED_NAMES = frozenset(Path(path).name for path in _REMOVED)
_RETAINED = (
    "README.md",
    "CLAUDE.md",
    "LICENSE",
    "CHANGELOG.md",
    ".claude/architecture-handshake.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".onex_ratchets/canonical_file_shape_baseline.txt",
    ".onex_ratchets/node_home_baseline.txt",
)
_PROSE_SUFFIXES = frozenset({".md", ".mdx", ".rst", ".adoc", ".txt"})
_MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _assert_removed_prose_absent() -> None:
    present = [path for path in _REMOVED if (_REPO_ROOT / path).exists()]
    assert not present, f"unread prose still present: {present}"


class TestRemovedProse(unittest.TestCase):
    """Unread prose is gone and the README is a short knowledge-base pointer."""

    def test_removed_prose_is_absent(self) -> None:
        _assert_removed_prose_absent()

    def test_readme_is_a_short_knowledge_base_pointer(self) -> None:
        readme = (_REPO_ROOT / "README.md").read_text()
        lines = readme.splitlines()
        assert 3 <= len(lines) <= 5, f"README has {len(lines)} lines"
        assert _KNOWLEDGE_BASE_URL in readme


class TestRetainedReaders(unittest.TestCase):
    """Retained reader inputs exist and nothing still points at removed files."""

    def test_retained_reader_inputs_exist(self) -> None:
        missing = [path for path in _RETAINED if not (_REPO_ROOT / path).is_file()]
        assert not missing, f"retained reader inputs missing: {missing}"

    def test_no_local_markdown_links_to_removed_files(self) -> None:
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=_REPO_ROOT, text=True
        ).split("\0")
        stale = [
            f"{path}: {target}"
            for path in tracked
            if Path(path).suffix.lower() in _PROSE_SUFFIXES
            and (_REPO_ROOT / path).is_file()
            for target in _MARKDOWN_LINK.findall((_REPO_ROOT / path).read_text())
            if "://" not in target and Path(target.split("#")[0]).name in _REMOVED_NAMES
        ]
        assert not stale, f"local links to removed files: {stale}"

    def test_hygiene_allowlist_drops_security_md(self) -> None:
        hygiene = (_REPO_ROOT / ".public-repo-hygiene.yaml").read_text()
        assert '"SECURITY.md"' not in hygiene


class TestReleaseMetadataAfterRemoval(unittest.TestCase):
    """The OMN-18595 release metadata checks still pass once the prose is removed."""

    def test_release_metadata_checks_pass_after_removal(self) -> None:
        _assert_removed_prose_absent()
        spec = importlib.util.spec_from_file_location(
            "release_metadata_omn18595",
            _REPO_ROOT / "tests" / "test_release_metadata_omn18595.py",
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        assert suite.countTestCases() == 3, suite.countTestCases()
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        assert result.wasSuccessful(), result.failures + result.errors


if __name__ == "__main__":
    unittest.main()
