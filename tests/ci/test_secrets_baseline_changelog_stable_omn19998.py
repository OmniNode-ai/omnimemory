# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19998: a release's CHANGELOG entry does not move a tracked secret finding.

The release train prepends a section to CHANGELOG.md. A baseline finding keyed
by CHANGELOG.md carries a line number, so every release shifted it, the
detect-secrets hook rewrote .secrets.baseline and failed ONEX Validation on the
release PR (omnimemory#719: line 365 became 377; release 0.18.9 needed a manual
baseline commit for the same reason). The one finding is a documentation
placeholder, so it is allowlisted on its own line and the baseline holds no
CHANGELOG.md entry to go stale.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from detect_secrets.pre_commit_hook import main as secrets_hook

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRAGMA = "pragma: allowlist secret"
PLACEHOLDER_URL = "OMNIMEMORY_DB_URL=postgresql://user:"


def test_baseline_holds_no_changelog_finding() -> None:
    baseline = json.loads((REPO_ROOT / ".secrets.baseline").read_text(encoding="utf-8"))
    assert "CHANGELOG.md" not in baseline["results"], (
        "a CHANGELOG.md baseline entry is line-number keyed and goes stale "
        "on every release that prepends a section"
    )


def test_placeholder_credential_url_lines_are_allowlisted_in_place() -> None:
    lines = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    placeholder_lines = [line for line in lines if PLACEHOLDER_URL in line]
    assert placeholder_lines, "positive control: the placeholder line must exist"
    unmarked = [line[:80] for line in placeholder_lines if PRAGMA not in line]
    assert unmarked == [], f"placeholder credential URL without {PRAGMA!r}: {unmarked}"


@pytest.mark.parametrize("prepended_lines", [0, 12])
def test_release_prepend_leaves_baseline_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepended_lines: int
) -> None:
    baseline = tmp_path / ".secrets.baseline"
    before = (REPO_ROOT / ".secrets.baseline").read_bytes()
    baseline.write_bytes(before)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "\n" * prepended_lines
        + (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # The hook asks git which files are changed but unstaged; it needs a repo.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    args = ["--baseline", ".secrets.baseline", "CHANGELOG.md"]
    assert secrets_hook(args) == 0
    assert baseline.read_bytes() == before

    # Positive control: the same hook must still reject an unmarked credential.
    changelog.write_text(changelog.read_text().replace(PRAGMA, ""), encoding="utf-8")
    assert secrets_hook(args) == 1
    assert baseline.read_bytes() == before
