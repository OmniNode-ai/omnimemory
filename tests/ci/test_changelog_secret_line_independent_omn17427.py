# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CHANGELOG.md must not depend on a line number in .secrets.baseline (OMN-17427).

The release train prepends a section to the top of CHANGELOG.md on every release.
The only credential-shaped text in the file is a historical documentation example
(a database URL with a user and a password in it). While it was tracked in
``.secrets.baseline`` by line number, every release shifted it, the
detect-secrets hook rewrote the baseline and exited 3, and the release PR failed
the pre-commit, Detect Secrets and ONEX Validation jobs until someone refreshed
the number by hand (omnimemory#709 did, omnimemory#719 did not).

The example is allowlisted inline instead, so a prepend can never move a baseline
entry: CHANGELOG.md has no baseline entry, and every credential-shaped line in it
carries the inline marker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# The URL-credential shape detect-secrets' BasicAuthDetector reports.
_BASIC_AUTH_RE = re.compile(r"://[^\s:/?#\[\]@]+:[^\s:/?#\[\]@]+@")
_ALLOWLIST_RE = re.compile(r"pragma: ?allowlist[ -]secret")


def test_changelog_has_no_baseline_entry() -> None:
    results = json.loads((_ROOT / ".secrets.baseline").read_text())["results"]
    assert "CHANGELOG.md" not in results, (
        "CHANGELOG.md is tracked by line number in .secrets.baseline; the next release "
        "shifts it and the detect-secrets hook fails the release PR"
    )


def test_every_credential_shaped_changelog_line_is_allowlisted_inline() -> None:
    offenders = [
        number
        for number, line in enumerate(
            (_ROOT / "CHANGELOG.md").read_text().splitlines(), start=1
        )
        if _BASIC_AUTH_RE.search(line) and not _ALLOWLIST_RE.search(line)
    ]
    assert offenders == [], f"CHANGELOG.md lines {offenders} need an inline allowlist"
