# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22385: the running configuration names no omnibase-compat dependency.

omnibase_compat is retired. The uv override list, the sibling lock refresh
default package list and the dep-provenance governed list must not name it.
The pattern is a character-class regex, so this file does not match itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPAT = re.compile(r"omnibase[_-]compat")

RUNNING_FILES = (
    "pyproject.toml",
    ".github/workflows/sibling-lock-refresh.yml",
)


@pytest.mark.parametrize("relative", RUNNING_FILES)
def test_running_file_names_no_retired_compat_package(relative: str) -> None:
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), start=1)
        if COMPAT.search(line) and not line.lstrip().startswith("#")
    ]
    assert hits == [], f"{relative} still names the retired package: {hits}"
