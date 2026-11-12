# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23949: the omnibase-core source names one revision everywhere.

pyproject.toml, uv.lock and the check-handshake workflow must all name the
dev-20261111183916-a85cb916d4c1 tag. The lock additionally records the
resolved commit, which the handshake workflow checks out by raw sha, so the
three references are only consistent when the sha extends the tag suffix.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CORE_TAG = "dev-20261111183916-a85cb916d4c1"
CORE_GIT_URL = "https://github.com/OmniNode-ai/omnibase_core.git"


def _lock_core_commit() -> str:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    entries = [p for p in lock["package"] if p["name"] == "omnibase-core"]
    assert len(entries) == 1, (
        f"expected exactly one omnibase-core package entry, got {len(entries)}"
    )
    source = entries[0]["source"]["git"]
    prefix = f"{CORE_GIT_URL}?tag={EXPECTED_CORE_TAG}#"
    assert source.startswith(prefix), f"unexpected omnibase-core source: {source}"
    commit = source[len(prefix) :]
    assert len(commit) == 40, f"expected a 40-hex commit after '#', got: {commit}"
    assert set(commit) <= set("0123456789abcdef"), f"commit is not hex: {commit}"
    return commit


def test_pyproject_sources_name_the_expected_tag() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    source = pyproject["tool"]["uv"]["sources"]["omnibase-core"]
    assert source["git"] == CORE_GIT_URL
    assert source["tag"] == EXPECTED_CORE_TAG


def test_lock_resolves_core_to_the_tagged_commit() -> None:
    commit = _lock_core_commit()
    tag_suffix = EXPECTED_CORE_TAG.rsplit("-", 1)[-1]
    assert commit.startswith(tag_suffix), (
        f"lock commit {commit} does not extend the tag suffix {tag_suffix}"
    )


def test_handshake_workflow_pins_the_same_commit() -> None:
    commit = _lock_core_commit()
    workflow = (REPO_ROOT / ".github/workflows/check-handshake.yml").read_text(
        encoding="utf-8"
    )
    ref_lines = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("ref:")
    ]
    assert f"ref: {commit}" in ref_lines, (
        f"check-handshake.yml has no 'ref: {commit}' line; found {ref_lines}"
    )
