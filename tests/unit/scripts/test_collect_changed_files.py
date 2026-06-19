# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for CI changed-file collection."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.ci.collect_changed_files import collect_changed_files


def _git(repo_root: Path, *args: str) -> str:
    executable = shutil.which("git")
    assert executable is not None
    result = subprocess.run(
        [executable, *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _write(repo_root: Path, relative_path: str, contents: str) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _commit(repo_root: Path, message: str) -> str:
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD")


def _init_repo(repo_root: Path) -> None:
    _git(repo_root, "init", "-b", "dev")
    _git(repo_root, "config", "user.email", "ci@example.test")
    _git(repo_root, "config", "user.name", "CI Test")


@pytest.mark.unit
def test_pull_request_merge_checkout_uses_merge_parents(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _write(repo_root, "README.md", "base\n")
    _commit(repo_root, "base")

    _git(repo_root, "checkout", "-b", "feature")
    _write(repo_root, "src/omnimemory/models/model_example.py", "value = 1\n")
    _commit(repo_root, "feature")

    _git(repo_root, "checkout", "dev")
    _git(repo_root, "merge", "--no-ff", "feature", "-m", "merge feature")

    assert collect_changed_files(
        repo_root=repo_root,
        event_name="pull_request",
        base_ref="dev",
    ) == ["src/omnimemory/models/model_example.py"]


@pytest.mark.unit
def test_pull_request_head_checkout_uses_event_shas(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _write(repo_root, "README.md", "base\n")
    base_sha = _commit(repo_root, "base")

    _git(repo_root, "checkout", "-b", "feature")
    _write(repo_root, "tests/unit/scripts/test_example.py", "def test_ok(): pass\n")
    head_sha = _commit(repo_root, "feature")

    assert collect_changed_files(
        repo_root=repo_root,
        event_name="pull_request",
        base_ref="dev",
        base_sha=base_sha,
        head_sha=head_sha,
    ) == ["tests/unit/scripts/test_example.py"]


@pytest.mark.unit
def test_non_pr_initial_commit_returns_empty_change_set(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _write(repo_root, "README.md", "base\n")
    _commit(repo_root, "base")

    assert collect_changed_files(repo_root=repo_root, event_name="push") == []
