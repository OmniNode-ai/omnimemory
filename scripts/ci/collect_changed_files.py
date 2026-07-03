# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Collect changed file paths for CI smart test selection."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git executable not found on PATH")
    return executable


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [_git_executable(), *args],
        cwd=repo_root,
        check=check,
        text=True,
        capture_output=True,
    )


def _revision_exists(repo_root: Path, revision: str) -> bool:
    result = _run_git(
        repo_root,
        ["rev-parse", "--verify", "--quiet", revision],
        check=False,
    )
    return result.returncode == 0


def _diff_names(repo_root: Path, left_revision: str, right_revision: str) -> list[str]:
    result = _run_git(
        repo_root,
        ["diff", "--name-only", left_revision, right_revision],
    )
    return [line for line in result.stdout.splitlines() if line]


def _fetch_base_ref(repo_root: Path, base_ref: str) -> None:
    _run_git(
        repo_root,
        [
            "fetch",
            "origin",
            f"+refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
            "--no-tags",
            "--prune",
        ],
    )


def collect_changed_files(
    *,
    repo_root: Path,
    event_name: str,
    base_ref: str | None = None,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> list[str]:
    """Return changed file paths for the current GitHub Actions checkout."""
    if event_name == "pull_request":
        if _revision_exists(repo_root, "HEAD^1") and _revision_exists(
            repo_root, "HEAD^2"
        ):
            return _diff_names(repo_root, "HEAD^1", "HEAD^2")

        if (
            base_sha
            and head_sha
            and _revision_exists(repo_root, base_sha)
            and _revision_exists(repo_root, head_sha)
        ):
            return _diff_names(repo_root, base_sha, head_sha)

        if base_ref:
            _fetch_base_ref(repo_root, base_ref)
            return _diff_names(repo_root, f"origin/{base_ref}", "HEAD")

        raise RuntimeError("pull_request changed-file collection needs a base ref")

    if _revision_exists(repo_root, "HEAD~1"):
        return _diff_names(repo_root, "HEAD~1", "HEAD")
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    changed_files = collect_changed_files(
        repo_root=args.repo_root,
        event_name=args.event_name,
        base_ref=args.base_ref or None,
        base_sha=args.base_sha or None,
        head_sha=args.head_sha or None,
    )
    contents = "\n".join(changed_files)
    if contents:
        contents += "\n"
    args.output.write_text(contents, encoding="utf-8")

    sys.stdout.write("Changed files:\n")
    sys.stdout.write(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
