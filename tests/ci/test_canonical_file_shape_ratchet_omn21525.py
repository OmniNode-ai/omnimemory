# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21525: the canonical-file-shape ratchet holds in this repository.

The tracked tree is copied into a throwaway git repository, committed as the
base, and one change is committed on top. The entry is the Canonical File Shape
job's own: ``node_canonical_file_shape_check_compute`` from the omnibase_core
dev-merge tag in ``ci.yml``, run with ``--rules file-shape --base <base>``.
The locked omnibase-core predates that node, so it runs through ``uv run --with``
as ``test_imperative_contract_guard_caller_omn20943.py`` does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODULE = (
    "omnibase_core.nodes.node_canonical_file_shape_check_compute."
    "runtime_canonical_file_shape_check"
)
BASELINE = ".onex_ratchets/canonical_file_shape_baseline.txt"
BURNDOWN = ".onex_ratchets/canonical_file_shape_burndown_baseline.txt"
PLANTED_HANDLER = "src/omnimemory/handlers/handler_zz_planted.py"
BASE_REF = "base-ref"
IDENTITY = ("-c", "user.name=t", "-c", "user.email=t@example.com")


def _ci_core_pin() -> str:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    pin = workflow["jobs"]["canonical-file-shape"]["env"]["CANONICAL_FILE_SHAPE_CORE"]
    assert isinstance(pin, str)
    return pin


def _git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *IDENTITY, *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _entries(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = tmp_path_factory.mktemp("omn21525") / "repo"
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for rel in filter(None, tracked):
        source = ROOT / rel
        if source.is_file():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, repo / rel)
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "branch", "-f", BASE_REF)
    return repo


def _check(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            _ci_core_pin(),
            "python",
            "-m",
            RUNTIME_MODULE,
            "--rules",
            "file-shape",
            "--repo-root",
            str(repo),
            "--base",
            base,
            "--head",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        cwd=repo,
    )


def _commit_on_top(repo: Path, name: str) -> str:
    _git(repo, "checkout", "-q", "-B", name, BASE_REF)
    return _git(repo, "rev-parse", BASE_REF)


def test_burndown_baseline_names_the_ticket_and_holds_the_measured_findings() -> None:
    burndown = ROOT / BURNDOWN
    assert (
        burndown.read_text().splitlines()[0].startswith("# Burn-down ticket: OMN-21525")
    )
    entries = _entries(burndown)
    assert entries == sorted(set(entries))
    assert not set(entries) & set(_entries(ROOT / BASELINE))


def test_clean_tree_is_accepted(base_repo: Path) -> None:
    base = _commit_on_top(base_repo, "clean")
    _git(base_repo, "commit", "-q", "--allow-empty", "-m", "empty")
    result = _check(base_repo, base)
    assert result.returncode == 0, result.stdout + result.stderr


def test_new_unbound_handler_is_refused_as_handler_outside_node(
    base_repo: Path,
) -> None:
    base = _commit_on_top(base_repo, "planted-handler")
    planted = base_repo / PLANTED_HANDLER
    planted.write_text("def handle(request: object) -> object:\n    return request\n")
    _git(base_repo, "add", PLANTED_HANDLER)
    _git(base_repo, "commit", "-q", "-m", "plant one unbound handler")
    result = _check(base_repo, base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{PLANTED_HANDLER}: [handler_outside_node]" in result.stdout + result.stderr


def test_burndown_line_for_an_existing_refused_file_is_refused_as_growth(
    base_repo: Path,
) -> None:
    base = _commit_on_top(base_repo, "burndown-growth")
    moved = _entries(base_repo / BASELINE)[0]
    for rel, keep in ((BASELINE, False), (BURNDOWN, True)):
        path = base_repo / rel
        lines = path.read_text().splitlines()
        lines = [line for line in lines if line != moved]
        if keep:
            lines.append(moved)
        path.write_text("\n".join(lines) + "\n")
    _git(base_repo, "add", BASELINE, BURNDOWN)
    _git(base_repo, "commit", "-q", "-m", "move one baseline line into the burn-down")
    result = _check(base_repo, base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        f"{BURNDOWN}: [baseline-growth] burn-down entry '{moved}'"
        in result.stdout + result.stderr
    )
