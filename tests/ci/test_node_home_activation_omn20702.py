# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The node-home ratchet is activated in this repository (OMN-20702).

A new ONEX node directory or ``onex.nodes`` entry point outside omnimarket is
refused against the shrink-only ``.onex_ratchets/node_home_baseline.txt``. The
check is the ``node-home`` rule group of the omnibase_core node
``node_canonical_file_shape_check_compute`` (OMN-21525). The locked omnibase-core
predates that node, so the entry runs through the Canonical File Shape job's pin.
"""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODULE = (
    "omnibase_core.nodes.node_canonical_file_shape_check_compute."
    "runtime_canonical_file_shape_check"
)
RUNTIME_COMMAND = f"{RUNTIME_MODULE} --rules node-home"


@cache
def _ci_core_pin() -> str:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    env = workflow["jobs"]["canonical-file-shape"]["env"]
    pin = env["CANONICAL_FILE_SHAPE_CORE"]
    assert isinstance(pin, str)
    return pin


def _uv_core(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--no-project", "--with", _ci_core_pin(), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        cwd=ROOT,
    )


@cache
def _node_home_baseline_path() -> str:
    result = _uv_core(
        "python",
        "-c",
        "from omnibase_core.validation.checker_canonical_file_shape import "
        "load_rule_set; "
        "from omnibase_core.validation.validator_canonical_file_shape import "
        "ValidatorCanonicalFileShape; "
        "print(load_rule_set(ValidatorCanonicalFileShape().contract)"
        ".parameters.node_home_baseline_path)",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _baseline_entries() -> list[str]:
    node_home_baseline = _node_home_baseline_path()
    baseline = ROOT / node_home_baseline
    assert baseline.is_file(), f"{node_home_baseline} is missing"
    return [
        line
        for line in baseline.read_text().splitlines()
        if line and not line.startswith("#")
    ]


def test_baseline_lists_only_existing_sorted_node_directories() -> None:
    entries = _baseline_entries()
    assert entries
    assert entries == sorted(set(entries))
    missing = [entry for entry in entries if not (ROOT / entry).is_dir()]
    assert missing == []


def test_check_accepts_the_tree_against_head() -> None:
    result = _uv_core(
        "python",
        "-m",
        RUNTIME_MODULE,
        "--rules",
        "node-home",
        "--repo-root",
        str(ROOT),
        "--base",
        "HEAD",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pre_commit_runs_the_check_from_omnibase_core() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [
        (repo["repo"], hook["id"])
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
    ]
    assert (
        "https://github.com/OmniNode-ai/omnibase_core",
        "check-node-home",
    ) in hooks


def test_ci_runs_the_check_against_the_merge_base_with_a_positive_control() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert f"python -m {RUNTIME_COMMAND}" in workflow
    assert 'node_home --base "$base"' in workflow
    assert "POSITIVE CONTROL FAILED: planted" in workflow
