# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: accepting retirement requires a complete enforcing replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.util_occ_retirement import occ_callers_retired

pytestmark = pytest.mark.unit


def _cutover_tree(root: Path) -> dict[str, Any]:
    workflows = root / ".github/workflows"
    workflows.mkdir(parents=True)
    caller = {
        "on": {"pull_request_target": {"branches": ["dev", "main"]}},
        "jobs": {
            "repo-evidence": {
                "uses": "OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml@"
                + "a" * 40,
                "with": {
                    "evidence-source": "caller",
                    "shadow": "false",
                    "compare-with-occ": "false",
                },
            }
        },
    }
    (workflows / "call-repo-evidence-gate.yml").write_text(yaml.safe_dump(caller))
    (workflows / "ci.yml").write_text(yaml.safe_dump({"jobs": {"tests": {}}}))
    return caller


def test_complete_cutover_is_accepted(tmp_path: Path) -> None:
    _cutover_tree(tmp_path)
    assert occ_callers_retired(tmp_path)


def test_active_caller_keeps_the_original_tests(tmp_path: Path) -> None:
    workflows = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    (workflows / "call-occ-companion-effect.yml").write_text("name: OCC\n")
    assert not occ_callers_retired(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("shadow", "true"), ("compare-with-occ", "true"), ("evidence-source", "occ")],
)
def test_retirement_refuses_inactive_replacement(
    tmp_path: Path, key: str, value: str
) -> None:
    caller = _cutover_tree(tmp_path)
    caller["jobs"]["repo-evidence"]["with"][key] = value
    (tmp_path / ".github/workflows/call-repo-evidence-gate.yml").write_text(
        yaml.safe_dump(caller)
    )
    with pytest.raises(AssertionError, match=key):
        occ_callers_retired(tmp_path)


@pytest.mark.parametrize("fault", ["trigger", "pin", "condition", "missing"])
def test_retirement_refuses_broken_admission(tmp_path: Path, fault: str) -> None:
    caller = _cutover_tree(tmp_path)
    path = tmp_path / ".github/workflows/call-repo-evidence-gate.yml"
    if fault == "missing":
        path.unlink()
    else:
        if fault == "trigger":
            caller["on"] = {"workflow_dispatch": {}}
        elif fault == "pin":
            caller["jobs"]["repo-evidence"]["uses"] = "other.yml@dev"
        else:
            caller["jobs"]["repo-evidence"]["if"] = "false"
        path.write_text(yaml.safe_dump(caller))
    with pytest.raises(AssertionError):
        occ_callers_retired(tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/call-occ-autobind.yml",
        "scripts/ci/occ_manual_replay_precheck.py",
    ],
)
def test_retirement_refuses_orphaned_occ_files(tmp_path: Path, path: str) -> None:
    _cutover_tree(tmp_path)
    leftover = tmp_path / path
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text("orphan\n")
    with pytest.raises(AssertionError, match="remains"):
        occ_callers_retired(tmp_path)


@pytest.mark.parametrize(
    "job",
    [
        {"uses": "OmniNode-ai/omniclaude/.github/workflows/occ-preflight.yml@dev"},
        {"needs": "occ-eligibility"},
        {"needs": ["tests", "occ-preflight"]},
        {"if": "needs.occ-preflight.result == 'success'"},
        {"steps": [{"run": "echo ${{ needs.occ-eligibility.result }}"}]},
        {"steps": [{"run": "python3 scripts/ci/occ_manual_replay_precheck.py"}]},
    ],
)
def test_retirement_refuses_occ_workflow_dependencies(
    tmp_path: Path, job: dict[str, Any]
) -> None:
    _cutover_tree(tmp_path)
    (tmp_path / ".github/workflows/ci.yml").write_text(
        yaml.safe_dump({"jobs": {"tests": job}})
    )
    with pytest.raises(AssertionError, match="OCC"):
        occ_callers_retired(tmp_path)


def test_retirement_refuses_occ_job(tmp_path: Path) -> None:
    _cutover_tree(tmp_path)
    (tmp_path / ".github/workflows/ci.yml").write_text(
        yaml.safe_dump({"jobs": {"occ-preflight": {}}})
    )
    with pytest.raises(AssertionError, match="OCC job"):
        occ_callers_retired(tmp_path)


def test_retirement_refuses_missing_ci(tmp_path: Path) -> None:
    _cutover_tree(tmp_path)
    (tmp_path / ".github/workflows/ci.yml").unlink()
    with pytest.raises(AssertionError, match="CI workflow is missing"):
        occ_callers_retired(tmp_path)
