# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the required CI Summary's eligibility barrier (OMN-18854)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github/workflows/ci.yml"


@pytest.mark.unit
def test_eligibility_uses_the_shared_validator_at_an_immutable_revision() -> None:
    jobs = yaml.safe_load(_WORKFLOW.read_text())["jobs"]
    assert "occ-preflight" in jobs, "CI must call the existing eligibility validator"
    gate = jobs["occ-preflight"]
    revision = re.fullmatch(
        r"OmniNode-ai/omnibase_core/\.github/workflows/occ-preflight\.yml@([0-9a-f]{40})",
        gate["uses"],
    )
    assert revision is not None
    assert gate["with"]["core-ref"] == revision[1]
    assert gate["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert gate["if"] == (
        "github.event_name == 'pull_request' || github.event_name == 'merge_group'"
    )
    assert "needs" not in gate
    assert "occ-eligibility" in jobs["ci-summary"]["needs"]
    assert jobs["ci-summary"]["if"] == "always()"
    assert "needs.occ-preflight." not in str(jobs["ci-summary"])
    assert jobs["occ-eligibility"]["needs"] == "occ-preflight"
    assert jobs["occ-eligibility"]["if"] == "always()"
    triggers = yaml.safe_load(_WORKFLOW.read_text())[True]
    assert "dev" in triggers["pull_request"]["branches"]
    assert set(triggers["pull_request"]["types"]) >= {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    }
    assert "paths" not in triggers["pull_request"]
    assert "paths-ignore" not in triggers["pull_request"]
    assert "merge_group" in triggers
    # Shape detectors must still run when evidence is pending or red.
    for name in ("shape-gate-independence", "canonical-file-shape"):
        assert "needs" not in jobs[name]


def _run_verdict(
    job_id: str, event: str, results: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"][job_id]
    script = next(step["run"] for step in job["steps"] if "run" in step)

    # Execute the actual workflow shell with all unrelated predecessors passing.
    def render(match: re.Match[str]) -> str:
        expression = match[1].strip()
        if expression == "github.event_name":
            return event
        assert re.fullmatch(r"needs\.[a-z-]+\.result", expression), expression
        return results.get(expression, "success")

    rendered = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", render, script)
    return subprocess.run(
        ["bash", "-c", rendered], capture_output=True, text=True, check=False
    )


@pytest.mark.unit
@pytest.mark.parametrize("event", ["pull_request", "merge_group"])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", ""])
def test_summary_requires_success_for_eligibility(event: str, result: str) -> None:
    admission = _run_verdict(
        "occ-eligibility", event, {"needs.occ-preflight.result": result}
    )
    expected_code = 0 if result == "success" else 1
    assert admission.returncode == expected_code, admission.stdout + admission.stderr
    completed = _run_verdict(
        "ci-summary",
        event,
        {
            "needs.occ-eligibility.result": "success"
            if admission.returncode == 0
            else "failure"
        },
    )
    assert completed.returncode == (0 if result == "success" else 1), (
        completed.stdout + completed.stderr
    )
    if result != "success":
        assert "occ-preflight" in admission.stdout
        assert "occ-eligibility" in completed.stdout


@pytest.mark.unit
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", ""])
def test_summary_requires_success_for_the_admission_job(result: str) -> None:
    completed = _run_verdict(
        "ci-summary", "pull_request", {"needs.occ-eligibility.result": result}
    )
    assert completed.returncode == (0 if result == "success" else 1), (
        completed.stdout + completed.stderr
    )


@pytest.mark.unit
@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_summary_preserves_non_pr_events(event: str) -> None:
    admission = _run_verdict(
        "occ-eligibility", event, {"needs.occ-preflight.result": "skipped"}
    )
    assert admission.returncode == 0, admission.stdout + admission.stderr
    completed = _run_verdict(
        "ci-summary", event, {"needs.occ-eligibility.result": "success"}
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.unit
def test_eligibility_enforcement_is_documented_and_checked_before_commit() -> None:
    manifest = yaml.safe_load((_ROOT / ".github/required-checks.yaml").read_text())
    gate = next(
        (
            check
            for check in manifest["required_checks"]
            if check["job_id"] == "occ-preflight"
        ),
        None,
    )
    assert gate is not None, "the required umbrella must document eligibility"
    assert gate["workflow"] == "ci.yml"
    assert "CI Summary" in gate["rationale"]
    config = yaml.safe_load((_ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        (
            hook
            for repo in config["repos"]
            for hook in repo["hooks"]
            if hook["id"] == "occ-eligibility-enforcement"
        ),
        None,
    )
    assert hook is not None, "the regression must run before commit"
    assert "tests/scripts/test_occ_eligibility_enforcement.py" in hook["entry"]
    assert hook["pass_filenames"] is False
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["onex-validation"]["steps"]
    assert any(
        "pytest tests/scripts/test_occ_eligibility_enforcement.py"
        in step.get("run", "")
        for step in steps
    ), "the pre-commit regression also runs as a CI validation"
    pattern = re.compile(hook["files"])
    for path in (
        ".github/workflows/ci.yml",
        ".github/required-checks.yaml",
        ".pre-commit-config.yaml",
        "tests/scripts/test_occ_eligibility_enforcement.py",
    ):
        assert pattern.search(path), path
