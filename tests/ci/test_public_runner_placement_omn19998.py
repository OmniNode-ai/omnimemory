# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Public-repository runner placement in omnimemory (OMN-19998).

RULING 2026-10-16T07:07:09Z: "Ci runners should be for private repos only."
omnimemory is public, so:

* no routing-seam fallback in a workflow names the self-hosted fleet;
* omnibase_core's check-public-runner-placement runs as a CI job that CI
  Summary requires and as a pre-commit hook, at the same core release.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"

FLEET_FALLBACK = """'["self-hosted","omnibase-ci"]'"""
JOB_ID = "public-runner-placement"
JOB_NAME = "Public Runner Placement (OMN-19998)"
HOOK_ID = "check-public-runner-placement"
CORE_RELEASE = "0.47.46"


def _load(path: Path) -> dict[object, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def fleet_fallbacks(text: str) -> list[str]:
    """Code lines that fall back to the self-hosted fleet."""
    return [
        line.strip()
        for line in text.splitlines()
        if FLEET_FALLBACK in line and not line.lstrip().startswith("#")
    ]


def test_fleet_fallback_detector_positive_control() -> None:
    planted = (
        "    runs-on: ${{ fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || "
        """'["self-hosted","omnibase-ci"]') }}\n"""
    )
    hosted = (
        "    runs-on: ${{ fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || "
        """'["ubuntu-latest"]') }}\n"""
    )
    commented = """    # falls back to '["self-hosted","omnibase-ci"]' once\n"""
    assert fleet_fallbacks(planted) == [planted.strip()]
    assert fleet_fallbacks(hosted) == []
    assert fleet_fallbacks(commented) == []


def test_no_workflow_falls_back_to_the_self_hosted_fleet() -> None:
    offenders = {
        path.name: found
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
        if (found := fleet_fallbacks(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, (
        "A public repository's routing fallback must be GitHub-hosted "
        "(RULING 2026-10-16T07:07:09Z): " + repr(offenders)
    )


def test_every_job_runs_on_a_literal_github_hosted_label() -> None:
    """The core check refuses a computed runs-on in a repository it has no baseline for."""
    offenders: dict[str, str] = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        jobs = _load(path).get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if not isinstance(job, dict) or "runs-on" not in job:
                continue
            label = job["runs-on"]
            if not (isinstance(label, str) and label.startswith("ubuntu-")):
                offenders[f"{path.name}:{job_id}"] = repr(label)
    assert offenders == {}, repr(offenders)


def _ci_job() -> dict[str, object]:
    jobs = _load(CI_WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)
    job = jobs[JOB_ID]
    assert isinstance(job, dict)
    return job


def _hooks() -> list[tuple[str, dict[str, object]]]:
    repos = _load(PRECOMMIT)["repos"]
    assert isinstance(repos, list)
    return [
        (str(repo["rev"]), hook)
        for repo in repos
        if isinstance(repo, dict)
        for hook in repo.get("hooks", [])
        if isinstance(hook, dict) and hook.get("id") == HOOK_ID
    ]


def test_ci_job_runs_the_core_check_unconditionally_on_a_hosted_runner() -> None:
    job = _ci_job()
    assert job["name"] == JOB_NAME
    assert job["runs-on"] == "ubuntu-latest"
    assert "needs" not in job
    assert "if" not in job
    steps = job["steps"]
    assert isinstance(steps, list)
    runs = "\n".join(str(step.get("run", "")) for step in steps)
    assert "omnibase_core.handlers.handler_public_runner_placement_cli" in runs
    assert "POSITIVE CONTROL FAILED" in runs
    assert "--base-ref" in runs
    assert "--repo omnimemory --visibility public" in runs


def test_ci_job_and_hook_run_the_same_core_release() -> None:
    """The CI job installs 0.47.46 and the hook's rev is that release's tag."""
    env = _ci_job()["env"]
    assert isinstance(env, dict)
    assert env["RUNNER_PLACEMENT_CORE"] == f"omnibase-core=={CORE_RELEASE}"
    hooks = _hooks()
    assert len(hooks) == 1, f"expected one {HOOK_ID} hook, found {len(hooks)}"
    assert hooks[0][0] == f"v{CORE_RELEASE}"


def test_ci_summary_requires_the_job_to_succeed() -> None:
    jobs = _load(CI_WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)
    summary = jobs["ci-summary"]
    assert isinstance(summary, dict)
    assert JOB_ID in summary["needs"]
    steps = summary["steps"]
    assert isinstance(steps, list)
    script = "\n".join(str(step.get("run", "")) for step in steps)
    assert f'needs.{JOB_ID}.result }}}}" != "success"' in script


def test_precommit_hook_names_this_public_repository() -> None:
    hooks = _hooks()
    assert hooks
    assert hooks[0][1]["args"] == ["--repo", "omnimemory", "--visibility", "public"]
