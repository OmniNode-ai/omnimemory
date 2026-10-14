# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Require a complete repo-evidence cutover before accepting OCC retirement."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def occ_callers_retired(repo_root: Path) -> bool:
    """Return false for an active caller; assert the replacement when retired.

    A missing caller alone is a wiring fault. Its absence is accepted only when
    the other caller and replay script are gone and repo-owned admission enforces
    without shadowing or waiting for OCC.
    """
    workflows = repo_root / ".github" / "workflows"
    if (workflows / "call-occ-companion-effect.yml").is_file():
        return False

    assert not (workflows / "call-occ-autobind.yml").exists(), (
        "OCC autobind caller remains after companion-effect retirement"
    )
    assert not (repo_root / "scripts/ci/occ_manual_replay_precheck.py").exists(), (
        "OCC replay precheck remains without its caller"
    )
    caller_path = workflows / "call-repo-evidence-gate.yml"
    assert caller_path.is_file(), "OCC retirement requires repo-owned evidence"
    caller: dict[str, Any] = yaml.safe_load(caller_path.read_text(encoding="utf-8"))
    triggers = caller.get("on", caller.get(True))
    assert isinstance(triggers, dict), "repo-evidence trigger block is missing"
    assert "pull_request_target" in triggers, (
        "repo-evidence must run from the base branch"
    )
    job = caller["jobs"]["repo-evidence"]
    assert re.fullmatch(
        r"OmniNode-ai/omnibase_core/\.github/workflows/receipt-gate\.yml@[0-9a-f]{40}",
        job.get("uses", ""),
    ), "repo-evidence must call the receipt gate at an immutable pin"
    assert not job.get("if"), "repo-evidence admission must run unconditionally"
    inputs = caller["jobs"]["repo-evidence"]["with"]
    for key, expected in (
        ("evidence-source", "caller"),
        ("shadow", "false"),
        ("compare-with-occ", "false"),
    ):
        assert inputs.get(key) == expected, f"repo-evidence must set {key}={expected}"

    assert (workflows / "ci.yml").is_file(), "CI workflow is missing"
    for path in workflows.glob("*.y*ml"):
        workflow: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            assert not job_id.startswith("occ-"), (
                f"OCC job remains: {path.name}:{job_id}"
            )
            uses = str(job.get("uses", "")).split("@", 1)[0].rsplit("/", 1)[-1]
            assert not uses.startswith(("occ-", "call-occ-")), (
                f"OCC reusable remains: {path.name}:{job_id}"
            )
            needs = job.get("needs") or []
            if isinstance(needs, str):
                needs = [needs]
            assert not any(need.startswith("occ-") for need in needs), (
                f"OCC dependency remains: {path.name}:{job_id}"
            )
            assert "needs.occ-" not in str(job.get("if", "")), (
                f"OCC condition remains: {path.name}:{job_id}"
            )
            for step in job.get("steps") or []:
                run = str(step.get("run", ""))
                assert "needs.occ-" not in run, f"OCC verdict read remains: {path.name}"
                assert "occ_manual_replay_precheck.py" not in run, (
                    f"OCC replay reference remains: {path.name}"
                )
    return True
