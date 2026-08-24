# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Workflow-shape guard: pytest timeout method must be ``signal`` (OMN-16348).

OMN-15977 (omnibase_core) banned pytest-timeout's ``thread`` method: its
watcher thread fires only when the GIL is released, so a CPU-bound pure-Python
runaway holds the GIL continuously and the declared ``--timeout`` ceiling
silently never fires (the config behind the 2026-08-12 46/53-minute pre-push
runaways in omnibase_core that needed manual SIGKILL). The only remaining
backstop is the job's ``timeout-minutes``, which cancels the whole shard with
no attributable test. ``signal`` delivers SIGALRM at the next bytecode
boundary regardless of GIL contention.

The original guards were per-file, which is exactly why additional surfaces
stayed invisible — so this assertion is per-invocation-surface: it scans every
run step of every workflow file, and none may pass ``--timeout-method=`` with
any value other than ``signal``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

_TIMEOUT_METHOD_RE = re.compile(r"--timeout-method=(\S+)")


def _has_pytest_token(run: str) -> bool:
    """True when a non-comment line of the run script invokes pytest."""
    return any(
        token == "pytest" or token.endswith("/pytest")
        for line in run.splitlines()
        if not line.lstrip().startswith("#")
        for token in line.split()
    )


def _all_workflow_run_commands() -> list[tuple[str, str, str]]:
    """Every ``(workflow file, job, run script)`` triple in .github/workflows."""
    commands: list[tuple[str, str, str]] = []
    for path in sorted(_WORKFLOWS_DIR.glob("*.yml")) + sorted(
        _WORKFLOWS_DIR.glob("*.yaml")
    ):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if isinstance(step, dict) and "run" in step:
                    commands.append((path.name, str(job_name), str(step["run"])))
    return commands


def test_no_workflow_pytest_invocation_uses_thread_timeout_method() -> None:
    """OMN-16348: every ``--timeout-method`` in any workflow must be ``signal``."""
    commands = _all_workflow_run_commands()

    # Positive control: the scanner must actually be seeing ci.yml's pytest
    # steps — an empty scan would vacuously pass while enforcing nothing.
    assert any(
        source == "ci.yml" and _has_pytest_token(run) for source, _, run in commands
    )

    violations = [
        f"{source}::{job}: {line.strip()}"
        for source, job, run in commands
        for line in run.splitlines()
        for method in _TIMEOUT_METHOD_RE.findall(line)
        if method != "signal"
    ]
    assert violations == [], (
        "workflow passes a non-signal --timeout-method (banned by OMN-15977; "
        "a CLI flag overrides any addopts signal default): "
        f"{violations}"
    )
