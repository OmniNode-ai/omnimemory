# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The core dependency-pin check is a CI Summary member here (OMN-22360)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CORE = "OmniNode-ai/omnibase_core"
WORKFLOW = f"{CORE}/.github/workflows/dependency-pins.yml@"


def _jobs() -> dict[str, dict[str, object]]:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = ci["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _core_hook_revs() -> set[str]:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    return {
        str(entry["rev"])
        for entry in config["repos"]
        if str(entry.get("repo", "")).rstrip("/").endswith(CORE)
    }


def test_dependency_pins_job_calls_the_core_workflow() -> None:
    assert str(_jobs()["dependency-pins"]["uses"]).startswith(WORKFLOW)


def test_dependency_pins_job_uses_the_core_hook_tag() -> None:
    tag = str(_jobs()["dependency-pins"]["uses"]).removeprefix(WORKFLOW)
    assert _core_hook_revs() == {tag}


def test_ci_summary_needs_dependency_pins() -> None:
    needs = _jobs()["ci-summary"]["needs"]
    assert isinstance(needs, list)
    assert "dependency-pins" in needs
