# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Sibling-pin guard: CI reads sibling repos only at a pinned rev (OMN-20001).

Ruling 2026-10-01: each repo checks only itself, and any read of a sibling repo
uses the rev this repo has pinned, never the sibling's live branch, so a merge
in one repo cannot turn another repo red. This scans every workflow for:

* ``uses: OmniNode-ai/<sibling>/...@<ref>`` (reusable workflows, composite
  actions) whose ref is not a 40-hex commit sha, and
* ``actions/checkout`` steps with ``repository: OmniNode-ai/<sibling>`` whose
  ``ref`` is missing (default branch tip), a branch name, or anything other
  than a 40-hex sha or the output of a ``*-pin`` resolve step that reads the
  rev this repo records (pyproject.toml / check-handshake.yml).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_SELF = "omnimemory"
_SHA = re.compile(r"[0-9a-f]{40}")
_PIN_OUTPUT = re.compile(r"\$\{\{ steps\.[a-z-]+-pin\.outputs\.rev \}\}")
_USES = re.compile(r"OmniNode-ai/([\w.-]+)/\S+@(\S+)")


def _violations(workflow: dict[str, object], name: str) -> list[str]:
    found: list[str] = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return found
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        entries: list[dict[str, object]] = [job]
        steps = job.get("steps")
        if isinstance(steps, list):
            entries.extend(s for s in steps if isinstance(s, dict))
        for entry in entries:
            uses = entry.get("uses")
            if not isinstance(uses, str):
                continue
            match = _USES.match(uses)
            if match and match.group(1) != _SELF and not _SHA.fullmatch(match.group(2)):
                found.append(f"{name}:{job_name}: uses {uses}")
            with_ = entry.get("with")
            if (
                uses.startswith("actions/checkout@")
                and isinstance(with_, dict)
                and isinstance(with_.get("repository"), str)
                and with_["repository"].startswith("OmniNode-ai/")
                and with_["repository"] != f"OmniNode-ai/{_SELF}"
            ):
                ref = str(with_.get("ref", ""))
                if not (_SHA.fullmatch(ref) or _PIN_OUTPUT.fullmatch(ref)):
                    found.append(
                        f"{name}:{job_name}: checkout {with_['repository']} ref={ref!r}"
                    )
    return found


def _all_violations() -> list[str]:
    found: list[str] = []
    for path in sorted(_WORKFLOWS_DIR.glob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            found.extend(_violations(data, path.name))
    return found


def test_no_workflow_reads_a_sibling_at_a_live_ref() -> None:
    assert _all_violations() == []


@pytest.mark.parametrize(
    "snippet",
    [
        "jobs:\n  j:\n    uses: OmniNode-ai/omniclaude/.github/workflows/x.yml@main\n",
        "jobs:\n  j:\n    steps:\n      - uses: OmniNode-ai/onex_change_control/.github/actions/a@dev\n",
        "jobs:\n  j:\n    steps:\n      - uses: actions/checkout@v7\n        with:\n          repository: OmniNode-ai/omnibase_core\n",
        "jobs:\n  j:\n    steps:\n      - uses: actions/checkout@v7\n        with:\n          repository: OmniNode-ai/omnimarket\n          ref: dev\n",
    ],
)
def test_guard_flags_live_sibling_reads(snippet: str) -> None:
    """Positive control: the scan reports each mutable-ref shape it exists to catch."""
    assert _violations(yaml.safe_load(snippet), "synthetic.yml") != []
