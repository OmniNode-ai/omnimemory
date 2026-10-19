# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20997: preserve the title context and the old bash verdicts with a release node.

The bash oracle below is the decision portion of the old reusable pinned at
725d2967b031d7065efd41d0751f903638202cc1. The node runs from a PyPI release in
an isolated venv, as the workflow does; no sibling clone or checkout is needed.

OMN-17427: the job body now lives in omnibase_core's reusable
``pr-title-check.yml``, which every repository of the same type calls. This
repository's caller passes only the omnimarket version; the caller job id
``pr-title`` and the reusable's job ``check-title`` compose the same context.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/pr-title-check.yml"
MANIFEST = ROOT / ".github/required-checks.yaml"
CONTEXT = "pr-title / check-title"
NODE_MODULE = "omnimarket.nodes.node_pr_title_check_compute"
OMNIMARKET_VERSION = "0.4.310"
CORE_REUSABLE = re.compile(
    r"OmniNode-ai/omnibase_core/\.github/workflows/pr-title-check\.yml@[0-9a-f]{40}"
)
RUNS_ON = """${{
  (github.event_name == 'pull_request' &&
   github.event.pull_request.head.repo.full_name != github.repository)
  && fromJSON(vars.OMNI_PUBLIC_PR_RUNS_ON_JSON || '["ubuntu-latest"]')
  || fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || '["self-hosted","omnibase-ci"]')
}}"""
OLD_BASH = r"""
if [[ -z "$PR_TITLE" ]]; then
  exit 1
fi
TITLE_LOWER=$(echo "$PR_TITLE" | tr '[:upper:]' '[:lower:]')
if [[ "$PR_AUTHOR" == *"[bot]" ]]; then
  exit 0
fi
if [[ "$TITLE_LOWER" =~ ^(chore\(deps|build\(deps|bump ) ]]; then
  exit 0
fi
if [[ "$TITLE_LOWER" =~ ^(chore:\ release|chore\(release\)|release:) ]]; then
  exit 0
fi
if echo "$PR_TITLE" | grep -qE 'OMN-[0-9]+'; then
  exit 0
fi
exit 1
"""


def _workflow() -> dict[Any, Any]:
    loaded: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return loaded


def test_workflow_calls_the_core_reusable_with_only_its_parameters() -> None:
    workflow = _workflow()
    assert list(workflow["jobs"]) == ["pr-title"]
    job = workflow["jobs"]["pr-title"]
    assert "onex_change_control" not in WORKFLOW.read_text(encoding="utf-8")
    assert CORE_REUSABLE.fullmatch(job["uses"])
    assert job["with"] == {"omnimarket-version": OMNIMARKET_VERSION}
    assert job["permissions"] == {"contents": "read"}
    assert not {"steps", "runs-on", "if", "needs", "name"} & job.keys()
    assert workflow[True] == {
        "pull_request": {"types": ["opened", "edited", "synchronize", "reopened"]},
        "merge_group": {"types": ["checks_requested"]},
    }


def test_call_composes_the_unchanged_required_context() -> None:
    # A reusable call's context is "<caller job name or id> / <callee job name or id>";
    # the core reusable's one job is check-title.
    (caller_id,) = _workflow()["jobs"]
    assert f"{caller_id} / check-title" == CONTEXT


def test_manifest_names_the_core_reusable_producer() -> None:
    rows = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["required_checks"]
    matching = [row for row in rows if row["name"] == CONTEXT]
    assert len(matching) == 1
    row = matching[0]
    assert row["producer_kind"] == "cross_repo"
    assert row["workflow"] == "pr-title-check.yml"
    assert row["job_id"] == "pr-title"
    assert row["conditional"] is False
    assert row["cross_repo_ref"] == _workflow()["jobs"]["pr-title"]["uses"]
    assert "onex_change_control" not in row["rationale"]


@pytest.fixture(scope="session")
def node_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install the same pinned wheel and minimal dependency as the workflow."""
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to exercise the workflow's release node"
    venv = tmp_path_factory.mktemp("pr-title-node") / "venv"
    python = venv / "bin/python"
    for argv in (
        [uv, "venv", "--quiet", "--python", "3.12", str(venv)],
        [
            uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            "--no-deps",
            f"omnimarket=={OMNIMARKET_VERSION}",
        ],
        [uv, "pip", "install", "--quiet", "--python", str(python), "pydantic>=2.11"],
    ):
        result = subprocess.run(
            argv, text=True, capture_output=True, check=False, timeout=90
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return python


@pytest.mark.parametrize(
    ("title", "author", "expected"),
    [
        ("Update dependencies", "dependabot[bot]", 0),
        ("Bare title", "manual[bot]", 0),
        ("chore(deps): update libraries", "human", 0),
        ("CHORE(DEPS): update libraries", "human", 0),
        ("chore(deps-extra): update libraries", "human", 0),
        ("build(deps): update libraries", "human", 0),
        ("BUILD(DEPS): update libraries", "human", 0),
        ("bump package", "human", 0),
        ("BUMP package", "human", 0),
        ("chore: release 1.2.3", "human", 0),
        ("CHORE: RELEASE 1.2.3", "human", 0),
        ("chore(release): 1.2.3", "human", 0),
        ("CHORE(RELEASE): 1.2.3", "human", 0),
        ("release: 1.2.3", "human", 0),
        ("RELEASE: 1.2.3", "human", 0),
        ("ci(OMN-20997): port title check", "human", 0),
        ("ticket OMN-1suffix", "human", 0),
        ("two references OMN-1 OMN-2", "human", 0),
        ("Bare title", "human", 1),
        ("lowercase omn-20997", "human", 1),
        ("missing digits OMN-", "human", 1),
        ("prefix chore(deps): update", "human", 1),
        ("chore(release-extra): update", "human", 1),
        ("bump", "human", 1),
        (" bare title", "human[bot]suffix", 1),
        ("Bare title", "human[BOT]", 1),
        ("", "dependabot[bot]", 1),
        ("", "human", 1),
        ('quotes " and $(false) OMN-20997', "human", 0),
    ],
)
def test_verdicts_equal_the_old_reusable(
    node_python: Path, title: str, author: str, expected: int
) -> None:
    old = subprocess.run(
        ["/bin/bash", "-c", OLD_BASH],
        env={**os.environ, "PR_TITLE": title, "PR_AUTHOR": author},
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    node = subprocess.run(
        [str(node_python), "-I", "-m", NODE_MODULE],
        input=json.dumps({"title": title, "author": author}),
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert old.returncode == expected, old.stdout + old.stderr
    assert node.returncode == old.returncode, node.stdout + node.stderr


def test_malformed_request_refuses_with_exit_two(node_python: Path) -> None:
    result = subprocess.run(
        [str(node_python), "-I", "-m", NODE_MODULE],
        input="{",
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2, result.stdout + result.stderr
