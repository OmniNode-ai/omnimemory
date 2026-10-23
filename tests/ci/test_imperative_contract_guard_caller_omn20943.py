# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20943: the caller owns the unchanged allowlist and uses core's guard.

The four recorded paths come from the guard's omnimemory allowlist at
725d2967b031d7065efd41d0751f903638202cc1. The project locks core 0.47.39,
which predates the ratchet handler, so behavioural probes use an isolated
0.47.41 validator environment without changing the application's lock.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
CALLER = WORKFLOWS / "imperative-contract-guard.yml"
ALLOWLIST = ROOT / "imperative-contract-allowlist.yaml"
CORE_SHA_PREFIX = "031148215"
CORE_USES = "OmniNode-ai/omnibase_core/.github/workflows/imperative-contract-guard.yml"
RECORDED_PATHS = [
    "src/omnimemory/handlers/adapters/adapter_valkey.py",
    "src/omnimemory/nodes/node_memory_retrieval_effect/adapters/adapter_embedding_client.py",
    "src/omnimemory/runtime/plugin.py",
    "src/omnimemory/topics.py",
]
LIVE_HANDLER = "src/omnimemory/nodes/node_memory_retrieval_effect/handlers/handler_memory_retrieval.py"


def _mapping(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _core(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--no-project", "--with", "omnibase-core==0.47.41", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        cwd=ROOT,
    )


def _caller_sha() -> str:
    jobs = _mapping(CALLER)["jobs"]
    assert isinstance(jobs, dict)
    uses = jobs["imperative-contract-guard"]["uses"]
    assert isinstance(uses, str)
    repository, sha = uses.split("@")
    assert repository == CORE_USES
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert sha.startswith(CORE_SHA_PREFIX)
    return sha


def test_caller_uses_core_and_keeps_the_job_and_context() -> None:
    workflow = _mapping(CALLER)
    assert workflow["name"] == "Imperative Contract Guard"
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"imperative-contract-guard"}
    job = jobs["imperative-contract-guard"]
    assert job["uses"] == f"{CORE_USES}@{_caller_sha()}"
    assert job["with"] == {"allowlist-path": ALLOWLIST.name}
    assert workflow["permissions"] == {"contents": "read"}
    for path in WORKFLOWS.iterdir():
        if path.suffix in {".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8")
            assert not re.search(
                r"onex_change_control/\.github/workflows/imperative-contract-guard\.yml",
                text,
            ), path


def test_canonical_shape_hook_uses_the_exemption_commit() -> None:
    repos = _mapping(ROOT / ".pre-commit-config.yaml")["repos"]
    assert isinstance(repos, list)
    (hook_repo,) = [
        repo
        for repo in repos
        if any(hook["id"] == "canonical-file-shape" for hook in repo["hooks"])
    ]
    assert hook_repo["repo"] == "https://github.com/OmniNode-ai/omnibase_core"
    assert hook_repo["rev"] == _caller_sha()


def test_own_allowlist_matches_the_recorded_paths() -> None:
    entries = _mapping(ALLOWLIST)["allowlisted_handlers"]
    assert isinstance(entries, list)
    assert [entry["path"] for entry in entries] == RECORDED_PATHS


def test_core_ratchet_accepts_unchanged_and_shrinking_but_refuses_an_extra_path() -> (
    None
):
    text = ALLOWLIST.read_text(encoding="utf-8")
    entries = _mapping(ALLOWLIST)["allowlisted_handlers"]
    assert isinstance(entries, list)
    extra = "src/omnimemory/nodes/node_planted_effect/handlers/handler_planted.py"
    request = {
        "base": text,
        "head": yaml.safe_dump({"allowlisted_handlers": [*entries, {"path": extra}]}),
        "shrunk": yaml.safe_dump({"allowlisted_handlers": entries[:-1]}),
    }
    result = _core(
        "python",
        "-c",
        "import json, sys; "
        "from omnibase_core.handlers.handler_imperative_allowlist_ratchet import "
        "HandlerImperativeAllowlistRatchet; "
        "from omnibase_core.models.nodes.imperative_contract_guard."
        "model_allowlist_ratchet_input import ModelAllowlistRatchetInput; "
        "p=json.load(sys.stdin); h=HandlerImperativeAllowlistRatchet(); "
        "reports=[h.handle(ModelAllowlistRatchetInput(base_text=p['base'], "
        "head_text=p[k])) for k in ('base', 'shrunk', 'head')]; "
        "print(json.dumps([dict(r.model_dump(), admitted=r.admitted) for r in reports]))",
        stdin=json.dumps(request),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    unchanged, shrunk, planted = json.loads(result.stdout)
    assert unchanged == {
        "added": [],
        "removed": [],
        "bootstrap": False,
        "admitted": True,
    }
    assert shrunk["admitted"] is True
    assert shrunk["removed"] == [RECORDED_PATHS[-1]]
    assert planted["admitted"] is False
    assert planted["bootstrap"] is False
    assert planted["added"] == [extra]


def test_core_guard_accepts_clean_tree_and_refuses_a_planted_live_handler(
    tmp_path: Path,
) -> None:
    # Use the caller's input; the merge-base caller cannot provide it.
    jobs = _mapping(CALLER)["jobs"]
    assert isinstance(jobs, dict)
    assert jobs["imperative-contract-guard"]["with"]["allowlist-path"] == ALLOWLIST.name
    root = tmp_path / "omnimemory"
    shutil.copytree(
        ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copyfile(ALLOWLIST, root / ALLOWLIST.name)
    args = (
        "check-imperative-contracts",
        "--repo-root",
        str(root),
        "--allowlist-path",
        ALLOWLIST.name,
        "--scan-freestanding",
        "--json",
    )
    clean = _core(*args)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    handler = root / LIVE_HANDLER
    original = handler.read_text(encoding="utf-8")
    # The scanner reads this source; it does not execute a network request.
    handler.write_text(
        original + '\nimport httpx\nhttpx.get("http://example.invalid/planted")\n',
        encoding="utf-8",
    )
    planted = _core(*args)
    assert planted.returncode == 1, planted.stdout + planted.stderr
    report = json.loads(planted.stdout)[0]
    assert report["new_violation_count"] > 0
    (violation,) = [
        violation
        for violation in report["new_violations"]
        if violation["handler_path"].endswith("handlers/handler_memory_retrieval.py")
    ]
    assert violation["reachability"] == "live"
    assert "HTTP" in violation["undeclared_transports"]
    assert "undeclared_transport" in violation["violations"]
