# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20943: omnimemory's imperative contract guard calls omnibase_core's reusable.

The caller used to call ``onex_change_control``'s reusable guard, which read
omnimemory's allowlist from a central ``allowlists/omnimemory.yaml`` in that
repository. It now calls ``omnibase_core``'s reusable at a full sha and names
omnimemory's own allowlist, which sits in this repository and carries the four
entries the change-control guard used at 725d2967, unchanged. The reusable
refuses an allowlist that gains a path against the merge base, so the list only
shrinks.

Three criteria are bound here:

* AC1: the caller workflow, the unchanged job and context names, and no workflow
  that still reads ``onex_change_control`` for this guard.
* AC2: the allowlist equals the recorded 725d2967 entries path for path, and the
  core ratchet handler (and the guard CLI's own ratchet step) refuses an
  allowlist that gains a path.
* AC3: the core guard exits 0 over this tree with the allowlist, non-zero over a
  planted raw HTTP call in the live retrieval handler, and non-zero when the
  allowlist is absent. The comparison with the change-control script's verdicts
  is a lab run, recorded on the pull request.

The canonical-file-shape hook and job also run at the core sha that admits the
allowlist a caller declares, so the new file is not refused by name.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from omnibase_core.handlers.handler_imperative_allowlist_ratchet import (
    HandlerImperativeAllowlistRatchet,
    allowlisted_paths,
)
from omnibase_core.handlers.handler_imperative_contract_guard_cli import (
    main,
    run_ratchet,
)
from omnibase_core.models.nodes.imperative_contract_guard.model_allowlist_ratchet_input import (
    ModelAllowlistRatchetInput,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CALLER = WORKFLOWS / "imperative-contract-guard.yml"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
UV_LOCK = REPO_ROOT / "uv.lock"

ALLOWLIST_PATH = "imperative-contract-allowlist.yaml"
ALLOWLIST = REPO_ROOT / ALLOWLIST_PATH

WORKFLOW_NAME = "Imperative Contract Guard"
JOB_KEY = "imperative-contract-guard"
CORE_REUSABLE = re.compile(
    r"OmniNode-ai/omnibase_core/\.github/workflows/imperative-contract-guard\.yml"
    r"@(?P<sha>[0-9a-f]{40})"
)
OCC_GUARD = re.compile(
    r"OmniNode-ai/onex_change_control/\.github/workflows/imperative-contract-guard\.yml"
)
CORE_WITH_RATCHET_HANDLER = (0, 47, 41)

# The omnimemory entries of onex_change_control 725d2967 allowlists/omnimemory.yaml,
# recorded here so the comparison needs no other repository.
RECORDED_725D2967_ENTRIES: list[dict[str, object]] = [
    {
        "path": "src/omnimemory/handlers/adapters/adapter_valkey.py",
        "violations": ["direct_db_cache_connection"],
        "ticket": "OMN-12673",
        "note": (
            "Existing live freestanding runtime adapter debt; "
            "dependency sweep must not broaden it."
        ),
    },
    {
        "path": (
            "src/omnimemory/nodes/node_memory_retrieval_effect/adapters/"
            "adapter_embedding_client.py"
        ),
        "violations": ["raw_http_call"],
        "ticket": "OMN-12673",
        "note": (
            "Existing live freestanding runtime adapter debt; "
            "dependency sweep must not broaden it."
        ),
    },
    {
        "path": "src/omnimemory/runtime/plugin.py",
        "violations": ["hardcoded_private_ip"],
        "ticket": "OMN-12673",
        "note": (
            "Existing live freestanding runtime plugin debt; "
            "dependency sweep must not broaden it."
        ),
    },
    {
        "path": "src/omnimemory/topics.py",
        "violations": ["hardcoded_topic"],
        "ticket": "OMN-12673",
        "note": (
            "Existing live topic registry literals; "
            "dependency sweep must not broaden them."
        ),
    },
]
RECORDED_725D2967_PATHS = [str(entry["path"]) for entry in RECORDED_725D2967_ENTRIES]

PLANTED_PATH = "src/omnimemory/planted_extra_imperative_io.py"
_PLANTED_ENTRY = f"""\
  - path: {PLANTED_PATH}
    violations:
      - raw_http_call
    ticket: OMN-12673
    note: Planted extra path; the ratchet must refuse it.
"""


def _workflow(path: Path) -> dict[object, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _caller_job() -> dict[str, object]:
    jobs = _workflow(CALLER)["jobs"]
    assert isinstance(jobs, dict)
    job = jobs[JOB_KEY]
    assert isinstance(job, dict)
    return job


def _allowlist_entries(text: str) -> list[object]:
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    entries = data["allowlisted_handlers"]
    assert isinstance(entries, list)
    return entries


def _references_occ_guard(text: str) -> bool:
    return OCC_GUARD.search(text) is not None or "onex-change-control-ref" in text


def _ratchet(base_text: str | None, head_text: str | None) -> tuple[bool, list[str]]:
    report = HandlerImperativeAllowlistRatchet().handle(
        ModelAllowlistRatchetInput(base_text=base_text, head_text=head_text)
    )
    return report.admitted, report.added


# --- AC1: the caller workflow ----------------------------------------------------


def test_caller_uses_the_core_reusable_at_a_full_sha() -> None:
    uses = _caller_job()["uses"]
    assert isinstance(uses, str)
    assert CORE_REUSABLE.fullmatch(uses), uses


def test_caller_passes_omnimemorys_own_allowlist_path() -> None:
    assert _caller_job()["with"] == {"allowlist-path": ALLOWLIST_PATH}
    assert ALLOWLIST.is_file()


def test_caller_keeps_the_workflow_job_and_context_names() -> None:
    workflow = _workflow(CALLER)
    assert workflow["name"] == WORKFLOW_NAME
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {JOB_KEY}
    assert "name" not in _caller_job(), "the reusable's job name is the context name"
    assert workflow["permissions"] == {"contents": "read"}
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert set(triggers) == {"pull_request", "push"}


def test_no_workflow_reads_onex_change_control_for_this_guard() -> None:
    workflows = sorted(WORKFLOWS.glob("*.y*ml"))
    assert CALLER in workflows
    offenders = [
        path.name
        for path in workflows
        if _references_occ_guard(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_occ_reference_check_has_a_positive_control() -> None:
    caller = CALLER.read_text(encoding="utf-8")
    planted_uses = caller.replace(
        "OmniNode-ai/omnibase_core/", "OmniNode-ai/onex_change_control/", 1
    )
    assert _references_occ_guard(planted_uses)
    assert _references_occ_guard("with:\n  onex-change-control-ref: 725d2967\n")


# --- AC2: the allowlist and the shrink-only ratchet ------------------------------


def test_allowlist_entries_equal_the_recorded_725d2967_entries() -> None:
    entries = _allowlist_entries(ALLOWLIST.read_text(encoding="utf-8"))
    assert entries == RECORDED_725D2967_ENTRIES


def test_allowlist_paths_equal_the_recorded_725d2967_paths_in_order() -> None:
    paths = allowlisted_paths(ALLOWLIST.read_text(encoding="utf-8"))
    assert paths == RECORDED_725D2967_PATHS


def test_ratchet_refuses_an_allowlist_that_gains_a_planted_path() -> None:
    base = ALLOWLIST.read_text(encoding="utf-8")
    admitted, added = _ratchet(base, base + _PLANTED_ENTRY)
    assert not admitted
    assert added == [PLANTED_PATH]


def test_ratchet_admits_the_unchanged_allowlist_and_a_shrunk_one() -> None:
    base = ALLOWLIST.read_text(encoding="utf-8")
    assert _ratchet(base, base) == (True, [])
    shrunk = yaml.safe_dump(
        {"allowlisted_handlers": RECORDED_725D2967_ENTRIES[:-1]}, sort_keys=False
    )
    assert _ratchet(base, shrunk) == (True, [])


def test_ratchet_admits_the_first_landing_of_the_allowlist() -> None:
    assert _ratchet(None, ALLOWLIST.read_text(encoding="utf-8")) == (True, [])


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=omn20943",
            "-c",
            "user.email=omn20943@example.invalid",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def allowlist_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repository whose HEAD commit carries omnimemory's allowlist."""
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    _git(tmp_path, "init", "-q", "-b", "dev")
    (tmp_path / ALLOWLIST_PATH).write_text(
        ALLOWLIST.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _git(tmp_path, "add", ALLOWLIST_PATH)
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_guard_ratchet_step_refuses_a_planted_path_against_the_merge_base(
    allowlist_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    allowlist = allowlist_repo / ALLOWLIST_PATH
    allowlist.write_text(
        allowlist.read_text(encoding="utf-8") + _PLANTED_ENTRY, encoding="utf-8"
    )
    assert run_ratchet(allowlist_repo, allowlist, "HEAD") == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert f"+ {PLANTED_PATH}" in out


def test_guard_ratchet_step_admits_the_unchanged_allowlist(
    allowlist_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_ratchet(allowlist_repo, allowlist_repo / ALLOWLIST_PATH, "HEAD") == 0
    assert "REFUSED" not in capsys.readouterr().out


# --- AC3: the core guard still refuses what it refused on omnimemory's tree -------

LIVE_HANDLER = (
    "src/omnimemory/nodes/node_memory_retrieval_effect/handlers/"
    "handler_memory_retrieval.py"
)
_PLANTED_HTTP_CALL = """

def _planted_raw_http_call() -> None:
    import httpx

    httpx.get("https://planted.example.invalid/planted")
"""


def _guard_exit(repo_root: Path, allowlist: str) -> int:
    return main(
        [
            "--repo-root",
            str(repo_root),
            "--allowlist-path",
            allowlist,
            "--scan-freestanding",
        ]
    )


@pytest.fixture
def tree_copy(tmp_path: Path) -> Path:
    """A copy of this repository's guard inputs: ``src``, ``pyproject.toml``, the allowlist."""
    shutil.copytree(
        REPO_ROOT / "src",
        tmp_path / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy(ALLOWLIST, tmp_path / ALLOWLIST_PATH)
    return tmp_path


def test_core_guard_exits_zero_over_the_clean_tree(
    tree_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _guard_exit(tree_copy, ALLOWLIST_PATH) == 0
    assert "Blocking LIVE" not in capsys.readouterr().out


def test_core_guard_exits_nonzero_over_a_planted_raw_http_call(
    tree_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handler = tree_copy / LIVE_HANDLER
    handler.write_text(
        handler.read_text(encoding="utf-8") + _PLANTED_HTTP_CALL, encoding="utf-8"
    )
    assert _guard_exit(tree_copy, ALLOWLIST_PATH) == 1
    out = capsys.readouterr().out
    assert "Blocking LIVE" in out
    assert "handler_memory_retrieval.py" in out


def test_core_guard_depends_on_the_allowlist_it_is_given(
    tree_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tree_copy / ALLOWLIST_PATH).unlink()
    assert _guard_exit(tree_copy, ALLOWLIST_PATH) == 1
    out = capsys.readouterr().out
    # runtime/plugin.py no longer violates (OMN-20936 removed its private-network
    # default), so its entry is stale and the ratchet lets it shrink away later.
    for path in RECORDED_725D2967_PATHS:
        if path != "src/omnimemory/runtime/plugin.py":
            assert path in out


# --- the canonical-file-shape rev that admits the allowlist ------------------------


def _canonical_file_shape_hook_rev() -> str:
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    (rev,) = [
        repo["rev"]
        for repo in config["repos"]
        if "canonical-file-shape" in [hook["id"] for hook in repo["hooks"]]
    ]
    assert isinstance(rev, str)
    return rev


def _canonical_file_shape_job_rev() -> str:
    job = _workflow(CI_WORKFLOW)["jobs"]
    assert isinstance(job, dict)
    env = job["canonical-file-shape"]["env"]["CANONICAL_FILE_SHAPE_CORE"]
    assert isinstance(env, str)
    return env.rsplit("@", 1)[1]


def test_canonical_file_shape_hook_and_job_run_at_the_core_sha_the_caller_pins() -> (
    None
):
    uses = _caller_job()["uses"]
    assert isinstance(uses, str)
    match = CORE_REUSABLE.fullmatch(uses)
    assert match, uses
    assert _canonical_file_shape_hook_rev() == match["sha"]
    assert _canonical_file_shape_job_rev() == match["sha"]


def test_locked_core_carries_the_ratchet_handler() -> None:
    block = UV_LOCK.read_text(encoding="utf-8").split(
        'name = "omnibase-core"\nversion = "', 1
    )[1]
    version = tuple(int(part) for part in block.split('"', 1)[0].split("."))
    assert version >= CORE_WITH_RATCHET_HANDLER, version
