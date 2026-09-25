# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
REQUIRED_CHECKS = REPO_ROOT / ".github" / "required-checks.yaml"

JOB_ID = "onex-validation"
JOB_NAME = "ONEX Validation"

pytestmark = pytest.mark.unit


def test_filename_scanners_only_check_supplied_file(tmp_path: Path) -> None:
    clean = tmp_path / "model_clean.py"
    dirty = tmp_path / "model_dirty.py"
    clean.write_text("class ModelClean:\n    pass\n", encoding="utf-8")
    dirty.write_text("# deprecated\n", encoding="utf-8")

    clean_run = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/validation/validate_no_backward_compatibility.py",
            str(clean),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    dirty_run = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/validation/validate_no_backward_compatibility.py",
            str(dirty),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    assert clean_run.returncode == 0
    assert dirty_run.returncode == 1


def test_shell_scanner_only_checks_supplied_file(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    dirty = tmp_path / "dirty.py"
    clean.write_text("value = 'ok'\n", encoding="utf-8")
    dirty.write_text("value = 'bolt://localhost'\n", encoding="utf-8")

    clean_run = subprocess.run(
        ["bash", "scripts/check_no_hardcoded_bolt.sh", str(clean)],
        cwd=REPO_ROOT,
        check=False,
    )
    dirty_run = subprocess.run(
        ["bash", "scripts/check_no_hardcoded_bolt.sh", str(dirty)],
        cwd=REPO_ROOT,
        check=False,
    )
    assert clean_run.returncode == 0
    assert dirty_run.returncode == 1


def _load_mapping(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path} did not parse to a mapping"
    return loaded


def _workflow() -> dict:
    return _load_mapping(CI_WORKFLOW)


def _job() -> dict:
    job = _workflow()["jobs"][JOB_ID]
    assert job["name"] == JOB_NAME
    return job


def _suite_step() -> dict:
    matches = [
        step
        for step in _job()["steps"]
        if "pre-commit run --all-files" in str(step.get("run", ""))
    ]
    assert len(matches) == 1, (
        f"expected one whole-tree pre-commit step in {JOB_ID}, found {len(matches)}"
    )
    return matches[0]


def _staged_scoped_hook_ids() -> set[str]:
    """Return pre-commit-stage hooks whose file input is the staged diff."""
    config = _load_mapping(PRECOMMIT_CONFIG)
    default_stages = config.get("default_stages", ["pre-commit"])
    hook_ids: set[str] = set()
    for repository in config["repos"]:
        for hook in repository.get("hooks", []):
            stages = hook.get("stages", default_stages)
            if "pre-commit" not in stages:
                continue
            if hook.get("pass_filenames", True) is False:
                continue
            hook_ids.add(str(hook["id"]))
    return hook_ids


def test_every_staged_scoped_hook_has_whole_tree_coverage() -> None:
    staged_ids = _staged_scoped_hook_ids()
    assert staged_ids, "expected at least one staged-file-scoped hook"

    step = _suite_step()
    command = str(step["run"])
    assert "SKIP" not in _job().get("env", {})
    assert "SKIP" not in step.get("env", {})
    assert "SKIP=" not in command

    for hook_id in sorted(staged_ids):
        assert "pre-commit run --all-files" in command, (
            f"{hook_id} is staged-file-scoped but has no whole-tree counterpart"
        )


def test_whole_tree_job_and_step_are_unconditional_and_blocking() -> None:
    job = _job()
    step = _suite_step()
    assert "if" not in job
    assert "needs" not in job
    assert job.get("continue-on-error") is not True
    assert "if" not in step
    assert step.get("continue-on-error") is not True


def test_whole_tree_job_is_required_and_summary_wired() -> None:
    manifest = _load_mapping(REQUIRED_CHECKS)
    matches = [
        check
        for check in manifest["required_checks"]
        if check.get("name") == JOB_NAME and check.get("job_id") == JOB_ID
    ]
    assert len(matches) == 1
    assert matches[0].get("conditional") is False

    excluded_names = {
        str(check.get("name")) for check in manifest.get("excluded_checks", [])
    }
    assert JOB_NAME not in excluded_names

    summary = _workflow()["jobs"]["ci-summary"]
    assert JOB_ID in summary["needs"]
    summary_commands = "\n".join(str(step.get("run", "")) for step in summary["steps"])
    assert "needs.onex-validation.result" in summary_commands


def test_workflow_has_no_pull_request_paths_filter() -> None:
    workflow = _workflow()
    triggers = workflow[True] if True in workflow else workflow["on"]
    pull_request = triggers.get("pull_request") or {}
    assert "paths" not in pull_request
    assert "paths-ignore" not in pull_request


@pytest.mark.parametrize(
    ("script_path", "violation"),
    [
        (
            "scripts/check_no_hardcoded_bolt.sh",
            "value = 'bolt://localhost:7687'\n",
        ),
        (
            "scripts/validation/check_kafka_no_hardcoded_fallback.sh",
            'value = os.getenv("KAFKA_HOST", "localhost:9092")\n',
        ),
        (
            "scripts/validation/check_no_internal_ips.sh",
            "value = '192." + "168.1.10'\n",
        ),
    ],
)
def test_shell_backstop_defaults_to_whole_tree_scan(
    script_path: str, violation: str, tmp_path: Path
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    candidate = source / "candidate.py"
    candidate.write_text("value = 'clean'\n", encoding="utf-8")
    script = REPO_ROOT / script_path

    clean_run = subprocess.run(["bash", str(script)], cwd=tmp_path, check=False)
    candidate.write_text(violation, encoding="utf-8")
    dirty_run = subprocess.run(["bash", str(script)], cwd=tmp_path, check=False)

    assert clean_run.returncode == 0
    assert dirty_run.returncode == 1
