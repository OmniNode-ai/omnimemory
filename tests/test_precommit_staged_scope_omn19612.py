# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


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


def _hook_script_path(hook_id: str) -> str:
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    hook = re.search(
        rf"(?ms)^      - id: {re.escape(hook_id)}\n(?P<body>.*?)(?=^      - id:|\Z)",
        config,
    )
    assert hook is not None, f"missing pre-commit hook: {hook_id}"

    entry = re.search(r"(?m)^        entry:\s*(?P<entry>.+)$", hook["body"])
    assert entry is not None, f"missing entry for pre-commit hook: {hook_id}"
    script = re.search(r"(?:[\w.-]+/)+[\w.-]+\.(?:py|sh)", entry["entry"])
    assert script is not None, f"hook entry has no script path: {hook_id}"
    return script.group(0)


def _onex_validation_commands() -> list[str]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["onex-validation"]
    assert job["name"] == "ONEX Validation"
    return [step["run"] for step in job["steps"] if "run" in step]


def test_staged_hooks_have_whole_tree_ci_backstops() -> None:
    hooks = {
        "no-hardcoded-bolt-fallback": "scripts/check_no_hardcoded_bolt.sh",
        "kafka-no-hardcoded-fallback": (
            "scripts/validation/check_kafka_no_hardcoded_fallback.sh"
        ),
        "no-internal-ips": "scripts/validation/check_no_internal_ips.sh",
        "validate-model-locations": ("scripts/validation/validate_model_locations.py"),
    }
    commands = "\n".join(_onex_validation_commands())

    for hook_id, expected_script in hooks.items():
        script = _hook_script_path(hook_id)
        assert script == expected_script
        assert script in commands, (
            f"{hook_id} has no whole-tree CI backstop in onex-validation"
        )


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
