# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
