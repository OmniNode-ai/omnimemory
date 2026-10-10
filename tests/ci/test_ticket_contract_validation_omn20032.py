# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the caller's pinned checkout and ticket validation shell step."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
# OCC commit carrying the binds_ac schema, matching the verifier-floor test
# convention of checking a known prefix and requiring a full commit SHA.
PIN_PREFIX = "802b5f71"


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/contract-validation.yml").read_text()
    )
    steps: list[dict[str, Any]] = workflow["jobs"]["contract-validation"]["steps"]
    return steps


def test_validators_checkout_uses_the_schema_pin_explicitly() -> None:
    checkouts = [
        step
        for step in _steps()
        if step.get("with", {}).get("repository") == "OmniNode-ai/onex_change_control"
    ]
    assert len(checkouts) == 1
    ref = checkouts[0]["with"]["ref"]
    assert re.fullmatch(r"[0-9a-f]{40}", ref)
    assert ref.startswith(PIN_PREFIX)
    assert checkouts[0]["with"]["path"] == ".onex_change_control_validators"


@pytest.mark.parametrize(
    ("branch", "filename", "validator_exit", "status", "ticket"),
    [
        ("jonah/omn-20032-repin", "OMN-20032.yaml", 0, "passed", "OMN-20032"),
        ("jonah/omn-20032-repin", "OMN-20032.yaml", 1, "failed", "OMN-20032"),
        ("jonah/omn-20032-repin", "omn-20032.yaml", 0, "passed", "OMN-20032"),
        ("release/0.18.4", "OMN-20032.yaml", 1, "skipped", ""),
        ("jonah/omn-20032-repin", None, 1, "skipped", "OMN-20032"),
        ("omn-20032-omn-99999", "OMN-20032.yaml", 0, "passed", "OMN-20032"),
    ],
)
def test_validation_step(
    tmp_path: Path,
    branch: str,
    filename: str | None,
    validator_exit: int,
    status: str,
    ticket: str,
) -> None:
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    if filename:
        (contracts / filename).write_text("schema_version: 1.0.0\n")
    validators = tmp_path / ".onex_change_control_validators"
    validators.mkdir()
    tools = tmp_path / "bin"
    tools.mkdir()
    invocation = tmp_path / "invocation"
    uv = tools / "uv"
    uv.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$PWD" "$@" > "$INVOCATION"\n'
        'exit "$VALIDATOR_EXIT"\n'
    )
    uv.chmod(0o755)
    output = tmp_path / "output"
    step = next(step for step in _steps() if step.get("id") == "validate-contract")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "BRANCH_NAME": branch,
            "GITHUB_OUTPUT": str(output),
            "INVOCATION": str(invocation),
            "VALIDATOR_EXIT": str(validator_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (1 if status == "failed" else 0), result.stderr
    assert output.read_text().splitlines() == [
        f"ticket-id={ticket}",
        f"validation-status={status}",
    ]
    if status == "skipped":
        assert not invocation.exists()
    else:
        arguments = invocation.read_text().splitlines()
        assert arguments[:3] == [str(validators), "run", "validate-yaml"]
        # The lab's macOS filesystem may resolve the upper-case spelling of
        # a lower-case filename; Linux must take the lower-case fallback.
        assert Path(arguments[3]).samefile(contracts / str(filename))
