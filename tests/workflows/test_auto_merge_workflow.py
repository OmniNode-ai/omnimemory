# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``.github/workflows/auto-merge.yml`` (OMN-16509).

The ``Enable auto-merge`` step calls ``gh pr merge --auto``. Passing an
explicit merge method is *rejected* on a queue-controlled branch (OMN-13214),
so the bare form must stay the first attempt. But ``dev`` is no longer
queue-controlled -- verified live for this repo on 2026-08-28:

* ``gh api graphql`` -> ``repository.mergeQueue(branch:"dev")`` is ``null``
* ``gh api repos/OmniNode-ai/omnimemory`` -> ``{"auto": true, "squash": true,
  "merge": false, "rebase": false}`` (squash is the one enabled method)

and gh's own CLI refuses a method-less ``--auto`` when run non-interactively::

    --merge, --rebase, or --squash required when not running interactively

This is the OMN-16501 defect class (fixed in omniclaude#2033, merge
bfa618e22); OMN-16509 ports that fix here.

These tests extract the actual Bash from the workflow YAML, stub the ``gh``
CLI on PATH, and assert the retry behavior. Pulling the snippet straight from
the YAML keeps the tests bound to the deployed logic rather than to a
re-implementation of it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "auto-merge.yml"


def _extract_step_script(step_name_marker: str) -> str:
    """Pull the inline Bash ``run:`` body from a named workflow step.

    Failing to extract a valid script is itself a test failure -- it means the
    YAML structure drifted and the test is no longer bound to the workflow.
    """
    lines = WORKFLOW_PATH.read_text().splitlines(keepends=True)
    in_step = False
    in_run = False
    body_lines: list[str] = []
    for line in lines:
        if not in_step:
            if step_name_marker in line:
                in_step = True
            continue
        if not in_run:
            if line.strip() == "run: |":
                in_run = True
            continue
        # Inside the run block. The body is indented to 10 spaces; the block
        # ends at the first non-blank line that is less indented.
        if line.strip() == "":
            body_lines.append(line)
            continue
        if line.startswith("          "):  # 10 spaces
            body_lines.append(line)
            continue
        break
    assert body_lines, (
        f"Could not extract '{step_name_marker}' step script from "
        "auto-merge.yml; workflow YAML structure changed. Test must be "
        "updated to match."
    )
    return dedent("".join(body_lines))


def _extract_enable_auto_merge_step_script() -> str:
    """Pull the inline Bash from the ``Enable auto-merge`` step."""
    return _extract_step_script("- name: Enable auto-merge")


@pytest.fixture
def gh_merge_stub_dir(tmp_path: Path) -> Path:
    """Stub the ``gh`` CLI for the step's two possible ``gh pr merge`` calls.

    The bare ``--auto`` invocation and the ``--auto --squash`` retry are
    scripted independently via env vars, so a test can prove the retry was
    *not* reached by arming it to succeed and asserting the step still failed.
    """
    stub = tmp_path / "gh"
    stub.write_text(
        dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            args="$*"
            case "$args" in
              *"--auto --squash"*)
                if [ "${STUB_SQUASH_RESULT:-success}" = "success" ]; then
                  echo "${STUB_SQUASH_OUTPUT:-auto-merge enabled}"
                  exit 0
                else
                  echo "${STUB_SQUASH_OUTPUT:-squash attempt failed}" >&2
                  exit 1
                fi
                ;;
              *"--auto"*)
                if [ "${STUB_BARE_RESULT:-success}" = "success" ]; then
                  echo "${STUB_BARE_OUTPUT:-auto-merge enabled}"
                  exit 0
                else
                  echo "${STUB_BARE_OUTPUT:-bare attempt failed}" >&2
                  exit 1
                fi
                ;;
              *)
                echo "unexpected gh invocation: $args" >&2
                exit 99
                ;;
            esac
            """
        )
    )
    stub.chmod(0o755)
    return tmp_path


def _run_enable_auto_merge(
    *,
    gh_merge_stub_dir: Path,
    bare_result: str = "success",
    bare_output: str = "",
    squash_result: str = "success",
    squash_output: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run the extracted ``Enable auto-merge`` Bash against the stub."""
    script = _extract_enable_auto_merge_step_script()
    env = {
        # Force PATH to contain only the stub + system essentials so the
        # script cannot accidentally reach the real `gh` binary.
        "PATH": f"{gh_merge_stub_dir}:/usr/bin:/bin",
        "GH_TOKEN": "stub-token",
        "GH_REPO": "OmniNode-ai/omnimemory",
        "PR": "456",
        "STUB_BARE_RESULT": bare_result,
        "STUB_BARE_OUTPUT": bare_output,
        "STUB_SQUASH_RESULT": squash_result,
        "STUB_SQUASH_OUTPUT": squash_output,
    }
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.unit
class TestAutoMergeEnableStep:
    """Behavioral coverage for the ``Enable auto-merge`` retry logic."""

    def test_bare_auto_success_does_not_retry(self, gh_merge_stub_dir: Path) -> None:
        """Queue-controlled regime (OMN-13214): a bare ``--auto`` that
        succeeds must not reach the ``--squash`` fallback at all."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="success",
            bare_output="Auto-merge enabled",
        )
        assert result.returncode == 0, result.stderr
        assert "auto-merge enabled:" in result.stdout
        assert "(squash)" not in result.stdout

    def test_already_enqueued_is_benign_no_retry(self, gh_merge_stub_dir: Path) -> None:
        """A benign 'already enqueued' race must exit 0 without invoking the
        ``--squash`` retry -- the squash stub is armed to fail, so reaching it
        would turn this green case red."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="failure",
            bare_output="pull request already enqueued",
            squash_result="failure",
            squash_output="must not be invoked",
        )
        assert result.returncode == 0, result.stderr
        assert "not newly enabled (expected)" in result.stdout

    def test_no_active_queue_retries_with_squash(self, gh_merge_stub_dir: Path) -> None:
        """OMN-16509 reproduction: a bare ``--auto`` rejected because gh needs
        an explicit method non-interactively (no active merge queue) must
        retry with ``--squash`` and succeed."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="failure",
            bare_output=(
                "--merge, --rebase, or --squash required when not running interactively"
            ),
            squash_result="success",
            squash_output="Auto-merge enabled",
        )
        assert result.returncode == 0, result.stderr
        assert "bare --auto rejected" in result.stdout
        assert "auto-merge enabled (squash):" in result.stdout

    def test_squash_retry_benign_race_is_tolerated(
        self, gh_merge_stub_dir: Path
    ) -> None:
        """If the ``--squash`` retry loses an 'already enqueued' race, that is
        still a benign outcome and must exit 0, matching the bare-path
        tolerance rather than red-noising the check."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="failure",
            bare_output=(
                "--merge, --rebase, or --squash required when not running interactively"
            ),
            squash_result="failure",
            squash_output="auto-merge already enabled on this pull request",
        )
        assert result.returncode == 0, result.stderr
        assert "not newly enabled (expected)" in result.stdout

    def test_squash_retry_failure_still_propagates(
        self, gh_merge_stub_dir: Path
    ) -> None:
        """If the ``--squash`` retry fails for a real reason, the step must
        fail loudly rather than swallow the error."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="failure",
            bare_output=(
                "--merge, --rebase, or --squash required when not running interactively"
            ),
            squash_result="failure",
            squash_output="some unrelated permanent error",
        )
        assert result.returncode == 1
        assert "auto-merge failed:" in result.stdout

    def test_queue_controlled_rejection_is_not_retried_with_squash(
        self, gh_merge_stub_dir: Path
    ) -> None:
        """A genuine queue-controlled rejection ('merge strategy ... set by
        the merge queue') is a DIFFERENT error from gh's non-interactive
        method requirement and must NOT trigger the ``--squash`` retry --
        passing an explicit method on a queue-controlled branch is itself
        rejected (OMN-13214). The squash stub is armed to succeed, so the step
        failing proves the retry was never reached."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="failure",
            bare_output="The merge strategy for dev is set by the merge queue",
            squash_result="success",
        )
        assert result.returncode == 1
        assert "auto-merge failed:" in result.stdout
        assert "(squash)" not in result.stdout

    def test_unrelated_failure_is_not_retried_with_squash(
        self, gh_merge_stub_dir: Path
    ) -> None:
        """The retry is gated on one specific gh-CLI error string, not on any
        ``gh pr merge`` failure. An unrelated permanent error must fail
        immediately without a method retry."""
        result = _run_enable_auto_merge(
            gh_merge_stub_dir=gh_merge_stub_dir,
            bare_result="failure",
            bare_output="GraphQL: Pull request is in unmergeable state",
            squash_result="success",
        )
        assert result.returncode == 1
        assert "auto-merge failed:" in result.stdout
        assert "(squash)" not in result.stdout


@pytest.mark.unit
class TestAutoMergeWorkflowYaml:
    """YAML-level invariants protecting the retry shape from regressions."""

    def test_merge_command_tries_bare_auto_first(self) -> None:
        """OMN-13214: queue-controlled branches reject an explicit merge
        method, so the first attempt must still be the bare ``--auto`` form."""
        text = WORKFLOW_PATH.read_text()
        assert 'gh pr merge "$PR" --repo "$GH_REPO" --auto 2>&1' in text, (
            "auto-merge.yml must still try bare --auto first (queue-controlled path)"
        )

    def test_merge_command_has_squash_fallback_for_no_queue(self) -> None:
        """OMN-16509 (porting OMN-16501): the workflow must retry with
        ``--squash`` gated on gh's specific non-interactive-method error, so a
        genuine queue-controlled rejection is never method-retried."""
        text = WORKFLOW_PATH.read_text()
        assert 'gh pr merge "$PR" --repo "$GH_REPO" --auto --squash 2>&1' in text, (
            "auto-merge.yml must retry with --squash when no merge queue is active"
        )
        assert "required when not running interactively" in text, (
            "the --squash retry must be gated on gh's specific "
            "non-interactive-method error, not a blanket catch-all"
        )


if __name__ == "__main__":  # pragma: no cover - manual run helper
    sys.exit(pytest.main([__file__, "-v"]))
