# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16665: the OCC manual-replay precheck exists here, and refuses correctly.

Two things are pinned, and the first matters more than it looks.

**The script exists and is importable from the path the workflow names.**
`call-occ-companion-effect.yml` has run `python3 scripts/ci/occ_manual_replay_precheck.py`
since the OMN-14993 fan-out, but the script was only ever committed to
omnibase_infra, omniclaude and omniweb. Every manual replay dispatched in this
repo died at `[Errno 2] No such file or directory` (live: run 33019346585, exit
2). A referenced-but-absent CI script is invisible until someone needs it, which
is precisely when they cannot afford the dead end — so the path is asserted, not
just the behavior.

**The three-way refusal.** GitHub's `gh pr view --json state` returns exactly
one of OPEN/CLOSED/MERGED, and CLOSED and MERGED are NOT interchangeable here:

* closed-unmerged is a dead target — no evidence was lost, refuse always;
* merged is a permanent evidence hole — refuse by default, allow when the
  operator explicitly arms the OMN-16665 override;
* draft is a live hold — refuse, the ready_for_review trigger handles it.

The middle case is the one this ticket added. Conflating it with the first is
what let omnimemory#447 merge unbound with no recovery path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "ci"
    / "occ_manual_replay_precheck.py"
)


def _load() -> ModuleType:
    """Import the script by path — the same path the workflow step invokes."""
    spec = importlib.util.spec_from_file_location("occ_manual_replay_precheck", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pr(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "number": 447,
        "state": "OPEN",
        "isDraft": False,
        "headRefOid": "a" * 40,
        "headRefName": "jonah/omn-16669-fix",
        "title": "docs(OMN-16669): correct the settings docstring default",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_the_script_the_workflow_references_actually_exists() -> None:
    """The OMN-16665 repair itself. Without this file the workflow step exits 2
    before the publish, and the documented remediation is a dead end."""
    assert _SCRIPT.is_file(), f"{_SCRIPT} is referenced by the workflow but absent"


@pytest.mark.unit
class TestRefusals:
    def test_open_non_draft_pr_is_eligible(self) -> None:
        """Control: the ordinary replay path still passes, so every refusal
        below is caused by the state under test."""
        _load().check_replay_eligible(_pr())

    def test_closed_unmerged_pr_is_refused(self) -> None:
        module = _load()
        with pytest.raises(
            module.ManualReplayRefusedError, match="closed without merging"
        ):
            module.check_replay_eligible(_pr(state="CLOSED"))

    def test_closed_unmerged_pr_is_refused_even_when_merged_replay_is_armed(
        self,
    ) -> None:
        """The override is scoped to MERGED. A blanket state override would
        re-open occ#4333, the incident F-17 was added to close."""
        module = _load()
        with pytest.raises(
            module.ManualReplayRefusedError, match="closed without merging"
        ):
            module.check_replay_eligible(_pr(state="CLOSED"), allow_merged_replay=True)

    def test_merged_pr_is_refused_by_default(self) -> None:
        module = _load()
        with pytest.raises(module.ManualReplayRefusedError) as excinfo:
            module.check_replay_eligible(_pr(state="MERGED"))
        # The refusal must name the way out, or it is the same dead end with a
        # better error message.
        assert "allow_merged_replay=true" in str(excinfo.value)

    def test_merged_pr_is_eligible_when_armed(self) -> None:
        _load().check_replay_eligible(_pr(state="MERGED"), allow_merged_replay=True)

    def test_draft_pr_is_refused(self) -> None:
        module = _load()
        with pytest.raises(module.ManualReplayRefusedError, match="draft"):
            module.check_replay_eligible(_pr(isDraft=True))

    def test_draft_is_refused_independently_of_the_merged_override(self) -> None:
        module = _load()
        with pytest.raises(module.ManualReplayRefusedError, match="draft"):
            module.check_replay_eligible(_pr(isDraft=True), allow_merged_replay=True)

    def test_state_comparison_is_case_insensitive(self) -> None:
        """gh emits uppercase; a lowercase-only comparison would silently admit
        every refused state."""
        module = _load()
        with pytest.raises(module.ManualReplayRefusedError):
            module.check_replay_eligible(_pr(state="merged"))


@pytest.mark.unit
class TestCliExitCodes:
    """The workflow reads the exit code, so it is part of the contract."""

    def test_eligible_pr_exits_zero(self, tmp_path: Path) -> None:
        module = _load()
        state_file = tmp_path / "pr_state.json"
        state_file.write_text(json.dumps(_pr()), encoding="utf-8")

        assert module.main(["precheck", str(state_file)]) == 0

    def test_refused_pr_exits_one(self, tmp_path: Path) -> None:
        module = _load()
        state_file = tmp_path / "pr_state.json"
        state_file.write_text(json.dumps(_pr(state="MERGED")), encoding="utf-8")

        assert module.main(["precheck", str(state_file)]) == 1

    def test_armed_merged_pr_exits_zero(self, tmp_path: Path) -> None:
        module = _load()
        state_file = tmp_path / "pr_state.json"
        state_file.write_text(json.dumps(_pr(state="MERGED")), encoding="utf-8")

        assert module.main(["precheck", str(state_file), "--allow-merged-replay"]) == 0

    def test_unreadable_state_file_exits_two(self, tmp_path: Path) -> None:
        """Distinct from the exit-1 refusal: a wiring fault must not read as a
        policy decision."""
        module = _load()

        assert module.main(["precheck", str(tmp_path / "missing.json")]) == 2
