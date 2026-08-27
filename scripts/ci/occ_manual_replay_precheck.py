#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Precheck for the OCC born-path manual replay entrypoint (OMN-14993, OMN-16665).

`call-occ-companion-effect.yml` carries a `workflow_dispatch` trigger so an
operator can manually re-request a machine mint for a PR whose original
`pull_request` event never produced a companion. This script runs BEFORE the
publish step, reads the live PR state the operator would otherwise have to check
by hand, and refuses with an actionable message instead of letting the request
travel to the bus only to be absorbed downstream.

**This file was missing in this repo until OMN-16665.** `call-occ-companion-
effect.yml` referenced it from the day the OMN-14993 fan-out landed, but the
script itself was only ever committed to omnibase_infra, omniclaude and omniweb.
Every manual replay dispatched here therefore died at
`python3: can't open file '.../scripts/ci/occ_manual_replay_precheck.py':
[Errno 2] No such file or directory` (live: run 33019346585, exit 2). It failed
CLOSED — the publish step never ran, so nothing bad was published — but the
documented remediation for a missing companion was a dead end in this repo.

## What is refused, and why

Two of the three refusals mirror the F-17 guard in omnimarket's
`handler_occ_companion_compute.py`, which deliberately declines to author a
companion for a PR that is not open (added after incident occ#4333, a companion
generated for a closed draft):

* **closed, not merged** — a dead target. Nothing was lost by not authoring a
  companion for an abandoned PR, and authoring one now would produce queue noise
  plus a failing obsolete companion. Refused unconditionally.
* **draft** — a live hold, not a loss. The `ready_for_review` trigger re-fires
  the born path on its own.

The third is new, and is the reason this script gained an argument:

* **merged** — refused BY DEFAULT, but eligible with `--allow-merged-replay`.
  OMN-16665 established that a merged PR without a companion is not the same
  situation as an abandoned one: it is a permanent evidence hole. The live case
  is the merge/queue-latency race — omnimemory#447 opened 19:35:00Z and merged
  19:46:09Z while the self-hosted runner fleet held the publisher job until
  19:47:34Z, so the mint command reached the broker after the PR had already
  merged. The publisher was green (its contract is "delivered to broker"), the
  compute did a correct live read, and the companion was lost with no signal.

  Earlier revisions of this script said a merged replay "would require a new,
  deliberately-scoped override of F-17 ... which is real design work, not a
  workflow trigger change". OMN-16665 did that work:
  `ModelOccCompanionRequest.allow_merged_replay` authors the companion for a
  MERGED PR only, never for a closed-unmerged one. This flag is the operator
  half of that override, and it stays opt-in so an ordinary replay cannot arm it
  by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ManualReplayRefusedError(Exception):
    """Raised when the target PR cannot be mint-replayed via this entrypoint."""


def check_replay_eligible(
    pr_state: dict[str, Any], *, allow_merged_replay: bool = False
) -> None:
    """Raise :class:`ManualReplayRefusedError` if ``pr_state`` cannot be replayed.

    ``pr_state`` is the parsed JSON of
    ``gh pr view <n> --json number,state,isDraft,headRefOid,headRefName,title``.
    ``gh`` returns exactly one of ``OPEN``/``CLOSED``/``MERGED`` for ``state``,
    compared case-insensitively here — note that this differs from the REST
    ``pulls`` payload, where a merged PR reports ``state == "closed"`` and only
    ``merged``/``merged_at`` discriminates. The ``gh`` form is what this
    entrypoint reads, so ``MERGED`` is directly observable.

    Args:
        pr_state: Parsed ``gh pr view --json ...`` output for the target PR.
        allow_merged_replay: Permit a MERGED PR (the OMN-16665 recovery path).
            Has no effect on a closed-unmerged or draft PR.
    """
    number = pr_state.get("number")
    state = str(pr_state.get("state", "")).strip().lower()
    is_draft = bool(pr_state.get("isDraft", False))

    if state == "merged" and not allow_merged_replay:
        raise ManualReplayRefusedError(
            f"PR #{number} is merged. A merged PR with no OCC companion is a "
            f"permanent evidence hole, not a dead target, and OMN-16665 added a "
            f"deliberately-scoped F-17 override to recover it. Re-dispatch this "
            f"workflow with allow_merged_replay=true to author the missing "
            f"contract + receipt against the merged PR. Refusing here rather "
            f"than publishing, because without that flag the request would be "
            f"declined downstream by the F-17 guard in omnimarket's "
            f"handler_occ_companion_compute.py."
        )

    if state == "closed":
        raise ManualReplayRefusedError(
            f"PR #{number} is closed without merging -- the OCC born-path "
            f"manual replay entrypoint cannot mint a companion for a dead "
            f"target, and allow_merged_replay does NOT relax this. The "
            f"downstream F-17 guard in omnimarket's "
            f"handler_occ_companion_compute.py deliberately refuses to author a "
            f"companion for an abandoned PR (added after incident occ#4333); no "
            f"evidence was lost by not authoring one. If the PR is reopened and "
            f"merged, the born path mints normally."
        )

    if is_draft:
        raise ManualReplayRefusedError(
            f"PR #{number} is a draft -- the OCC born-path manual replay "
            f"entrypoint cannot mint a companion for a draft PR (same F-17 "
            f"guard, draft-suppression branch, in handler_occ_companion_"
            f"compute.py). Mark the PR ready for review -- the ready_for_review "
            f"trigger (OMN-14987) re-fires the born path on its own -- or retry "
            f"this dispatch once it is out of draft."
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Refuse an ineligible OCC manual mint replay before publish."
    )
    parser.add_argument("pr_state_json", help="Path to gh pr view --json output.")
    parser.add_argument(
        "--allow-merged-replay",
        action="store_true",
        help=(
            "Permit replay for a MERGED PR whose companion was lost (OMN-16665). "
            "Does not permit a closed-unmerged or draft PR."
        ),
    )
    args = parser.parse_args(argv[1:])

    pr_state_path = Path(args.pr_state_json)
    try:
        pr_state = json.loads(pr_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to read/parse {pr_state_path}: {exc}", file=sys.stderr)
        return 2

    try:
        check_replay_eligible(pr_state, allow_merged_replay=args.allow_merged_replay)
    except ManualReplayRefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    state = str(pr_state.get("state", "")).strip().lower()
    qualifier = (
        "MERGED (allow_merged_replay armed -- OMN-16665 recovery)"
        if state == "merged"
        else "OPEN and non-draft"
    )
    print(f"PR #{pr_state.get('number')} is {qualifier} -- manual replay eligible.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
