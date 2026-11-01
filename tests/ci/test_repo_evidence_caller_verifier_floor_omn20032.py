# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20032: pin the receipt-gate reusable that refuses a verifier below the floor.

OMN-21858: the floor is now an ancestry fact checked by the reusable on the caller's
``verifier-tag``; the caller passes no ``verifier-version``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"

# The first omnibase_core dev-merge tag whose commit descends from #1932, the commit
# that adds VERIFIER_FLOOR to the reusable's "Install the pinned verifier" step. The
# dev-merge tag embeds its UTC creation time, so a tag at or after this one carries it.
FLOOR_REUSABLE_TAG = "dev-20261018024445-b6ee996befa1"

# The first omnimarket dev-merge tag whose commit descends from the floor commit
# e565ba5ba00c (omnimarket#3511, the refusal of a contract that leaves an acceptance
# criterion unbound). The tag embeds its UTC creation time, so a tag at or after this
# one carries the floor.
FLOOR_VERIFIER_TAG = "dev-20261016123308-0a385ce46cdf"
DEV_MERGE_TAG = r"dev-\d{14}-[0-9a-f]{12}"


def _job() -> dict[str, Any]:
    doc = yaml.safe_load(CALLER.read_text(encoding="utf-8"))
    job: dict[str, Any] = doc["jobs"]["repo-evidence"]
    return job


def test_caller_pins_the_reusable_that_carries_the_verifier_floor() -> None:
    uses = _job()["uses"]
    assert re.fullmatch(
        r"OmniNode-ai/omnibase_core/\.github/workflows/receipt-gate\.yml@dev-\d{14}-[0-9a-f]{12}",
        uses,
    )
    assert uses.split("@", 1)[1] >= FLOOR_REUSABLE_TAG, (
        "pin the omnibase_core receipt-gate reusable at or past omnibase_core#1932"
    )


def test_caller_verifier_tag_is_at_or_above_the_floor() -> None:
    inputs = _job()["with"]
    assert "verifier-version" not in inputs
    assert re.fullmatch(DEV_MERGE_TAG, inputs["verifier-tag"])
    assert inputs["verifier-tag"] >= FLOOR_VERIFIER_TAG, (
        "verifier-tag must be an omnimarket dev-merge tag at or after the floor commit"
    )


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("dev-20261001000000-000000000000", False),
        ("dev-20261016123307-ffffffffffff", False),
        (FLOOR_VERIFIER_TAG, True),
        ("dev-20261101000200-949993edf3b7", True),
    ],
)
def test_floor_comparison_has_a_refusing_and_an_admitting_case(
    tag: str, expected: bool
) -> None:
    assert (tag >= FLOOR_VERIFIER_TAG) is expected
