# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26318: omnimemory commits its ledger-file census ratchet row and per-file table.

The file does not exist at the merge base, so every test here fails there and
passes at the head that adds it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
RATCHET = ROOT / "config/ledger_file_census_ratchet.yaml"
REPO = "omnimemory"


def _load() -> dict[str, object]:
    assert RATCHET.is_file(), f"{RATCHET} is not committed"
    data = yaml.safe_load(RATCHET.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_ratchet_holds_a_row_for_this_repository() -> None:
    ratchet = _load()["ratchet"]
    assert isinstance(ratchet, dict)
    assert isinstance(ratchet[REPO], int)
    assert ratchet[REPO] >= 0


def test_per_file_table_sums_to_the_ratchet_row() -> None:
    data = _load()
    ratchet = data["ratchet"]
    files = data["files"]
    assert isinstance(ratchet, dict)
    assert isinstance(files, dict)
    table = files[REPO]
    assert isinstance(table, dict)
    assert len(table) == ratchet[REPO]
    assert all(isinstance(lines, int) and lines > 0 for lines in table.values())


def test_every_listed_file_exists_in_the_repository() -> None:
    files = _load()["files"]
    assert isinstance(files, dict)
    table = files[REPO]
    assert isinstance(table, dict)
    for path in table:
        assert (ROOT / path).is_file(), f"census lists {path}, which is not a file"
