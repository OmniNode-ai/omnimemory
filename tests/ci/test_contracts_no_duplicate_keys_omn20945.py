# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20945: no ticket contract repeats a YAML mapping key.

PyYAML's ``safe_load`` keeps the last value of a repeated key and says nothing,
so a dropped list-item header can merge two ``evidence_requirements`` entries
into one without any test noticing. omnibase_core's strict loader refuses the
repeat, which is what the OCC-retirement cut-over relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibase_core.errors.error_duplicate_yaml_mapping_key import (
    DuplicateYamlMappingKeyError,
)
from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates

pytestmark = pytest.mark.unit

CONTRACTS = sorted((Path(__file__).resolve().parents[2] / "contracts").glob("*.yaml"))


def test_contracts_directory_is_not_empty() -> None:
    assert CONTRACTS, "contracts/ holds no ticket contract; the glob is wrong"


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.name)
def test_contract_repeats_no_mapping_key(path: Path) -> None:
    load_yaml_mapping_no_duplicates(
        path.read_text(encoding="utf-8"),
        source=f"contracts/{path.name}",
    )


def test_strict_loader_refuses_a_repeated_key() -> None:
    planted = (
        "evidence_requirements:\n"
        "  - kind: tests\n"
        "    description: first\n"
        "    command: one\n"
        "    description: second\n"
    )
    with pytest.raises(
        DuplicateYamlMappingKeyError, match="duplicate key 'description'"
    ):
        load_yaml_mapping_no_duplicates(planted, source="planted.yaml")
