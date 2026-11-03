# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnimemory declares no dependency on omnimarket (OMN-22360).

omnimarket sits above omnimemory in the declared layer order and depends on
omninode-memory, so any declaration of omnimarket here is an upward edge and a
cycle. The core dependency-pin validator refuses it once this repository wires
it; this test holds the manifest until then.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
UPPER = "omnimarket"


def _table(document: object, *keys: str) -> object:
    node = document
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def _requirements(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _requirements(nested)]
    return []


def _declared_names(document: dict[str, object]) -> set[str]:
    requirements = (
        _requirements(_table(document, "project", "dependencies"))
        + _requirements(_table(document, "project", "optional-dependencies"))
        + _requirements(_table(document, "dependency-groups"))
        + _requirements(_table(document, "tool", "uv", "dev-dependencies"))
        + _requirements(_table(document, "tool", "uv", "override-dependencies"))
    )
    return {
        re.split(r"[<>=!~\[; @]", r.strip(), maxsplit=1)[0].lower()
        for r in requirements
    }


def _manifest() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_manifest_declares_no_omnimarket_dependency() -> None:
    assert UPPER not in _declared_names(_manifest())


def test_manifest_has_no_omnimarket_source() -> None:
    sources = _table(_manifest(), "tool", "uv", "sources")
    assert not (isinstance(sources, dict) and UPPER in sources)
