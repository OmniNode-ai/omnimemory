# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23140: memory modules do not import the seams-verify overlay module.

The rebuild plan's section 4.1 row "intelligence-memory to seams-verify" counted the memory
consumers' imports of ``omnibase_infra.runtime.overlay``. Overlay resolution lives in the
primitive layer (``omnibase_core.overlays``) and the consumers call it there. The count is read
from this tree by an ast walk over every import form (top level, nested, lazy); the planted
control shows the same walk reads one when the import is present.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[3] / "src" / "omnimemory"
_SEAMS_VERIFY_MODULE = "omnibase_infra.runtime.overlay"
_PRIMITIVE_MODULE = "omnibase_core.overlays.contract_env_ref"


def _import_statements(source: str) -> list[list[str]]:
    """Per import statement, every dotted name it can land on."""
    found: list[list[str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(
                [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            )
        elif isinstance(node, ast.Import):
            found.append([alias.name for alias in node.names])
    return found


def _imported_modules(source: str) -> list[str]:
    return [name for names in _import_statements(source) for name in names]


def _count_of_relation(source: str) -> int:
    return sum(
        1
        for names in _import_statements(source)
        if any(
            name == _SEAMS_VERIFY_MODULE or name.startswith(f"{_SEAMS_VERIFY_MODULE}.")
            for name in names
        )
    )


def _importers_of_expand() -> list[Path]:
    return [
        path
        for path in sorted(_SRC.rglob("*.py"))
        if "expand_contract_env_refs" in path.read_text(encoding="utf-8")
        and _PRIMITIVE_MODULE in _imported_modules(path.read_text(encoding="utf-8"))
    ]


def test_relation_count_memory_seams_verify_reads_zero_in_this_tree() -> None:
    total = {
        path.relative_to(_SRC).as_posix(): _count_of_relation(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(_SRC.rglob("*.py"))
    }
    assert {name: n for name, n in total.items() if n} == {}


def test_relation_count_memory_seams_verify_control_reads_one_for_a_planted_import() -> (
    None
):
    planted = (
        "def endpoint():\n"
        "    from omnibase_infra.runtime.overlay.contract_env_ref import (\n"
        "        expand_contract_env_refs,\n"
        "    )\n"
        '    return expand_contract_env_refs("${env.HOST}")\n'
    )
    assert _count_of_relation(planted) == 1


def test_relation_count_memory_seams_verify_consumers_call_the_primitive_layer() -> (
    None
):
    """The modules that expand environment references are the ones the cut moved."""
    names = {path.relative_to(_SRC).as_posix() for path in _importers_of_expand()}
    assert len(names) == 8, "the walk found no consumer: it is not reading the tree"
    assert "runtime/plugin.py" in names
