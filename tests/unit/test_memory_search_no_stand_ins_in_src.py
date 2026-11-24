# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The environment switch is gone and the stand-ins are importable from tests only (OMN-26659).

AC3 falsifier: a search of ``src`` finds the switch or a mock binding.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"

_FORBIDDEN = {
    "the environment switch": re.compile(r"OMNIMEMORY_USE_STUB_HANDLERS"),
    "the stub flag": re.compile(r"use_stub_handlers"),
    "a mock retrieval handler": re.compile(
        r"HandlerQdrantMock|HandlerDbMock|HandlerGraphMock|MockRetrievalStore"
    ),
    "a mock binder": re.compile(r"bind_(?:qdrant|db|graph)_mock"),
    "a mock config model": re.compile(
        r"ModelHandler(?:Qdrant|Db|Graph)MockConfig|ModelHandlerMemoryRetrievalConfig"
    ),
}

_REMOVED_MODULES = (
    "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant_mock",
    "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_db_mock",
    "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_graph_mock",
    "omnimemory.nodes.node_memory_retrieval_effect.adapters.adapter_mock_store",
    "omnimemory.nodes.node_memory_retrieval_effect.models.model_handler_memory_retrieval_config",
)


def _src_files() -> list[Path]:
    return [
        path
        for path in _SRC.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    ]


@pytest.mark.unit
@pytest.mark.parametrize("what", list(_FORBIDDEN))
def test_src_has_no_switch_or_mock_binding(what: str) -> None:
    pattern = _FORBIDDEN[what]
    hits = [
        f"{path.relative_to(_SRC)}:{number}"
        for path in _src_files()
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if pattern.search(line)
    ]

    assert hits == [], f"{what} found in src: {hits}"


@pytest.mark.unit
def test_search_probe_would_find_a_planted_switch(tmp_path: Path) -> None:
    """Positive control: the scan above does match when the switch is present."""
    planted = tmp_path / "dispatch_handlers.py"
    planted.write_text('os.getenv("OMNIMEMORY_USE_STUB_HANDLERS", "true")\n')

    assert _FORBIDDEN["the environment switch"].search(planted.read_text())


@pytest.mark.unit
@pytest.mark.parametrize("module", _REMOVED_MODULES)
def test_stand_in_modules_are_gone_from_the_package(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.unit
def test_stand_ins_import_from_tests() -> None:
    stand_in = importlib.import_module("tests.stand_ins.stand_in_vector_store")

    assert hasattr(stand_in, "HandlerQdrantMock")


@pytest.mark.unit
def test_switch_is_not_read_by_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the old switch to false changes nothing: there is no stub mode left."""
    from omnimemory.nodes.node_memory_retrieval_effect.container.registry_memory_retrieval import (
        bind_memory_retrieval,
    )
    from omnimemory.nodes.node_memory_retrieval_effect.models.model_retrieval_store_settings import (
        ModelRetrievalStoreSettings,
    )

    monkeypatch.setenv("OMNIMEMORY_USE_STUB_HANDLERS", "true")
    for name in ("QDRANT_HOST", "QDRANT_PORT", "LLM_EMBEDDING_URL"):
        monkeypatch.delenv(name, raising=False)

    assert bind_memory_retrieval(ModelRetrievalStoreSettings()).configured_stores == (
        frozenset()
    )
