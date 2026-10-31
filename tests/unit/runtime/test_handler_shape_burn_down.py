# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler-shape findings are gone from the files of slice rf1400-s2 (OMN-21762).

The core ``ValidatorHandlerShape`` judges every handler class. These tests run it
over the files the slice touched and exercise the collaborators that took over the
state the mock and semantic handlers used to hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibase_core.validation.validator_handler_shape import ValidatorHandlerShape

from omnimemory.nodes.node_semantic_analyzer_compute.adapters.adapter_embedding_cache import (
    EmbeddingLruCache,
)
from omnimemory.runtime.handler_lifecycle import HandlerMemoryLifecycle

_SRC = Path(__file__).resolve().parents[3] / "src" / "omnimemory"

_SLICE_FILES = (
    "nodes/node_agent_learning_retrieval_effect/adapters/adapter_learning_scoring.py",
    "nodes/node_memory_lifecycle_orchestrator/handlers/handler_memory_tick.py",
    "nodes/node_memory_retrieval_effect/handlers/handler_db_mock.py",
    "nodes/node_memory_retrieval_effect/handlers/handler_graph_mock.py",
    "nodes/node_memory_retrieval_effect/handlers/handler_memory_retrieval.py",
    "nodes/node_memory_retrieval_effect/handlers/handler_qdrant_mock.py",
    "nodes/node_semantic_analyzer_compute/handlers/handler_semantic_compute.py",
    "protocols/protocol_memory_expire_handler.py",
    "runtime/dispatch_handlers.py",
    "runtime/handler_lifecycle.py",
)


@pytest.mark.unit
def test_slice_files_have_no_handler_shape_findings() -> None:
    """The core handler-shape validator reports nothing for the slice files."""
    validator = ValidatorHandlerShape()
    findings = [
        f"{issue.file_path}:{issue.line_number}: {issue.code}"
        for relative in _SLICE_FILES
        for issue in validator.validate_file(_SRC / relative).issues
    ]
    assert findings == []


@pytest.mark.unit
def test_validator_still_flags_a_mutating_public_method(tmp_path: Path) -> None:
    """Positive control: the validator does report a handler that mutates itself."""
    module = tmp_path / "handler_probe.py"
    module.write_text(
        "class HandlerProbe:\n"
        "    def __init__(self) -> None:\n"
        "        self._items: list[int] = []\n"
        "    def seed(self, item: int) -> None:\n"
        "        self._items.append(item)\n"
    )
    codes = [i.code for i in ValidatorHandlerShape().validate_file(module).issues]
    assert "mutating_public_method" in codes


@pytest.mark.unit
def test_embedding_cache_round_trip_and_eviction() -> None:
    """The cache returns copies, evicts past its size and empties on reset."""
    cache = EmbeddingLruCache(max_size=1)
    cache.put("a", [1.0])
    assert cache.contains("a")
    copy = cache.get("a")
    copy.append(2.0)
    assert cache.get("a") == [1.0]
    cache.put("b", [3.0])
    assert not cache.contains("a")
    assert (cache.size, cache.max_size) == (1, 1)
    cache.reset()
    assert cache.size == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifecycle_initialize_and_shutdown_track_started() -> None:
    """initialize reports each provided component and shutdown stops the handler."""
    lifecycle = HandlerMemoryLifecycle(graph_memory_adapter=object())
    assert await lifecycle.initialize() == {"graph_memory": "initialized"}
    assert lifecycle.is_started()
    await lifecycle.shutdown()
    assert not lifecycle.is_started()
