# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for graphify_to_qdrant script.

Tests build_embedding_text and chunk_nodes without requiring live services.
"""

from __future__ import annotations

import pytest

from scripts.graphify_to_qdrant import build_embedding_text, chunk_nodes

SAMPLE_NODE_FULL = {
    "id": "omniclaude::node_delegation_orchestrator",
    "label": "node_delegation_orchestrator",
    "repo": "omniclaude",
    "source_file": "src/omniclaude/nodes/node_delegation_orchestrator/__init__.py",
    "node_type": "handler",
    "community": "delegation",
    "relations": ["calls:node_quality_gate", "subscribes:onex.cmd.delegate.v1"],
    "protocols": ["DelegationProtocol"],
    "contract_refs": ["contract.yaml#dispatch"],
}

SAMPLE_NODE_MINIMAL = {
    "id": "omniclaude::bare_node",
    "label": "bare_node",
}


@pytest.mark.unit
def test_build_embedding_text_includes_label() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "node_delegation_orchestrator" in text


@pytest.mark.unit
def test_build_embedding_text_includes_repo() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "omniclaude" in text


@pytest.mark.unit
def test_build_embedding_text_includes_source_file() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "node_delegation_orchestrator/__init__.py" in text


@pytest.mark.unit
def test_build_embedding_text_includes_community() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "delegation" in text


@pytest.mark.unit
def test_build_embedding_text_includes_node_type() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "handler" in text


@pytest.mark.unit
def test_build_embedding_text_includes_relations() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "calls:node_quality_gate" in text


@pytest.mark.unit
def test_build_embedding_text_includes_protocols() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "DelegationProtocol" in text


@pytest.mark.unit
def test_build_embedding_text_includes_contract_refs() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert "contract.yaml#dispatch" in text


@pytest.mark.unit
def test_build_embedding_text_handles_missing_optional_fields() -> None:
    text = build_embedding_text(SAMPLE_NODE_MINIMAL)
    assert "bare_node" in text
    assert "omniclaude" in text


@pytest.mark.unit
def test_build_embedding_text_returns_string() -> None:
    text = build_embedding_text(SAMPLE_NODE_FULL)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.unit
def test_chunk_nodes_empty_list() -> None:
    result = list(chunk_nodes([], batch_size=10))
    assert result == []


@pytest.mark.unit
def test_chunk_nodes_single_batch() -> None:
    nodes = [{"id": f"n{i}"} for i in range(5)]
    chunks = list(chunk_nodes(nodes, batch_size=10))
    assert len(chunks) == 1
    assert chunks[0] == nodes


@pytest.mark.unit
def test_chunk_nodes_exact_multiple() -> None:
    nodes = [{"id": f"n{i}"} for i in range(10)]
    chunks = list(chunk_nodes(nodes, batch_size=5))
    assert len(chunks) == 2
    assert chunks[0] == nodes[:5]
    assert chunks[1] == nodes[5:]


@pytest.mark.unit
def test_chunk_nodes_remainder() -> None:
    nodes = [{"id": f"n{i}"} for i in range(7)]
    chunks = list(chunk_nodes(nodes, batch_size=3))
    assert len(chunks) == 3
    assert chunks[0] == nodes[:3]
    assert chunks[1] == nodes[3:6]
    assert chunks[2] == nodes[6:]


@pytest.mark.unit
def test_chunk_nodes_batch_size_one() -> None:
    nodes = [{"id": f"n{i}"} for i in range(3)]
    chunks = list(chunk_nodes(nodes, batch_size=1))
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk == [nodes[i]]
