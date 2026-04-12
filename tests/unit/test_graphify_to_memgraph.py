# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for graphify_to_memgraph script.

Tests parse_graphify_json, build_node_cypher, and build_edge_cypher
without requiring a live Memgraph connection.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.graphify_to_memgraph import (
    build_edge_cypher,
    build_node_cypher,
    parse_graphify_json,
)

SAMPLE_GRAPH = {
    "nodes": [
        {
            "id": "omniclaude::node_delegation_orchestrator",
            "label": "node_delegation_orchestrator",
            "repo": "omniclaude",
            "source_file": "src/omniclaude/nodes/node_delegation_orchestrator/__init__.py",
            "node_type": "handler",
            "community": "delegation",
        },
        {
            "id": "omniclaude::node_quality_gate",
            "label": "node_quality_gate",
            "repo": "omniclaude",
            "source_file": "src/omniclaude/nodes/node_quality_gate/__init__.py",
            "node_type": "handler",
            "community": "quality",
        },
    ],
    "links": [
        {
            "source": "omniclaude::node_delegation_orchestrator",
            "target": "omniclaude::node_quality_gate",
            "relation": "calls",
            "confidence_score": 0.95,
        }
    ],
}


@pytest.fixture
def graph_json_file(tmp_path: Path) -> Path:
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(SAMPLE_GRAPH))
    return graph_file


@pytest.mark.unit
def test_parse_graphify_json_nodes(graph_json_file: Path) -> None:
    nodes, links = parse_graphify_json(graph_json_file)
    assert len(nodes) == 2
    assert nodes[0]["id"] == "omniclaude::node_delegation_orchestrator"
    assert nodes[1]["label"] == "node_quality_gate"


@pytest.mark.unit
def test_parse_graphify_json_links(graph_json_file: Path) -> None:
    nodes, links = parse_graphify_json(graph_json_file)
    assert len(links) == 1
    assert links[0]["source"] == "omniclaude::node_delegation_orchestrator"
    assert links[0]["target"] == "omniclaude::node_quality_gate"
    assert links[0]["relation"] == "calls"
    assert links[0]["confidence_score"] == 0.95


@pytest.mark.unit
def test_parse_graphify_json_empty_keys() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({}, f)
        path = Path(f.name)
    nodes, links = parse_graphify_json(path)
    assert nodes == []
    assert links == []


@pytest.mark.unit
def test_build_node_cypher_uses_id_as_merge_key() -> None:
    node = SAMPLE_GRAPH["nodes"][0]
    cypher, params = build_node_cypher(node)
    assert "MERGE (n:Node {id: $id})" in cypher
    assert params["id"] == "omniclaude::node_delegation_orchestrator"


@pytest.mark.unit
def test_build_node_cypher_sets_all_fields() -> None:
    node = SAMPLE_GRAPH["nodes"][0]
    cypher, params = build_node_cypher(node)
    assert params["label"] == "node_delegation_orchestrator"
    assert params["repo"] == "omniclaude"
    assert (
        params["source_file"]
        == "src/omniclaude/nodes/node_delegation_orchestrator/__init__.py"
    )
    assert params["node_type"] == "handler"
    assert params["community"] == "delegation"


@pytest.mark.unit
def test_build_node_cypher_missing_optional_fields() -> None:
    node = {"id": "some::node"}
    cypher, params = build_node_cypher(node)
    assert params["id"] == "some::node"
    assert params["label"] == ""
    assert params["repo"] == ""
    assert params["source_file"] == ""
    assert params["node_type"] == ""
    assert params["community"] == ""


@pytest.mark.unit
def test_build_node_cypher_does_not_use_label_as_merge_key() -> None:
    node = SAMPLE_GRAPH["nodes"][0]
    cypher, _ = build_node_cypher(node)
    assert "MERGE (n:Node {label:" not in cypher
    assert "MERGE (n:Node {name:" not in cypher


@pytest.mark.unit
def test_build_edge_cypher_structure() -> None:
    edge = SAMPLE_GRAPH["links"][0]
    cypher, params = build_edge_cypher(edge)
    assert "MATCH (a:Node {id: $source})" in cypher
    assert "MATCH" in cypher
    assert "(b:Node {id: $target})" in cypher
    assert "MERGE (a)-[r:RELATES" in cypher
    assert params["source"] == "omniclaude::node_delegation_orchestrator"
    assert params["target"] == "omniclaude::node_quality_gate"
    assert params["relation"] == "calls"
    assert params["confidence_score"] == 0.95


@pytest.mark.unit
def test_build_edge_cypher_missing_optional_fields() -> None:
    edge = {"source": "a::node", "target": "b::node"}
    cypher, params = build_edge_cypher(edge)
    assert params["relation"] == ""
    assert params["confidence_score"] == 0.0
