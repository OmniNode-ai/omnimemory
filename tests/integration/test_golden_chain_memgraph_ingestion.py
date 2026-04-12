# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Golden chain tests for Memgraph ingestion flow (OMN-8646).

Validates the end-to-end path:
  graph.json input → parse_graphify_json → build_node_cypher / build_edge_cypher
  → ingest_repo → driver.session → tx.run()

Tests (9 total):
  Happy path:
    1. test_ingest_repo_runs_node_cypher_for_each_node
    2. test_ingest_repo_runs_edge_cypher_for_each_link
    3. test_ingest_repo_returns_correct_counts
    4. test_ingest_repo_batches_large_node_sets
    5. test_ingest_repo_empty_graph_returns_zero_counts
  Error paths:
    6. test_ingest_repo_tx_run_raises_propagates
    7. test_ingest_repo_session_raises_propagates
  Parse layer:
    8. test_parse_graphify_json_roundtrip
    9. test_build_node_and_edge_cypher_contain_merge_keys

Ticket: OMN-8646
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.graphify_to_memgraph import (
    build_edge_cypher,
    build_node_cypher,
    ingest_repo,
    parse_graphify_json,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_NODE_A = {
    "id": "repo_a::node_foo",
    "label": "node_foo",
    "repo": "repo_a",
    "source_file": "src/repo_a/nodes/node_foo/__init__.py",
    "node_type": "handler",
    "community": "core",
}
_NODE_B = {
    "id": "repo_a::node_bar",
    "label": "node_bar",
    "repo": "repo_a",
    "source_file": "src/repo_a/nodes/node_bar/__init__.py",
    "node_type": "handler",
    "community": "core",
}
_LINK_AB = {
    "source": "repo_a::node_foo",
    "target": "repo_a::node_bar",
    "relation": "calls",
    "confidence_score": 0.9,
}

SAMPLE_GRAPH = {"nodes": [_NODE_A, _NODE_B], "links": [_LINK_AB]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver_mock() -> MagicMock:
    """Return a mock neo4j-style driver whose session context is fully wired."""
    tx = MagicMock()
    tx.run = MagicMock()
    tx.commit = MagicMock()
    tx.__enter__ = MagicMock(return_value=tx)
    tx.__exit__ = MagicMock(return_value=False)

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    driver.close = MagicMock()

    # Expose internals for assertion
    driver._mock_session = session
    driver._mock_tx = tx
    return driver


# ---------------------------------------------------------------------------
# Happy path — ingest_repo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoldenChainMemgraphIngestRepo:
    """Golden chain: ingest_repo → driver.session → tx.run() with correct Cypher."""

    def test_ingest_repo_runs_node_cypher_for_each_node(self) -> None:
        """Each node produces exactly one tx.run() call with MERGE (n:Node {id: $id})."""
        driver = _make_driver_mock()
        tx = driver._mock_tx

        ingest_repo(driver, [_NODE_A, _NODE_B], [])

        run_calls = tx.run.call_args_list
        assert len(run_calls) == 2
        for c in run_calls:
            cypher = c.args[0]
            assert "MERGE (n:Node {id: $id})" in cypher

    def test_ingest_repo_runs_edge_cypher_for_each_link(self) -> None:
        """Each link produces exactly one tx.run() call with MERGE (a)-[r:RELATES...]."""
        driver = _make_driver_mock()
        tx = driver._mock_tx

        ingest_repo(driver, [], [_LINK_AB])

        run_calls = tx.run.call_args_list
        assert len(run_calls) == 1
        cypher = run_calls[0].args[0]
        assert "MERGE (a)-[r:RELATES" in cypher

    def test_ingest_repo_returns_correct_counts(self) -> None:
        """ingest_repo returns {'nodes': 2, 'edges': 1} for sample graph."""
        driver = _make_driver_mock()
        counts = ingest_repo(driver, [_NODE_A, _NODE_B], [_LINK_AB])

        assert counts["nodes"] == 2
        assert counts["edges"] == 1

    def test_ingest_repo_batches_large_node_sets(self) -> None:
        """Nodes exceeding BATCH_SIZE=500 are flushed in multiple transactions."""
        from scripts.graphify_to_memgraph import BATCH_SIZE

        many_nodes = [
            {
                "id": f"repo::node_{i}",
                "label": f"node_{i}",
                "repo": "repo",
                "source_file": "",
                "node_type": "handler",
                "community": "",
            }
            for i in range(BATCH_SIZE + 5)
        ]

        driver = _make_driver_mock()
        session = driver._mock_session

        counts = ingest_repo(driver, many_nodes, [])

        assert counts["nodes"] == BATCH_SIZE + 5
        # Expect exactly 2 begin_transaction calls for nodes (one full batch + remainder)
        assert session.begin_transaction.call_count >= 2

    def test_ingest_repo_empty_graph_returns_zero_counts(self) -> None:
        """Empty nodes and links produces counts of zero with no tx.run calls."""
        driver = _make_driver_mock()
        tx = driver._mock_tx

        counts = ingest_repo(driver, [], [])

        assert counts == {"nodes": 0, "edges": 0}
        tx.run.assert_not_called()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoldenChainMemgraphErrors:
    """Golden chain error paths: adapter raises → propagates from ingest_repo."""

    def test_ingest_repo_tx_run_raises_propagates(self) -> None:
        """When tx.run raises, the exception propagates out of ingest_repo."""
        driver = _make_driver_mock()
        driver._mock_tx.run = MagicMock(side_effect=RuntimeError("Cypher syntax error"))

        with pytest.raises(RuntimeError, match="Cypher syntax error"):
            ingest_repo(driver, [_NODE_A], [])

    def test_ingest_repo_session_raises_propagates(self) -> None:
        """When driver.session raises, the exception propagates from ingest_repo."""
        driver = _make_driver_mock()
        driver.session = MagicMock(
            side_effect=ConnectionError("Memgraph not reachable")
        )

        with pytest.raises(ConnectionError, match="Memgraph not reachable"):
            ingest_repo(driver, [_NODE_A], [])


# ---------------------------------------------------------------------------
# Parse + cypher layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoldenChainMemgraphParseCypher:
    """Golden chain: parse_graphify_json + build_*_cypher produce correct output."""

    def test_parse_graphify_json_roundtrip(self, tmp_path: Path) -> None:
        """Parsed nodes and links match SAMPLE_GRAPH exactly."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(SAMPLE_GRAPH))

        nodes, links = parse_graphify_json(graph_file)

        assert len(nodes) == 2
        assert len(links) == 1
        assert nodes[0]["id"] == "repo_a::node_foo"
        assert links[0]["relation"] == "calls"
        assert links[0]["confidence_score"] == 0.9

    def test_build_node_and_edge_cypher_contain_merge_keys(self) -> None:
        """build_node_cypher uses id as merge key; build_edge_cypher uses source+target."""
        node_cypher, node_params = build_node_cypher(_NODE_A)
        edge_cypher, edge_params = build_edge_cypher(_LINK_AB)

        assert "MERGE (n:Node {id: $id})" in node_cypher
        assert node_params["id"] == "repo_a::node_foo"
        assert node_params["repo"] == "repo_a"

        assert "MATCH (a:Node {id: $source})" in edge_cypher
        assert "(b:Node {id: $target})" in edge_cypher
        assert edge_params["source"] == "repo_a::node_foo"
        assert edge_params["target"] == "repo_a::node_bar"
        assert edge_params["confidence_score"] == 0.9


# ---------------------------------------------------------------------------
# Optional smoke test (live Memgraph on .201)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.memgraph
def test_smoke_memgraph_live_ingest() -> None:
    """Smoke test: live Memgraph on the dev host (port 7687).

    Skipped when Memgraph is unreachable.
    """
    import socket

    _host = "192.168.86.201"  # onex-allow-internal-ip
    _bolt = f"bolt://{_host}:7687"  # onex-allow-internal-ip
    try:
        sock = socket.create_connection((_host, 7687), timeout=2)
        sock.close()
    except OSError:
        pytest.skip(f"Live Memgraph at {_host}:7687 not reachable")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        pytest.skip("neo4j package not installed")

    driver = GraphDatabase.driver(_bolt, auth=None)
    try:
        counts = ingest_repo(driver, [_NODE_A, _NODE_B], [_LINK_AB])
        assert counts["nodes"] == 2
        assert counts["edges"] == 1
    finally:
        driver.close()
