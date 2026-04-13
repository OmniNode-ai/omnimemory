#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Ingest graphify graph.json files into Memgraph via Bolt protocol.

Graph.json schema:
  nodes: list of {id, label, repo, source_file, node_type, ...}
  links: list of {source, target, relation, confidence_score, ...}

Usage:
    uv run python scripts/graphify_to_memgraph.py \\
        --graph-dir /path/to/graphify-graphs/ \\
        --bolt-uri bolt://localhost:7687

    # --graph-dir defaults to $OMNI_HOME/.onex_state/graphify-graphs/ (or
    # path relative to this script's repo root if OMNI_HOME is not set).

    # Or via env var (e.g. when Memgraph runs on a remote host):
    MEMGRAPH_URL=bolt://192.168.1.201:7687 \\
    uv run python scripts/graphify_to_memgraph.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# scripts/ is one level below the repo root (omnimemory/)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_graph_dir() -> Path:
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        return Path(omni_home) / ".onex_state" / "graphify-graphs"
    return _REPO_ROOT.parent / ".onex_state" / "graphify-graphs"


def parse_graphify_json(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a graph.json file and return (nodes, edges).

    Nodes are under the 'nodes' key; edges are under the 'links' key.
    """
    with open(path) as f:
        data = json.load(f)
    nodes: list[dict[str, Any]] = data.get("nodes", [])
    links: list[dict[str, Any]] = data.get("links", [])
    return nodes, links


def build_node_cypher(node: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a (cypher_statement, params) pair for MERGEing a node.

    Identity key is 'id' — unique per graphify output across repos.
    """
    cypher = (
        "MERGE (n:Node {id: $id}) "
        "SET n.label = $label, n.repo = $repo, n.source_file = $source_file, "
        "n.node_type = $node_type, n.community = $community"
    )
    params: dict[str, Any] = {
        "id": node["id"],
        "label": node.get("label", ""),
        "repo": node.get("repo", ""),
        "source_file": node.get("source_file", ""),
        "node_type": node.get("node_type", ""),
        "community": node.get("community", ""),
    }
    return cypher, params


def build_edge_cypher(edge: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a (cypher_statement, params) pair for MERGEing an edge."""
    cypher = (
        "MATCH (a:Node {id: $source}), (b:Node {id: $target}) "
        "MERGE (a)-[r:RELATES {relation: $relation}]->(b) "
        "SET r.confidence_score = $confidence_score"
    )
    params: dict[str, Any] = {
        "source": edge["source"],
        "target": edge["target"],
        "relation": edge.get("relation", ""),
        "confidence_score": edge.get("confidence_score", 0.0),
    }
    return cypher, params


def _run_batch(session: Any, statements: list[tuple[str, dict[str, Any]]]) -> None:
    with session.begin_transaction() as tx:
        for cypher, params in statements:
            tx.run(cypher, params)
        tx.commit()


def ingest_repo(
    driver: Any, nodes: list[dict[str, Any]], links: list[dict[str, Any]]
) -> dict[str, int]:
    """Ingest a single repo's nodes and edges into Memgraph.

    Returns counts of nodes and edges ingested.
    """
    with driver.session() as session:
        # Ingest nodes in batches
        node_count = 0
        batch: list[tuple[str, dict[str, Any]]] = []
        for node in nodes:
            batch.append(build_node_cypher(node))
            if len(batch) >= BATCH_SIZE:
                _run_batch(session, batch)
                node_count += len(batch)
                batch = []
        if batch:
            _run_batch(session, batch)
            node_count += len(batch)

        # Ingest edges in batches
        edge_count = 0
        batch = []
        for link in links:
            batch.append(build_edge_cypher(link))
            if len(batch) >= BATCH_SIZE:
                _run_batch(session, batch)
                edge_count += len(batch)
                batch = []
        if batch:
            _run_batch(session, batch)
            edge_count += len(batch)

    return {"nodes": node_count, "edges": edge_count}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Ingest graphify graph.json files into Memgraph"
    )
    parser.add_argument(
        "--graph-dir",
        default=None,
        help=(
            "Directory containing per-repo graph.json files. "
            "Defaults to $OMNI_HOME/.onex_state/graphify-graphs/ "
            "or <repo-root>/../.onex_state/graphify-graphs/ if OMNI_HOME is unset."
        ),
    )
    parser.add_argument(
        "--bolt-uri",
        default=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        help="Memgraph Bolt URI (env: MEMGRAPH_URL, default: bolt://localhost:7687)",
    )
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir) if args.graph_dir is not None else _default_graph_dir()
    if not graph_dir.is_dir():
        logger.error("graph-dir does not exist: %s", graph_dir)
        sys.exit(1)

    graph_files = sorted(graph_dir.rglob("graph.json"))
    if not graph_files:
        logger.error("No graph.json files found under %s", graph_dir)
        sys.exit(1)

    try:
        from neo4j import GraphDatabase  # type: ignore[import-untyped]
    except ImportError:
        logger.error("neo4j package not installed — run: uv sync --extra graph")
        sys.exit(1)

    correlation_id = str(uuid.uuid4())
    logger.info("Starting ingestion run correlation_id=%s", correlation_id)

    driver = GraphDatabase.driver(args.bolt_uri, auth=None)
    try:
        total_nodes = 0
        total_edges = 0
        for graph_file in graph_files:
            repo_name = graph_file.parent.name
            logger.info("Ingesting %s ...", repo_name)
            nodes, links = parse_graphify_json(graph_file)
            counts = ingest_repo(driver, nodes, links)
            logger.info(
                "  %s: %d nodes, %d edges", repo_name, counts["nodes"], counts["edges"]
            )
            total_nodes += counts["nodes"]
            total_edges += counts["edges"]
        logger.info(
            "Done. correlation_id=%s total_nodes=%d total_edges=%d repos=%d",
            correlation_id,
            total_nodes,
            total_edges,
            len(graph_files),
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
