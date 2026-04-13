#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Embed graphify graph.json node descriptions into Qdrant.

For each node, builds a rich embedding text from label, source_file, community,
repo, node_type, relations, protocols, and contract refs, then embeds via the
gte-Qwen2-1.5B-instruct model (default; configurable via --embedding-model) and upserts
into a Qdrant collection.

Usage:
    uv run python scripts/graphify_to_qdrant.py \\
        --graph-dir /path/to/graphify-graphs/ \\
        --qdrant-url http://localhost:6333 \\
        --embedding-url http://localhost:8100 \\
        --collection onex-graphify

    # Or via env vars:
    QDRANT_URL=http://localhost:6333 \\
    EMBEDDING_MODEL_URL=http://localhost:8100 \\
    uv run python scripts/graphify_to_qdrant.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100
EMBED_RETRY_ATTEMPTS = 3
EMBED_BACKOFF_SECONDS = [1, 2, 4]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_graph_dir() -> Path:
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        return Path(omni_home) / ".onex_state" / "graphify-graphs"
    return _REPO_ROOT.parent / ".onex_state" / "graphify-graphs"


_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def screen_or_redact_pii(text: str) -> str:
    """Strip emails, phone numbers, and SSN patterns from text before embedding."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _SSN_RE.sub("[SSN]", text)
    return text


def build_embedding_text(node: dict[str, Any]) -> str:
    """Build a rich text representation of a node for embedding.

    Combines label, source_file, community, repo, node_type, relations,
    protocols, and contract_refs into a single string.
    """
    parts: list[str] = []

    label = node.get("label", "")
    if label:
        parts.append(f"label: {label}")

    source_file = node.get("source_file", "")
    if source_file:
        parts.append(f"source: {source_file}")

    community = node.get("community", "")
    if community:
        parts.append(f"community: {community}")

    repo = node.get("repo", node.get("id", "").split("::")[0] if "::" in node.get("id", "") else "")
    if repo:
        parts.append(f"repo: {repo}")

    node_type = node.get("node_type", "")
    if node_type:
        parts.append(f"type: {node_type}")

    relations: list[str] = node.get("relations", [])
    if relations:
        parts.append("relations: " + ", ".join(relations))

    protocols: list[str] = node.get("protocols", [])
    if protocols:
        parts.append("protocols: " + ", ".join(protocols))

    contract_refs: list[str] = node.get("contract_refs", [])
    if contract_refs:
        parts.append("contracts: " + ", ".join(contract_refs))

    return " | ".join(parts)


def chunk_nodes(
    nodes: list[dict[str, Any]], batch_size: int = UPSERT_BATCH_SIZE
) -> Generator[list[dict[str, Any]], None, None]:
    """Yield successive batch_size-sized chunks from nodes."""
    for i in range(0, len(nodes), batch_size):
        yield nodes[i : i + batch_size]


def embed_batch(
    texts: list[str],
    embedding_url: str,
    model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
) -> list[list[float]]:
    """Call the embedding API with retry/backoff. Returns list of embedding vectors."""
    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed — run: uv sync --extra graph")
        sys.exit(1)

    payload = {"model": model, "input": texts}
    last_exc: Exception | None = None

    for attempt, backoff in enumerate(EMBED_BACKOFF_SECONDS):
        try:
            resp = httpx.post(
                f"{embedding_url}/v1/embeddings",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding response size mismatch: expected {len(texts)}, got {len(embeddings)}"
                )
            return embeddings
        except Exception as exc:
            last_exc = exc
            if attempt < EMBED_RETRY_ATTEMPTS - 1:
                logger.warning(
                    "Embedding attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1,
                    EMBED_RETRY_ATTEMPTS,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.warning(
                    "Embedding exhausted %d attempts for batch of %d texts: %s — skipping",
                    EMBED_RETRY_ATTEMPTS,
                    len(texts),
                    exc,
                )

    raise RuntimeError(f"Embedding failed after {EMBED_RETRY_ATTEMPTS} attempts") from last_exc


def _stable_point_id(node_id: str) -> int:
    """Return a stable, deterministic Qdrant point ID for a node ID string.

    Uses blake2b so the same node_id produces the same integer across
    interpreter processes (unlike Python's salted built-in hash).
    """
    digest = hashlib.blake2b(node_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**63)


def upsert_to_qdrant(
    client: Any,
    collection: str,
    nodes: list[dict[str, Any]],
    vectors: list[list[float]],
) -> int:
    """Upsert node+vector pairs into Qdrant. Returns count upserted."""
    from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

    points = [
        PointStruct(
            id=_stable_point_id(node["id"]),
            vector=vector,
            payload={
                "id": node["id"],
                "label": screen_or_redact_pii(node.get("label", "")),
                "repo": node.get("repo", ""),
                "source_file": screen_or_redact_pii(node.get("source_file", "")),
                "node_type": node.get("node_type", ""),
                "community": node.get("community", ""),
            },
        )
        for node, vector in zip(nodes, vectors, strict=True)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def _ensure_collection(client: Any, collection: str, vector_size: int) -> None:
    from qdrant_client.models import (  # type: ignore[import-untyped]
        Distance,
        VectorParams,
    )

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d)", collection, vector_size)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Embed graphify graph.json nodes into Qdrant"
    )
    parser.add_argument(
        "--graph-dir",
        default=None,
        help=(
            "Directory containing per-repo graph.json files. "
            "Defaults to $OMNI_HOME/.onex_state/graphify-graphs/"
        ),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant base URL (env: QDRANT_URL)",
    )
    _embedding_url = os.environ.get("EMBEDDING_MODEL_URL")
    parser.add_argument(
        "--embedding-url",
        default=_embedding_url,
        required=_embedding_url is None,
        help="Embedding model base URL (env: EMBEDDING_MODEL_URL)",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "onex-graphify"),
        help="Qdrant collection name (env: QDRANT_COLLECTION)",
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
        from qdrant_client import QdrantClient  # type: ignore[import-untyped]
    except ImportError:
        logger.error("qdrant-client not installed — run: uv sync --extra graph")
        sys.exit(1)

    correlation_id = str(uuid.uuid4())
    logger.info("Starting embedding run correlation_id=%s", correlation_id)

    qdrant = QdrantClient(url=args.qdrant_url)
    collection_created = False
    total_upserted = 0

    for graph_file in graph_files:
        repo_name = graph_file.parent.name
        with open(graph_file) as f:
            data = json.load(f)
        nodes: list[dict[str, Any]] = data.get("nodes", [])
        if not nodes:
            logger.info("  %s: no nodes, skipping", repo_name)
            continue

        logger.info("  %s: embedding %d nodes ...", repo_name, len(nodes))
        repo_upserted = 0

        for batch in chunk_nodes(nodes, batch_size=UPSERT_BATCH_SIZE):
            texts = [screen_or_redact_pii(build_embedding_text(n)) for n in batch]
            try:
                vectors = embed_batch(texts, args.embedding_url)
            except RuntimeError as exc:
                logger.warning("Skipping batch in %s: %s", repo_name, exc)
                continue

            if not collection_created:
                _ensure_collection(qdrant, args.collection, len(vectors[0]))
                collection_created = True

            repo_upserted += upsert_to_qdrant(qdrant, args.collection, batch, vectors)

        logger.info("  %s: upserted %d points", repo_name, repo_upserted)
        total_upserted += repo_upserted

    logger.info(
        "Done. correlation_id=%s total_upserted=%d repos=%d",
        correlation_id,
        total_upserted,
        len(graph_files),
    )


if __name__ == "__main__":
    main()
