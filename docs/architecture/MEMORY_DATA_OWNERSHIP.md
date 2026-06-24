> **Navigation**: [Home](../INDEX.md) > Architecture > Memory Data Ownership

# Memory Data Ownership

**Owner:** `omnimemory`
**Last verified:** 2026-06-21 — node count (15 with `contract.yaml`), protocol names, and Valkey access path verified against `src/omnimemory/` source.
**Verification:** `find src/omnimemory/nodes -name contract.yaml`, `src/omnimemory/protocols/__init__.py`, `docker-compose.yml`, `pyproject.toml` entry-point check
**Source plans:**
- `omni_home/docs/plans/2026-04-07-plan-omnimemory-architecture.md`
- `omni_home/docs/plans/2026-04-10-omnimemory-to-omnimarket-migration.md`

---

## Summary

OmniMemory owns the **memory-layer data services**: Qdrant, Memgraph, Valkey, and Kreuzberg. These services form the storage substrate for all memory domain operations. Platform-wide infrastructure (Kafka/Redpanda, PostgreSQL) is owned by `omnibase_infra`.

---

## Services Owned by omnimemory

| Service | Container | Default Port | Purpose | Ownership rationale |
|---------|-----------|--------------|---------|---------------------|
| Qdrant | `omnimemory-qdrant` | 6333 (HTTP), 6334 (gRPC) | Vector database for semantic memory, embedding search, and agent learning retrieval | Memory-domain-specific; no other platform component requires a vector store |
| Memgraph | `omnimemory-memgraph` | 7687 (Bolt), 7444 (HTTP) | Graph database for relationship and intent queries | Memory-domain-specific; used exclusively by intent storage and agent coordination nodes |
| Valkey | `omnimemory-valkey` | 6379 | In-memory cache and session storage for memory operations | Memory-domain cache; scoped to memory retrieval latency optimization |
| Kreuzberg | `omnimemory-kreuzberg-parser` | 8090 | Document text extraction service for filesystem crawl pipeline | Required exclusively by the crawl pipeline; not shared across platform |

All four services are declared in `docker-compose.yml` at the omnimemory repo root.

---

## Services NOT Owned by omnimemory

| Service | Owner | Why it is not omnimemory's |
|---------|-------|---------------------------|
| Kafka / Redpanda | [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra) | Platform-wide event bus shared by all ONEX services |
| PostgreSQL | [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra) | Platform-wide relational database; used for `agent_learnings`, metadata, and lifecycle state, but the schema and host are infra-owned |

Memory nodes write memory metadata (lifecycle state, promotion tiers, association weights) to the infra-owned PostgreSQL instance. The schema migrations for memory tables (`agent_learnings`, `memory_cubes`, `memory_associations`) are authored in `omnibase_infra` but represent memory-domain data.

**To start infra-owned services:**
```bash
# From any terminal with ONEX shell functions loaded
infra-up   # starts Kafka + PostgreSQL
```

**To start memory-owned services:**
```bash
# From omnimemory repo root
docker compose up -d
```

---

## Qdrant

**Role:** Primary vector store for all memory search operations.

**Collections:**
- `document_memory` — document chunks indexed during filesystem crawl
- `agent_learnings_error` — error-pattern learnings from resolved agent sessions
- `agent_learnings_context` — task-context learnings from completed agent sessions

**Embedding dimensions:** 1024-dim vectors from the platform embedding service (see `LLM_EMBEDDING_URL` in `~/.omnibase/.env`).

**Access pattern:** All reads and writes go through `node_memory_retrieval_effect` and `node_memory_storage_effect` handlers. No component accesses Qdrant directly outside of these nodes.

**Migration note:** After the omnimarket node migration, Qdrant access will route through omnimarket-hosted handlers. The Qdrant service itself remains omnimemory-owned.

---

## Memgraph

**Role:** Graph database for intent relationship queries and agent coordination.

**Data model:** Intent nodes, session nodes, and relationship edges. Enables graph-traversal queries that flat SQL cannot express efficiently.

**Optional:** Memgraph is not required for baseline memory operation. Start with `infra-up-memory` (the runtime bundle that includes Memgraph) or with the omnimemory `docker compose up -d`.

**Access pattern:** Accessed exclusively by `node_intent_storage_effect` and `node_intent_query_effect` via the `ProtocolIntentGraphAdapter` protocol and the concrete Memgraph adapter in `nodes/node_intent_storage_effect/adapters/`.

---

## Valkey

**Role:** In-memory cache for session storage and retrieval latency optimization.

**Usage:** Session state caching and short-lived ephemeral storage. Data is not durably persisted; Valkey is complementary to Qdrant and PostgreSQL.

**Access pattern:** Accessed through handler-level caching utilities. Not accessed directly from outside omnimemory.

---

## Kreuzberg

**Role:** Document text extraction service for the filesystem crawl pipeline.

**Usage:** `node_kreuzberg_parse_effect` calls Kreuzberg over HTTP to extract structured text from documents before embedding. Supports PDF, DOCX, HTML, and other formats.

**Access pattern:** `node_kreuzberg_parse_effect` is the only caller. Kreuzberg is stateless; requests are isolated per document.

**Migration note:** After the omnimarket node migration, `node_kreuzberg_parse_effect` will live in omnimarket. The Kreuzberg service container remains omnimemory-owned.

---

## What Moves vs. What Stays

See [Market Migration Boundary](../migrations/MARKET_MIGRATION_BOUNDARY.md) for the full boundary definition.

**Short version:**

| Layer | Stays in omnimemory | Moves to omnimarket |
|-------|--------------------|--------------------|
| Storage services | Qdrant, Memgraph, Valkey, Kreuzberg | — |
| Node handlers | — | All 15 runnable nodes (those with `contract.yaml`) |
| Protocols | All (`ProtocolEmbeddingClient`, `ProtocolEmbeddingProvider`, `ProtocolIntentGraphAdapter`, etc.) | — |
| Models | All domain models | — |
| Adapters | All concrete adapter implementations | — |
| Runtime plugin | PluginMemory entry point | — |
