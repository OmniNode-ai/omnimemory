> **Navigation**: [Home](../INDEX.md) > Migrations > Market Migration Boundary

# Market Migration Boundary

**Owner:** `omnimemory`
**Last verified:** 2026-04-29
**Verification:** `pyproject.toml` entry-points + node inventory in `src/omnimemory/nodes/`
**Source plan:** `omni_home/docs/plans/2026-04-10-omnimemory-to-omnimarket-migration.md` (OMN-8295 epic)

---

## Migration Goal

Migrate all 17 runnable ONEX nodes from `omnimemory` into `omnimarket`, leaving `omnimemory` as a **pure primitives package**: models, protocols, and storage adapters.

After migration, nodes auto-wire via the ONEX runtime's `importlib.metadata` discovery using the `onex.nodes` entry point — exactly like every other omnimarket node. `omnimemory` retains the `onex.domain_plugins` entry point (`PluginMemory`) for runtime plugin lifecycle management.

**Epic:** OMN-8295

---

## What Stays in omnimemory

These components are the permanent core of omnimemory. They do not move to omnimarket.

### Protocols

All protocol interfaces remain in `src/omnimemory/protocols/`:

- `ProtocolEmbedding`, `ProtocolEmbeddingProvider`
- `ProtocolCrawlStateRepository`
- `ProtocolHandlerIntent`
- `ProtocolIntentGraphAdapter`
- `ProtocolSecretsProvider`
- `base_protocols.py`, `data_models.py`, `error_models.py`

**Why:** Handlers in omnimarket depend on these protocols for dependency injection. Moving protocols to omnimarket would create a circular import (`omnimarket` depends on `omnimemory` protocols — if protocols moved to omnimarket, `omnimemory` adapters would need to import from omnimarket, reversing the dependency direction).

### Models

All domain models remain in `src/omnimemory/models/`:

- Crawl models (`ModelCrawlTickCommand`, `ModelDocumentDiscoveredEvent`, etc.)
- Memory models (`ModelMemoryItem`, `ModelMemoryQuery`, `ModelSimilarityResult`, etc.)
- Persona models (`ModelPersonaSignal`, `ModelUserPersonaV1`, etc.)
- Foundation, intelligence, scoring, subscription, events, service, config models

**Why:** Models are the shared language between omnimarket handlers and omnibase_infra adapters. Keeping them in omnimemory means both parties can depend on `omninode-memory` without circularity.

### Adapters

All concrete protocol implementations remain in omnimemory (per-node `adapters/` subdirectories):

- Qdrant adapter (`handlers/adapters/`)
- Filesystem adapter (`handlers/adapters/`, `nodes/node_memory_storage_effect/adapters/`)
- Memgraph adapter (`nodes/node_intent_storage_effect/adapters/`)
- Archive adapters (`nodes/node_memory_lifecycle_orchestrator/adapters/`)
- Persona storage adapters (`nodes/node_persona_storage_effect/adapters/`)

**Why:** Adapters are concrete infrastructure implementations. They belong with the domain package that owns the storage contract, not in omnimarket's portable workflow layer.

### Runtime Infrastructure

- `bootstrap.py`, `settings.py`, `secrets.py`
- `runtime/` — plugin entry point, DI container wiring (`PluginMemory`)
- `audit/` — I/O audit tooling
- `enums/`, `errors/`, `tools/`, `utils/`

### Storage Services

Qdrant, Memgraph, Valkey, and Kreuzberg remain omnimemory-owned. See [Memory Data Ownership](../architecture/MEMORY_DATA_OWNERSHIP.md).

---

## What Moves to omnimarket

All 17 runnable nodes (those with a `contract.yaml`) move to `omnimarket/src/omnimarket/nodes/`:

| Node | Type | Domain |
|------|------|--------|
| `node_filesystem_crawler_effect` | EFFECT | Crawl |
| `node_kreuzberg_parse_effect` | EFFECT | Crawl |
| `node_memory_storage_effect` | EFFECT | Memory |
| `node_memory_retrieval_effect` | EFFECT | Memory |
| `node_memory_lifecycle_orchestrator` | ORCHESTRATOR | Memory |
| `node_semantic_analyzer_compute` | COMPUTE | Intelligence |
| `node_similarity_compute` | COMPUTE | Intelligence |
| `node_agent_coordinator_orchestrator` | ORCHESTRATOR | Coordination |
| `node_agent_learning_retrieval_effect` | EFFECT | Learning |
| `node_intent_event_consumer_effect` | EFFECT | Intent |
| `node_intent_query_effect` | EFFECT | Intent |
| `node_intent_storage_effect` | EFFECT | Intent |
| `node_navigation_history_reducer` | REDUCER | Navigation |
| `node_persona_builder_compute` | COMPUTE | Persona |
| `node_persona_lifecycle_orchestrator` | ORCHESTRATOR | Persona |
| `node_persona_retrieval_effect` | EFFECT | Persona |
| `node_persona_storage_effect` | EFFECT | Persona |

**What moves with each node:** handler implementations, node-local models, `contract.yaml`, clients, registry, utils, validators.

**What stays in omnimemory:** each node's `adapters/` subdirectory (concrete protocol implementations injected at runtime via DI).

**Stub nodes (no contract.yaml) — not migrated:**
- `node_memory_consolidator_reducer` — `__init__.py` skeleton only
- `node_statistics_reducer` — `__init__.py` skeleton only

---

## Migration Waves

| Wave | Nodes | Linear |
|------|-------|--------|
| 1 | `node_similarity_compute`, `node_persona_builder_compute`, `node_semantic_analyzer_compute` | OMN-8297 |
| 2 | `node_memory_storage_effect`, `node_memory_retrieval_effect`, `node_persona_storage_effect`, `node_persona_retrieval_effect`, `node_agent_learning_retrieval_effect` | OMN-8298 |
| 3 | `node_filesystem_crawler_effect`, `node_kreuzberg_parse_effect` | OMN-8299 |
| 4 | `node_intent_storage_effect`, `node_intent_query_effect`, `node_intent_event_consumer_effect` | OMN-8300 |
| 5 | `node_memory_lifecycle_orchestrator`, `node_navigation_history_reducer`, `node_persona_lifecycle_orchestrator`, `node_agent_coordinator_orchestrator` | OMN-8301 |

Pre-work: add `omninode-memory` dependency to omnimarket (OMN-8296, blocks Wave 1).

---

## Import Path Changes After Migration

| Before migration | After migration |
|-----------------|----------------|
| `omnimemory.nodes.<node>.*` | `omnimarket.nodes.<node>.*` |
| `omnimemory.protocols.*` | `omnimemory.protocols.*` (unchanged) |
| `omnimemory.models.*` | `omnimemory.models.*` (unchanged) |

`contract.yaml` schema refs must update:
- `omnimemory.nodes.<node>.models.*` → `omnimarket.nodes.<node>.models.*`
- `omnimemory.nodes.<node>.handlers.*` → `omnimarket.nodes.<node>.handlers.*`

---

## Circular Dependency Guard

| Rule | Rationale |
|------|-----------|
| `omnimarket` MAY import `omnimemory` (for protocols, models, adapters) | Correct direction: market node handlers depend on domain primitives |
| `omnimemory` MUST NOT import `omnimarket` | Would create circular dependency |
| Adapters stay in omnimemory and are injected at runtime via DI | Preserves the protocol-adapter-handler separation |

---

## Post-Migration omnimemory State

After all 17 nodes move:

- `nodes/` retains only adapter subdirectories and stub nodes
- The `onex.node_package` entry point (if present) is removed from `pyproject.toml`
- The `onex.domain_plugins` entry point (`PluginMemory`) remains
- Package description: "OmniNode memory primitives — models, protocols, and storage adapters for Qdrant, Memgraph, and PostgreSQL backends."

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Runtime consumers pick up stale entry points from old omnimemory | After migration, remove `onex.node_package` from omnimemory and re-deploy via Kafka rebuild command |
| Schema refs in `contract.yaml` not updated | Update all `schema_ref: omnimemory.nodes.*` to `omnimarket.nodes.*` per node |
| Adapter injection breaks post-migration | Verify via golden chain tests per node; DI container (omnimemory bootstrap) must inject adapters into omnimarket handler constructors |
| `omnimarket/pyproject.toml` does not yet depend on omnimemory | Add `omninode-memory` to omnimarket dependencies before Wave 1 (OMN-8296) |
