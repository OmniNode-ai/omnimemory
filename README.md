# OmniMemory

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ONEX 4.0](https://img.shields.io/badge/ONEX-4.0-purple.svg)](https://github.com/OmniNode-ai/omnibase_core)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Memory persistence, recall, and semantic retrieval for the OmniNode platform.** OmniMemory provides ONEX (OmniNode eXecution)-compliant nodes and handlers for storing agent context, indexing embeddings, querying intent graphs, and managing the full memory lifecycle across distributed omni agents.

## Where This Fits

OmniMemory holds three distinct ownership roles in the ONEX platform:

1. **Domain owner for memory persistence and retrieval semantics.** OmniMemory defines the authoritative models, protocols, and lifecycle rules for all memory operations: what a memory item is, how it is stored, how it is retrieved, and how it ages. These primitives do not move to other packages.

2. **Runtime plugin owner for memory nodes.** `PluginMemory` (registered as the `onex.domain_plugins` entry point) is the kernel lifecycle hook for the memory domain. It wires message types, verifies handler imports, initializes the dispatch engine, and subscribes to Kafka topics at runtime.

3. **Storage integration owner for Qdrant, Memgraph, Valkey, and Kreuzberg adapters.** The concrete adapter implementations for all memory-layer storage backends live in `omnimemory`. These adapters implement the domain protocols and are injected at runtime via the DI container.

**What is migrating to omnimarket:** The 15 runnable ONEX node handler implementations (those with `contract.yaml`) are moving to `omnimarket`. Protocols, models, adapters, and the runtime plugin stay here. See [docs/migrations/MARKET_MIGRATION_BOUNDARY.md](docs/migrations/MARKET_MIGRATION_BOUNDARY.md).

---

## What This Repo Owns

- **Domain models** — all memory, crawl, persona, intent, and intelligence Pydantic models in `src/omnimemory/models/`
- **Protocol interfaces** — `ProtocolEmbeddingClient`, `ProtocolEmbeddingProvider`, `ProtocolIntentGraphAdapter`, `ProtocolSecretsProvider`, and all base protocols in `src/omnimemory/protocols/`
- **Storage adapters** — Qdrant, Memgraph, Valkey, and filesystem adapter implementations in `handlers/adapters/` and `nodes/*/adapters/`
- **Runtime plugin** — `PluginMemory` in `src/omnimemory/runtime/` registered as `onex.domain_plugins`
- **Memory-layer data services** — Qdrant, Memgraph, Valkey, Kreuzberg (owned via `docker-compose.yml`)
- **Node handlers** — 15 runnable nodes in `src/omnimemory/nodes/` (migrating to omnimarket)

## What This Repo Does Not Own

| Resource | Owner |
|----------|-------|
| Kafka / Redpanda (platform event bus) | [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra) |
| PostgreSQL (platform relational DB) | [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra) |
| ONEX kernel, node execution, contracts | [`omnibase_core`](https://github.com/OmniNode-ai/omnibase_core) |
| Protocol interfaces for platform boundaries | [`omnibase_spi`](https://github.com/OmniNode-ai/omnibase_spi) |
| Portable workflow packages and node runtime (post-migration) | [`omnimarket`](https://github.com/OmniNode-ai/omnimarket) |
| Dashboard projections and read-model surfaces | [`omnidash`](https://github.com/OmniNode-ai/omnidash) |

---

## Architecture

Follows the [ONEX Four-Node Architecture](https://github.com/OmniNode-ai/omnibase_core/blob/main/docs/architecture/ONEX_FOUR_NODE_ARCHITECTURE.md) (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR) applied to memory operations.

### Node inventory

- **Effect nodes** — `memory_storage_effect`, `memory_retrieval_effect`, `agent_learning_retrieval_effect`, `intent_storage_effect`, `intent_query_effect`, `intent_event_consumer_effect`, `filesystem_crawler_effect`, `kreuzberg_parse_effect`, `persona_storage_effect`
- **Compute nodes** — `semantic_analyzer_compute`, `similarity_compute`, `persona_builder_compute`
- **Reducer nodes** — `navigation_history_reducer`, `memory_consolidator_reducer` (stub — no contract.yaml)
- **Orchestrator nodes** — `memory_lifecycle_orchestrator`, `agent_coordinator_orchestrator`

> `node_persona_lifecycle_orchestrator` and `node_persona_retrieval_effect` were decommissioned in an earlier cleanup pass; they never had a `contract.yaml` and are no longer present in the repository.

> _Verified against `src/omnimemory/nodes/` on 2026-06-21: 16 node directories, 15 with `contract.yaml`; `node_memory_consolidator_reducer` is the contract-less stub._

### Memory evolution (planned phases)

The architecture plan (`omni_home/docs/plans/2026-04-07-plan-omnimemory-architecture.md`) describes five enhancement phases: surprise gating on the write path, activation decay for retrieval ranking, memory cube isolation for multi-agent boundaries, Hebbian association strengthening, and hybrid vector+FTS search. These are planned, not yet implemented.

---

## Infrastructure Ownership

OmniMemory's `docker-compose.yml` owns the **memory-layer data services**:

| Service | Container | Default Port | Purpose |
|---------|-----------|--------------|---------|
| Qdrant | `omnimemory-qdrant` | 6333 (HTTP), 6334 (gRPC) | Vector database for semantic memory |
| Memgraph | `omnimemory-memgraph` | 7687 (Bolt), 7444 (HTTP) | Graph database for relationship/intent queries |
| Valkey | `omnimemory-valkey` | 6379 | In-memory cache and session storage |
| Kreuzberg | `omnimemory-kreuzberg-parser` | 8090 | Document text extraction service |

**Not owned here** — these services are managed by other repositories:

| Service | Owner Repository | Why |
|---------|-----------------|-----|
| Kafka / Redpanda | [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra) | Platform-wide event bus, shared by all services |
| PostgreSQL | [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra) | Platform-wide relational database, shared by all services |

See [docs/architecture/MEMORY_DATA_OWNERSHIP.md](docs/architecture/MEMORY_DATA_OWNERSHIP.md) for detailed service boundaries and adapter ownership.

---

## Quick Start

### Memory services only

```bash
git clone https://github.com/OmniNode-ai/omnimemory.git
cd omnimemory

# Start platform infra first (Kafka + PostgreSQL — owned by omnibase_infra)
infra-up

# Start memory data services
docker compose up -d

# Verify all services are healthy
docker compose ps
```

Default service ports (all configurable via `.env`):
- Qdrant REST: `localhost:6333`
- Memgraph Bolt: `localhost:7687`
- Valkey: `localhost:6379`
- Kreuzberg parser: `localhost:8090`

See [docs/runbooks/STARTING_MEMORY_SERVICES.md](docs/runbooks/STARTING_MEMORY_SERVICES.md) for the full startup runbook including health checks and troubleshooting.

### Install and run tests

```bash
uv sync --group dev
uv run pytest tests/ -m unit
```

For configuration options see [docs/environment_variables.md](docs/environment_variables.md).

### Minimal usage example

```python
import asyncio
from uuid import uuid4

from omnibase_core.container import ModelONEXContainer
from omnimemory.handlers.adapters.models import ModelIntentClassificationOutput
from omnimemory.handlers.handler_intent import HandlerIntent


async def main() -> None:
    container = ModelONEXContainer()
    handler = HandlerIntent(container)

    await handler.initialize(connection_uri="bolt://localhost:7687")

    result = await handler.store_intent(
        session_id="session_123",
        intent_data=ModelIntentClassificationOutput(
            intent_category="debugging",
            confidence=0.92,
            keywords=["error", "traceback"],
        ),
        correlation_id=str(uuid4()),
    )

    query_result = await handler.query_session(
        session_id="session_123",
        min_confidence=0.5,
    )

    await handler.shutdown()


asyncio.run(main())
```

---

## Directory Structure

```text
src/omnimemory/
├── audit/              # I/O audit logging
├── enums/              # Domain enumerations (memory types, operation types, lifecycle states)
├── errors/             # Structured error types
├── handlers/           # HandlerIntent, HandlerSubscription + adapters
├── models/             # Pydantic models (memory, crawl, persona, intent, intelligence)
├── nodes/              # EFFECT, COMPUTE, REDUCER, ORCHESTRATOR node implementations
│   └── <node>/
│       ├── adapters/   # Stays in omnimemory (protocol implementations)
│       └── handlers/   # Migrating to omnimarket
├── adapters/           # Shared utilities with adapter_* prefix (PII detection, retry, health, metrics)
├── protocols/          # Protocol interfaces (embedding, intent graph, secrets)
├── runtime/            # PluginMemory, DI container wiring, dispatch, introspection
└── tools/              # Contract linter and validators
```

---

## Development and Test Commands

```bash
# Install all dependencies
uv sync --group dev

# Format and lint
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/

# Type checking
uv run mypy src/omnimemory/ --strict

# Run all tests
uv run pytest tests/ -v

# Unit tests only (no external services required)
uv run pytest tests/ -m unit

# Pre-commit validation
pre-commit run --all-files
```

---

## Migration Status

Nodes are migrating to omnimarket. The migration preserves the protocol-adapter-handler split: handlers move, adapters stay.

See [docs/migrations/MARKET_MIGRATION_BOUNDARY.md](docs/migrations/MARKET_MIGRATION_BOUNDARY.md).

---

## Documentation

Full documentation index: **[docs/INDEX.md](docs/INDEX.md)**

Key docs:
- [Architecture: ONEX Four-Node Pattern](docs/architecture/ONEX_FOUR_NODE_ARCHITECTURE.md)
- [Architecture: Memory Data Ownership](docs/architecture/MEMORY_DATA_OWNERSHIP.md)
- [Migration: Market Migration Boundary](docs/migrations/MARKET_MIGRATION_BOUNDARY.md)
- [Runbook: Starting Memory Services](docs/runbooks/STARTING_MEMORY_SERVICES.md)
- [Reference: Environment Variables](docs/environment_variables.md)
- [Runtime: Plugin System](docs/runtime/RUNTIME_PLUGINS.md)

---

## Security, Contributing, and License

- [SECURITY.md](SECURITY.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [LICENSE](LICENSE)

Open an issue or email contact@omninode.ai.
