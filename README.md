# OmniMemory

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ONEX 4.0](https://img.shields.io/badge/ONEX-4.0-purple.svg)](https://github.com/OmniNode-ai/omnibase_core)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Memory persistence, recall, and semantic retrieval for the OmniNode platform.** OmniMemory provides ONEX-compliant nodes and handlers for storing agent context, indexing embeddings, querying intent graphs, and managing the full memory lifecycle across distributed omni agents.

## Four-Node Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     EFFECT      │───▶│     COMPUTE     │───▶│     REDUCER     │───▶│  ORCHESTRATOR   │
│  (store/fetch)  │    │ (embed/analyze) │    │  (consolidate)  │    │  (coordinate)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

- **EFFECT**: Memory storage, retrieval, and intent query against external backends
- **COMPUTE**: Semantic analysis, similarity scoring, embedding generation
- **REDUCER**: Memory consolidation, statistics aggregation, lifecycle state management
- **ORCHESTRATOR**: Agent coordination, multi-step memory lifecycle workflows

## What This Repo Provides

- **Memory nodes** — `memory_storage_effect`, `memory_retrieval_effect`, `intent_storage_effect`, `intent_query_effect`, `intent_event_consumer_effect`
- **Compute nodes** — `semantic_analyzer_compute`, `similarity_compute`
- **Reducer nodes** — `memory_consolidator_reducer`, `statistics_reducer`
- **Orchestrator nodes** — `memory_lifecycle_orchestrator`, `agent_coordinator_orchestrator`
- **Intent handlers** — `handler_intent`, `handler_subscription` with protocol-driven adapters
- **Protocol interfaces** — embedding provider, intent graph adapter, secrets provider
- **Audit layer** — I/O audit logging via `audit/`
- **Runtime plugin** — registered as `onex.domain_plugins` entry point (`PluginMemory`)

## Infrastructure Ownership

OmniMemory's `docker-compose.yml` owns the **memory-layer data services**. These are the services you need to run omnimemory locally:

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

If you need Kafka or Postgres, start the `omnibase_infra` stack first:
```bash
docker compose -f /path/to/omnibase_infra/docker/docker-compose.infra.yml up -d
```

## Quick Start

### Memory services only

To run just the omnimemory data services (Qdrant, Memgraph, Valkey, Kreuzberg):

```bash
git clone https://github.com/OmniNode-ai/omnimemory.git
cd omnimemory

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

### Install and run tests

```bash
uv sync
uv run pytest tests/ -m unit
```

For configuration options see [docs/environment_variables.md](docs/environment_variables.md).

Minimal example using the intent handler:
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

    # Store an intent
    result = await handler.store_intent(
        session_id="session_123",
        intent_data=ModelIntentClassificationOutput(
            intent_category="debugging",
            confidence=0.92,
            keywords=["error", "traceback"],
        ),
        correlation_id=str(uuid4()),
    )

    # Query session intents
    query_result = await handler.query_session(
        session_id="session_123",
        min_confidence=0.5,
    )

    await handler.shutdown()


asyncio.run(main())
```

## Directory Structure

```text
src/omnimemory/
├── audit/              # I/O audit logging
├── enums/              # Domain enumerations (memory types, operation types, lifecycle states)
├── errors/             # Structured error types
├── handlers/           # HandlerIntent, HandlerSubscription + adapters
├── models/             # Pydantic models (core, memory, intelligence, service, container, contracts)
├── nodes/              # EFFECT, COMPUTE, REDUCER, ORCHESTRATOR node implementations
├── protocols/          # Protocol interfaces (embedding, intent graph, secrets)
├── runtime/            # Plugin registration, wiring, dispatch, introspection
├── tools/              # Contract linter and stubs
└── utils/              # Shared utilities (audit logger, PII detection, retry, health)
```

## Development

Uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync
uv run pytest tests/ -m unit
uv run mypy src/omnimemory/ --strict
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Documentation

**Reference**: [docs/](docs/)

Open an issue or email contact@omninode.ai.
