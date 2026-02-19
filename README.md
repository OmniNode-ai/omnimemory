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

- **Memory nodes** — `memory_storage_effect`, `memory_retrieval_effect`, `intent_storage_effect`, `intent_query_effect`
- **Compute nodes** — `semantic_analyzer_compute`, `similarity_compute`
- **Reducer nodes** — `memory_consolidator_reducer`, `statistics_reducer`
- **Orchestrator nodes** — `memory_lifecycle_orchestrator`, `agent_coordinator_orchestrator`
- **Intent handlers** — `handler_intent`, `handler_subscription` with protocol-driven adapters
- **Protocol interfaces** — embedding provider, intent graph adapter, secrets provider
- **Audit layer** — I/O audit logging via `audit/`
- **Runtime plugin** — registered as `onex.domain_plugins` entry point (`PluginMemory`)

## Quick Start

Install:
```bash
poetry add omnimemory
```

Minimal example using the intent handler:
```python
from omnimemory.handlers.handler_intent import HandlerIntent
from omnimemory.models.core.model_memory_operation import ModelMemoryOperation

handler = HandlerIntent(container=container)
result = await handler.handle(ModelMemoryOperation(...))
```

Run tests:
```bash
poetry run pytest
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

Uses [Poetry](https://python-poetry.org/) for package management.

```bash
poetry install
poetry run pytest tests/
poetry run mypy src/omnimemory/
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

## Documentation

**Reference**: [docs/](docs/)
