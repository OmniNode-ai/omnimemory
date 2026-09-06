<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/omninode-inline-white.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/omninode-inline-full-color.svg">
    <img alt="omninode" src="docs/assets/brand/omninode-inline-full-color.svg" width="420">
  </picture>
</p>

# OmniMemory

[![CI](https://github.com/OmniNode-ai/omnimemory/actions/workflows/ci.yml/badge.svg)](https://github.com/OmniNode-ai/omnimemory/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONEX 4.0](https://img.shields.io/badge/ONEX-4.0-purple.svg)](https://github.com/OmniNode-ai/omnibase_core)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Memory persistence, recall, and semantic retrieval for the OmniNode platform.** OmniMemory provides ONEX (OmniNode eXecution)-compliant nodes and handlers for storing agent context, indexing embeddings, querying intent graphs, and managing the full memory lifecycle across distributed omni agents.

## Documentation

**Every OmniMemory document lives in the OmniNode knowledge base, not in this repository.** This README is the landing page and the full index; there are no `docs/` pages here to read.

- **[OmniNode knowledge base](https://github.com/OmniNode-ai/knowledge-base)** — public documentation home.

**Architecture**
- [ONEX Four-Node Architecture](https://github.com/OmniNode-ai/knowledge-base/blob/main/architecture/omnimemory-four-node-architecture.md) — the EFFECT / COMPUTE / REDUCER / ORCHESTRATOR archetypes applied to memory
- [ARCH-002: Kafka Abstraction Rule](https://github.com/OmniNode-ai/knowledge-base/blob/main/architecture/omnimemory-arch-002-kafka-abstraction.md) — why nodes never speak to a broker directly

**Reference**
- [Environment Variables](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-environment-variables.md) — every setting, type, default and constraint
- [Memory Data Ownership](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-memory-data-ownership.md) — which repository owns which storage service
- [Runtime Plugin System](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-runtime-plugins.md) — how `PluginMemory` wires into the ONEX kernel
- [Handler Reuse Matrix](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-handler-reuse-matrix.md) — which `omnibase_infra` handler each memory node reuses

**Guides**
- [PII Handling](https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-pii-handling.md) — detection, sanitization and storage-path integration
- [Performance Testing](https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-performance-testing.md) — SLA targets, benchmarks and how to read them
- [Market Migration Boundary](https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-market-migration-boundary.md) — what moves to `omnimarket` and what stays

**Runbooks**
- [Starting OmniMemory Services](https://github.com/OmniNode-ai/knowledge-base/blob/main/runbooks/omnimemory-starting-memory-services.md) — bringing the storage layer up, health checks, troubleshooting

Only this README, [CLAUDE.md](CLAUDE.md), [CHANGELOG.md](CHANGELOG.md), [LICENSE](LICENSE), [SECURITY.md](SECURITY.md) and the `.claude/` and `.github/` trees carry markdown in this repository. The `kb-doc-gate` required check runs in `strict` mode (see [`.kb-doc-gate.yaml`](.kb-doc-gate.yaml)) and fails any PR that reintroduces documentation here.

---

## What This Repo Owns

- **Domain models** — memory, crawl, persona, intent, and intelligence Pydantic models in `src/omnimemory/models/`
- **Protocol interfaces** — `ProtocolEmbeddingClient`, `ProtocolEmbeddingProvider`, `ProtocolIntentGraphAdapter`, `ProtocolSecretsProvider` and the base protocols in `src/omnimemory/protocols/`
- **Storage adapters** — Qdrant, Memgraph, Valkey and filesystem implementations in `handlers/adapters/` and `nodes/*/adapters/`
- **Runtime plugin** — `PluginMemory` in `src/omnimemory/runtime/`, registered as `onex.domain_plugins`
- **Memory-layer data services** — Qdrant, Memgraph, Valkey, Kreuzberg (owned via `docker-compose.yml`)
- **Node handlers** — contract-carrying nodes in `src/omnimemory/nodes/`, migrating to `omnimarket`

Not owned here: Kafka/Redpanda and PostgreSQL ([`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra)), the ONEX kernel and contracts ([`omnibase_core`](https://github.com/OmniNode-ai/omnibase_core)), platform-boundary protocols ([`omnibase_spi`](https://github.com/OmniNode-ai/omnibase_spi)), the post-migration node runtime ([`omnimarket`](https://github.com/OmniNode-ai/omnimarket)), and dashboard read models ([`omnidash`](https://github.com/OmniNode-ai/omnidash)). See [Memory Data Ownership](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-memory-data-ownership.md) for the full boundary table, and [Market Migration Boundary](https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-market-migration-boundary.md) for what is moving.

---

## Quick Start

```bash
git clone https://github.com/OmniNode-ai/omnimemory.git
cd omnimemory

# Start platform infra first (Kafka + PostgreSQL — owned by omnibase_infra)
infra-up

# Start the memory data services (Qdrant 6333, Memgraph 7687, Valkey 6379, Kreuzberg 8090)
docker compose up -d
docker compose ps

# Install and run the fast test suite
uv sync --group dev
uv run pytest tests/ -m unit
```

Ports and every other setting are configurable — see [Environment Variables](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-environment-variables.md). The full startup runbook, with health checks and troubleshooting, is [Starting OmniMemory Services](https://github.com/OmniNode-ai/knowledge-base/blob/main/runbooks/omnimemory-starting-memory-services.md).

## Development

```bash
uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/
uv run mypy src/omnimemory/ --strict
uv run pytest tests/ -v
pre-commit run --all-files
```

Repo invariants, enforced conventions and the pre-commit/pre-push split are in [CLAUDE.md](CLAUDE.md).

---

## Security, Contributing, and License

- [SECURITY.md](SECURITY.md)
- [CONTRIBUTING.md](.github/CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)
- [LICENSE](LICENSE)

Open an issue or email <contact@omninode.ai>.
