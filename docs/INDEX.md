> **Navigation**: Home (You are here)
>
> _Verified against code on 2026-06-21: node inventory (16 dirs, 15 with `contract.yaml`), runtime plugin entry point, protocol names, and storage-service ownership confirmed against `src/omnimemory/`._

# OmniMemory Documentation

Welcome to the OmniMemory documentation. This is the navigation hub — all documentation starts here.

## Documentation Authority Model

| Source | Authority | Contains |
|--------|-----------|----------|
| **[CLAUDE.md](../CLAUDE.md)** | **Hard constraints** | Invariants, forbidden patterns, zero-backwards-compat policy, quick reference |
| **docs/** | **Explanations** | Architecture, guides, conventions, reference, migrations, runbooks |
| **[README.md](../README.md)** | **First contact** | Elevator pitch, quick start, ownership summary, project overview |

**When in conflict, CLAUDE.md takes precedence.** The docs directory provides depth and context; CLAUDE.md provides the enforceable rules.

**Quick Reference:**
- Need a rule or constraint? Check [CLAUDE.md](../CLAUDE.md)
- Need an explanation or deep dive? Check [docs/](.)
- Need environment setup? Check [environment_variables.md](environment_variables.md)
- Need to start services? Check [runbooks/STARTING_MEMORY_SERVICES.md](runbooks/STARTING_MEMORY_SERVICES.md)

---

## Start Here

| I want to... | Go to |
|---|---|
| Understand what omnimemory owns | [README.md — Where This Fits](../README.md#where-this-fits) |
| Start memory services locally | [Runbook: Starting Memory Services](runbooks/STARTING_MEMORY_SERVICES.md) |
| Understand the memory architecture | [ONEX Four-Node Architecture](architecture/ONEX_FOUR_NODE_ARCHITECTURE.md) |
| Understand storage backend ownership | [Memory Data Ownership](architecture/MEMORY_DATA_OWNERSHIP.md) |
| Understand what moves to omnimarket | [Market Migration Boundary](migrations/MARKET_MIGRATION_BOUNDARY.md) |
| Understand Kafka/event bus integration | [ARCH-002 Kafka Abstraction](architecture/ARCH_002_KAFKA_ABSTRACTION.md) |
| Set up environment variables | [Environment Variables](environment_variables.md) |
| Understand PII detection and privacy | [PII Handling Guide](pii_handling.md) |
| Run performance benchmarks | [Performance Testing Guide](PERFORMANCE_TESTING.md) |
| Work with runtime plugins | [Runtime Plugins](runtime/RUNTIME_PLUGINS.md) |
| Run CI locally or debug CI failures | [CI Monitoring Guide](ci/CI_MONITORING_GUIDE.md) |
| Find handler reuse opportunities | [Handler Reuse Matrix](handler_reuse_matrix.md) |
| Understand stub protocols and compat layer | [Stub Protocols](stub_protocols.md) |

---

## Current Architecture

System design, data flow, ownership boundaries, and architectural decisions.

| Document | Description |
|---|---|
| [ONEX Four-Node Architecture](architecture/ONEX_FOUR_NODE_ARCHITECTURE.md) | EFFECT, COMPUTE, REDUCER, ORCHESTRATOR archetypes in OmniMemory |
| [ARCH-002 Kafka Abstraction](architecture/ARCH_002_KAFKA_ABSTRACTION.md) | Kafka/Redpanda event bus integration and abstraction layer |
| [Memory Data Ownership](architecture/MEMORY_DATA_OWNERSHIP.md) | Qdrant, Memgraph, Valkey, Kreuzberg ownership and infra boundary |

---

## Migrations

Migration context, boundary definitions, and durable upgrade guidance.

| Document | Description |
|---|---|
| [Market Migration Boundary](migrations/MARKET_MIGRATION_BOUNDARY.md) | What moves to omnimarket, what stays, wave plan, import path changes |

---

## Runbooks

Current operational procedures with commands and expected evidence.

| Document | Description |
|---|---|
| [Starting Memory Services](runbooks/STARTING_MEMORY_SERVICES.md) | Start Qdrant, Memgraph, Valkey, Kreuzberg; health checks; stop procedure |

---

## Reference

Stable configuration, environment, and operational reference.

| Document | Description |
|---|---|
| [Environment Variables](environment_variables.md) | All environment variables for configuring OmniMemory |
| [Handler Reuse Matrix](handler_reuse_matrix.md) | Maps `omnibase_infra` handlers to Core 8 memory nodes |
| [Performance Testing Guide](PERFORMANCE_TESTING.md) | Running and interpreting OmniMemory performance benchmarks |
| [PII Handling Guide](pii_handling.md) | PII detection system, privacy compliance, and data security |
| [Stub Protocols](stub_protocols.md) | Compatibility layer stubs and their migration path to `omnibase_core` |

---

## Runtime

Runtime plugin system and extension points.

| Document | Description |
|---|---|
| [Runtime Plugins](runtime/RUNTIME_PLUGINS.md) | Plugin architecture, registration, and lifecycle management |

---

## CI

Continuous integration monitoring, failure analysis, and tooling.

| Document | Description |
|---|---|
| [CI Monitoring Guide](ci/CI_MONITORING_GUIDE.md) | CI performance monitoring, failure triage, and local reproduction |

---

## Historical Context

Point-in-time audit records and historical design context. These are not current source of truth.

| Document | Description |
|---|---|
| [FK Audit](db-split/fk-audit.md) | Foreign key audit of migration files (2026-02-10, DB split) |

**Architecture plans (context only, not promoted):**
- `omni_home/docs/plans/2026-04-07-plan-omnimemory-architecture.md` — memory evolution design (surprise gating, activation decay, memory cubes, Hebbian associations). Planned, not yet implemented.
- `omni_home/docs/plans/2026-04-10-omnimemory-to-omnimarket-migration.md` — node migration plan. Promoted summary: [Market Migration Boundary](migrations/MARKET_MIGRATION_BOUNDARY.md).

---

## Document Status

| Section | Status | Coverage |
|---|---|---|
| Architecture | Partial | 3 docs: Four-Node, Kafka, Data Ownership |
| Migrations | Initial | 1 doc: Market Migration Boundary |
| Runbooks | Initial | 1 doc: Starting Memory Services |
| Reference | Partial | 5 docs |
| Runtime | Initial | 1 doc |
| CI | Initial | 1 doc |
| Historical | Archive | DB split audit |
| Getting Started | Not started | 0 of ~3 needed |
| Guides | Not started | 0 of ~8 needed |
| Decisions (ADRs) | Not started | 0 of ~10 needed |
