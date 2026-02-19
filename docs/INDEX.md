> **Navigation**: Home (You are here)

# OmniMemory Documentation

Welcome to the OmniMemory documentation. This is the navigation hub — all documentation starts here.

## Documentation Authority Model

| Source | Authority | Contains |
|--------|-----------|----------|
| **[CLAUDE.md](../CLAUDE.md)** | **Hard constraints** | Invariants, forbidden patterns, zero-backwards-compat policy, quick reference |
| **docs/** | **Explanations** | Architecture, guides, conventions, reference, ADRs |
| **[README.md](../README.md)** | **First contact** | Elevator pitch, quick start, project overview |

**When in conflict, CLAUDE.md takes precedence.** The docs directory provides depth and context; CLAUDE.md provides the enforceable rules.

**Quick Reference:**
- Need a rule or constraint? Check [CLAUDE.md](../CLAUDE.md)
- Need an explanation or deep dive? Check [docs/](.)
- Need environment setup? Check [environment_variables.md](environment_variables.md)

---

## Quick Navigation

| I want to... | Go to |
|---|---|
| Understand the memory architecture | [ONEX Four-Node Architecture](architecture/ONEX_FOUR_NODE_ARCHITECTURE.md) |
| Understand Kafka/event bus integration | [ARCH-002 Kafka Abstraction](architecture/ARCH_002_KAFKA_ABSTRACTION.md) |
| Set up environment variables | [Environment Variables](environment_variables.md) |
| Understand PII detection and privacy | [PII Handling Guide](pii_handling.md) |
| Run performance benchmarks | [Performance Testing Guide](PERFORMANCE_TESTING.md) |
| Work with runtime plugins | [Runtime Plugins](runtime/RUNTIME_PLUGINS.md) |
| Run CI locally or debug CI failures | [CI Monitoring Guide](ci/CI_MONITORING_GUIDE.md) |
| Find handler reuse opportunities | [Handler Reuse Matrix](handler_reuse_matrix.md) |
| Understand stub protocols and compat layer | [Stub Protocols](stub_protocols.md) |

---

## Documentation Structure

### Architecture

System design, data flow, and architectural decisions.

| Document | Description |
|---|---|
| [ONEX Four-Node Architecture](architecture/ONEX_FOUR_NODE_ARCHITECTURE.md) | EFFECT, COMPUTE, REDUCER, ORCHESTRATOR archetypes in OmniMemory |
| [ARCH-002 Kafka Abstraction](architecture/ARCH_002_KAFKA_ABSTRACTION.md) | Kafka/Redpanda event bus integration and abstraction layer |

### CI

Continuous integration monitoring, failure analysis, and tooling.

| Document | Description |
|---|---|
| [CI Monitoring Guide](ci/CI_MONITORING_GUIDE.md) | CI performance monitoring, failure triage, and local reproduction |

### Runtime

Runtime plugin system and extension points.

| Document | Description |
|---|---|
| [Runtime Plugins](runtime/RUNTIME_PLUGINS.md) | Plugin architecture, registration, and lifecycle management |

### Reference

Configuration, environment, and operational reference.

| Document | Description |
|---|---|
| [Environment Variables](environment_variables.md) | All environment variables for configuring OmniMemory |
| [Handler Reuse Matrix](handler_reuse_matrix.md) | Maps `omnibase_infra` handlers to Core 8 memory nodes |
| [Performance Testing Guide](PERFORMANCE_TESTING.md) | Running and interpreting OmniMemory performance benchmarks |
| [PII Handling Guide](pii_handling.md) | PII detection system, privacy compliance, and data security |
| [Stub Protocols](stub_protocols.md) | Compatibility layer stubs and their migration path to `omnibase_core` |

### DB Split (Audit Archive)

Point-in-time audit records from the OMN-2054 database split work. These are historical records, not living documentation.

| Document | Description |
|---|---|
| [FK Audit: OMN-2069](db-split/fk-audit-omn-2069.md) | Foreign key audit of migration files (2026-02-10) |

---

## Document Status

| Section | Status | Coverage |
|---|---|---|
| Architecture | Initial | 2 of ~15 needed |
| CI | Initial | 1 of ~5 needed |
| Runtime | Initial | 1 of ~3 needed |
| Reference | Partial | 5 of ~10 needed |
| DB Split | Archive | Complete (point-in-time audit) |
| Getting Started | Not started | 0 of ~3 needed |
| Guides | Not started | 0 of ~8 needed |
| Patterns | Not started | 0 of ~5 needed |
| Decisions (ADRs) | Not started | 0 of ~10 needed |
