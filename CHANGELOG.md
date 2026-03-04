# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-03-04

### Dependencies
- `omnibase-core` bumped to >=0.23.0,<0.24.0 (was >=0.22.0,<0.23.0) (OMN-3565)
- `omnibase-infra` bumped to >=0.15.0,<0.16.0 (was >=0.14.0,<0.15.0) (OMN-3565)

## [0.6.1] - 2026-02-28

### Added
- Contract topic validation tests for `intent_query_effect` (OMN-1538, #95)
- AI-slop checker Phase 2 rollout (#97)

### Changed
- Replace omninode_bridge db references with omnimemory in docs (#96)
- Replace Step N narration with intent comments in handler docs (#98)

### Dependencies
- `omnibase-core` bumped to >=0.22.0,<0.23.0 (was ==0.21.0)
- `omnibase-spi` bumped to >=0.15.0,<0.16.0 (was ==0.14.0)
- `omnibase-infra` bumped to >=0.13.0,<0.14.0 (was >=0.11.0,<0.12.0)
- Removed `[tool.uv] override-dependencies` (no longer needed with omnibase-infra>=0.13.0)

## [0.6.0] - 2026-02-27

### Changed
- Version bump as part of coordinated OmniNode platform release (release-20260227-eceed7)

### Dependencies
- omnibase-spi pinned to 0.14.0
- omnibase-core pinned to 0.21.0

## [0.4.0] - 2026-02-24

### Added
- MIT LICENSE and SPDX copyright headers
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- GitHub issue templates and PR template
- `.github/dependabot.yml` for automated dependency updates
- `no-internal-ips` pre-commit hook for CI enforcement

### Changed
- Bumped `omnibase-core` to 0.19.0, `omnibase-spi` to 0.12.0, `omnibase-infra` to 0.10.0
- Replaced hardcoded internal IP addresses with environment variable defaults
- Standardized pre-commit hook IDs to canonical names (`mypy-type-check`, `pyright-type-check`, `onex-validate-naming`, `onex-validate-clean-root`)

### Fixed
- Documentation cleanup: removed internal references, updated Quick Start section with `uv` commands
- SPDX headers applied to all source files

## [0.3.0] - 2026-02-19

### Added

- Add `event_bus` topic declarations to all effect node contracts ([OMN-2212](https://linear.app/omninode/issue/OMN-2212), [#37](https://github.com/OmniNode-ai/omnimemory/pull/37))
- Add contract-driven topic discovery for runtime event-bus wiring ([OMN-2213](https://linear.app/omninode/issue/OMN-2213), [#38](https://github.com/OmniNode-ai/omnimemory/pull/38))
- Migrate omnimemory to standard `event_bus.subscribe_topics` contract field ([OMN-1746](https://linear.app/omninode/issue/OMN-1746), [#39](https://github.com/OmniNode-ai/omnimemory/pull/39))
- ARCH-002 compliance — abstract Kafka from handlers behind `ProtocolEventBusPublish` ([OMN-2214](https://linear.app/omninode/issue/OMN-2214), [#40](https://github.com/OmniNode-ai/omnimemory/pull/40))
- Add `MessageDispatchEngine` integration for handler-level event dispatch ([OMN-2215](https://linear.app/omninode/issue/OMN-2215), [#41](https://github.com/OmniNode-ai/omnimemory/pull/41))
- Add CI Kafka import lint guard to enforce ARCH-002 (no direct `kafka` imports in `src/omnimemory/nodes/`) ([OMN-1750](https://linear.app/omninode/issue/OMN-1750), [#42](https://github.com/OmniNode-ai/omnimemory/pull/42))
- Add `PluginMemory` runtime plugin for memory subsystem registration ([OMN-2216](https://linear.app/omninode/issue/OMN-2216), [#43](https://github.com/OmniNode-ai/omnimemory/pull/43))
- Make `AdapterIntentGraph` conform to `ProtocolIntentGraph` from omnibase-spi ([OMN-1476](https://linear.app/omninode/issue/OMN-1476), [#44](https://github.com/OmniNode-ai/omnimemory/pull/44))
- Add wire model registration and entry point declaration for plugin discovery ([OMN-2217](https://linear.app/omninode/issue/OMN-2217), [#45](https://github.com/OmniNode-ai/omnimemory/pull/45))
- Add CI infrastructure alignment tooling for cross-repo consistency checks ([OMN-2218](https://linear.app/omninode/issue/OMN-2218), [#46](https://github.com/OmniNode-ai/omnimemory/pull/46))
- Apply required status checks to branch protection rules ([OMN-2186](https://linear.app/omninode/issue/OMN-2186), [#47](https://github.com/OmniNode-ai/omnimemory/pull/47))
- Add `safe_db_url_display` utility for masking credentials in logs and diagnostics ([OMN-2220](https://linear.app/omninode/issue/OMN-2220), [#50](https://github.com/OmniNode-ai/omnimemory/pull/50))

### Changed

- Switch database connection to `OMNIMEMORY_DB_URL` environment variable ([OMN-2060](https://linear.app/omninode/issue/OMN-2060), [#49](https://github.com/OmniNode-ai/omnimemory/pull/49))
- Extract shared CLAUDE.md rules into `~/.claude/CLAUDE.md` and replace local copy with a reference ([OMN-2164](https://linear.app/omninode/issue/OMN-2164), [#51](https://github.com/OmniNode-ai/omnimemory/pull/51))

### Breaking Changes

- **`frozen=True` on boundary-crossing models** ([OMN-2219](https://linear.app/omninode/issue/OMN-2219), [#48](https://github.com/OmniNode-ai/omnimemory/pull/48)): All Pydantic models that cross subsystem boundaries now have `frozen=True`. Any code that mutates these models after construction will raise a `ValidationError`. Callers must construct new instances instead of modifying existing ones.
- **`OMNIMEMORY_DB_URL` replaces previous database URL configuration** ([OMN-2060](https://linear.app/omninode/issue/OMN-2060), [#49](https://github.com/OmniNode-ai/omnimemory/pull/49)): The database connection is now read exclusively from `OMNIMEMORY_DB_URL`. Update your `.env` file accordingly (e.g. `OMNIMEMORY_DB_URL=postgresql://user:pass@host:port/db`).

## [0.2.0] - 2026-02-13

### Changed

- **omnibase-core**: `^0.13.1` -> `^0.17.0`
- **omnibase-spi**: `^0.6.4` -> `^0.8.0`
- **omnibase-infra**: `^0.3.2` -> `^0.7.0`

### Breaking Changes (from upstream)

- **ModelBaseError removed** (core 0.17.0): Replaced with `ModelErrorDetails` from
  `omnibase_core.models.core.model_error_details`. The new model uses `error_message`,
  `error_code`, and `error_type` fields instead of `message`, `code`, and `details`.
- **Realm-agnostic event topics** (infra 0.4.0+): Event topic prefixes changed from
  `dev.` to `onex.` (e.g., `onex.omnimemory.intent.stored.v1`). Realm isolation is
  now handled by the envelope identity, not topic names.

### Migration Notes

- `ModelBaseError(message=..., code=..., details=...)` ->
  `ModelErrorDetails(error_message=..., error_code=..., error_type=..., component=...)`
- Event topic assertions updated from `dev.omnimemory.*` to `onex.omnimemory.*`

### New Capabilities Available (upstream)

These features are now available from the updated dependencies:

- **Cryptography module** (core 0.17.0): Blake3 hashing and Ed25519 signing
- **Message envelope system** (core 0.17.0): `ModelMessageEnvelope`, `ModelEmitterIdentity`
- **Agent definition models** (core 0.17.0): `ModelAgentDefinition` for agent YAML validation
- **Canonical status enums** (core 0.17.0): `EnumExecutionStatus`, `EnumOperationStatus`,
  `EnumWorkflowStatus`, `EnumHealthStatus`
- **Intelligence protocols** (spi 0.8.0): `ProtocolIntentClassifier`, `ProtocolIntentGraph`,
  `ProtocolPatternExtractor`
- **Pipeline contracts** (spi 0.8.0): 14 models for workflow execution
- **Validation contracts** (spi 0.8.0): 6 models for validation runs
- **Measurement contracts** (spi 0.8.0): 8 models for quality metrics
- **Event ledger sinks** (infra 0.7.0): File-based and in-memory event ledgers
- **Validation framework** (infra 0.7.0): ONEX compliance validators
- **Gateway module** (infra 0.7.0): API gateway infrastructure
- **Slack integration** (infra 0.4.0): Webhook handler with Block Kit formatting

## [0.1.0] - 2025-09-13

### Added

- Initial release of OmniMemory
- 26 Pydantic models with zero `Any` types
- ONEX 4-node architecture (Effect, Compute, Reducer, Orchestrator)
- Protocol definitions for memory operations
- Core 8 node implementations
- Vector, graph, and relational memory storage adapters
- Intent event consumer for Kafka event ingestion
- Memory lifecycle orchestration (expire, archive, tick)
- Semantic analysis compute node
- Similarity compute node
