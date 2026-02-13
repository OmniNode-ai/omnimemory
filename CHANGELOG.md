# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
